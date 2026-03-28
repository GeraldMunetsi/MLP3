"""
step1_generate_data_SIR3param.py
=================================
Adaptive Sobol data generation for the 3-parameter SIR emulator.

Parameters
----------
    tau   (τ)   per-contact transmission rate
    gamma (γ)   recovery rate
    rho   (ρ)   initial infected seed fraction

Epidemic threshold (Barabasi-Albert network)
---------------------------------------------
    R₀ = τ/γ × ⟨k²⟩/⟨k⟩

Output pickle format (compatible with utils_SIR.py)
----------------------------------------------------
    sim['params'] = {'tau': float, 'gamma': float, 'rho': float}
    sim['output'] = {'t': array, 'S': array, 'I': array, 'R': array, ...}
"""

import numpy as np
import networkx as nx
import EoN
import pickle
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from pathlib import Path
import argparse
from tqdm import tqdm


# GLOBAL CONSTANTS
n_timepoints=400
N = 10000   # network size (nodes)
m =5      # Barabasi-Albert attachment parameter
PARAM_NAMES = ['tau', 'gamma', 'rho']


PARAM_RANGES = {
    'tau'  : (0.0024, 0.017),  # Expected range: R₀ ∈ [0.12, 4.98] #   recovery rate
    'gamma': (0.07,  0.5),  # Infectious period 2-10 days
    'rho'  : (0.001, 0.010),
},


# BA NETWORK STATISTICS  (computed once, cached)

_NETWORK_STATS_CACHE = {}


def get_ba_network_stats(N=N, m=m, seed=42):
    """
    Compute and cache BA-network degree statistics needed for R0.

    R0 formula for heterogeneous networks (configuration model):

        R0 = tau/gamma x <k2>/<k>

    The ratio <k2>/<k> is the degree-heterogeneity correction factor.
    Hub nodes in a BA graph dramatically lower the epidemic threshold
    compared to a homogeneous (Erdos-Renyi) network.

    Results are cached so the network is built only once per run.
    """
    cache_key = (N, m, seed)

    if cache_key not in _NETWORK_STATS_CACHE:
        print(f"\n  Computing BA network statistics (N={N:,}, m={m}) ...")
        G       = nx.barabasi_albert_graph(N, m, seed=seed)
        degrees = np.array([d for _, d in G.degree()])

        k_avg  = degrees.mean()
        k2_avg = (degrees ** 2).mean()
        ratio  = k2_avg / k_avg          # <k2>/<k>

        _NETWORK_STATS_CACHE[cache_key] = {
            'k_avg' : k_avg,
            'k2_avg': k2_avg,
            'ratio' : ratio,
            'k_std' : degrees.std(),
            'k_max' : degrees.max(),
        }

        print(f"    <k>       = {k_avg:.2f}")
        print(f"    <k2>      = {k2_avg:.2f}")
        print(f"    <k2>/<k>  = {ratio:.2f}   (R0 multiplier)")
        print(f"    sigma_k   = {degrees.std():.2f}")
        print(f"    k_max     = {degrees.max()}")

    return _NETWORK_STATS_CACHE[cache_key]



# SOBOL SAMPLING  (d = 3)


def generate_sobol_samples(n_samples, seed=42, scramble=True):
    """
    Generate Sobol quasi-random samples over the (tau, gamma, rho) space.

    Why Sobol?
    ----------
    Sobol sequences are 'low-discrepancy': they fill 3-D space more
    evenly than plain random (Monte Carlo) sampling. This matters because
    near the epidemic threshold R0 = 1 we need dense, uniform coverage
    to capture bifurcation behaviour, and we want to do so with as few
    expensive simulations as possible.

    Args:
        n_samples : number of parameter sets to generate
        seed      : scrambling seed (for reproducibility)
        scramble  : recommended True — removes inter-dimension correlations

    Returns:
        samples : [n_samples, 3]   each row = [tau, gamma, rho]
    """
    print(f"\nGenerating {n_samples} Sobol samples  (d=3: tau, gamma, rho) ...")

    sampler = qmc.Sobol(d=3, scramble=scramble, seed=seed)

    # Sobol quality is best at exact powers of 2, Using powers of 2 preserves balance properties.
    
    n_pow2 = 2 ** int(np.ceil(np.log2(max(n_samples, 2))))    

    if n_pow2 > n_samples:
        print(f"  Generating {n_pow2} (next power of 2), selecting {n_samples}")
        samples_unit = sampler.random(n=n_pow2)
        indices      = np.linspace(0, n_pow2 - 1, n_samples, dtype=int)
        samples_unit = samples_unit[indices]
    else:
        samples_unit = sampler.random(n=n_samples)

    # Scale unit-cube [0,1]^3 to physical parameter ranges
    samples = np.zeros_like(samples_unit)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi       = PARAM_RANGES[name]
        samples[:, i] = samples_unit[:, i] * (hi - lo) + lo

    discrepancy = qmc.discrepancy(samples_unit)
    print(f"  Star discrepancy = {discrepancy:.6f}  (lower = better coverage)")

    return samples


def _sobol_candidates(n_candidates=4098, seed=None):
    """Generate a Sobol candidate pool for adaptive selection (internal)."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    n_pow2  = 2 ** int(np.ceil(np.log2(max(n_candidates, 2))))
    unit    = sampler.random(n=n_pow2)

    if n_pow2 > n_candidates:
        idx  = np.linspace(0, n_pow2 - 1, n_candidates, dtype=int)
        unit = unit[idx]

    candidates =np.zeros_like(unit)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi            = PARAM_RANGES[name]
        candidates[:, i]  = unit[:, i] * (hi - lo) + lo

    return candidates



# ADAPTIVE SELECTION  (R0-aware)


def estimate_errors(simulations, N=N, m=m):
    """
    Score each Sobol candidate by expected prediction error.

    Score = (coverage gap) x (1 + threshold proximity)

    Coverage gap:
        Normalised min-distance from the candidate to all existing
        parameter sets. Large gap = sparse region = high priority.

    Threshold proximity:
        Gaussian weight centred at R0 = 1. Near the bifurcation point
        stochastic variability is highest and prediction is hardest.

    R0 for each candidate:
        R0 = tau/gamma x <k2>/<k>

    Returns:
        candidates   : [n_candidates, 3]
        error_scores : [n_candidates]    (higher = higher priority)
    """
    existing = np.array([[s['params'][n] for n in PARAM_NAMES]
                         for s in simulations])

    candidates = _sobol_candidates()

    # Normalise by range width so all three dimensions contribute equally
    scales          = np.array([PARAM_RANGES[n][1] - PARAM_RANGES[n][0]
                                 for n in PARAM_NAMES])
    existing_norm   = existing    / scales
    candidates_norm = candidates  / scales

    min_distances = cdist(candidates_norm, existing_norm).min(axis=1)

    # R0 = tau/gamma x <k2>/<k>, R₀ derived from actual BA network structure
    net    = get_ba_network_stats(N=N, m=m)
    R0     = (candidates[:, 0] / candidates[:, 1]) * net['ratio']

    near   = (R0 >=0.8) & (R0 <= 1.2)
    print(f"  Candidate R0 range  : [{R0.min():.2f}, {R0.max():.2f}]")
    print(f"  Near R0=1 (0.8-1.2) : {near.sum()} ({100*near.mean():.1f}%)")

    # Gaussian weight — peak at R0=1, decays away from threshold
    threshold_weight = np.exp(-8* (R0 - 1) ** 2)

    # Final score: far-from-existing AND near-threshold = highest priority
    scores = min_distances * (1 + threshold_weight)

    return candidates, scores


def select_next_samples(existing_sims, n_new, percentile=75, N=N, m=m):
    """
    Select the next batch using maximin adaptive sampling.

    Algorithm:
        1. Score all Sobol candidates with estimate_errors()
        2. Keep the top (100-percentile)% — the 'high-error pool'
        3. From the pool, greedily pick n_new points that are farthest
           from all existing + already-selected points (maximin)

    Maximin criterion ensures the new points are spread out within the
    high-error region rather than clustering together.

    Args:
        existing_sims : list of simulation dicts already completed
        n_new         : how many new parameter sets to add
        percentile    : score threshold (75 = top 25% kept)

    Returns:
        selected : [n_new, 3]   array of [tau, gamma, rho]
    """
    print(f"\n  Selecting {n_new} new samples (R0-aware adaptive) ...")

    candidates, scores = estimate_errors(existing_sims, N, m)

    pool = candidates[scores >= np.percentile(scores, percentile)]
    print(f"  High-error pool (top {100-percentile}%): {len(pool)} candidates")

    if len(pool) < n_new * 2:
        print("  Pool too small — relaxing threshold to 50th percentile ...")
        pool = candidates[scores >= np.percentile(scores, 50)]

    existing = np.array([[s['params'][n] for n in PARAM_NAMES]
                          for s in existing_sims])

    # Maximin greedy selection
    selected  = []
    remaining = pool.copy()

    for _ in range(n_new):
        if len(remaining) == 0:
            break

        reference = (np.vstack([existing] + [selected])
                     if selected else existing)

        min_dist  = cdist(remaining, reference).min(axis=1)
        best      = np.argmax(min_dist)

        selected.append(remaining[best])
        remaining = np.delete(remaining, best, axis=0)

    print(f"  Selected {len(selected)} new parameter sets")
    return np.array(selected)



# NETWORK GENERATION


def generate_network(N=N, m=m, seed=42):
    """Build the Barabasi-Albert network and warm the stats cache."""
    print(f"\nBuilding BA network  N={N:,}, m={m} ...")
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    get_ba_network_stats(N, m, seed)
    return G



# SIR SIMULATION


def run_sir_replicates(G, tau, gamma, rho,n_replicates=3, tmax=50, n_timepoints=n_timepoints):
    """
    Run n_replicates stochastic SIR simulations and average their outputs.

    Why average replicates?
    -----------------------git remote add origin https://github.com/GeraldMunetsi/MLP3.git
    Near R0 = 1 the SIR process is highly stochastic — two runs with
    identical parameters can produce very different I(t) curves due to
    random transmission events. Averaging n_replicates runs reduces this
    variance without requiring extra unique parameter sets, which are the
    expensive part of the study.

    EoN.fast_SIR implements Gillespie's exact stochastic algorithm on
    the network, so each replicate is a genuinely independent sample
    from the correct stochastic process.

    Args:
        G            : fixed NetworkX BA graph
        tau          : transmission rate
        gamma        : recovery rate
        rho          : initial seed fraction
        n_replicates : number of independent stochastic runs
        tmax         : simulation end time
        n_timepoints : resolution of the output time grid

    Returns:
        dict with t, S, I, R (means) and S_std, I_std, R_std
    """
    t_fixed = np.linspace(0, tmax, n_timepoints)

    try:
        S_runs, I_runs, R_runs = [], [], []

        for _ in range(n_replicates):
            t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho, tmax=tmax)
            S_runs.append(np.interp(t_fixed, t, S))
            I_runs.append(np.interp(t_fixed, t, I))
            R_runs.append(np.interp(t_fixed, t, R))

        return {
            't'           : t_fixed,
            'S'           : np.mean(S_runs, axis=0),
            'I'           : np.mean(I_runs, axis=0),
            'R'           : np.mean(R_runs, axis=0),
            'S_std'       : np.std(S_runs, axis=0),
            'I_std'       : np.std(I_runs, axis=0),
            'R_std'       : np.std(R_runs, axis=0),
            'n_replicates': n_replicates,
        }

    except Exception as e:
        print(f"    Warning: simulation failed — {e}")
        zeros = np.zeros(n_timepoints)
        return {
            't': t_fixed, 'S': zeros.copy(), 'I': zeros.copy(), 'R': zeros.copy(),
            'S_std': zeros.copy(), 'I_std': zeros.copy(), 'R_std': zeros.copy(),
            'n_replicates': 0,
        }


def run_batch(G, params_array, n_replicates=5, tmax=120, n_timepoints=n_timepoints):
    """
    Simulate a batch of parameter sets and return a list of result dicts.

    Output format matches utils_SIR.EpidemicDatasetSIR:
        {'params': {'tau': float, 'gamma': float, 'rho': float},
         'output': {'t': ..., 'S': ..., 'I': ..., 'R': ...}}

    Args:
        params_array : [B, 3]   rows = [tau, gamma, rho]
    """
    results = []

    for row in tqdm(params_array,
                    desc=f"  Simulating ({n_replicates} reps/set)"):
        tau, gamma, rho = float(row[0]), float(row[1]), float(row[2])

        output = run_sir_replicates(
            G, tau, gamma, rho,
            n_replicates=n_replicates,
            tmax=tmax,
            n_timepoints=n_timepoints,
        )

        results.append({
            'params': {'tau': tau, 'gamma': gamma, 'rho': rho},
            'output': output,
        })

    return results



# MAIN DATASET GENERATION LOOP


def generate_dataset(
    initial_samples=100,
    batch_size=30,
    n_rounds=5,
    n_replicates=5,
    N=N,
    m=m,
    tmax=20,
    n_timepoints=n_timepoints, #
):
    """
    Build the full training dataset via two-phase adaptive Sobol sampling.

    Phase 1 — Initial coverage
        Spread initial_samples points uniformly across the 3-D
        (tau, gamma, rho) cube using a Sobol sequence.

    Phase 2 — Adaptive refinement  (n_rounds iterations)
        Each round adds batch_size new points where:
            (a) coverage is sparse, OR
            (b) R0 is near 1  (bifurcation zone, highest uncertainty)
        Selection is driven by R0 = tau/gamma x <k2>/<k>.

    Final result:
        Broad uniform coverage + concentrated sampling near R0=1,
        achieved with the minimum number of expensive ABM simulations.
    """
    total = initial_samples + batch_size * n_rounds

  
    print("STEP 1 — SIR DATA GENERATION")
    print("3-Parameter model: tau, gamma, rho")
    print("R0 = tau/gamma x <k2>/<k>")

    print(f"\n  Parameter ranges (fixed in PARAM_RANGES):")
    for name in PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        print(f"    {name:5s}  in [{lo}, {hi}]")
    print(f"\n  Network      : BA(N={N:,}, m={m})")
    print(f"  Strategy     : Adaptive Sobol (d=3)")
    print(f"  Initial      : {initial_samples} samples")
    print(f"  Adaptive     : {n_rounds} rounds x {batch_size}")
    print(f"  Total sets   : {total}")
    print(f"  Replicates   : {n_replicates}/set  ->  "
          f"{total * n_replicates:,} individual runs")
    print(f"  Time grid    : tmax={tmax},  {n_timepoints} points")

    # Build the fixed network
    G = generate_network(N=N, m=m)

    # Phase 1 — Initial Sobol
  
    print(f"PHASE 1 — INITIAL SOBOL  ({initial_samples} samples)")
   

    all_sims = run_batch(
        G, generate_sobol_samples(initial_samples),
        n_replicates, tmax, n_timepoints
    )
    print(f"  Phase 1 done — {len(all_sims)} simulations")

    # Phase 2 — Adaptive rounds
    for rnd in range(1, n_rounds + 1):
        print("\n" + "=" * 70)
        print(f"PHASE 2 — ADAPTIVE ROUND {rnd}/{n_rounds}  (+{batch_size} samples)")
        print("=" * 70)

        new_params = select_next_samples(all_sims, n_new=batch_size, N=N, m=m)
        all_sims.extend(
            run_batch(G, new_params, n_replicates, tmax, n_timepoints)
        )
        print(f"  Round {rnd} done — total: {len(all_sims)} simulations")

    # Summary
    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)

    net = get_ba_network_stats(N, m)
    print(f"\n  Network: <k>={net['k_avg']:.2f}, "
          f"<k2>={net['k2_avg']:.2f}, "
          f"<k2>/<k>={net['ratio']:.2f}")

    tau_arr   = np.array([s['params']['tau']   for s in all_sims])
    gamma_arr = np.array([s['params']['gamma'] for s in all_sims])
    R0_arr    = (tau_arr / gamma_arr) * net['ratio']


    #This counts how many of your simulations fall near the epidemic threshold(R₀ ≈ 1).
    #Quality check , did my sampler work?  


    n_thresh = ((R0_arr > 0.8) & (R0_arr < 1.2)).sum() #
    print(f"\n  R0 distribution  (R0 = tau/gamma x <k2>/<k>):")
    print(f"    min  = {R0_arr.min():.2f}")
    print(f"    max  = {R0_arr.max():.2f}")
    print(f"    mean = {R0_arr.mean():.2f}")
    print(f"    Near R0=1 (0.8-1.2): {n_thresh} "
          f"({100*n_thresh/len(all_sims):.1f}%)  <- adaptive focus")

    return {
        'simulations': all_sims,
        'network': {
            'graph' : G,
            'N'     : N,
            'm'     : m,
            'type'  : 'barabasi_albert',
            'k_avg' : net['k_avg'],
            'k2_avg': net['k2_avg'],
            'ratio' : net['ratio'],
        },
        'metadata': {
            'n_samples'        : len(all_sims),
            'n_replicates'     : n_replicates,
            'has_std'          : True,
            'noise_reduction'  : 'averaged_replicates',
            'model_type'       : 'step0_model.py',
            'dimensionality'   : 3,
            'param_names'      : PARAM_NAMES,
            'param_ranges'     : PARAM_RANGES,
            'total_population' : N,
            'm'                : m,
            'tmax'             : tmax,
            'n_timepoints'     : n_timepoints,
            'sampling_strategy': 'adaptive_sobol',                                                                                             
            'r0_formula'       : 'R0 = tau/gamma * k2_avg/k_avg (BARABASI-ALBERT NETWORK)',
            'initial_samples'  : initial_samples,
            'batch_size'       : batch_size,
            'n_rounds'         : n_rounds,
        },
    }



# SAVE


def save_dataset(dataset, filepath):
    """Pickle the dataset to disk and print a summary."""
    filepath = Path(filepath)

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = filepath.stat().st_size / (1024 ** 2)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(f"  File       : {filepath}  ({size_mb:.2f} MB)")
    print(f"  Samples    : {dataset['metadata']['n_samples']}")
    print(f"  Parameters : {dataset['metadata']['param_names']}")
    print(f"  R0 formula : {dataset['metadata']['r0_formula']}")
    print(f"\n  Compatible with:")
    print(f"    utils_SIR.py              EpidemicDatasetSIR")
    print(f"    step2_split_data.py       train/val/test split")
    print(f"    step3_train_SIR3param.py  model training")



# ENTRY POINT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3-parameter SIR training data via adaptive Sobol sampling.\n"
            "R0 = tau/gamma x <k2>/<k>\n"
            "Parameter ranges are fixed in PARAM_RANGES at the top of the file."
        )
    )
    # Sampling strategy
    parser.add_argument('--initial_samples', type=int, default=500,
                        help='Initial Sobol samples (default: 500)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='New samples per adaptive round  (default: 32)')
    parser.add_argument('--n_rounds', type=int, default= 10,
                        help='Number of adaptive rounds       (default: 10)')
    parser.add_argument('--n_replicates',type=int, default=3,
                        help='Stochastic replicates per set   (default: 5)')
    # Network
    parser.add_argument('--N',type=int,  default=10000,
                        help='Network size (default: 10000)')
    parser.add_argument('--m', type=int,default=5,
                        help='BA attachment parameter (default: 5)')
    # Simulation
    parser.add_argument('--tmax',type=float, default=50.0,
                        help='Simulation end time(default: 120)')
    parser.add_argument('--n_timepoints',type=int,  default=n_timepoints,
                        help='Output time resolution(default: 50)')
    # Output
    parser.add_argument('--output', type=str,
                        default='epidemic_data_age_adaptive_sobol.pkl',
                        help='Output filename')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("3-PARAMETER SIR DATA GENERATION")
    print("R0 = tau/gamma x <k2>/<k>")
    print("=" * 70)
    print("Parameter ranges are set in PARAM_RANGES (top of file):")
    for name in PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        print(f"  {name:5s}  in [{lo}, {hi}]")

    dataset = generate_dataset(
        initial_samples=args.initial_samples,
        batch_size=args.batch_size,
        n_rounds=args.n_rounds,
        n_replicates=args.n_replicates,
        N=args.N,
        m=args.m,
        tmax=args.tmax,
        n_timepoints=args.n_timepoints,
    )

    save_dataset(dataset, args.output)


    #CSV output for visualisation in R or Excel (optional)
    import pandas as pd

def save_parameters_csv(dataset, filepath_csv):
    """
    Save one row per simulation:
    tau, gamma, rho, R0
    """
    net = dataset['network']
    ratio = net['ratio']

    rows = []
    for sim in dataset['simulations']:
        tau = sim['params']['tau']
        gamma = sim['params']['gamma']
        rho = sim['params']['rho']
        R0 = (tau / gamma) * ratio

        rows.append({
            'tau': tau,
            'gamma': gamma,
            'rho': rho,
            'R0': R0
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath_csv, index=False)
    print(f"Saved parameters CSV → {filepath_csv}")

    csv_name = Path(args.output).with_suffix("_parameters.csv")
    save_parameters_csv(dataset, csv_name)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nPipeline:")
    print(f"  1.  step1_generate_data_SIR3param.py  ->  {args.output}")
    print(f"  2.  python step2_split_data.py --input {args.output}")
    print(f"  3.  python step3_train_SIR3param.py")
    print(f"  4.  python step4_validate_SIR3param.py")
    print(f"  5.  python step5_test_SIR3param.py")
