

import numpy as np
import networkx as nx
import EoN
import pickle
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from pathlib import Path
import argparse
from tqdm import tqdm
from numpy.random import default_rng


# GLOBAL CONSTANTS
N = 10000   # network size (nodes)
m = 5       # Barabasi-Albert attachment parameter

PARAM_NAMES = ['tau', 'gamma', 'rho']

# Fixed parameter space 
PARAM_RANGES = {
    'tau'  : (0.0024, 0.034),  # Expected range: R₀ ∈ [0.12, 4.98] #   recovery rate
    'gamma': (0.07,  1),  # Infectious period 2-10 days
    'rho'  : (0.001, 0.010),
}

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


#Random Sampling
seed=4849
def random_sampling(n_samples, param_ranges=PARAM_RANGES, seed=None):
    """
    Generate random parameter samples uniformly over PARAM_RANGES.
    Returns array of shape [n_samples, 3].
    """
    rng = default_rng(seed)

    samples = np.zeros((n_samples, len(PARAM_NAMES)))

    for i, name in enumerate(PARAM_NAMES):
        low, high = param_ranges[name]
        samples[:, i] = rng.uniform(low, high, n_samples)

    return samples

# NETWORK GENERATION

def generate_network(N=N, m=m, seed=42):
    """Build the Barabasi-Albert network and warm the stats cache."""
    print(f"\nBuilding BA network  N={N:,}, m={m} ...")
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    get_ba_network_stats(N, m, seed)
    return G

# SIR SIMULATION

def run_sir_replicates(G, tau, gamma, rho,
                        n_replicates=5, tmax=50, n_timepoints=100):
    """
    Run n_replicates stochastic SIR simulations and average their outputs.

    Why average replicates?
    -----------------------
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
    t_fixed = np.linspace(0, tmax, n_timepoints)   #time grid construction for interpolation of S, I, R curves, t(fixed)​={0,…,tmax​}
    # if tmax=20 and n_timepoints=50, then t_fixed will be an array of 50 equally spaced time points from 0 to 20, inclusive. This is the time grid on which the S, I, R curves will be evaluated and averaged across replicates.
    #You cannot directly average trajectories unless they’re aligned on the same time grid.
    # So you create a common evaluation grid.
    
    try:
        S_runs, I_runs, R_runs = [], [], []

        for _ in range(n_replicates):
            t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho, tmax=tmax) # we run stochastic SIR simulations using EoN.fast_SIR, which returns arrays of time points (t) and corresponding S, I, R values. Each run will produce different t arrays due to the stochastic nature of the simulation.
            S_runs.append(np.interp(t_fixed, t, S)) # we use np.interp to interpolate the S, I, R values onto the common time grid t_fixed. This allows us to average the trajectories across replicates even though they were generated on different time grids.
            I_runs.append(np.interp(t_fixed, t, I))# Interpolation So you're constructing a continuous approximation of a jump process.
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


def run_batch(G, params_array, n_replicates=5, tmax=50, n_timepoints=100):
    """
    Simulate a batch of parameter sets and return a list of result dicts.

    Output format matches utils_SIR.EpidemicDatasetSIR:
        {'params': {'tau': float, 'gamma': float, 'rho': float},
         'output': {'t': ..., 'S': ..., 'I': ..., 'R': ...}}

    Args:
        params_array : [B, 3]   rows = [tau, gamma, rho] , B=number of rows in params_array
    """
    results = []

    #The Loop Over Parameter Sets
    for row in tqdm(params_array,                                          # tqdm counts how many iterations have occurred
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
    initial_samples=500,
    batch_size=32,
    n_replicates=5,
    N=N,
    m=m,
    tmax=50,
    n_timepoints=100,
):
    """
    Build the full training dataset via two-phase adaptive Sobol sampling.

    Phase 1 — Initial coverage
        Spread initial_samples points uniformly across the 3-D
        (tau, gamma, rho) cube using a Sobol sequence.


    """
    total = initial_samples 

  
    print("STEP 1 — SIR DATA GENERATION")
    print("3-Parameter model: tau, gamma, rho")
    print("R0 = tau/gamma x <k2>/<k>")
    print("=" * 70)
    print(f"\n  Parameter ranges (fixed in PARAM_RANGES):")
    for name in PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        print(f"    {name:5s}  in [{lo}, {hi}]")
    print(f"\n  Network      : BA(N={N:,}, m={m})")
    print(f"  Strategy     : Random sampling (d=3)")
    print(f"  Initial      : {initial_samples} samples")
    print(f"  Total sets   : {total}")
    print(f"  Replicates   : {n_replicates}/set  ->  "
          f"{total * n_replicates:,} individual runs")
    print(f"  Time grid    : tmax={tmax},  {n_timepoints} points")

    # Build the fixed network
    G = generate_network(N=N, m=m)

    # Phase 1 — Initial Sobol
   
    print(f"Phase 1 — Random samples ({initial_samples} samples)")
  

    params_array = random_sampling(initial_samples, seed=4849)

    all_sims = run_batch(
    G, params_array,
    n_replicates, tmax, n_timepoints
)
    
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

    n_thresh = ((R0_arr > 0.5) & (R0_arr < 1.5)).sum()
    print(f"\n  R0 distribution  (R0 = tau/gamma x <k2>/<k>):")
    print(f"    min  = {R0_arr.min():.2f}")
    print(f"    max  = {R0_arr.max():.2f}")
    print(f"    mean = {R0_arr.mean():.2f}")
   
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
            'model_type'       : 'sir_3param',
            'dimensionality'   : 3,
            'param_names'      : PARAM_NAMES,
            'param_ranges'     : PARAM_RANGES,
            'total_population' : N,
            'm'                : m,
            'tmax'             : tmax,
            'n_timepoints'     : n_timepoints,
            'sampling_strategy': 'random sampling',
            'r0_formula'       : 'R0 = tau/gamma * k2_avg/k_avg ' ,
            'initial_samples'  : initial_samples,
            'batch_size'       : batch_size,
            
        },
    }


# ============================================================================
# SAVE
# ============================================================================

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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3-parameter SIR training data via adaptive Sobol sampling.\n"
            "R0 = tau/gamma x <k2>/<k>\n"
            "Parameter ranges are fixed in PARAM_RANGES at the top of the file."
        )
    )
    # Sampling strategy
    parser.add_argument('--initial_samples', type=int, default=820,
                        help='Random samples(default: 500)')
    parser.add_argument('--batch_size',type=int, default=32,
                        help='New samples per adaptive round  (default: 32)')
    parser.add_argument('--n_replicates', type=int, default=3,
                        help='Stochastic replicates per set (default: 5)')
    # Network
    parser.add_argument('--N',type=int,   default=10000,
                        help='Network size (default: 10000)')
    parser.add_argument('--m',type=int,default=5,
                        help='BA attachment parameter (default: 5)')
    # Simulation
    parser.add_argument('--tmax',type=float, default=20.0,
                        help='Simulation end time (default: 20)')
    parser.add_argument('--n_timepoints',type=int,default=100,
                        help='Output time resolution(default: 50)')
    # Output
    parser.add_argument('--output',type=str,
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
        n_replicates=args.n_replicates,
        N=args.N,
        m=args.m,
        tmax=args.tmax,
        n_timepoints=args.n_timepoints,
    )

    save_dataset(dataset, args.output)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nPipeline:")
    print(f"  1.  step1_generate_data_SIR3param.py  ->  {args.output}")
    print(f"  2.  python step2_split_data.py --input {args.output}")
    print(f"  3.  python step3_train_SIR3param.py")
    print(f"  4.  python step4_validate_SIR3param.py")
    print(f"  5.  python step5_test_SIR3param.py")
