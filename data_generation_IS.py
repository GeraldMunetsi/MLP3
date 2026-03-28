"""
step1_generate_data_SIR3param_IS.py
=====================================
Weighted Importance Sampling (IS) adaptive data generation
for the 3-parameter SIR emulator.

Parameters
----------
    tau   (τ)  – per-contact transmission rate
    gamma (γ)  – recovery rate
    rho   (ρ)  – initial infected seed fraction

Epidemic threshold (Barabasi-Albert network)
---------------------------------------------
    R₀ = τ/γ × ⟨k²⟩/⟨k⟩

Adaptive strategy: Weighted IS + Resampling


    New approach (THIS FILE):
        Probabilistic — define target π(θ) concentrated near R₀ = 1
        Compute IS weights:  w_i = π(θ_i) / q(θ_i)
        Resample from existing sims proportional to w_i
        Add kernel jitter to resampled points → next batch
        ESS monitors sampling efficiency every round

    Why this is better:
        • Weights depend on what the ABM PRODUCED, not just where we sampled
        • ESS gives a formal, quantitative measure of efficiency
        • Bayesian-ready: π(θ) can be swapped for a likelihood-weighted
          posterior when real epidemic data becomes available
        • Kernel smoothing prevents sample impoverishment

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
from pathlib import Path
import argparse
from tqdm import tqdm


# CONSTANTS


N = 10000   # network size (nodes)
m = 5       # Barabasi-Albert attachment parameter
initial_samples=40 #basically my proporsal distribution
n_replicates=5 
tmax=50 
n_timepoints=50
n_rounds=50
sharpness=5.0
kernel_bandwith=0.05
batch_size= 32 # 32

PARAM_NAMES = ['tau', 'gamma', 'rho']

PARAM_RANGES = {
    'tau'  : (0.0024, 0.05),    # Expected range: R₀ ∈ [0.12, 4.98] #   recovery rate
    'gamma': (0.07,0.5),  # Infectious period 2-10 days
    'rho'  : (0.001,0.010),
}

#tau was 0.034


# BA NETWORK STATISTICS  (computed once, cached)

_NETWORK_STATS_CACHE = {}


def get_ba_network_stats(N=N, m=m, seed=42):
    """
    Compute and cache BA-network degree statistics needed for R0.
    R0 = tau/gamma × <k²>/<k>
    Results are cached so the network is built only once per run.
    """
    cache_key = (N, m, seed)

    if cache_key not in _NETWORK_STATS_CACHE:
        print(f"\n  Computing BA network statistics (N={N:,}, m={m}) ...")
        G       = nx.barabasi_albert_graph(N, m, seed=seed)
        degrees = np.array([d for _, d in G.degree()])

        k_avg  = degrees.mean()
        k2_avg = (degrees ** 2).mean()
        ratio  = k2_avg / k_avg          # <k²>/<k>

        _NETWORK_STATS_CACHE[cache_key] = {
            'k_avg' : k_avg,
            'k2_avg': k2_avg,
            'ratio' : ratio,
            'k_std' : degrees.std(),
            'k_max' : degrees.max(),
        }

        print(f"<k> = {k_avg:.2f}")
        print(f"<k²>= {k2_avg:.2f}")
        print(f"<k²>/<k>= {ratio:.2f}   (R0 multiplier)")
        print(f"sigma_k= {degrees.std():.2f}")
        print(f"k_max= {degrees.max()}")

    return _NETWORK_STATS_CACHE[cache_key]



# SOBOL SAMPLING  

def generate_sobol_samples(n_samples, seed=42, scramble=True):
    """
    Generate Sobol quasi-random samples over the (tau, gamma, rho) space.

    Why Sobol?
    ----------
    Sobol sequences are 'low-discrepancy': they fill 3-D space more
    evenly than plain random (Monte Carlo) sampling. This is our
    PROPOSAL DISTRIBUTION q(θ) — approximately uniform over the
    parameter cube. The IS weights will then correct for the fact
    that uniform coverage is not our scientific goal.

    Args:
        n_samples : number of parameter sets to generate
        seed      : scrambling seed (for reproducibility)
        scramble  : recommended True — removes inter-dimension correlations

    Returns:
        samples : [n_samples, 3]   each row = [tau, gamma, rho]
    """
    print(f"\nGenerating {n_samples} Sobol samples  (d=3: tau, gamma, rho) ...")

    sampler = qmc.Sobol(d=3, scramble=scramble, seed=seed)

    # Sobol quality is best at exact powers of 2
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
        lo, hi        = PARAM_RANGES[name]
        samples[:, i] = samples_unit[:, i] * (hi - lo) + lo

    discrepancy = qmc.discrepancy(samples_unit)
    print(f"  Star discrepancy = {discrepancy:.6f}  (lower = better coverage)")

    return samples



# WEIGHTED IMPORTANCE SAMPLING  (replaces estimate_errors + select_next_samples)

def compute_r0(params_array, N=N, m=m):
    """
    Compute R₀ for each row of params_array.
     Args:
        params_array : [n, 3]   rows = [tau, gamma, rho]

    Returns:
        r0 : [n]   R₀ values
    """
    net   = get_ba_network_stats(N=N, m=m)
    tau   = params_array[:, 0]
    gamma = params_array[:, 1]
    return (tau / gamma) * net['ratio']


def compute_log_target(r0, sharpness=sharpness):
    """
    Log of the target distribution π(θ).

    π(θ) ∝ exp( -sharpness × (R₀(θ) - 1)² )

    This is a Gaussian in R₀-space, peaked at the epidemic threshold
    R₀ = 1. Regions far from the threshold get exponentially down-
    weighted.

    Why this target?
    Near R₀ = 1 the SIR process is most stochastic ,smallchanges in
    τ or γ flip the system between extinction and epidemic. The emulator
    must be most accurate here. Concentrating training data at R₀ ≈ 1
    is our scientific priority, encoded as a probability distribution.

    Changing π(θ) is the ONLY change needed to redirect the sampler.
    For example, to target a calibrated posterior once real data arrives,
    replace this Gaussian with  log π(θ) = log L(data | θ) + log p(θ).

    Args:
        r0        : [n]   R₀ values
        sharpness : controls width of the Gaussian (higher = narrower)

    Returns:
        log_pi : [n]   log target density (unnormalised)
    """
    return -sharpness * (r0 - 1.0) ** 2


def compute_log_proposal(params_array):
    """
    Log of the proposal distribution q(θ).

    The Sobol sequence gives approximately uniform coverage over the
    parameter cube. Because the cube is compact and Sobol is uniform,
    q(θ) is approximately constant for all θ in the cube.

    log q(θ) = -log(Volume)   same for all points, cancels in weights

    We compute log q(θ) properly as the log-density of the uniform
    distribution over each parameter's range, so the code is correct
    even if ranges change.

    Args:
        params_array : [n, 3]   rows = [tau, gamma, rho]

    Returns:
        log_q : [n]   log proposal density (same for all points if uniform)
    """
    log_volume = 0.0
    for name in PARAM_NAMES:
        lo, hi      = PARAM_RANGES[name]
        log_volume += np.log(hi - lo)

    # Constant for all points — but returned as array for correctness
    return np.full(len(params_array), -log_volume)


def compute_importance_weights(simulations, N=N, m=m, sharpness=sharpness): #8.-0
    """
    Compute normalised importance weights for all existing simulations.

    IS weight formula:
        w_i  = π(θ_i) / q(θ_i)                  (unnormalised)
        w̃_i = w_i / Σ_j w_j                      (normalised, sums to 1)

    In log space (numerically stable):
        log w_i = log π(θ_i) − log q(θ_i)
        log w̃_i = log w_i − log_sum_exp(log w)

    Since q is uniform, log q is constant and cancels in the ratio,
    so the weights are proportional purely to π(θ_i) — the target.
    Points near R₀ = 1 get high weight; points far away get low weight.

    ESS (Effective Sample Size):
        ESS = 1 / Σ w̃_i²

    ESS = N means all weights are equal (uniform → wasteful).
    ESS << N means a few points dominate (weight collapse → danger).
    We want ESS to be reasonably large relative to N each round.

    Args:
        simulations : list of simulation dicts
        N, m        : network parameters for R₀ calculation
        sharpness   : Gaussian sharpness in target π(θ)

    Returns:
        weights     : [n]  normalised IS weights (sum to 1)
        ess         : float  effective sample size
        r0          : [n]  R₀ values (for diagnostics)
        log_weights : [n]  unnormalised log weights (for debugging)
    """
    params_array = np.array([
        [s['params'][name] for name in PARAM_NAMES]
        for s in simulations
    ])

    r0       = compute_r0(params_array, N=N, m=m)

    log_pi   = compute_log_target(r0, sharpness=sharpness)
    log_q    = compute_log_proposal(params_array)

    # Unnormalised log weights
    log_w    = log_pi - log_q

    # ── log-sum-exp trick: prevents numerical overflow/underflow ──────────────
    # Problem: exp(large number) = inf,  exp(very negative) = 0
    # Solution: subtract max before exponentiating, add it back in log space
    #
    # log Σ exp(wᵢ) = max(w) + log Σ exp(wᵢ - max(w))
    #
    # The shifted terms  wᵢ - max(w) are all ≤ 0, so exp() is safe.
    max_log_w = np.max(log_w)
    log_sum_w    = max_log_w + np.log(np.sum(np.exp(log_w - max_log_w)))
    log_w_norm   = log_w - log_sum_w

    weights      = np.exp(log_w_norm)           # normalised, sums to 1

    # ESS = 1 / Σ w̃ᵢ²
    ess= 1.0 / np.sum(weights ** 2)

    return weights, ess, r0, log_w


def resample_with_kernel_smoothing(
    simulations,
    weights,
    n_new,
    kernel_bandwidth=kernel_bandwith,
    rng=None,
):
    """
    Resample from existing simulations proportional to IS weights,
    then add kernel (Gaussian) jitter to the resampled parameters.

    Why resampling?
 
    After computing IS weights, high-weight simulations contain most
    of the statistical information. Resampling draws n_new parameter
    sets from the existing pool, with high-weight points drawn more
    often. This is the 'S' step in Sequential Importance Resampling.

    Why kernel smoothing (jitter) from Alex
   
    Naive resampling with replacement causes SAMPLE IMPOVERISHMENT:
    the same parameter set can appear many times, reducing diversity.
    Adding small Gaussian noise (a kernel) around each resampled point
    spreads the samples while keeping them near the high-weight region.
    This is Liu & West (2001) kernel smoothing for particle filters.

    Bandwidth choice:
    bandwidth controls the std of the jitter as a fraction of each
    parameter's range.  Too small → impoverishment persists.
    Too large → samples escape the high-weight region.
    Default 0.05 = 5% of each parameter's range, a safe starting point.

    Parameters are clipped to PARAM_RANGES after jitter.

    Args:
        simulations      : list of existing simulation dicts
        weights          : [n]  normalised IS weights from compute_IS_weights
        n_new            : number of new parameter sets to generate
        kernel_bandwidth : Gaussian jitter width (fraction of param range)
        rng              : numpy random Generator (for reproducibility)

    Returns:
        new_params : [n_new, 3]   new [tau, gamma, rho] parameter sets
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    params_array = np.array([
        [s['params'][name] for name in PARAM_NAMES]
        for s in simulations
    ])
    n_existing = len(simulations)

    # ── Step 1: Weighted resampling with replacement ──────────────────────────
    # np.random.choice draws indices from [0, n_existing) with probability
    # equal to the normalised weights. High-weight simulations are drawn
    # more often; low-weight ones may not appear at all.
    indices    = rng.choice(n_existing, size=n_new, replace=True, p=weights)
    resampled  = params_array[indices].copy()   # [n_new, 3]

    # ── Step 2: Kernel smoothing — add Gaussian jitter ───────────────────────
    # std of jitter for each parameter = bandwidth × (hi - lo)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi    = PARAM_RANGES[name]
        param_std = kernel_bandwidth * (hi - lo)

        jitter            = rng.normal(0.0, param_std, size=n_new)
        resampled[:, i]  += jitter

        # Clip to valid range — jitter must not push parameters out of bounds
        resampled[:, i]   = np.clip(resampled[:, i], lo, hi)

    return resampled


def select_next_samples(existing_sims, n_new, N=N, m=m,
                         sharpness=sharpness, kernel_bandwidth=kernel_bandwith):
    """
    IS-based adaptive selection — replaces the geometry-based method.

    Algorithm

    1. Compute IS weights:     w_i = π(θ_i) / q(θ_i)   for all existing sims
       where π(θ) ∝ exp(-sharpness × (R₀(θ) - 1)²)
       and   q(θ) ≈ Uniform (Sobol proposal)

    2. Compute ESS to diagnose weight health.
       ESS < 30% of N  weights are collapsing → bandwidth may need tuning.

    3. Resample n_new parameter sets from existing sims with prob ∝ w_i.
       This concentrates new simulations where the target π(θ) is high
       (near R₀ = 1) automatically — no hand-crafted score function needed.

    4. Add Gaussian kernel jitter to each resampled point so we get
       n_new DISTINCT parameter sets rather than duplicates.

    Comparison with old geometry-based method
   
    Old:  score = min_distance × (1 + exp(-8(R₀-1)²))
          → fills coverage gaps AND weights threshold proximity
          → maximin greedy pick (deterministic)
          → driven by WHERE we have sampled (space geometry)

    New:  w_i = π(θ_i) / q(θ_i)  = exp(-8(R₀-1)²) / (1/Volume)
          → driven by WHAT the ABM produced (probability mismatch)
          → resampling is stochastic, principled, and ESS-monitored
          → π(θ) is modular: swap for posterior when data arrives

    Args:
        existing_sims     : list of simulation dicts already completed
        n_new             : how many new parameter sets to generate
        N, m              : network parameters for R₀
        sharpness         : Gaussian sharpness in π(θ) (default 8.0)
        kernel_bandwidth  : jitter width as fraction of param range

    Returns:
        new_params : [n_new, 3]   array of [tau, gamma, rho]
    """
    print(f"\n  IS-based adaptive selection: {n_new} new samples ...")

    # ── Step 1: Compute IS weights ────────────────────────────────────────────
    weights, ess, r0, log_w = compute_importance_weights(
        existing_sims, N=N, m=m, sharpness=sharpness
    )
    n_existing = len(existing_sims)
    ess_pct    = 100.0 * ess / n_existing

    print(f"  Existing simulations    : {n_existing}")
    print(f"  R₀ range (existing)     : [{r0.min():.2f}, {r0.max():.2f}]")
    print(f"  R₀ near 1 (0.8–1.8)    : "
          f"{((r0 >= 0.8) & (r0 <= 1.8)).sum()} "
          f"({100*((r0 >=0.8) & (r0 <= 1.8)).mean():.1f}%)")
    print(f"  Max IS weight           : {weights.max():.6f}")
    print(f"  ESS                     : {ess:.1f} / {n_existing}  "
          f"({ess_pct:.1f}%)")

    if ess_pct < 20.0:
        print(f" ESS < 20% — weight collapse detected.")
        print(f"     Consider: wider π(θ), lower sharpness, or wider bandwidth.")
    elif ess_pct < 50.0:
        print(f" ESS < 50% — moderate weight concentration.")
    else:
        print(f" ESS healthy — IS weights well-distributed.")

    # ── Step 2: Resample + kernel smooth → new parameter sets ─────────────────
    rng        = np.random.default_rng(seed=42 + n_existing)
    new_params = resample_with_kernel_smoothing(
        existing_sims, weights, n_new,
        kernel_bandwidth=kernel_bandwidth,
        rng=rng,
    )

    # Compute R₀ for selected new params (for diagnostic printout)
    new_r0         = compute_r0(new_params, N=N, m=m)
    new_near_thresh = ((new_r0 > 0.8) & (new_r0 < 1.8)).sum()

    print(f"  New params R₀ range     : [{new_r0.min():.2f}, {new_r0.max():.2f}]")
    print(f"  New near R₀=1 (0.8–1.8): {new_near_thresh} "
          f"({100*new_near_thresh/n_new:.1f}%)  IS focus")

    return new_params


# NETWORK GENERATION


def generate_network(N=N, m=m, seed=42):
    """Build the Barabasi-Albert network and warm the stats cache."""
    print(f"\nBuilding BA network  N={N:,}, m={m} ...")
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    get_ba_network_stats(N, m, seed)
    return G


# SIR SIMULATION


def run_sir_replicates(G, tau, gamma, rho, n_replicates= n_replicates, tmax=tmax, n_timepoints=n_timepoints):
    """
    Run n_replicates stochastic SIR simulations and average their outputs.

    Why average replicates?
   
    Near R₀ = 1 the SIR process is highly stochastic — two runs with
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


def run_batch(G, params_array, n_replicates=n_replicates, tmax=tmax, n_timepoints= n_timepoints):
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
    initial_samples=initial_samples,
    batch_size=batch_size,
    n_rounds=n_rounds,
    n_replicates= n_replicates,
    sharpness=sharpness,
    kernel_bandwidth=kernel_bandwith,
    N=N,
    m=m,
    tmax=tmax,
    n_timepoints=n_timepoints,
):
    """
    Build the full training dataset via IS-adaptive Sobol sampling.

    Phase 1 — Initial coverage
        Spread initial_samples points uniformly across the 3-D
        (tau, gamma, rho) cube using a Sobol sequence. This is our
        proposal distribution q(θ) ≈ Uniform.

    Phase 2 — IS-Adaptive refinement  (n_rounds iterations)
        Each round:
            (a) Compute IS weights: w_i = π(θ_i) / q(θ_i)
                where π(θ) ∝ exp(-sharpness × (R₀(θ) - 1)²)
            (b) Compute ESS — diagnose weight health
            (c) Resample batch_size points from existing sims
                with prob ∝ w_i
            (d) Add Gaussian kernel jitter (kernel_bandwidth)
                to produce distinct, non-collapsed parameter sets
            (e) Run ABM at new parameter sets → extend dataset

    Result: uniform base coverage + IS-concentrated samples near R₀ = 1,
    achieved with the minimum number of expensive ABM simulations,
    with formal ESS monitoring every round.

    Args:
        initial_samples  : Phase 1 Sobol samples (broad coverage)
        batch_size       : new parameter sets per adaptive round
        n_rounds         : number of IS-adaptive rounds
        n_replicates     : stochastic replicates per parameter set
        sharpness        : Gaussian sharpness in π(θ) at R₀ = 1
        kernel_bandwidth : jitter width for kernel smoothing (fraction of range)
        N, m             : BA network parameters
        tmax             : simulation end time
        n_timepoints     : output time grid resolution
    """
    total = initial_samples + batch_size * n_rounds


    print("STEP 1 — SIR DATA GENERATION  (IS-Adaptive Sobol)")
    print("Adaptive method: Weighted Importance Sampling + Resampling")

    print(f"\n  Parameter ranges:")
    for name in PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        print(f"    {name:5s}  in [{lo}, {hi}]")
    print(f"\n  Network        : BA(N={N:,}, m={m})")
    print(f"  Strategy       : IS-Adaptive Sobol (d=3)")
    print(f"  Target π(θ)    : Gaussian at R₀=1, sharpness={sharpness}")
    print(f"  Proposal q(θ)  : Uniform (Sobol)")
    print(f"  Kernel bw      : {kernel_bandwidth} × param range")
    print(f"  Initial        : {initial_samples} samples")
    print(f"  Adaptive       : {n_rounds} rounds × {batch_size}")
    print(f"  Total sets     : {total}")
    print(f"  Replicates     : {n_replicates}/set   "
          f"{total * n_replicates:,} individual runs")
    print(f"  Time grid      : tmax={tmax},  {n_timepoints} points")

    # Build the fixed network
    G = generate_network(N=N, m=m)

    # Phase 1 — Initial Sobol (proposal distribution)
  
    print(f"PHASE 1 — INITIAL SOBOL  ({initial_samples} samples)  [proposal q(θ)]")
   

    all_sims = run_batch(
        G, generate_sobol_samples(initial_samples),
        n_replicates, tmax, n_timepoints
    )
    print(f"  Phase 1 done — {len(all_sims)} simulations")

    # Phase 2 — IS-Adaptive rounds
    ess_history = []   # track ESS across rounds for metadata

    for rnd in range(1, n_rounds + 1):
      
        print(f"PHASE 2 — IS-ADAPTIVE ROUND {rnd}/{n_rounds}  "
              f"(+{batch_size} samples)")
     

        new_params = select_next_samples(
            all_sims, n_new=batch_size,
            N=N, m=m,
            sharpness=sharpness,
            kernel_bandwidth=kernel_bandwidth,
        )

        all_sims.extend(
            run_batch(G, new_params, n_replicates, tmax, n_timepoints)
        )

        # Recompute ESS after this round for metadata record
        _, ess, _, _ = compute_importance_weights(all_sims, N=N, m=m,
                                                   sharpness=sharpness)
        ess_history.append(float(ess))
        print(f"  Round {rnd} done — total: {len(all_sims)} simulations  "
              f"| ESS: {ess:.1f}/{len(all_sims)} "
              f"({100*ess/len(all_sims):.1f}%)")

    # Summary

    print("DATA GENERATION COMPLETE")
    net = get_ba_network_stats(N, m)
    print(f"\n  Network: <k>={net['k_avg']:.2f}, "
          f"<k²>={net['k2_avg']:.2f}, "
          f"<k²>/<k>={net['ratio']:.2f}")

    tau_arr   = np.array([s['params']['tau']   for s in all_sims])
    gamma_arr = np.array([s['params']['gamma'] for s in all_sims])
    R0_arr    = (tau_arr / gamma_arr) * net['ratio']
    n_thresh  = ((R0_arr >= 0.8) & (R0_arr <= 1.8)).sum()

    print(f"\n  R₀ distribution  (R₀ = tau/gamma × <k²>/<k>):")
    print(f"    min  = {R0_arr.min():.2f}")
    print(f"    max  = {R0_arr.max():.2f}")
    print(f"    mean = {R0_arr.mean():.2f}")
    print(f"    Near R₀=1 (0.8–1.8): {n_thresh} "
          f"({100*n_thresh/len(all_sims):.1f}%)  IS focus")

    print(f"\n  ESS per round: "
          + "  ".join([f"R{i+1}:{e:.0f}" for i, e in enumerate(ess_history)]))

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
            'sampling_strategy': 'weighted_IS_resampling',   # updated
            'r0_formula'       : 'R0 = tau/gamma * k2_avg/k_avg (BARABASI-ALBERT NETWORK)',
            'initial_samples'  : initial_samples,
            'batch_size'       : batch_size,
            'n_rounds'         : n_rounds,
            # IS-specific metadata — new fields
            'IS_target'        : 'Gaussian at R0=1',
            'IS_sharpness'     : sharpness,
            'IS_proposal'      : 'Uniform_Sobol',
            'IS_kernel_bw'     : kernel_bandwidth,
            'IS_ess_history'   : ess_history,
        },
    }

# SAVE


def save_dataset(dataset, filepath):
    """Pickle the dataset to disk and print a summary."""
    filepath = Path(filepath)

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = filepath.stat().st_size / (1024 ** 2)

 
    print("SAVED")
    print(f"  File            : {filepath}  ({size_mb:.2f} MB)")
    print(f"  Samples         : {dataset['metadata']['n_samples']}")
    print(f"  Parameters      : {dataset['metadata']['param_names']}")
    print(f"  R₀ formula      : {dataset['metadata']['r0_formula']}")
    print(f"  Sampling method : {dataset['metadata']['sampling_strategy']}")
    print(f"  IS target       : {dataset['metadata']['IS_target']}")
    print(f"  ESS history     : {dataset['metadata']['IS_ess_history']}")
    


# ENTRY POINT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3-parameter SIR training data via IS-adaptive Sobol sampling.\n"
            "Adaptive method: Weighted Importance Sampling + Kernel Resampling\n"
            "Target π(θ) ∝ exp(-sharpness × (R₀(θ) - 1)²)\n"
            "Proposal q(θ) ≈ Uniform (Sobol sequence)\n"
            "Parameter ranges are fixed in PARAM_RANGES at the top of the file."
        )
    )
    # Sampling strategy
    parser.add_argument('--initial_samples', type=int, default=initial_samples,
                        help='Initial Sobol samples ')
    parser.add_argument('--batch_size',type=int, default=batch_size,
                        help='New samples per IS-adaptive round')
    parser.add_argument('--n_rounds',        type=int, default=n_rounds,
                        help='Number of IS-adaptive rounds )')
    parser.add_argument('--n_replicates',    type=int, default=n_replicates,
                        help='Stochastic replicates per set ')
    # IS-specific
    parser.add_argument('--sharpness',type=float, default=sharpness,
                        help='Gaussian sharpness in target π(θ) ')
    parser.add_argument('--kernel_bandwidth', type=float, default=kernel_bandwith,
                        help='Kernel jitter bandwidth, fraction of param range '
                             , )
    # Network
    parser.add_argument('--N', type=int,   default=N,
                        help='Network size')
    parser.add_argument('--m', type=int,   default=m,
                        help='BA attachment parameter ')
    # Simulation
    parser.add_argument('--tmax',type=float, default=tmax,
                        help='Simulation end time ')
    parser.add_argument('--n_timepoints',type=int,   default=n_timepoints,
                        help='Output time resolution')
    # Output
    parser.add_argument('--output', type=str,
                        default='epidemic_data_age_adaptive_sobol.pkl',
                        help='Output filename')

    args = parser.parse_args()


    print("3-PARAMETER SIR DATA GENERATION  (IS-Adaptive)")
    print("Target  π(θ) ∝ exp(-sharpness × (R₀ - 1)²)")
    print("Proposal q(θ) ≈ Uniform (Sobol)")

    print("Parameter ranges set in PARAM_RANGES (top of file):")
    for name in PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        print(f"  {name:5s}  in [{lo}, {hi}]")

    dataset = generate_dataset(
        initial_samples  = args.initial_samples,
        batch_size       = args.batch_size,
        n_rounds         = args.n_rounds,
        n_replicates     = args.n_replicates,
        sharpness        = args.sharpness,
        kernel_bandwidth = args.kernel_bandwidth,
        N                = args.N,
        m                = args.m,
        tmax             = args.tmax,
        n_timepoints     = args.n_timepoints,
    )

    save_dataset(dataset, args.output)

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
    print(f"Saved parameters CSV  {filepath_csv}")


    print("DONE")
    
    print(f"  1.  step1_generate_data_SIR3param_IS.py  ->  {args.output}")
    print(f"  2.  python step2_split_data.py --input {args.output}")
    