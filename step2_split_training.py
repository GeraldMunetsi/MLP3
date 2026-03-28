
import pickle
import numpy as np
import pandas as pd
import argparse
import networkx as nx
from pathlib import Path



# NETWORK HELPERS


def compute_network_moments(network: dict) -> tuple[float, float]:
    """
    Extracts <k> and <k²> from the network dict.

    R0 = (tau/gamma) * (<k²>/<k>)   [Pastor-Satorras & Vespignani 2001]
    """
    graph = network.get('graph', None)
    if graph is not None and isinstance(graph, nx.Graph):
        degrees = np.array([d for _, d in graph.degree()])
    else:
        N = network.get('N', 10000)
        m = network.get('m', 5)
        print("  No graph object — reconstructing BA graph for moments.")
        g = nx.barabasi_albert_graph(N, m, seed=42)
        degrees = np.array([d for _, d in g.degree()])
    return float(degrees.mean()), float((degrees**2).mean())


def compute_R0(tau, gamma, mean_k, mean_k2):
    if gamma == 0 or mean_k == 0:
        return float('nan')
    return (tau / gamma) * (mean_k2 / mean_k)



# SPLIT BY PARAMETER SET  (the fix)


def split_dataset(dataset,
                  train_ratio=0.60,
                  val_ratio=0.20,
                  test_ratio=0.20,
                  seed=42,
                  stratify=True):
    """
    Split by PARAMETER SET, not by simulation.

    Algorithm
  
    1. Group all simulations by their (tau, gamma, rho) tuple.
    2. Randomly assign each GROUP to train / val / test.
    3. All replicates of a param set go to the SAME split.
    4. Optionally stratify groups by R0 band so each split has
       proportional coverage of sub-threshold / near-threshold /
       above-threshold / large-epidemic scenarios.

    Parameters
   
    dataset      : dict with keys 'simulations', 'network', 'metadata'
    train_ratio  : fraction of PARAMETER SETS in training
    val_ratio    : fraction of PARAMETER SETS in validation
    test_ratio   : fraction of PARAMETER SETS in test
    seed         : random seed
    stratify     : if True, preserve R0 distribution across splits

    Returns
    -------
    split_data : dict with keys 'train', 'val', 'test', 'network', 'metadata'
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1.0"

    np.random.seed(seed)
    sims = dataset['simulations']

    # ── Step 1: Compute R0 for stratification ────────────────────────────────
    network = dataset.get('network', {})
    if not isinstance(network, dict):
        N = dataset['metadata'].get('total_population', 10000)
        m = dataset['metadata'].get('m', 5)
        network = {'graph': nx.barabasi_albert_graph(N, m, seed=42), 'N': N, 'm': m}

    mean_k, mean_k2 = compute_network_moments(network)
    K2K = mean_k2 / mean_k
    print(f"  Network: <k>={mean_k:.3f}  <k²>={mean_k2:.3f}  <k²>/<k>={K2K:.3f}")

    # ── Step 2: Group simulations by param set ────────────────────────────────
    # Key = (tau, gamma, rho) rounded to avoid float noise
    param_to_indices = {}
    for i, sim in enumerate(sims):
        p = sim['params']
        tau   = float(p.get('tau',   p.get('beta',  0)))
        gamma = float(p.get('gamma', p.get('mu',    0)))
        rho   = float(p.get('rho',   0))
        key   = (round(tau, 8), round(gamma, 8), round(rho, 8))
        if key not in param_to_indices:
            param_to_indices[key] = []
        param_to_indices[key].append(i)

    param_keys = list(param_to_indices.keys())
    n_param_sets = len(param_keys)

    print(f"\n  Found {n_param_sets} unique parameter sets")
    print(f"  Total simulations: {len(sims)}")
    reps_per_set = len(sims) / n_param_sets
    print(f"  Avg replicates per set: {reps_per_set:.1f}")

    # ── Step 3: R0-stratified split ───────────────────────────────────────────
    R0_per_set = np.array([
        compute_R0(k[0], k[1], mean_k, mean_k2) for k in param_keys
    ])

    if stratify:
        # 4 strata: sub-threshold / near-threshold / moderate / large
        # Ensures each split has proportional R0 coverage
        strata = np.digitize(R0_per_set, bins=[0.8, 1.2, 3.0])
        # strata values: 0=R0<0.8, 1=0.8-1.2, 2=1.2-3.0, 3=R0>3
        stratum_labels, stratum_counts = np.unique(strata, return_counts=True)
        print(f"\n  R0 strata for stratified split:")
        stratum_names = {0:"R0<0.8", 1:"R0 0.8-1.2", 2:"R0 1.2-3", 3:"R0>3"}
        for sl, sc in zip(stratum_labels, stratum_counts):
            print(f"    {stratum_names.get(sl, sl)}: {sc} param sets ({100*sc/n_param_sets:.1f}%)")

        train_param_idx, val_param_idx, test_param_idx = [], [], []

        for stratum in stratum_labels:
            stratum_idx = np.where(strata == stratum)[0]
            np.random.shuffle(stratum_idx)
            n_s      = len(stratum_idx)
            n_tr_s   = max(1, int(n_s * train_ratio))
            n_va_s   = max(1, int(n_s * val_ratio))
            # remainder goes to test
            train_param_idx.extend(stratum_idx[:n_tr_s].tolist())
            val_param_idx.extend(stratum_idx[n_tr_s:n_tr_s+n_va_s].tolist())
            test_param_idx.extend(stratum_idx[n_tr_s+n_va_s:].tolist())

    else:
        # Simple random split by param set
        all_param_idx = np.random.permutation(n_param_sets)
        n_tr = int(n_param_sets * train_ratio)
        n_va = int(n_param_sets * val_ratio)
        train_param_idx = all_param_idx[:n_tr].tolist()
        val_param_idx   = all_param_idx[n_tr:n_tr+n_va].tolist()
        test_param_idx  = all_param_idx[n_tr+n_va:].tolist()

    # ── Step 4: Collect simulation indices for each split ─────────────────────
    def collect_sims(param_idx_list):
        sim_indices = []
        for pi in param_idx_list:
            sim_indices.extend(param_to_indices[param_keys[pi]])
        return sim_indices

    train_sim_idx = collect_sims(train_param_idx)
    val_sim_idx   = collect_sims(val_param_idx)
    test_sim_idx  = collect_sims(test_param_idx)

    train_sims = [sims[i] for i in train_sim_idx]
    val_sims   = [sims[i] for i in val_sim_idx]
    test_sims  = [sims[i] for i in test_sim_idx]

    # ── Step 5: Verify zero leakage ───────────────────────────────────────────
    train_params_set = set(param_keys[pi] for pi in train_param_idx)
    val_params_set   = set(param_keys[pi] for pi in val_param_idx)
    test_params_set  = set(param_keys[pi] for pi in test_param_idx)

    leak_tr_te = train_params_set & test_params_set
    leak_tr_va = train_params_set & val_params_set
    leak_va_te = val_params_set   & test_params_set

    print(f"\n  === LEAKAGE CHECK ===")
    print(f"  Train ∩ Test param sets: {len(leak_tr_te)}  (must be 0)")
    print(f"  Train ∩ Val  param sets: {len(leak_tr_va)}  (must be 0)")
    print(f"  Val   ∩ Test param sets: {len(leak_va_te)}  (must be 0)")
    if len(leak_tr_te) > 0 or len(leak_tr_va) > 0 or len(leak_va_te) > 0:
        raise RuntimeError("BUG: Parameter leakage detected! Aborting.")
    print(f"  ✓ Zero leakage confirmed — all splits have disjoint param sets")

    # ── Step 6: R0 distribution report ───────────────────────────────────────
    print(f"\n  === SPLIT SUMMARY ===")
    N = network.get('N', 10000)
    for name, sim_list, param_idx_list in [
        ('train', train_sims, train_param_idx),
        ('val',   val_sims,   val_param_idx),
        ('test',  test_sims,  test_param_idx),
    ]:
        r0_vals = [R0_per_set[pi] for pi in param_idx_list]
        n_sims  = len(sim_list)
        n_psets = len(param_idx_list)
        print(f"\n  {name:5s}: {n_psets:4d} param sets  |  {n_sims:5d} simulations")
        print(f"         R0 min={min(r0_vals):.3f}  mean={np.mean(r0_vals):.3f}  max={max(r0_vals):.3f}")
        for lo, hi, lbl in [(0,0.8,"R0<0.8"),(0.8,1.2,"R0 0.8-1.2"),(1.2,3.0,"R0 1.2-3.0"),(3.0,99,"R0>3")]:
            cnt = sum(lo <= r < hi for r in r0_vals)
            print(f"         {lbl:12s}: {cnt:4d} param sets  ({100*cnt/n_psets:.1f}%)")

    split_data = {
        'train'   : {'simulations': train_sims, 'indices': train_sim_idx,
                     'param_indices': train_param_idx},
        'val'     : {'simulations': val_sims,   'indices': val_sim_idx,
                     'param_indices': val_param_idx},
        'test'    : {'simulations': test_sims,  'indices': test_sim_idx,
                     'param_indices': test_param_idx},
        'network' : network,
        'metadata': dataset['metadata'],
        # Store for downstream verification
        'split_info': {
            'method'       : 'parameter_set_stratified',
            'n_param_sets' : n_param_sets,
            'K2K'          : float(K2K),
            'leakage_check': 'PASSED — zero shared param sets across splits',
            'stratified'   : stratify,
        }
    }
    return split_data


# ============================================================================
# CSV EXPORT
# ============================================================================

def export_training_csv(split_data: dict, output_csv_path: Path) -> pd.DataFrame:
    """Export training set parameters + epidemiological summaries to CSV."""
    network      = split_data['network']
    mean_k, mean_k2 = compute_network_moments(network)
    K2K = mean_k2 / mean_k
    N   = network.get('N', split_data['metadata'].get('total_population', 10_000))

    print(f"\n  Network moments:  <k>={mean_k:.4f}  <k²>={mean_k2:.4f}  K2K={K2K:.4f}")

    rows = []
    for local_i, (sim, orig_idx) in enumerate(
            zip(split_data['train']['simulations'],
                split_data['train']['indices'])):

        p     = sim['params']
        tau   = float(p.get('tau',   p.get('beta',  np.nan)))
        gamma = float(p.get('gamma', p.get('mu',    np.nan)))
        rho   = float(p.get('rho',   np.nan))
        R0    = compute_R0(tau, gamma, mean_k, mean_k2)

        out    = sim['output']
        I_mean = np.array(out.get('I_mean', out.get('I', [])))
        R_mean = np.array(out.get('R_mean', out.get('R', [])))

        if len(I_mean) > 0:
            peak_I      = float(I_mean.max())
            peak_time   = int(I_mean.argmax())
            final_R     = float(R_mean[-1]) if len(R_mean) > 0 else np.nan
            attack_rate = final_R / N if N > 0 else np.nan
        else:
            peak_I = peak_time = final_R = attack_rate = np.nan

        rows.append({
            'sim_index'   : int(orig_idx),
            'tau'         : tau,
            'gamma'       : gamma,
            'rho'         : rho,
            'R0'          : round(R0, 6),
            'peak_I'      : round(peak_I, 2)      if not np.isnan(peak_I)    else np.nan,
            'peak_time'   : peak_time,
            'final_R'     : round(final_R, 2)     if not np.isnan(final_R)   else np.nan,
            'attack_rate' : round(attack_rate, 6) if not np.isnan(attack_rate) else np.nan,
            'n_replicates': out.get('n_replicates', 1),
        })

    df = pd.DataFrame(rows)
    df['near_threshold'] = ((df['R0'] >= 0.8) & (df['R0'] <= 1.2)).astype(int)

    print(f"\n  Training CSV: {len(df)} rows")
    print(f"  R0 range: {df['R0'].min():.3f} – {df['R0'].max():.3f}")
    print(f"  Near threshold (0.8-1.2): {df['near_threshold'].sum()} ({100*df['near_threshold'].mean():.1f}%)")

    df.to_csv(output_csv_path, index=False)
    print(f"  Saved: {output_csv_path}")
    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split by PARAMETER SET (not simulation) to prevent data leakage."
    )
    parser.add_argument('--input',       type=str,   default="epidemic_data_age_adaptive_sobol.pkl")
    parser.add_argument('--output',      type=str,   default=None)
    parser.add_argument('--output_csv',  type=str,   default=None)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio',   type=float, default=0.15)
    parser.add_argument('--test_ratio',  type=float, default=0.15)
    parser.add_argument('--no_stratify', action='store_true',
                        help="Disable R0-stratified splitting (not recommended)")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 2: DATA SPLITTING — BY PARAMETER SET (LEAKAGE-FREE)")
    print("=" * 70)
    print()
    print("  METHOD: Group simulations by (tau,gamma,rho) → split groups")
    print("  This ensures test set contains UNSEEN parameter combinations.")
    print("  Replicates of the same param set always stay in the same split.")
    print()

    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)

    n_sims = len(dataset['simulations'])
    n_reps = dataset['simulations'][0]['output'].get('n_replicates', '?')
    print(f"  Loaded: {args.input}")
    print(f"  {n_sims} total simulations  (n_replicates={n_reps})")

    split_data = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=not args.no_stratify,
    )

    # Save pickle
    out_path = Path(args.output) if args.output else \
               Path(args.input).parent / (Path(args.input).stem + '_split.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Saved pickle: {out_path}  ({out_path.stat().st_size/(1024**2):.2f} MB)")

    # Export CSV
    csv_path = Path(args.output_csv) if args.output_csv else \
               Path(args.input).parent / (Path(args.input).stem + '_train_params.csv')
    df = export_training_csv(split_data, csv_path)

    print()
    print("=" * 70)
    print("  COMPLETE — leakage-free split saved")
    print(f"  Pickle : {out_path}")
    print(f"  CSV    : {csv_path}")
    print()
    print("  IMPORTANT: Re-run step3 training with the new split file.")
    print("  Your previous R2 scores were inflated due to parameter leakage.")
    print("  Expect R2_test to decrease by ~0.05-0.15 — this is the honest value.")
    print("=" * 70)