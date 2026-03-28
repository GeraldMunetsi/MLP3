"""
STEP 2: Split Data - FIXED
Preserves network structure and handles replicate data
Exports both .pkl (full data) and .csv (summary table)
"""

import pickle
import csv
import numpy as np
import argparse
from pathlib import Path


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """FIXED: Ensures network preserved as dict."""
    np.random.seed(seed)
    
    n_samples = len(dataset['simulations'])
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_sims = [dataset['simulations'][i] for i in train_idx]
    val_sims = [dataset['simulations'][i] for i in val_idx]
    test_sims = [dataset['simulations'][i] for i in test_idx]
    
    # FIXED: Ensure network is dict
    network = dataset.get('network', {})
    if not isinstance(network, dict):
        print(f"  Warning: Network is {type(network)}, fixing...")
        import networkx as nx
        N = dataset['metadata'].get('total_population', 10000)
        m = dataset['metadata'].get('m', 5)
        network = {
            'graph': nx.barabasi_albert_graph(N, m, seed=42),
            'N': N,
            'm': m
        }

    split_data = {
        'train': {'simulations': train_sims, 'indices': train_idx.tolist()},
        'val':   {'simulations': val_sims,   'indices': val_idx.tolist()},
        'test':  {'simulations': test_sims,  'indices': test_idx.tolist()},
        'network': network,
        'metadata': dataset['metadata']
    }
    
    return split_data


def extract_simulation_summary(sim, split_label, original_index):
    """
    Flattens one simulation into a single CSV row.

    What we extract:
    - Parameters: tau (transmission rate), gamma (recovery rate), rho (initial infected fraction)
    - Derived:    R0 (basic reproduction number)
    - Summary statistics from the output trajectories:
        * peak_infected    — highest number of infected at any time point
        * time_to_peak     — time step when infection peaks
        * final_susceptible— S at end of simulation (how many escaped infection)
        * final_recovered  — R at end of simulation (total who got infected)
        * attack_rate      — final_recovered / N  (fraction of population infected)
    - Metadata: split label (train/val/test), original index in dataset
    """
    row = {}

    # ── 1. Split label & index ──────────────────────────────────────────────
    row['split']          = split_label
    row['original_index'] = original_index

    # ── 2. Input parameters ─────────────────────────────────────────────────
    params = sim.get('params', {})
    row['tau']   = params.get('tau',   np.nan)   # transmission rate per edge per time step
    row['gamma'] = params.get('gamma', np.nan)   # recovery rate per time step
    row['rho']   = params.get('rho',   np.nan)   # initial fraction of population infected

    # R0 = (tau / gamma) × (⟨k²⟩ / ⟨k⟩)  for Barabási-Albert networks
    # We store it if it was pre-computed, otherwise leave NaN
    row['R0'] = params.get('R0', np.nan)

    # ── 3. Output trajectory summary ────────────────────────────────────────
    output = sim.get('output', {})

    # Mean trajectory (averaged over replicates if replicate data exists)
    S_mean = np.array(output.get('S_mean', output.get('S', [])))
    I_mean = np.array(output.get('I_mean', output.get('I', [])))
    R_mean = np.array(output.get('R_mean', output.get('R', [])))

    if len(I_mean) > 0:
        peak_idx            = int(np.argmax(I_mean))
        row['peak_infected']     = float(np.max(I_mean))
        row['time_to_peak']      = peak_idx
        row['final_susceptible'] = float(S_mean[-1]) if len(S_mean) > 0 else np.nan
        row['final_recovered']   = float(R_mean[-1]) if len(R_mean) > 0 else np.nan
        row['n_timepoints']      = len(I_mean)

        # Attack rate = proportion of population that ever got infected
        N = S_mean[0] + I_mean[0] + R_mean[0] if len(S_mean) > 0 else np.nan
        row['attack_rate'] = row['final_recovered'] / N if N > 0 else np.nan

        # Variability across replicates (useful for uncertainty analysis)
        I_std = output.get('I_std', None)
        row['peak_infected_std'] = float(np.max(I_std)) if I_std is not None else np.nan
    else:
        # If trajectory missing, fill with NaN so CSV row still exists
        for col in ['peak_infected', 'time_to_peak', 'final_susceptible',
                    'final_recovered', 'n_timepoints', 'attack_rate', 'peak_infected_std']:
            row[col] = np.nan

    row['n_replicates'] = output.get('n_replicates', 1)

    return row


def export_csv(split_data, csv_path):
    """
    Writes a CSV summary file from the split dataset.

    Structure:
        split | original_index | tau | gamma | rho | R0 |
        peak_infected | time_to_peak | final_susceptible |
        final_recovered | attack_rate | n_timepoints |
        peak_infected_std | n_replicates

    Why CSV alongside pickle?
    - Pickle stores everything (full trajectories, graph objects) — hard to inspect
    - CSV gives you a quick, human-readable summary for:
        * Sanity checking your parameter distribution
        * Plotting R0 vs attack_rate in R or Excel instantly
        * Sharing with collaborators who don't use Python
        * Quick EDA (exploratory data analysis) before training
    """
    all_rows = []

    for split_label in ['train', 'val', 'test']:
        simulations = split_data[split_label]['simulations']
        indices     = split_data[split_label]['indices']

        for sim, original_idx in zip(simulations, indices):
            row = extract_simulation_summary(sim, split_label, original_idx)
            all_rows.append(row)

    if not all_rows:
        print("  ⚠ No rows extracted — CSV not written.")
        return

    # Write using csv.DictWriter so column order is consistent
    fieldnames = list(all_rows[0].keys())

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  ✓ CSV saved: {csv_path}  ({len(all_rows)} rows × {len(fieldnames)} columns)")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',       type=str,   default="epidemic_data_age_adaptive_sobol.pkl")
    parser.add_argument('--output',      type=str,   default=None)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio',   type=float, default=0.15)
    parser.add_argument('--test_ratio',  type=float, default=0.15)
    parser.add_argument('--no_csv',      action='store_true', help="Skip CSV export")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("STEP 2: DATA SPLITTING (FIXED)")
    print("="*70)

    print(f"\nLoading: {args.input}")
    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)

    n_samples = len(dataset['simulations'])

    if 'S_std' in dataset['simulations'][0]['output']:
        n_reps = dataset['simulations'][0]['output'].get('n_replicates', '?')
        print(f"  ✓ {n_samples} samples ({n_reps} replicates each)")
    else:
        print(f"  ✓ {n_samples} samples")

    split_data = split_dataset(dataset, args.train_ratio, args.val_ratio, args.test_ratio)

    print(f"\n  Train: {len(split_data['train']['simulations'])}")
    print(f"  Val:   {len(split_data['val']['simulations'])}")
    print(f"  Test:  {len(split_data['test']['simulations'])}")

    # ── Pickle output path ───────────────────────────────────────────────────
    if args.output is None:
        output_path = Path(args.input).parent / (Path(args.input).stem + '_split.pkl')
    else:
        output_path = Path(args.output)

    print(f"\nSaving pickle: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f" Saved ({output_path.stat().st_size / (1024**2):.2f} MB)")

    # ── CSV export ───────────────────────────────────────────────────────────
    if not args.no_csv:
        csv_path = output_path.with_suffix('.csv')   # same name, .csv extension
        print(f"\nSaving CSV:{csv_path}")
        export_csv(split_data, csv_path)

    print("\n" + "="*70)
    print("COMPLETE — Network preserved as dict | CSV + Pickle exported")
    print("="*70)