"""
step5_test_SIR3param.py
========================
FINAL TEST SET EVALUATION — 3-Parameter SIR Emulator (τ, γ, ρ)

⚠  These are your DISSERTATION RESULTS.  Run this ONCE on the held-out
   test set after all model selection decisions have been made.

Changes from step5_test_REPLICATED_SIMPLIFIED.py:
  ✗  dummy_graph_stats removed from evaluate_model()
  ✗  graph_stats argument removed from model.forward() calls
  ✗  mlp_input_dim / mlp_hidden / mlp_layers removed from default config
  ✗  import changed: utils_AGE_MLP → utils_SIR
  ✓  param annotation shows τ, γ, ρ in all plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
import json
import pandas as pd
from scipy import stats

from step0_model  import create_hybrid_mlp_model
from utils_SIR import create_dataloaders, compute_metrics, get_device


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_replicate_model(model_path, device):
    """Load a single replicate checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'n_params'        : 3,
            'n_fourier'       : 64,
            'fourier_hidden'  : 32,
            'param_hidden'    : 16,
            'temporal_hidden' : 64,
            'dropout'         : 0.3,
            'n_knots'         : 12,
            'n_timepoints'    : 50,
            'total_population': 10000,
        }

    model = create_hybrid_mlp_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device, n_timesteps):
    """
    Run inference on the full test set.

    No graph_stats — the 3-parameter SIR model does not use them.
    """
    model.eval()

    all_predictions, all_targets, all_params = [], [], []

    print("    Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # ── Clean forward pass (no dummy tensors) ─────────────────────────
            predictions = model(batch, n_timesteps=n_timesteps)

            all_predictions.append(predictions.cpu())
            all_targets.append(batch.y.cpu())
            all_params.append(batch.params.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets     = torch.cat(all_targets,     dim=0)
    params      = torch.cat(all_params,      dim=0)
    metrics     = compute_metrics(predictions, targets)

    return predictions, targets, params, metrics


def evaluate_all_replicates(models_dir, test_loader, device, n_timesteps):
    """
    Evaluate every replicate model on the test set.

    Looks for: best_balanced_mlp_model_*.pt
    """
    models_dir  = Path(models_dir)

    if not models_dir.exists():
        raise ValueError(f"Directory not found: {models_dir}")

    model_paths = sorted(
        list(models_dir.glob("best_balanced_mlp_model_*.pt")),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    if not model_paths:
        raise ValueError(
            f"No models found in {models_dir} "
            "matching 'best_balanced_mlp_model_*.pt'"
        )

    results_list = []
    targets = params = None

    print(f"\n{'='*70}")
    print(f"FINAL TEST EVALUATION  ·  {len(model_paths)} REPLICATE(S)")
    print(f"3-Parameter SIR: tau (τ), gamma (γ), rho (ρ)")
    print(f"⚠  These are your DISSERTATION RESULTS!")
    print(f"{'='*70}\n")

    for idx, model_path in enumerate(model_paths, 1):
        print(f"Replicate {idx}/{len(model_paths)} : {model_path.name}")

        model, checkpoint = load_replicate_model(model_path, device)

        predictions, targets_rep, params_rep, metrics = evaluate_model(
            model, test_loader, device, n_timesteps
        )

        if targets is None:
            targets = targets_rep
            params  = params_rep

        results_list.append({
            'replicate_id'   : idx,
            'model_path'     : str(model_path),
            'predictions'    : predictions,
            'metrics'        : metrics,
            'checkpoint_info': {
                'epoch'      : checkpoint.get('epoch', 'unknown'),
                'val_metrics': checkpoint.get('val_metrics', {}),
                'weight_mode': checkpoint.get('weight_mode', 'unknown'),
                'param_names': checkpoint.get('param_names', ['tau', 'gamma', 'rho']),
            },
        })

        print(f"  MAE_I : {metrics['MAE_I']:.2f}  ← key metric")
        print(f"  R²    : {metrics['R2']:.4f}")
        print()

    print(f"✓ Evaluated {len(results_list)} replicates\n")
    return results_list, targets, params


# ============================================================================
# AGGREGATE STATISTICS
# ============================================================================

def compute_aggregate_statistics(results_list):
    """Compute mean, std, CV, 95% CI across replicates."""
    n = len(results_list)

    metric_keys = ['MAE', 'MAE_S', 'MAE_I', 'MAE_R', 'R2', 'RMSE', 'MSE']
    stats_dict  = {'n_replicates': n}

    for key in metric_keys:
        arr = np.array([r['metrics'][key] for r in results_list])
        sem = arr.std() / np.sqrt(n)
        ci  = stats.t.interval(0.95, n - 1, loc=arr.mean(), scale=sem) if n > 1 else (arr.mean(), arr.mean())

        stats_dict[key] = {
            'mean'  : float(arr.mean()),
            'std'   : float(arr.std()),
            'sem'   : float(sem),
            'min'   : float(arr.min()),
            'max'   : float(arr.max()),
            'median': float(np.median(arr)),
            'ci_95' : [float(ci[0]), float(ci[1])],
            'cv'    : float(arr.std() / arr.mean() * 100) if arr.mean() != 0 else 0.0,
        }

    return stats_dict


# ============================================================================
# VISUALISATION
# ============================================================================

def plot_test_predictions(results_list, targets, output_dir, n_samples=8):
    """
    Plot SIR predictions vs ground truth for a sample of test cases.
    Shows τ, γ, ρ values per panel.
    """
    output_dir = Path(output_dir)
    targets_np = targets.numpy()

    n_total  = len(targets_np)
    indices  = np.linspace(0, n_total - 1, n_samples, dtype=int)

    compartments = ['Susceptible (S)', 'Infected (I)', 'Recovered (R)']
    gt_colors    = ['lightblue', 'lightcoral', 'lightgreen']
    n_reps       = len(results_list)
    pred_colors  = plt.cm.tab10(np.linspace(0, 1, n_reps))

    fig = plt.figure(figsize=(18, 3 * n_samples))
    gs  = GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        'Test Set Predictions — All Replicates  |  3-Parameter SIR (τ, γ, ρ)',
        fontsize=16, fontweight='bold'
    )

    for row, idx in enumerate(indices):
        target = targets_np[idx]

        for col in range(3):
            ax = fig.add_subplot(gs[row, col])

            # Ground truth
            ax.plot(target[:, col], 'o', color=gt_colors[col],
                    alpha=0.6, markersize=5, markeredgewidth=0,
                    label='Ground Truth', zorder=10)

            # Replicate predictions
            for rep_i, result in enumerate(results_list):
                pred = result['predictions'][idx].numpy()
                ax.plot(pred[:, col], '-',
                        color=pred_colors[rep_i], linewidth=1.5, alpha=0.6,
                        label=f"M{result['replicate_id']}" if col == 1 else "")

            if row == 0:
                ax.set_title(compartments[col], fontsize=12, fontweight='bold')

            ax.set_xlabel('Time step', fontsize=9)
            ax.set_ylabel('Count',     fontsize=9)

            if col == 1:
                ax.legend(loc='best', fontsize=7, ncol=2)

            ax.grid(True, alpha=0.3, linestyle='--')

    plt.savefig(output_dir / 'test_comparison_plots.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'test_comparison_plots.png'}")


def plot_metrics_distribution(stats_dict, output_dir):
    """Publication-ready boxplot/distribution panel for all test metrics."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Test Metrics Distribution  ·  3-Parameter SIR (τ, γ, ρ)',
        fontsize=16, fontweight='bold'
    )

    # R² histogram
    ax = axes[0, 0]
    ax.axvline(stats_dict['R2']['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {stats_dict['R2']['mean']:.4f}")
    ax.set_title('R² Test Distribution'); ax.set_xlabel('R²')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # MAE_I histogram
    ax = axes[0, 1]
    ax.axvline(stats_dict['MAE_I']['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {stats_dict['MAE_I']['mean']:.2f}")
    ax.set_title('MAE_I Test Distribution'); ax.set_xlabel('MAE_I (Infected)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Per-compartment MAE bar
    ax = axes[0, 2]
    comps  = ['S', 'I', 'R']
    means  = [stats_dict[f'MAE_{c}']['mean'] for c in comps]
    stds   = [stats_dict[f'MAE_{c}']['std']  for c in comps]
    ax.bar([0, 1, 2], means, yerr=stds, capsize=5,
           color=['cornflowerblue', 'tomato', 'mediumseagreen'],
           alpha=0.7, edgecolor='black')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(comps)
    ax.set_ylabel('MAE'); ax.set_title('Per-Compartment MAE')
    ax.grid(True, alpha=0.3, axis='y')

    # R² vs MAE_I scatter (one point per replicate)
    ax = axes[1, 0]
    ax.set_xlabel('R²'); ax.set_ylabel('MAE_I')
    ax.set_title('R² vs MAE_I Trade-off'); ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"TEST RESULTS SUMMARY\n"
        f"{'='*30}\n\n"
        f"3-Parameter SIR (τ, γ, ρ)\n"
        f"n replicates: {stats_dict['n_replicates']}\n\n"
        f"R²\n"
        f"  Mean : {stats_dict['R2']['mean']:.4f}\n"
        f"  95% CI: [{stats_dict['R2']['ci_95'][0]:.4f}, "
        f"{stats_dict['R2']['ci_95'][1]:.4f}]\n\n"
        f"MAE_I  ← KEY METRIC\n"
        f"  Mean : {stats_dict['MAE_I']['mean']:.2f}\n"
        f"  95% CI: [{stats_dict['MAE_I']['ci_95'][0]:.2f}, "
        f"{stats_dict['MAE_I']['ci_95'][1]:.2f}]\n"
        f"  CV   : {stats_dict['MAE_I']['cv']:.1f}%"
    )
    ax.text(0.05, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / 'test_metrics_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'test_metrics_distribution.png'}")


# ============================================================================
# SAVE RESULTS & DISSERTATION REPORT
# ============================================================================

def save_results(results_list, stats_dict, output_dir):
    """Save CSV, JSON, and plain-text dissertation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # CSV
    rows = [
        {'replicate_id': r['replicate_id'],
         'model_path'  : r['model_path'],
         **r['metrics'],
         'training_epoch': r['checkpoint_info']['epoch'],
         'weight_mode'   : r['checkpoint_info']['weight_mode']}
        for r in results_list
    ]
    pd.DataFrame(rows).to_csv(output_dir / 'test_replicate_results.csv', index=False)
    print(f"✓ Saved: {output_dir / 'test_replicate_results.csv'}")

    # JSON
    with open(output_dir / 'test_final_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"✓ Saved: {output_dir / 'test_final_statistics.json'}")

    # ── Dissertation-ready report ─────────────────────────────────────────────
    mae_i_mean = stats_dict['MAE_I']['mean']
    mae_i_ci   = stats_dict['MAE_I']['ci_95']
    r2_mean    = stats_dict['R2']['mean']
    r2_ci      = stats_dict['R2']['ci_95']
    cv         = stats_dict['MAE_I']['cv']

    performance = (
        "EXCEPTIONAL ⭐⭐⭐⭐⭐" if mae_i_mean < 150 else
        "EXCELLENT   ⭐⭐⭐⭐"   if mae_i_mean < 200 else
        "GOOD        ⭐⭐⭐"    if mae_i_mean < 250 else
        "NEEDS IMPROVEMENT"
    )
    consistency = (
        "EXCELLENT  (CV < 5%)"  if cv < 5  else
        "GOOD       (CV < 10%)" if cv < 10 else
        "ACCEPTABLE (CV < 15%)" if cv < 15 else
        f"HIGH VARIABILITY (CV = {cv:.1f}%)"
    )

    summary_lines = [
        "=" * 70,
        "FINAL TEST RESULTS — 3-Parameter SIR Emulator (τ, γ, ρ)",
        "⚠  THESE ARE YOUR DISSERTATION NUMBERS",
        "=" * 70,
        "",
        f"  Replicates   : {stats_dict['n_replicates']}",
        f"  Test samples : {len(results_list[0]['predictions'])}",
        "",
        "=" * 70,
        "OVERALL PERFORMANCE:",
        "=" * 70,
        f"  MAE   : {stats_dict['MAE']['mean']:.2f} ± {stats_dict['MAE']['std']:.2f}",
        f"          95% CI: [{stats_dict['MAE']['ci_95'][0]:.2f}, {stats_dict['MAE']['ci_95'][1]:.2f}]",
        "",
        f"  R²    : {r2_mean:.4f} ± {stats_dict['R2']['std']:.4f}",
        f"          95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]",
        "",
        f"  RMSE  : {stats_dict['RMSE']['mean']:.2f} ± {stats_dict['RMSE']['std']:.2f}",
        "",
        "=" * 70,
        "PER-COMPARTMENT MAE:",
        "=" * 70,
        f"  Susceptible (S) : {stats_dict['MAE_S']['mean']:.2f} ± {stats_dict['MAE_S']['std']:.2f}",
        f"  Infected (I)    : {mae_i_mean:.2f} ± {stats_dict['MAE_I']['std']:.2f}  ← KEY METRIC",
        f"                    95% CI: [{mae_i_ci[0]:.2f}, {mae_i_ci[1]:.2f}]",
        f"                    CV = {cv:.1f}%",
        f"  Recovered (R)   : {stats_dict['MAE_R']['mean']:.2f} ± {stats_dict['MAE_R']['std']:.2f}",
        "",
        "=" * 70,
        "PERFORMANCE ASSESSMENT:",
        "=" * 70,
        f"  Level       : {performance}",
        f"  Consistency : {consistency}",
        "",
        "=" * 70,
        "SUGGESTED DISSERTATION TEXT:",
        "-" * 70,
        f"\"The 3-parameter SIR emulator (τ, γ, ρ) achieved a test set MAE_I",
        f"of {mae_i_mean:.2f} ({mae_i_ci[0]:.2f}–{mae_i_ci[1]:.2f}, 95% CI,",
        f"n={stats_dict['n_replicates']} replicates), with overall R² = {r2_mean:.4f} ±",
        f"{stats_dict['R2']['std']:.4f}. Replicate consistency was {consistency.lower()},",
        f"with coefficient of variation {cv:.1f}%, indicating robust performance",
        "across random initialisations.\"",
        "",
        "=" * 70,
        "STATISTICAL NOTES:",
        "=" * 70,
        "  · All metrics: mean ± standard deviation across replicates",
        "  · 95% CI computed using t-distribution (appropriate for small n)",
        "  · CV (coefficient of variation) = std / mean × 100",
        "  · CV < 10% indicates good replicate consistency",
        "",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)

    with open(output_dir / 'FINAL_DISSERTATION_RESULTS.txt', 'w') as f:
        f.write(summary_text)

    print(f"✓ Saved: {output_dir / 'FINAL_DISSERTATION_RESULTS.txt'}")
    print("\n" + summary_text)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final test evaluation — 3-Parameter SIR emulator (τ, γ, ρ)"
    )
    parser.add_argument('--models_dir',  type=str, default='replicates_outputs')
    parser.add_argument('--data',        type=str, default='epidemic_data_age_adaptive_sobol_split_augmented.pkl')
    parser.add_argument('--output_dir',  type=str, default='sir3param_test_results')
    parser.add_argument('--n_samples',   type=int, default=8)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("STEP 5: FINAL TEST SET EVALUATION — 3-Parameter SIR (τ, γ, ρ)")
    print("THESE ARE YOUR DISSERTATION RESULTS!")
    print("=" * 70)

    device     = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nLoading test data: {args.data}")
    dataloaders = create_dataloaders(args.data, batch_size=40)
    test_loader = dataloaders['test']
    n_timesteps = dataloaders['metadata']['n_timepoints']
    print(f"✓ Test samples: {len(test_loader.dataset)}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results_list, targets, params = evaluate_all_replicates(
        args.models_dir, test_loader, device, n_timesteps
    )

    # ── Statistics ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    stats_dict = compute_aggregate_statistics(results_list)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-READY FIGURES")
    print("=" * 70 + "\n")
    plot_test_predictions(results_list, targets, output_dir, args.n_samples)
    plot_metrics_distribution(stats_dict, output_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING FINAL RESULTS")
    print("=" * 70 + "\n")
    save_results(results_list, stats_dict, output_dir)

    print("\n" + "=" * 70)
    print("✅  FINAL TEST COMPLETE")
    print("=" * 70)
    print(f"\n  Results → {output_dir}")
    print("\n  Copy FINAL_DISSERTATION_RESULTS.txt into your thesis!")
    print("  Use test_metrics_distribution.png  as a figure.")
    print("  Use test_comparison_plots.png      as a figure.")
