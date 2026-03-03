"""
step4_validate_SIR3param.py
============================
Validation script for the 3-parameter SIR emulator.

Parameters:
    tau   - transmission rate
    gamma - recovery rate
    rho   - network rewiring / contact heterogeneity

Age-structured code has been fully removed.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from scipy import stats
import pandas as pd

from step0_model import create_hybrid_mlp_model          # your hybrid architecture
from utils_SIR   import create_dataloaders, compute_metrics, get_device   # renamed utils


# ============================================================================
# CONSTANTS
# ============================================================================

PARAM_NAMES   = ['tau', 'gamma', 'rho']          # 3-parameter SIR
N_PARAMS      = 3
COMPARTMENTS  = ['Susceptible (S)', 'Infected (I)', 'Recovered (R)']
COMP_COLORS   = ['lightblue', 'lightcoral', 'lightgreen']


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path, device):
    """
    Load a single replicate model checkpoint.

    Args:
        model_path : Path to .pt checkpoint file
        device     : torch.device

    Returns:
        model      : loaded model in eval mode
        checkpoint : raw checkpoint dict
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # ── Config ─────────────────────────────────────────────────────────────
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Minimal default config for 3-parameter SIR model
        config = {
            'n_params'       : N_PARAMS,     # tau, gamma, rho
            'n_fourier'      : 64,
            'fourier_hidden' : 32,
            'param_hidden'   : 16,
            'mlp_hidden'     : 32,
            'mlp_layers'     : 2,
            'temporal_hidden': 64,
            'dropout'        : 0.3,
            'n_knots'        : 12,
            'n_timepoints'   : 50,
            'total_population': 10000,
        }

    model = create_hybrid_mlp_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def find_replicate_models(models_dir):
    """
    Find all numbered replicate models inside *models_dir*.

    Pattern: best_balanced_mlp_model_*.pt

    Returns:
        List of Paths sorted by replicate number.
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise ValueError(f"Directory not found: {models_dir}")

    model_files = sorted(
        models_dir.glob("best_balanced_mlp_model_*.pt"),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    if len(model_files) == 0:
        raise ValueError(
            f"No models found in {models_dir} "
            "matching pattern 'best_balanced_mlp_model_*.pt'"
        )

    return model_files


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, val_loader, device, n_timesteps):
    """
    Run forward pass for every batch in *val_loader*.

    NOTE: graph_stats / age-structure arguments have been removed.
          The model now accepts only (batch, n_timesteps).

    Returns:
        predictions : Tensor [N, T, 3]
        targets     : Tensor [N, T, 3]
        params      : Tensor [N, 3]   → (tau, gamma, rho)
        metrics     : dict
    """
    model.eval()

    all_predictions = []
    all_targets     = []
    all_params      = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # ── Forward pass (no graph_stats, no age dummy) ────────────────
            predictions = model(batch, n_timesteps=n_timesteps)

            targets = batch.y

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_params.append(batch.params.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets     = torch.cat(all_targets,     dim=0)
    params      = torch.cat(all_params,      dim=0)

    metrics = compute_metrics(predictions, targets)

    return predictions, targets, params, metrics


def evaluate_all_replicates(models_dir, val_loader, device, n_timesteps):
    """
    Loop over every replicate model and collect predictions + metrics.

    Returns:
        results_list : list of dicts (one per replicate)
        targets      : Tensor – ground truth (identical across replicates)
        params       : Tensor – (tau, gamma, rho) for each validation sample
    """
    model_paths = find_replicate_models(models_dir)

    print(f"\n{'='*70}")
    print(f"EVALUATING {len(model_paths)} REPLICATE(S) — 3-PARAMETER SIR MODEL")
    print(f"Parameters: tau (τ), gamma (γ), rho (ρ)")
    print(f"{'='*70}\n")

    print(f"Models directory : {models_dir}")
    for mp in model_paths:
        print(f"  · {mp.name}")
    print()

    results_list = []
    targets      = None
    params       = None

    for idx, model_path in enumerate(model_paths, 1):
        print(f"Replicate {idx}/{len(model_paths)}: {model_path.name}")

        model, checkpoint = load_model(model_path, device)

        predictions, targets_rep, params_rep, metrics = evaluate_model(
            model, val_loader, device, n_timesteps
        )

        if targets is None:
            targets = targets_rep
            params  = params_rep

        result = {
            'replicate_id'   : checkpoint.get('replicate_id', idx),
            'seed'           : checkpoint.get('seed', None),
            'model_filename' : model_path.name,
            'predictions'    : predictions.numpy(),
            'metrics'        : metrics,
        }

        results_list.append(result)
        print(f"  R² = {metrics['R2']:.4f}  |  MAE_I = {metrics['MAE_I']:.2f}\n")

    return results_list, targets, params


# ============================================================================
# AGGREGATE STATISTICS
# ============================================================================

def compute_aggregate_statistics(results_list):
    """
    Compute mean, std, 95 % CI, CV across replicates for each metric.

    Returns:
        stats_dict : nested dict keyed by metric name
    """
    metric_names = ['MAE', 'RMSE', 'R2', 'MAE_S', 'MAE_I', 'MAE_R']
    stats_dict   = {}

    for metric_name in metric_names:
        values = np.array([r['metrics'][metric_name] for r in results_list])
        n      = len(values)

        mean = np.mean(values)
        std  = np.std(values, ddof=1) if n > 1 else 0.0
        sem  = stats.sem(values)      if n > 1 else 0.0
        ci   = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 else (mean, mean)

        stats_dict[metric_name] = {
            'values'  : values.tolist(),
            'n'       : int(n),
            'mean'    : float(mean),
            'std'     : float(std),
            'sem'     : float(sem),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'min'     : float(np.min(values)),
            'max'     : float(np.max(values)),
            'median'  : float(np.median(values)),
            'cv'      : float(std / mean * 100) if mean != 0 else None,
        }

    return stats_dict


# ============================================================================
# VISUALISATION
# ============================================================================

def load_training_histories(models_dir):
    """Load training_history_*.npy files (one per replicate)."""
    models_dir   = Path(models_dir)
    history_files = sorted(
        models_dir.glob("training_history_*.npy"),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    if len(history_files) == 0:
        return None

    histories = []
    for hist_file in history_files:
        try:
            history = np.load(hist_file, allow_pickle=True).item()
            histories.append(history)
        except Exception:
            histories.append(None)

    return histories


def plot_training_curves(results_list, models_dir, output_dir):
    """Training-curve panel for all replicates."""
    output_dir = Path(output_dir)
    histories  = load_training_histories(models_dir)

    if histories is None or all(h is None for h in histories):
        print("  ⚠  Training histories not found — skipping training-curves plot.")
        return

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 12))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle(
        'Training Curves — All Replicates  |  3-Parameter SIR (τ, γ, ρ)',
        fontsize=16, fontweight='bold'
    )

    n_replicates = len(results_list)
    colors       = plt.cm.tab10(np.linspace(0, 1, n_replicates))

    # ── 1. Validation R² ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_r2' in history:
            ax1.plot(range(1, len(history['val_r2']) + 1), history['val_r2'],
                     color=colors[i], alpha=0.7, linewidth=2,
                     label=f"Model {result['replicate_id']}")
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Validation R²', fontweight='bold')
    ax1.set_title('R² Evolution')
    ax1.legend(fontsize=8, ncol=2, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)

    # ── 2. Validation MAE_I ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_mae_i' in history:
            ax2.plot(range(1, len(history['val_mae_i']) + 1), history['val_mae_i'],
                     color=colors[i], alpha=0.7, linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation MAE_I', fontweight='bold')
    ax2.set_title('MAE_I Evolution')
    ax2.grid(True, alpha=0.3)

    # ── 3. Train vs Val Loss ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax3.plot(epochs, history['train_loss'],
                     color=colors[i], alpha=0.5, linewidth=1, linestyle='--')
            ax3.plot(epochs, history['val_loss'],
                     color=colors[i], alpha=0.7, linewidth=2)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Loss', fontweight='bold')
    ax3.set_title('Training (dashed) vs Validation (solid) Loss')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # ── 4. MAE_S ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_mae_s' in history:
            ax4.plot(range(1, len(history['val_mae_s']) + 1), history['val_mae_s'],
                     color=colors[i], alpha=0.7, linewidth=2)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Validation MAE_S', fontweight='bold')
    ax4.set_title('Susceptible (S) Error Evolution')
    ax4.grid(True, alpha=0.3)

    # ── 5. MAE_R ─────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_mae_r' in history:
            ax5.plot(range(1, len(history['val_mae_r']) + 1), history['val_mae_r'],
                     color=colors[i], alpha=0.7, linewidth=2)
    ax5.set_xlabel('Epoch', fontweight='bold')
    ax5.set_ylabel('Validation MAE_R', fontweight='bold')
    ax5.set_title('Recovered (R) Error Evolution')
    ax5.grid(True, alpha=0.3)

    # ── 6. Convergence — last 10 epochs ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    for i, (result, history) in enumerate(zip(results_list, histories)):
        if history is not None and 'val_r2' in history:
            last_10     = history['val_r2'][-10:]
            epochs_last = range(len(history['val_r2']) - 9, len(history['val_r2']) + 1)
            ax6.plot(epochs_last, last_10,
                     color=colors[i], alpha=0.7, linewidth=2, marker='o')
    ax6.set_xlabel('Epoch', fontweight='bold')
    ax6.set_ylabel('Validation R²', fontweight='bold')
    ax6.set_title('Convergence (Last 10 Epochs)')
    ax6.grid(True, alpha=0.3)

    # ── 7. Final metrics bar chart ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :])
    metrics_to_plot = ['R2', 'MAE_S', 'MAE_I', 'MAE_R']
    x     = np.arange(len(results_list))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        values = [r['metrics'][metric] for r in results_list]
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        ax7.bar(x + offset, values, width, label=metric, alpha=0.7)

    ax7.set_xlabel('Replicate Model Number', fontweight='bold')
    ax7.set_ylabel('Metric Value', fontweight='bold')
    ax7.set_title('Final Validation Metrics per Replicate  (τ, γ, ρ  model)')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f"Model {r['replicate_id']}" for r in results_list])
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3, axis='y')

    out_path = output_dir / 'training_curves_all_replicates.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_prediction_samples(results_list, targets, params, output_dir, n_samples=6):
    """
    Plot sample SIR predictions vs ground truth for all replicates.

    Parameters shown per sample: τ (tau), γ (gamma), ρ (rho).
    """
    output_dir = Path(output_dir)
    targets    = targets.numpy()
    params     = params.numpy()

    # Select evenly spaced samples from validation set
    n_total = len(targets)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 3 * n_samples))
    gs  = GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        'Validation Predictions: All Replicates vs Ground Truth  '
        '|  3-Parameter SIR (τ, γ, ρ)',
        fontsize=16, fontweight='bold'
    )

    n_replicates = len(results_list)
    pred_colors  = plt.cm.tab10(np.linspace(0, 1, n_replicates))

    for row, idx in enumerate(indices):
        target = targets[idx]           # shape [T, 3]
        param  = params[idx]            # shape [3]  → tau, gamma, rho

        tau_val   = param[0]
        gamma_val = param[1]
        rho_val   = param[2]

        for col in range(3):
            ax = fig.add_subplot(gs[row, col])

            # Ground truth
            ax.plot(target[:, col], 'o',
                    color=COMP_COLORS[col], label='Ground Truth',
                    alpha=0.6, markersize=6, markeredgewidth=0, zorder=10)

            # Each replicate
            for rep_idx, result in enumerate(results_list):
                pred = result['predictions'][idx]       # [T, 3]
                ax.plot(pred[:, col], '-',
                        color=pred_colors[rep_idx], linewidth=1.5, alpha=0.6,
                        label=f"Model {result['replicate_id']}" if col == 1 else "")

            if row == 0:
                ax.set_title(COMPARTMENTS[col], fontsize=12, fontweight='bold')

            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)

            if col == 1:
                ax.legend(loc='best', fontsize=7, ncol=2)

            ax.grid(True, alpha=0.3, linestyle='--')

            # Parameter annotation (first column only)
            if col == 0:
                info_text = (
                    f'Sample {row + 1}\n'
                    f'τ={tau_val:.3f}\n'
                    f'γ={gamma_val:.3f}\n'
                    f'ρ={rho_val:.3f}'
                )
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    out_path = output_dir / 'predictions_all_replicates.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" Saved: {out_path}")


def plot_validation_summary(results_list, stats_dict, output_dir):
    """Summary statistics panel across all replicates."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Validation Results — All Replicates  |  3-Parameter SIR (τ, γ, ρ)',
        fontsize=16, fontweight='bold'
    )

    # ── R² histogram ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    r2_values = stats_dict['R2']['values']
    ax.hist(r2_values, bins=min(10, len(r2_values)),
            color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(stats_dict['R2']['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {stats_dict['R2']['mean']:.4f}")
    ax.set_xlabel('R²')
    ax.set_ylabel('Frequency')
    ax.set_title('R² Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ── MAE_I histogram ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    mae_i_values = stats_dict['MAE_I']['values']
    ax.hist(mae_i_values, bins=min(10, len(mae_i_values)),
            color='salmon', edgecolor='black', alpha=0.7)
    ax.axvline(stats_dict['MAE_I']['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {stats_dict['MAE_I']['mean']:.2f}")
    ax.set_xlabel('MAE_I')
    ax.set_ylabel('Frequency')
    ax.set_title('MAE_I Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ── Box-plot: all metrics ─────────────────────────────────────────────────
    ax = axes[0, 2]
    metrics_data = {
        'R²'   : stats_dict['R2']['values'],
        'MAE'  : stats_dict['MAE']['values'],
        'MAE_S': stats_dict['MAE_S']['values'],
        'MAE_I': stats_dict['MAE_I']['values'],
        'MAE_R': stats_dict['MAE_R']['values'],
    }
    bp = ax.boxplot([metrics_data[k] for k in metrics_data],
                    labels=list(metrics_data.keys()),
                    patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Value')
    ax.set_title('Metrics Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Per-compartment MAE bar chart ─────────────────────────────────────────
    ax = axes[1, 0]
    comp_labels = ['S', 'I', 'R']
    means = [stats_dict[f'MAE_{c}']['mean'] for c in comp_labels]
    stds  = [stats_dict[f'MAE_{c}']['std']  for c in comp_labels]
    ax.bar(np.arange(3), means, yerr=stds, capsize=5,
           color=['cornflowerblue', 'tomato', 'mediumseagreen'],
           alpha=0.7, edgecolor='black')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel('MAE')
    ax.set_title('Per-Compartment MAE (Mean ± Std)')
    ax.grid(True, alpha=0.3, axis='y')

    # ── R² vs MAE_I scatter ───────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.scatter(stats_dict['R2']['values'], stats_dict['MAE_I']['values'],
               s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('R²')
    ax.set_ylabel('MAE_I')
    ax.set_title('R² vs MAE_I Trade-off')
    ax.grid(True, alpha=0.3)

    # ── Summary text box ──────────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis('off')

    s_r2   = stats_dict['R2']
    s_maei = stats_dict['MAE_I']

    summary_text = (
        f"VALIDATION SUMMARY\n"
        f"{'='*32}\n\n"
        f"Model: 3-Parameter SIR\n"
        f"Parameters: τ, γ, ρ\n"
        f"n replicates: {s_r2['n']}\n\n"
        f"R²:\n"
        f"  Mean : {s_r2['mean']:.4f}\n"
        f"  Std  : {s_r2['std']:.4f}\n"
        f"  95% CI: [{s_r2['ci_lower']:.4f}, {s_r2['ci_upper']:.4f}]\n\n"
        f"MAE_I:\n"
        f"  Mean : {s_maei['mean']:.2f}\n"
        f"  Std  : {s_maei['std']:.2f}\n"
        f"  CV   : {s_maei['cv']:.2f}%\n"
        f"  95% CI: [{s_maei['ci_lower']:.2f}, {s_maei['ci_upper']:.2f}]\n\n"
        f"Consistency: "
        f"{'EXCELLENT' if s_maei['cv'] < 5 else 'GOOD' if s_maei['cv'] < 10 else 'ACCEPTABLE'}"
    )

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    out_path = output_dir / 'validation_summary.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


# ============================================================================
# REPORTING
# ============================================================================

def convert_to_python_types(obj):
    """Recursively convert numpy scalars → Python native types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(i) for i in obj)
    return obj


def save_results(results_list, stats_dict, output_dir):
    """Persist results as JSON, CSV, and plain-text report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ── JSON ─────────────────────────────────────────────────────────────────
    results_data = {
        'model_description'   : '3-Parameter SIR Emulator (tau, gamma, rho)',
        'individual_results'  : [
            {
                'replicate_id'  : int(r['replicate_id']) if r['replicate_id'] is not None else None,
                'seed'          : int(r['seed'])          if r['seed']          is not None else None,
                'model_filename': str(r['model_filename']),
                'metrics'       : convert_to_python_types(r['metrics']),
            }
            for r in results_list
        ],
        'aggregate_statistics': convert_to_python_types(stats_dict),
    }

    json_path = output_dir / 'validation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = [
        {'replicate_id': r['replicate_id'],
         'seed'        : r['seed'],
         'model_filename': r['model_filename'],
         **r['metrics']}
        for r in results_list
    ]
    csv_path = output_dir / 'validation_results.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    # ── Plain-text report ────────────────────────────────────────────────────
    lines = [
        "=" * 70,
        "VALIDATION REPORT — 3-PARAMETER SIR EMULATOR",
        "Parameters: tau (τ)  |  gamma (γ)  |  rho (ρ)",
        "=" * 70,
        "",
        f"Number of replicates : {len(results_list)}",
        f"Model pattern        : best_balanced_mlp_model_*.pt",
        "",
        "=" * 70,
        "AGGREGATE STATISTICS",
        "=" * 70,
        "",
        "PRIMARY METRICS:",
        "-" * 70,
    ]

    for metric_name in ['R2', 'MAE_I']:
        s   = stats_dict[metric_name]
        fmt = ".4f" if metric_name == 'R2' else ".2f"
        lines += [
            f"\n{metric_name}:",
            f"  Mean    : {s['mean']:{fmt}}",
            f"  Std     : {s['std']:{fmt}}",
            f"  95% CI  : [{s['ci_lower']:{fmt}},  {s['ci_upper']:{fmt}}]",
            f"  CV      : {s['cv']:.2f}%" if s['cv'] else "  CV      : N/A",
            f"  Range   : [{s['min']:{fmt}},  {s['max']:{fmt}}]",
        ]

    lines += [
        "",
        "\nPER-COMPARTMENT MAE:",
        "-" * 70,
    ]
    for comp in ['S', 'I', 'R']:
        s = stats_dict[f'MAE_{comp}']
        lines.append(
            f"  MAE_{comp}: {s['mean']:7.2f} ± {s['std']:5.2f}"
            f"  (95% CI: [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}])"
        )

    lines += [
        "",
        "\nINDIVIDUAL REPLICATE RESULTS:",
        "-" * 70,
        f"{'ID':>3} {'Seed':>6} {'R²':>8} {'MAE_I':>8}  {'Model File':<40}",
        "-" * 70,
    ]
    for r in results_list:
        lines.append(
            f"{r['replicate_id']:3d} "
            f"{str(r['seed']) if r['seed'] else 'N/A':>6} "
            f"{r['metrics']['R2']:8.4f} "
            f"{r['metrics']['MAE_I']:8.2f}  "
            f"{r['model_filename']:<40}"
        )

    # Interpretation
    cv = stats_dict['MAE_I']['cv']
    consistency = (
        "EXCELLENT ⭐⭐⭐⭐⭐" if cv and cv < 5  else
        "GOOD      ⭐⭐⭐⭐"   if cv and cv < 10 else
        "ACCEPTABLE ⭐⭐⭐"   if cv and cv < 15 else
        "HIGH VARIABILITY ⚠"
    )
    performance = (
        "EXCEPTIONAL" if stats_dict['MAE_I']['mean'] < 150 else
        "EXCELLENT"   if stats_dict['MAE_I']['mean'] < 200 else
        "GOOD"        if stats_dict['MAE_I']['mean'] < 250 else
        "NEEDS IMPROVEMENT"
    )

    lines += [
        "",
        "=" * 70,
        "INTERPRETATION",
        "=" * 70,
        f"\nModel Consistency (CV) : {consistency}",
        f"Overall Performance    : {performance}",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    report_path = output_dir / 'VALIDATION_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"✓ Saved: {report_path}")
    print("\n" + report_text)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate replicate models — 3-Parameter SIR (tau, gamma, rho)"
    )
    parser.add_argument('--models_dir', type=str,
                        default='replicates_outputs',
                        help='Directory containing trained replicate models')
    parser.add_argument('--data', type=str,
                        default='epidemic_data_age_adaptive_sobol_split_augmented.pkl',
                        help='Dataset file (3-parameter SIR)')
    parser.add_argument('--output_dir', type=str,
                        default='sir3param_validation_results',
                        help='Output directory for plots and reports')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("STEP 4: VALIDATION — 3-PARAMETER SIR MODEL (τ, γ, ρ)")
    print("=" * 70)
    print(f"\n✨ Models directory : {args.models_dir}")
    print(f"   Pattern          : best_balanced_mlp_model_*.pt")
    print(f"   Parameters       : tau (τ), gamma (γ), rho (ρ)")

    device = get_device()

    # ── Load validation data ──────────────────────────────────────────────────
    print(f"\nLoading data: {args.data}")
    dataloaders = create_dataloaders(args.data, batch_size=40)
    val_loader  = dataloaders['val']
    n_timesteps = dataloaders['metadata']['n_timepoints']

    print(f"✓ Validation samples : {len(val_loader.dataset)}")

    # ── Evaluate all replicates ───────────────────────────────────────────────
    results_list, targets, params = evaluate_all_replicates(
        args.models_dir, val_loader, device, n_timesteps
    )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    stats_dict = compute_aggregate_statistics(results_list)

    print(f"\n✓ Statistics over {len(results_list)} replicate(s)")
    print(f"  Mean R²    : {stats_dict['R2']['mean']:.4f} ± {stats_dict['R2']['std']:.4f}")
    print(f"  Mean MAE_I : {stats_dict['MAE_I']['mean']:.2f} ± {stats_dict['MAE_I']['std']:.2f}")
    print(f"  CV (MAE_I) : {stats_dict['MAE_I']['cv']:.2f}%")

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING VISUALISATIONS")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n1. Training curves (if histories available)...")
    plot_training_curves(results_list, args.models_dir, output_dir)

    print("\n2. Prediction samples (all replicates)...")
    plot_prediction_samples(results_list, targets, params, output_dir, n_samples=6)

    print("\n3. Validation summary statistics...")
    plot_validation_summary(results_list, stats_dict, output_dir)

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70 + "\n")

    save_results(results_list, stats_dict, args.output_dir)

    print("\n" + "=" * 70)
    print("✓ VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to : {args.output_dir}")
    print(f"\n  Key files:")
    print(f"    · VALIDATION_REPORT.txt          (read this first)")
    print(f"    · validation_results.csv         (spreadsheet)")
    print(f"    · validation_results.json        (machine-readable)")
    print(f"\n  Plots:")
    print(f"    · training_curves_all_replicates.png")
    print(f"    · predictions_all_replicates.png")
    print(f"    · validation_summary.png")
    print(f"\n  Run command:")
    print(f"    python step4_validate_SIR3param.py --models_dir {args.models_dir}")
