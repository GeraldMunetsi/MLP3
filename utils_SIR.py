"""
utils_SIR.py  ·  Utility Functions for 3-Parameter SIR Emulator
================================================================
Parameters: tau (τ), gamma (γ), rho (ρ)

Removed from the old utils_AGE_MLP1.py:
  · GraphStatsNormalizer  (no graph-stat branch in the model)
  · graph_stats field in EpidemicDatasetMLP, BatchWrapper, collate_mlp
  · 7 age-structured parameters replaced by 3 SIR parameters
  · networkx graph stats computation
"""

import torch
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

from torch.utils.data import Dataset, DataLoader
# ============================================================================
# DATASET
# ============================================================================

class EpidemicDatasetSIR(Dataset):
    """
    PyTorch Dataset for the 3-parameter SIR emulator.

    Each item returns:
        params  : [3]          – [tau, gamma, rho]
        y       : [T, 3]       – [S(t), I(t), R(t)] trajectories
    """

    def __init__(self, simulations, n_timepoints):
        """
        Args:
            simulations  : list of simulation dicts (from pickle file)
            n_timepoints : number of time steps T
        """
        super().__init__()
        self.simulations  = simulations
        self.n_timepoints = n_timepoints

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        sim = self.simulations[idx]

        # ── 3-parameter vector [tau, gamma, rho] ─────────────────────────────
        params = np.array([
            sim['params']['tau'],
            sim['params']['gamma'],
            sim['params']['rho'],
        ], dtype=np.float32)

        # ── SIR trajectories [T, 3] ───────────────────────────────────────────
        S = sim['output']['S']
        I = sim['output']['I']
        R = sim['output']['R']
        y = np.stack([S, I, R], axis=1).astype(np.float32)

        return params, y


# ============================================================================
# BATCH WRAPPER
# ============================================================================

class BatchWrapper:
    """
    Thin wrapper so the training loop can use `batch.params` and `batch.y`,
    matching the API expected by HybridSplineFourierMLPPhysics.forward().
    """

    def __init__(self, params, y):
        self.params = params
        self.y      = y

    def to(self, device):
        self.params = self.params.to(device)
        self.y      = self.y.to(device)
        return self


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_sir(batch_list):
    """
    Custom collate for SIR batches (no graph stats).

    Args:
        batch_list : list of (params[3], y[T,3]) tuples

    Returns:
        BatchWrapper
    """
    params_list = [item[0] for item in batch_list]
    y_list      = [item[1] for item in batch_list]

    params = torch.FloatTensor(np.array(params_list))   # (B, 3)
    y      = torch.FloatTensor(np.array(y_list))         # (B, T, 3)

    return BatchWrapper(params, y)


# ============================================================================
# DATA LOADERS
# ============================================================================

def create_dataloaders(dataset_path, batch_size=32, num_workers=0):
    """
    Load the SIR dataset pickle and return train/val/test DataLoaders.

    Expected pickle structure
    ─────────────────────────
    {
      'train': {'simulations': [...]},
      'val':   {'simulations': [...]},
      'test':  {'simulations': [...]},
      'metadata': {'n_timepoints': int, ...}
    }

    Each simulation dict must contain:
      sim['params']  → {'tau': float, 'gamma': float, 'rho': float}
      sim['output']  → {'t': array, 'S': array, 'I': array, 'R': array}

    Args:
        dataset_path : path to .pkl file
        batch_size   : batch size (use drop_last=True for training to
                       avoid BatchNorm errors on single-sample batches)
        num_workers  : DataLoader workers

    Returns:
        dict with keys: 'train', 'val', 'test', 'metadata', 'n_timepoints'
    """
    print(f"\nLoading dataset: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # ── Infer number of time points from first simulation ────────────────────
    first_sim    = data['train']['simulations'][0]
    n_timepoints = len(first_sim['output']['t'])
    print(f"  n_timepoints : {n_timepoints}")

    # ── Build datasets ────────────────────────────────────────────────────────
    train_dataset = EpidemicDatasetSIR(data['train']['simulations'], n_timepoints)
    val_dataset   = EpidemicDatasetSIR(data['val']['simulations'],   n_timepoints)
    test_dataset  = EpidemicDatasetSIR(data['test']['simulations'],  n_timepoints)

    print(f"  Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # ── Build loaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True,           # prevents BatchNorm crash on 1-sample tail batch
        num_workers=num_workers,  collate_fn=collate_sir,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False,
        num_workers=num_workers,  collate_fn=collate_sir,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False,
        num_workers=num_workers,  collate_fn=collate_sir,
    )

    # Store n_timepoints in metadata so downstream scripts can read it
    metadata = data.get('metadata', {})
    metadata['n_timepoints'] = n_timepoints

    return {
        'train'       : train_loader,
        'val'         : val_loader,
        'test'        : test_loader,
        'metadata'    : metadata,
        'n_timepoints': n_timepoints,
    }


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(predictions, targets, prefix=''):
    """
    Compute a comprehensive set of regression metrics.

    Args:
        predictions : [N, T, 3]  or Tensor equivalent
        targets     : [N, T, 3]  or Tensor equivalent
        prefix      : optional string prefix for metric keys

    Returns:
        dict of float values (both lowercase and UPPERCASE keys for
        backward compatibility with existing plotting code)
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # ── Overall ───────────────────────────────────────────────────────────────
    mae  = np.abs(predictions - targets).mean()
    mse  = ((predictions - targets) ** 2).mean()
    rmse = np.sqrt(mse)

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2     = 1.0 - ss_res / (ss_tot + 1e-8)

    # ── Per-compartment MAE ───────────────────────────────────────────────────
    mae_s = np.abs(predictions[:, :, 0] - targets[:, :, 0]).mean()
    mae_i = np.abs(predictions[:, :, 1] - targets[:, :, 1]).mean()
    mae_r = np.abs(predictions[:, :, 2] - targets[:, :, 2]).mean()

    # ── Per-compartment R² ────────────────────────────────────────────────────
    def _r2(pred, true):
        ss_r = np.sum((true - pred) ** 2)
        ss_t = np.sum((true - true.mean()) ** 2)
        return 1.0 - ss_r / (ss_t + 1e-8)

    r2_s = _r2(predictions[:, :, 0], targets[:, :, 0])
    r2_i = _r2(predictions[:, :, 1], targets[:, :, 1])
    r2_r = _r2(predictions[:, :, 2], targets[:, :, 2])

    p = prefix
    return {
        # lowercase
        f'{p}mae'  : mae,   f'{p}mse'  : mse,   f'{p}rmse' : rmse,
        f'{p}r2'   : r2,
        f'{p}mae_s': mae_s, f'{p}mae_i': mae_i,  f'{p}mae_r': mae_r,
        f'{p}r2_s' : r2_s,  f'{p}r2_i' : r2_i,   f'{p}r2_r' : r2_r,
        # UPPERCASE (backward-compatible aliases)
        f'{p}MAE'  : mae,   f'{p}MSE'  : mse,   f'{p}RMSE' : rmse,
        f'{p}R2'   : r2,
        f'{p}MAE_S': mae_s, f'{p}MAE_I': mae_i,  f'{p}MAE_R': mae_r,
        f'{p}R2_S' : r2_s,  f'{p}R2_I' : r2_i,   f'{p}R2_R' : r2_r,
    }


# ============================================================================
# DEVICE HELPER
# ============================================================================

def get_device():
    """Return GPU device if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ Using GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\n⚠  Using CPU (GPU not available)")
    return device


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience  : epochs to wait after last improvement
        min_delta : minimum change to count as improvement
        mode      : 'min' (lower is better) or 'max' (higher is better)
    """

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.min_delta if self.mode == 'min'
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("utils_SIR.py  ·  3-Parameter SIR Dataset Utilities")
    print("=" * 70)
    print("\nKey classes / functions:")
    print("  EpidemicDatasetSIR  – params: [tau, gamma, rho]")
    print("  BatchWrapper        – .params (B,3), .y (B,T,3)")
    print("  collate_sir         – collate function (no graph stats)")
    print("  create_dataloaders  – returns train/val/test loaders")
    print("  compute_metrics     – MAE, RMSE, R², per-compartment metrics")
    print("  get_device          – GPU/CPU selection")
    print("  EarlyStopping       – patience-based early stopping")
    print("\nChanges from utils_AGE_MLP1.py:")
    print("  ✗ GraphStatsNormalizer      removed")
    print("  ✗ graph_stats in batch      removed")
    print("  ✗ 7 age-structured params   replaced by 3: tau, gamma, rho")
    print("  ✗ networkx dependency       removed")
    print("=" * 70)
