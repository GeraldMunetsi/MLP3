"""
utils_SIR.py  ·  Utility Functions for 3-Parameter SIR Emulator
================================================================
Parameters: tau (τ), gamma (γ), rho (ρ)

Changes from previous version:
  · Normalisation moved INTO __getitem__() — model never sees raw tau/gamma/rho
  · BatchWrapper now carries params_norm (→ Fourier encoder) and rho_raw (→ decoder IC)
  · PARAM_RANGES defined here as the SINGLE SOURCE OF TRUTH for normalisation
  · EarlyStopping fixed: monitors R² (mode='max'), patience=15, min_delta=1e-4
  · collate_sir updated to handle new batch fields
"""

import torch
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

from torch.utils.data import Dataset, DataLoader


# ============================================================================
# NORMALISATION CONSTANTS  ←  SINGLE SOURCE OF TRUTH
# ============================================================================
# These MUST match PARAM_RANGES in step1_data_generation.py.
# The model never sees these — normalisation happens in __getitem__().



PARAM_MINS = np.array([0.0005, 0.007,  0.001], dtype=np.float32)   # [tau, gamma, rho]
PARAM_MAXS = np.array([0.024,  0.5,  0.010], dtype=np.float32)


def normalise_params(params_raw: np.ndarray) -> np.ndarray:
    """
    Min-max normalise raw [tau, gamma, rho] to [0, 1]^3.

    Args:
        params_raw : (3,) or (N, 3) array of raw parameter values
    Returns:
        params_norm : same shape, each dimension in [0, 1]
    """
    return (params_raw - PARAM_MINS) / (PARAM_MAXS - PARAM_MINS + 1e-8)



# DATASET


class EpidemicDatasetSIR(Dataset):
    """
    PyTorch Dataset for the 3-parameter SIR emulator.

    Each item returns a dict with:
        params_norm : (3,)    normalised [tau, gamma, rho] in [0,1]
                              fed into the Fourier encoder
        rho_raw     : scalar  raw (un-normalised) rho in [0.001, 0.010]
                              used by decoder to pin S(0)=N(1-rho), I(0)=N*rho
        y           : (T, 3)  target [S(t), I(t), R(t)] trajectories
    """

    def __init__(self, simulations: list, n_timepoints: int):
        super().__init__()
        self.simulations  = simulations
        self.n_timepoints = n_timepoints

    def __len__(self) -> int:
        return len(self.simulations)

    def __getitem__(self, idx: int) -> dict:
        sim = self.simulations[idx]

        # ── Raw parameters ────────────────────────────────────────────────────
        params_raw = np.array([
            sim['params']['tau'],
            sim['params']['gamma'],
            sim['params']['rho'],
        ], dtype=np.float32)                                   # (3,)

        # ── Normalised parameters for Fourier encoder ─────────────────────────
        # Normalisation lives HERE — the model receives [0,1] inputs only.
        # Without this, cos(tiny * omega) ≈ 1 for all inputs → encoder collapse.
        params_norm = normalise_params(params_raw)             # (3,) in [0, 1]

        # ── Raw rho for decoder initial conditions ─────────────────────────────
        # S(0) = N*(1-rho) and I(0) = N*rho are hard constraints, not predictions.
        # The decoder needs RAW rho to compute actual counts in people.
        rho_raw = params_raw[2]                                # scalar float32

        # ── SIR trajectories ──────────────────────────────────────────────────
        S = sim['output']['S']
        I = sim['output']['I']
        R = sim['output']['R']
        y = np.stack([S, I, R], axis=1).astype(np.float32)    # (T, 3)

        return {
            'params_norm': params_norm,
            'rho_raw'    : rho_raw,
            'y'          : y,
        }


# ============================================================================
# BATCH WRAPPER
# ============================================================================

class BatchWrapper:
    """
    Thin wrapper for attribute-style batch access.

    Attributes:
        params_norm : (B, 3)    normalised params  → Fourier encoder
        rho_raw     : (B,)      raw rho            → decoder initial conditions
        y           : (B, T, 3) target trajectories
    """

    def __init__(self, params_norm, rho_raw, y):
        self.params_norm = params_norm
        self.rho_raw     = rho_raw
        self.y           = y

    def to(self, device):
        self.params_norm = self.params_norm.to(device)
        self.rho_raw     = self.rho_raw.to(device)
        self.y           = self.y.to(device)
        return self


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_sir(batch_list: list) -> BatchWrapper:
    """
    Custom collate: stacks dicts from EpidemicDatasetSIR into a BatchWrapper.

    Args:
        batch_list : list of dicts (one per sample)
    Returns:
        BatchWrapper
    """
    params_norm = torch.FloatTensor(
        np.stack([item['params_norm'] for item in batch_list])  # (B, 3)
    )
    rho_raw = torch.FloatTensor(
        np.array([item['rho_raw'] for item in batch_list])       # (B,)
    )
    y = torch.FloatTensor(
        np.stack([item['y'] for item in batch_list])             # (B, T, 3)
    )
    return BatchWrapper(params_norm, rho_raw, y)


# ============================================================================
# DATA LOADERS
# ============================================================================

def create_dataloaders(dataset_path: str, batch_size: int = 32,
                       num_workers: int = 0) -> dict:
    """
    Load the SIR dataset pickle and return train/val/test DataLoaders.

    Expected pickle structure
    ─────────────────────────
    {
      'train'   : {'simulations': [...]},
      'val'     : {'simulations': [...]},
      'test'    : {'simulations': [...]},
      'metadata': {'n_timepoints': int, ...}
    }

    Each simulation dict must contain:
      sim['params']  → {'tau': float, 'gamma': float, 'rho': float}
      sim['output']  → {'t': array, 'S': array, 'I': array, 'R': array}

    Args:
        dataset_path : path to .pkl file
        batch_size   : samples per batch
        num_workers  : DataLoader worker processes

    Returns:
        dict with keys: 'train', 'val', 'test', 'metadata', 'n_timepoints'
    """
    print(f"\nLoading dataset : {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # ── Infer number of time points from first simulation ────────────────────
    first_sim    = data['train']['simulations'][0]
    n_timepoints = len(first_sim['output']['t'])
    print(f"  n_timepoints  : {n_timepoints}")

    # ── Build datasets ────────────────────────────────────────────────────────
    train_dataset = EpidemicDatasetSIR(data['train']['simulations'], n_timepoints)
    val_dataset   = EpidemicDatasetSIR(data['val']['simulations'],   n_timepoints)
    test_dataset  = EpidemicDatasetSIR(data['test']['simulations'],  n_timepoints)

    print(f"  Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = True,          # prevents BatchNorm crash on 1-sample tail batch
        num_workers = num_workers,
        collate_fn  = collate_sir,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        num_workers = num_workers,
        collate_fn  = collate_sir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        num_workers = num_workers,
        collate_fn  = collate_sir,
    )

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

def compute_metrics(predictions, targets, prefix: str = '') -> dict:
    """
    Compute regression metrics for SIR trajectory predictions.

    Args:
        predictions : (N, T, 3) tensor or array  — predicted [S, I, R]
        targets     : (N, T, 3) tensor or array  — ground truth [S, I, R]
        prefix      : optional string prefix for all keys (e.g. 'val_')

    Returns:
        dict of floats with both lowercase and UPPERCASE keys
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    def _r2(pred, true):
        ss_r = np.sum((true - pred) ** 2)
        ss_t = np.sum((true - true.mean()) ** 2)
        return float(1.0 - ss_r / (ss_t + 1e-8))

    mae  = float(np.abs(predictions - targets).mean())
    mse  = float(((predictions - targets) ** 2).mean())
    rmse = float(np.sqrt(mse))
    r2   = _r2(predictions, targets)

    mae_s = float(np.abs(predictions[:, :, 0] - targets[:, :, 0]).mean())
    mae_i = float(np.abs(predictions[:, :, 1] - targets[:, :, 1]).mean())
    mae_r = float(np.abs(predictions[:, :, 2] - targets[:, :, 2]).mean())

    r2_s = _r2(predictions[:, :, 0], targets[:, :, 0])
    r2_i = _r2(predictions[:, :, 1], targets[:, :, 1])
    r2_r = _r2(predictions[:, :, 2], targets[:, :, 2])

    p = prefix
    return {
        f'{p}mae'  : mae,   f'{p}mse'  : mse,   f'{p}rmse' : rmse,
        f'{p}r2'   : r2,
        f'{p}mae_s': mae_s, f'{p}mae_i': mae_i,  f'{p}mae_r': mae_r,
        f'{p}r2_s' : r2_s,  f'{p}r2_i' : r2_i,   f'{p}r2_r' : r2_r,
        # UPPERCASE aliases for backward compatibility
        f'{p}MAE'  : mae,   f'{p}MSE'  : mse,   f'{p}RMSE' : rmse,
        f'{p}R2'   : r2,
        f'{p}MAE_S': mae_s, f'{p}MAE_I': mae_i,  f'{p}MAE_R': mae_r,
        f'{p}R2_S' : r2_s,  f'{p}R2_I' : r2_i,   f'{p}R2_R' : r2_r,
    }


# ============================================================================
# DEVICE HELPER
# ============================================================================

def get_device() -> torch.device:
    """Return CUDA GPU if available, otherwise CPU."""
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

    Correct usage — monitor val R², stop when it plateaus:

        stopper = EarlyStopping(patience=15, min_delta=1e-4, mode='max')

        for epoch in range(max_epochs):
            train_one_epoch(...)
            val_metrics = evaluate(model, val_loader)

            # Save checkpoint if this is the best epoch
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                torch.save(model.state_dict(), 'best_model.pt')

            # Check stopping criterion
            if stopper(val_metrics['r2']):
                print(f"Early stopping at epoch {epoch}")
                break

    Args:
        patience  : epochs to wait after last improvement before stopping
                    15 is appropriate for this model (converges ~20-40 epochs)
        min_delta : minimum absolute improvement to reset the counter
                    1e-4 prevents stopping on numerical noise in R²
        mode      : 'max' for R² (higher is better)
                    'min' for loss (lower is better)
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4,
                 mode: str = 'max'):
        assert mode in ('min', 'max'), "mode must be 'min' or 'max'"
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Args:
            score : current epoch's monitored metric (e.g. val_metrics['r2'])
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset state — use when resuming training from a checkpoint."""
        self.counter    = 0
        self.best_score = None
        self.early_stop = False


# ============================================================================
# QUICK TEST
# ============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("utils_SIR.py  ·  3-Parameter SIR Dataset Utilities")
#     print("=" * 70)

#     # ── Verify normalisation ─────────────────────────────────────────────────
#     print("\n── Normalisation check ──────────────────────────────────────────────")
#     test_cases = np.array([
#         [0.001, 0.01,  0.001],   # → [0, 0, 0]
#         [0.15,  0.50,  0.010],   # → [1, 1, 1]
#         [0.075, 0.255, 0.0055],  # → [~0.5, ~0.5, ~0.5]
#     ], dtype=np.float32)

#     normed = normalise_params(test_cases)
#     for raw, norm in zip(test_cases, normed):
#         print(f"  raw={raw}  →  norm={norm.round(3)}")

#     # ── Verify BatchWrapper fields ────────────────────────────────────────────
#     print("\n── BatchWrapper field check ─────────────────────────────────────────")
#     fake_batch = BatchWrapper(
#         params_norm = torch.zeros(4, 3),
#         rho_raw     = torch.ones(4) * 0.005,
#         y           = torch.zeros(4, 50, 3),
#     )
#     print(f"  params_norm : {fake_batch.params_norm.shape}   → Fourier encoder")
#     print(f"  rho_raw     : {fake_batch.rho_raw.shape}         → decoder IC")
#     print(f"  y           : {fake_batch.y.shape}  → training target")

#     # ── Verify EarlyStopping ─────────────────────────────────────────────────
#     print("\n── EarlyStopping (mode=max, patience=3, min_delta=1e-4) ─────────────")
#     stopper = EarlyStopping(patience=3, min_delta=1e-4, mode='max')
#     for ep, r2 in enumerate([0.80, 0.85, 0.86, 0.86, 0.86, 0.86]):
#         stop = stopper(r2)
#         print(f"  epoch {ep+1}: R²={r2:.4f}  counter={stopper.counter}  stop={stop}")
#         if stop:
#             break

#     # ── Show PARAM_MINS/MAXS ──────────────────────────────────────────────────
#     print("\n── PARAM_RANGES (single source of truth) ────────────────────────────")
#     for name, lo, hi in zip(['tau', 'gamma', 'rho'], PARAM_MINS, PARAM_MAXS):
#         print(f"  {name:<6} ∈ [{lo:.4f}, {hi:.4f}]")

#     print("=" * 70)