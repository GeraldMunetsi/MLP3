"""
step0_model.py — SIR Epidemic Emulator

Architecture overview

1. StandardRFF          : [tau,gamma,rho] > 128-dim Fourier embedding                     
2. Fusion MLP           : 128 → latent_dim (learns nonlinear combinations)
3. S decoder            : monotone-decreasing B-spline via cumprod retention
4. g(t) decoder         : free B-spline + sigmoid → g ∈ (0,1)
5. Conservation         : I = (N-S)·g,  R = (N-S)·(1-g)
                          → S+I+R = N exactly, I≥0, R≥0 always

Key design choices

• StandardRFF encodes all 3 params JOINTLY → captures τ/γ interactions for R₀
• S uses cumprod of sigmoid retention rates → guaranteed monotone decreasing
• g(0) pinned to ≈1 via first coefficient = 8.0 (sigmoid(8) = 0.9997)
  so I(0) = (N-S₀)·g(0) ≈ Nρ and R(0) ≈ 0 as required by SIR physics
• No softplus/clamp on R — conservation is exact by construction
• Free (unconstrained) g(t) allows bell-curve I(t) for R₀ > 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import BSpline

timepoints=80
N=100000
knots=5
class StandardRFF(nn.Module):
    def __init__(self, n_params=3, n_fourier=64, sigma=1.0):
        super().__init__()
        W = torch.randn(n_fourier, n_params) * sigma  # each row is a random 3D vector
        self.register_buffer('W', W)
        self.scale = (2.0 / n_fourier) ** 0.5        # RBF kernel normalisation
        self.output_dim = 2 * n_fourier

    def forward(self, x):
        z   = x @ self.W.T
        phi = self.scale * torch.cat([torch.cos(z), torch.sin(z)], dim=1)
        return phi    

   

# 2. B-SPLINE LAYER 

class BSplineLayer(nn.Module):
    """
    Differentiable B-spline evaluation layer

    Converts n_knots control-point coefficients into timepoints smooth values
    using a pre-computed (frozen) basis matrix B of shape (timepoints, n_knots).

    output[b, t] = Σ coeffs[b, k] × B[t, k]= coeffs @ B.T   (batch matrix multiply)
    The basis matrix is computed once at initilization using scipy and registered as a
    buffer (moves to my CPU automatically, never trained).
    """
    def __init__(self, n_knots, n_timepoints=timepoints, degree= 3):
        super().__init__()
        self.n_knots      = n_knots
        self.n_timepoints = timepoints

        # Build clamped B-spline knot vector
        # Clamped = curve passes exactly through first and last control points
        internal = np.linspace(0, 1, n_knots - degree + 1)
        knots    = np.concatenate([
            np.zeros(degree),
            internal,
            np.ones(degree),
        ]) #Repetition at ends ensures the curve touches the first and last control points

        # Evaluate all basis functions at each timestep
        t_eval  = np.linspace(0, 1, timepoints)
        B_np    = np.zeros((timepoints, n_knots))
        for k in range(n_knots):
            c = np.zeros(n_knots)
            c[k] = 1.0
            B_np[:, k] = BSpline(knots, c, degree)(t_eval)

        self.register_buffer('B', torch.tensor(B_np, dtype=torch.float32))

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs : (batch, n_knots)
        Returns:
            curve  : (batch, timepoints)
        """
        return coeffs @ self.B.T    # (batch, timepoints)




# 3. TEMPORAL DECODER

class TemporalDecoder(nn.Module):
    """
    Decodes latent vector z and unnormilized rho into SIR trajectories.

    S compartment
  
    Predicted via a monotone-decreasing B-spline.
    Uses cumprod of sigmoid retention rates so S can only ever go down:
        S_coeffs[k] = S₀ × r₁ × r₂ × ... × rₖ,   each rᵢ ∈ (0,1)
    Guarantees: S(t) ≤ S(0) = N(1-ρ) for all t.

    g(t) function
    g(t) = I(t) / (N - S(t)) = fraction of ever-infected still infectious.
    Predicted as a FREE B-spline , then we sigmoid it such that g ∈ (0,1).
    First spline coefficient is pinned to 8.0 so sigmoid(8) ≈ 1 → g(0) ≈ 1.

    I and R

    I(t) = (N - S(t)) × g(t)≥ 0 always (both factors ≥ 0)
    R(t) = (N - S(t)) × (1 - g(t))≥ 0 always (g < 1)
    S + I + R = S + (N-S)·[g+(1-g)] = N, exact conservation

    Args:
        latent_dim : dimension of input z from fusion MLP
        n_knots : number of B-spline control points
        n_timepoints : output timesteps (50)
        total_population: N (10000) 
        hidden_dim: hidden size for retention and g networks
    """
    def __init__(
        self,
        latent_dim      : int,
        n_knots         : int = knots,
        total_population: int = N,
        hidden_dim      : int = 64,
    ):
        super().__init__()
        self.N            = float(total_population)
        self.n_knots      = n_knots
        self.n_timepoints = timepoints

        # S decoder: predicts K-1 retention rates → monotone decreasing S(t)
        self.predict_S_retention = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_knots - 1),
        )

        # R decoder: predicts K-1 retention rates → monotone increasing R(t)
        self.predict_R_rates = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_knots - 1),
        )

        # B-spline layers — one per compartment
        self.spline_S = BSplineLayer(n_knots, timepoints)
        self.spline_R = BSplineLayer(n_knots, timepoints)


    def forward(self, z: torch.Tensor, rho_raw: torch.Tensor) -> tuple:
        batch_size = z.size(0)
        device     = z.device

        # ── S(t): monotone decreasing ─────────────────────────────────────────
        S_0 = ((1.0 - rho_raw) * self.N).unsqueeze(1)       # (batch, 1)

        retention_raw   = self.predict_S_retention(z)        # (batch, K-1)  ← Bug 2 fix
        retention_rates = torch.sigmoid(retention_raw)       # ∈ (0, 1)

        ones_S    = torch.ones(batch_size, 1, device=device)
        all_rates = torch.cat([ones_S, retention_rates], dim=1)  # (batch, K)  ← Bug 1 fix

        cum_product = torch.cumprod(all_rates, dim=1)        # (batch, K), decreasing
        S_coeffs    = S_0 * cum_product                      # (batch, K)
        S_pred      = self.spline_S(S_coeffs)                # (batch, T)

        # ── f(t) = R(t)/(N−S(t)): monotone increasing ────────────────────────
        r_raw = self.predict_R_rates(z)                      # (batch, K-1)  ← Bug 4 fix
        r     = torch.sigmoid(r_raw)                         # ∈ (0, 1)

        ones_R = torch.ones(batch_size, 1, device=device)
        all_r  = torch.cat([ones_R, r], dim=1)               # (batch, K)

        cum_r    = torch.cumprod(all_r, dim=1)               # (batch, K), decreasing ∈ (0,1]
        f_coeffs = 1.0 - cum_r                               # (batch, K), increasing ∈ [0,1)
        f_t      = self.spline_R(f_coeffs)                   # (batch, T), monotone increasing

        # ── I and R from conservation ─────────────────────────────────────────
        ever_infected = self.N - S_pred                      # ≥ 0 always

        R_pred = ever_infected * f_t                         # (batch, T) ≥ 0, monotone ↑
        I_pred = ever_infected * (1.0 - f_t)                 # (batch, T) ≥ 0, bell-shaped

        # S + I + R = S + (N-S)·f + (N-S)·(1-f) = S + (N-S) = N  ✓
        return S_pred, I_pred, R_pred
    

    # def forward(self, z: torch.Tensor, rho_raw: torch.Tensor) -> tuple:
    #     """
    #     Args:
    #         z       : (batch, latent_dim)
    #         rho_raw : (batch,) seed fraction

    #     Returns:
    #         S_pred, I_pred, R_pred  each (batch, n_timepoints)
    #     """

    #     batch_size = z.size(0)
    #     device     = z.device

    #     # S₀ = N(1-ρ)
    #     S_0 = ((1.0 - rho_raw) * self.N).unsqueeze(1)

    #     # -----------------------------
    #     # S(t) monotone spline
    #     # -----------------------------

    #     retention_raw   = self.predict_S_retention(z)
    #     retention_rates = torch.sigmoid(retention_raw)

    #     ones      = torch.ones(batch_size, 1, device=device)
    #     all_rates = torch.cat([ones, retention_rates], dim=1)

    #     cum_product = torch.cumprod(all_rates, dim=1)

    #     S_coeffs = S_0 * cum_product
    #     S_pred   = self.spline_S(S_coeffs)

    #     # -----------------------------
    #     # g(t) spline
    #     # -----------------------------

    #     g_coeffs = self.predict_g_coeffs(z)        # (batch, n_knots)

    #     h_t = self.spline_g(g_coeffs)              # (batch, T)

    #     # neural output → probability
    #     g_tilde = torch.sigmoid(h_t)

    #     # time grid (must exist in your model)
    #     t = self.t_grid.to(device)                 # (T,)

    #     # weight function ensuring g(0)=1
    #     w = 1 - torch.exp(-t)

    #     w = w.unsqueeze(0)                         # (1, T)

    #     # enforce g(0)=1 exactly
    #     g = 1 - (1 - g_tilde) * w

    #     # -----------------------------
    #     # Compute I(t), R(t)
    #     # -----------------------------

    #     ever_infected = self.N - S_pred

    #     I_pred = ever_infected * g
    #     R_pred = ever_infected * (1.0 - g)

    #     return S_pred, I_pred, R_pred
    

# 4. FULL MODEL


class HybridSIREmulator(nn.Module):
    """
    Full SIR emulator.

    Pipeline:
        params_norm (batch,3)
            ↓  StandardRFF
        phi (batch, 2×n_fourier = 128)
            ↓  Fusion MLP
        z   (batch, latent_dim)
            ↓  TemporalDecoder  [+ rho_raw]
        S_pred, I_pred, R_pred  (batch, timepoints)

    Args:
        config : dict with keys:
            n_params        (int)   = 3
            n_fourier       (int)   = 64
            sigma           (float) = 1.0
            fusion_hidden   (int)   = 128
            latent_dim      (int)   = 64
            n_knots         (int)   = knots
            n_timepoints    (int)   = timepoints
            total_population(int)   = 10000
            decoder_hidden  (int)   = 64
            dropout         (float) = 0.1
    """
    def __init__(self, config: dict):
        super().__init__()

        n_params         = config.get('n_params',          3)
        n_fourier        = config.get('n_fourier',        64)
        sigma            = config.get('sigma',           1.0)
        fusion_hidden    = config.get('fusion_hidden',   128)
        latent_dim       = config.get('latent_dim',       64)
        n_knots          = config.get('n_knots',          knots)
       # n_timepoints     = config.get('n_timepoints',     timepoints)
        total_population = config.get('total_population', N)
        decoder_hidden   = config.get('decoder_hidden',   64)
        dropout          = config.get('dropout',         0.1)

        rff_out = 2 * n_fourier   # 128

        # 1. Standard RFF 
        self.rff = StandardRFF(
            n_params  = n_params,
            n_fourier = n_fourier,
            sigma     = sigma,
        )

        # 2. Fusion MLP 
        # 128 → fusion_hidden → latent_dim
        # LayerNorm stabilises training after the unnormalised RFF output
        self.fusion = nn.Sequential(
            nn.Linear(rff_out, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, latent_dim),
            nn.ReLU(),
        )

        # 3. Temporal decoder 
        self.temporal_decoder = TemporalDecoder(
            latent_dim      = latent_dim,
            n_knots         = n_knots,
            total_population= total_population,
            hidden_dim      = decoder_hidden,
        )

        self.n_timepoints = timepoints

    def forward(self, data, n_timesteps=None) -> torch.Tensor:
        """
        Args:
            data        : BatchWrapper with fields .params_norm and .rho_raw
            n_timesteps : ignored (kept for API compatibility); always uses
                          self.n_timepoints from config

        Returns:
            predictions : (batch, n_timepoints, 3)  — [S, I, R] stacked
        """
        params_norm = data.params_norm    # (batch, 3)
        rho_raw     = data.rho_raw        # (batch,)

        # Fourier embedding
        phi = self.rff(params_norm)       # (batch, 128)

        # Latent vector
        z   = self.fusion(phi)            # (batch, latent_dim)

        # SIR trajectories
        S_pred, I_pred, R_pred = self.temporal_decoder(z, rho_raw)

        # Stack to (batch, T, 3) for loss function compatibility
        return torch.stack([S_pred, I_pred, R_pred], dim=2)

    def get_component_params(self) -> dict:
        """Return parameter counts per component (useful for logging)."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        rff_frozen = sum(
            b.numel() for b in self.rff.buffers()
        )

        return {
            'rff_trainable'   : count(self.rff),
            'rff_frozen'      : rff_frozen,
            'fusion'          : count(self.fusion),
            'temporal_decoder': count(self.temporal_decoder),
            'total'           : count(self),
        }


# 
# 5. FACTORY FUNCTION
# 

def create_hybrid_mlp_model(config: dict) -> HybridSIREmulator:
    """
    Build and return the SIR emulator from a config dict.

    Minimal config:
        config = {
            'n_params'        : 3,
            'n_fourier'       : 64,
            'sigma'           : 1.0,
            'fusion_hidden'   : 128,
            'latent_dim'      : 64,
            'n_knots'         : 12,
            'n_timepoints'    : n_timepoints,
            'total_population': 10000,
            'decoder_hidden'  : 64,
            'dropout'         : 0.1,
        }
    """
    model = HybridSIREmulator(config)
    return model


# 6. SMOKE TEST


if __name__ == '__main__':
    import types

    print("=" * 60)
    print("SMOKE TEST — step0_model.py")
    print("=" * 60)

    config = {
        'n_params'        : 3,
        'n_fourier'       : 64,
        'sigma'           : 1.0,
        'fusion_hidden'   : 128,
        'latent_dim'      : 64,
        'n_knots'         : knots,
        #'n_timepoints'    : timepoints,
        'total_population': 10000,
        'decoder_hidden'  : 64,
        'dropout'         : 0.1,
    }

    model = create_hybrid_mlp_model(config)
    model.eval()

    # Fake batch of size 4
    batch_size = 4
    batch = types.SimpleNamespace(
        params_norm = torch.rand(batch_size, 3),               # τ,γ,ρ in [0,1]
        rho_raw     = torch.FloatTensor(batch_size).uniform_(0.001, 0.010),
    )

    with torch.no_grad():
        out = model(batch)

    print(f"\n  Input  params_norm : {batch.params_norm.shape}")
    print(f"  Input  rho_raw     : {batch.rho_raw.shape}")
    print(f"  Output predictions : {out.shape}  (batch, T, 3)")
    print()

    S = out[:, :, 0]
    I = out[:, :, 1]
    R = out[:, :, 2]

    # Conservation check
    total = S + I + R
    print(f"  Conservation  S+I+R = N ?")
    print(f"    mean = {total.mean().item():.4f}  (should be 10000)")
    print(f"    max deviation from N: {(total - 10000).abs().max().item():.6f}")

    # Non-negativity
    print(f"\n  Non-negativity:")
    print(f"    I min = {I.min().item():.4f}  (should be ≥ 0)")
    print(f"    R min = {R.min().item():.4f}  (should be ≥ 0)")

    # Initial conditions
    print(f"\n  Initial conditions (t=0):")
    for i in range(batch_size):
        rho    = batch.rho_raw[i].item()
        I0_exp = rho * 10000
        I0_got = I[i, 0].item()
        R0_got = R[i, 0].item()
        print(f"    sample {i}: ρ={rho:.4f}  I(0) expected≈{I0_exp:.1f}  "
              f"got={I0_got:.1f}  R(0)={R0_got:.4f}")

    # S monotone check
    S_diffs = S[:, 1:] - S[:, :-1]
    n_violations = (S_diffs > 1e-4).sum().item()
    print(f"\n  S monotone decreasing: {n_violations} violations (should be 0)")

    # Parameter counts
    comp = model.get_component_params()
    print(f"\n  Parameter counts:")
    print(f"    RFF trainable  : {comp['rff_trainable']:>8,}  (zero — fully frozen)")
    print(f"    RFF frozen     : {comp['rff_frozen']:>8,}  (W matrix)")
    print(f"    Fusion MLP     : {comp['fusion']:>8,}")
    print(f"    Temporal decoder: {comp['temporal_decoder']:>8,}")
    print(f"    TOTAL trainable: {comp['total']:>8,}")

    print("\n  All checks passed")