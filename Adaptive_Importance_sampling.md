Sequential Importance Weighted Sampling
 
  ▼
PHASE 1 — Initial Sobol Sampling
  │
  ├── Generate Sobol points in parameter cube
  │      θ = (τ, γ, ρ)
  │
  ├
  │
  ├── Run ABM simulations
  │      EoN.fast_SIR()
  │
  ├── For each θ:
  │      run n_replicates simulations
  │
  └── Store results
        S(t), I(t), R(t)
        
  │
  ▼
Dataset now contains initial simulations
  │
  ▼
PHASE 2 — Adaptive Importance Sampling
  │
  │  (repeated for n_rounds)
  │
  ├──────────────────────────────────────┐
  │                                      │
  │   Compute Importance Weights         │
  │                                      │
  │   θ_i → compute R₀                   │
  │                                      │
  │   R₀ = τ/γ × <k²>/<k>                │
  │                                      │
  │   Target distribution                │
  │   π(θ) ∝ exp[-sharpness (R₀−1)²]     │
  │                                      │
  │   Proposal q(θ) ≈ sobol initial      │
  │                                      │
  │   IS weight                          │
  │   w_i = π(θ_i) / q(θ_i)              │
  │                                      │
  │   Normalize weights                  │
  │   log-sum-exp trick (making the weights probabilities )                 │
  │                                      │
  │   Compute ESS                        │
  │   ESS = 1 / Σ w²                     │    How many independent samples your weighted dataset is equivalent to. (also check Weight collapse)
  │                                      │
  └──────────────────────────────────────┘
  │
  ▼
Resampling Step with kernel_smoothing
  │
  ├── Sample parameter sets
  │    proportional to weights
  │
  ├── High-weight points chosen more
  │
  ▼
Jitter
  │
  ├── Add Gaussian noise
  │
  │     θ_new = θ_resampled + ε
  │
  │     ε ~ N(0, bandwidth × range)
  │
  ├── Clip to valid parameter bounds
  │
  ▼
New parameter batch
  │
  ▼
Run ABM simulations again
(EoN.fast_SIR)
  │
  ▼
Append simulations to dataset
(all_sims grows)
  │
  ▼
Recompute ESS
Store diagnostics
  │
  ▼
Repeat adaptive round
  │
  ▼
After final round
  │
  ▼
Dataset summary
  │
  ├── compute R₀ distribution
  ├── report ESS history
  ├── check threshold coverage
  │
  ▼
Save dataset
(pickle file)
  │
  ▼
Optional
save_parameters_csv()
  │
  ▼
END