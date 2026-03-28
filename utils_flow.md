step1_data_generation.py
        │
        │  saves pickle with sim['params']['rho'] = 0.005  (raw float)
        ▼
utils_SIR.py  →  EpidemicDatasetSIR.__getitem__()
        │
        │  reads sim['params']['rho']
        │  params_raw = np.array([tau, gamma, rho])   ← rho is index [2]
        │  rho_raw = params_raw[2]                    ← extracted HERE
        │  returns dict with 'rho_raw': rho_raw
        ▼
utils_SIR.py  →  collate_sir()
        │
        │  stacks rho_raw across the batch
        │  BatchWrapper.rho_raw = torch.FloatTensor([0.005, 0.003, ...])  shape (B,)
        ▼
step3_train.py  →  training loop
        │
        │  batch = next(train_loader)     ← BatchWrapper
        │  batch.to(device)
        │  pred = model(batch)            ← batch.rho_raw is inside here
        ▼
step0_model.py  →  HybridSplineFourierMLPPhysics.forward()
        │
        │  rho_raw = data.rho_raw         ← arrives from batch, NOT data generation
        │  params_norm = data.params_norm
        │  ...
        ▼
SplineTemporalDecoderPhysics.forward(z, rho_raw)
        │
        │  S_0 = (1 - rho_raw) * N       ← used here for hard constraint
        │  I_0 = rho_raw * N