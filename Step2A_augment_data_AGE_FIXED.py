"""
step2A_augment_data_SIR3param.py
=================================
Data augmentation for the 3-parameter SIR emulator.

Parameters: tau (τ), gamma (γ), rho (ρ)

Two augmentation strategies:
    1. Parameter augmentation  — add small noise to (tau, gamma, rho)
       while keeping the trajectory unchanged.  Teaches the model that
       nearby parameters produce similar trajectories.

    2. Compartment augmentation — add small noise to (S, I, R) values
       while enforcing the conservation law S + I + R = N.  Teaches
       the model robustness to stochastic trajectory variability.

Changes from Step2A_augment_data_AGE.py:
    7 age-structured param names replaced by ['tau', 'gamma', 'rho']
    param_ranges default updated to match PARAM_RANGES in step1
    class renamed: EpidemicAugmenterForSplitData_AGE
                 →  EpidemicAugmenterSIR
    all output key names (tau, gamma, rho) match utils_SIR.py
    default input/output filenames updated to sir3param convention
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

# ============================================================================
# AUGMENTER CLASS
# ============================================================================

class EpidemicAugmenterSIR:
    """
    Data augmenter for 3-parameter SIR simulations.

    Why augment?
    ------------
    Your ABM simulations are expensive. Augmentation artificially
    expands the training set by creating plausible variations of
    existing simulations, improving generalisation without running
    additional full simulations.

    Two strategies are implemented:

    Parameter augmentation
        Takes an existing (tau, gamma, rho) + trajectory pair and
        returns a new pair where the parameters have been perturbed by
        a small multiplicative noise.  The trajectory is kept unchanged
        — we are effectively telling the model "this trajectory is also
        consistent with these slightly different parameters".
        This is valid because the epidemic dynamics are smooth functions
        of the parameters (except very near R₀ = 1).

    Compartment augmentation
        Takes an existing trajectory and adds small noise to S, I, R
        independently, then re-normalises so S + I + R = N exactly.
        This teaches the model robustness to the stochastic variability
        inherent in the ABM.
    """

    def __init__(
        self,
        param_noise_std: float = 0.013,
        compartment_noise_std: float = 0.01,
        n_param_augment: int = 2,
        n_compartment_augment: int = 1,
        param_ranges: dict = None,
    ):
        """
        Args:
            param_noise_std         : std of multiplicative noise on params
            compartment_noise_std   : std of multiplicative noise on S, I, R
            n_param_augment         : how many parameter-noised copies per sim
            n_compartment_augment   : how many compartment-noised copies per sim
            param_ranges            : hard bounds to clip noised params
                                      (read from pickle metadata if None)
        """
        self.param_noise_std       = param_noise_std
        self.compartment_noise_std = compartment_noise_std
        self.n_param_augment       = n_param_augment
        self.n_compartment_augment = n_compartment_augment

        # Default ranges match PARAM_RANGES in step1
        self.param_ranges = param_ranges or {
        'tau'  : (0.001, 0.15),    # was 0.9 — reduced dramatically
        'gamma': (0.01,  0.50),    # mild reduction
        'rho'  : (0.001, 0.010),
    }

    # ── Public method 

    def augment_simulation(self, sim: dict) -> list:
        """
        Augment a single simulation.

        Returns a list of dicts:
            [original, param_aug_1, param_aug_2, ..., comp_aug_1, ...]

        Each dict has the same format as the original:
            {'params': {'tau': …, 'gamma': …, 'rho': …},
             'output': {'t': …, 'S': …, 'I': …, 'R': …, …}}
        """
        augmented = [deepcopy(sim)]

        for _ in range(self.n_param_augment):
            augmented.append(self._augment_parameters(sim))

        for _ in range(self.n_compartment_augment):
            augmented.append(self._augment_compartments(sim))

        return augmented

    # ── Private helpers ───────────────────────────────────────────────────────

    def _augment_parameters(self, sim: dict) -> dict:
        """
        Return a copy of sim with (tau, gamma, rho) slightly perturbed.

        Noise is multiplicative: new_param = param × (1 + N(0, std))
        Parameters are clipped to stay within self.param_ranges.
        """
        sim_noisy = deepcopy(sim)
        params    = sim['params']

        noisy_params = {}
        for name in ['tau', 'gamma', 'rho']:
            noise          = np.random.normal(0, self.param_noise_std)
            value_noisy    = params[name] * (1 + noise)

            if name in self.param_ranges:
                lo, hi     = self.param_ranges[name]
                value_noisy = float(np.clip(value_noisy, lo, hi))

            noisy_params[name] = float(value_noisy)

        sim_noisy['params'] = noisy_params
        return sim_noisy

    def _augment_compartments(self, sim: dict) -> dict:
        """
        Return a copy of sim with small noise added to S, I, R.

        After adding noise the compartments are re-normalised so that
        S + I + R = N exactly (conservation law preserved).
        Existing std fields (S_std, I_std, R_std) are carried over
        unchanged — they describe the original stochastic replicates
        and should not be perturbed.
        """
        sim_noisy = deepcopy(sim)
        output    = sim['output']

        S = output['S'].copy()
        I = output['I'].copy()
        R = output['R'].copy()

        # Multiplicative noise
        S_noisy = S * (1 + np.random.normal(0, self.compartment_noise_std, S.shape))
        I_noisy = I * (1 + np.random.normal(0, self.compartment_noise_std, I.shape))
        R_noisy = R * (1 + np.random.normal(0, self.compartment_noise_std, R.shape))


# Why multiplicative?
# Because epidemiological parameters are scale-sensitive and positive.
# Additive noise would distort small rho values.
        # Non-negativity
        S_noisy = np.maximum(S_noisy, 0)
        I_noisy = np.maximum(I_noisy, 0)
        R_noisy = np.maximum(R_noisy, 0)

        # Re-normalise to enforce S + I + R = N exactly
        N     = S[0] + I[0] + R[0]          # read N from t=0
        total = S_noisy + I_noisy + R_noisy
        factor = N / (total + 1e-8)
        S_noisy *= factor
        I_noisy *= factor
        R_noisy *= factor

        # Build new output dict
        new_output = {
            't': output['t'].copy(),
            'S': S_noisy,
            'I': I_noisy,
            'R': R_noisy,
        }

        # Carry over std fields from original replicates (if present)
        for key in ['S_std', 'I_std', 'R_std', 'n_replicates']:
            if key in output:
                new_output[key] = (output[key].copy()
                                   if hasattr(output[key], 'copy')
                                   else output[key])

        sim_noisy['output'] = new_output
        return sim_noisy


# ============================================================================
# AUGMENT A FULL SPLIT DATASET
# ============================================================================

#Split-level augmentation: apply the augmenter to every simulation in the requested splits (train).  The network and metadata are preserved without modification.  The augmented simulations are added to the original ones, and the new total count is recorded in metadata.
def augment_split_dataset(
    split_data: dict,
    augmenter: EpidemicAugmenterSIR,
    augment_train: bool = True, # aumenting th training set only not validation and test sets, because we want to preserve the true distribution of those splits for evaluation integrity.  The training set is where we want to expand the data to improve model learning.
    augment_val: bool   = False,   
    augment_test: bool  = False,
) -> dict:
    """
    Augment the train (and optionally val/test) split of a dataset.

    Only the training set is augmented by default.  Augmenting val/test
    would violate evaluation integrity — those splits must reflect the
    true distribution of the ABM, not artificially expanded versions.

    Args:
        split_data    : dict from step2_split_data.py
                        (keys: train, val, test, network, metadata)
        augmenter     : fitted EpidemicAugmenterSIR
        augment_train : augment the training split (default: True)
        augment_val   : augment the validation split (default: False)
        augment_test  : augment the test split (default: False)

    Returns:
        augmented_split : same structure as split_data, with simulations
                          expanded in the requested splits
    """
    print("\n" + "=" * 70)
    print("DATA AUGMENTATION 3-Parameter SIR (tau, gamma, rho)")
    print("=" * 70)
    print(f"\n  Parameter noise std    : {augmenter.param_noise_std}")
    print(f"  Compartment noise std  : {augmenter.compartment_noise_std}")
    print(f"  Param copies / sim     : {augmenter.n_param_augment}")
    print(f"  Compartment copies/sim : {augmenter.n_compartment_augment}")
    mult = 1 + augmenter.n_param_augment + augmenter.n_compartment_augment
    print(f"  Total multiplier       : {mult}x  (1 original + {mult-1} augmented)")

    # ── Preserve network ──────────────────────────────────────────────────────
    network = split_data.get('network', {})
    if not isinstance(network, dict):
        print(f"\n  Warning: network is {type(network)}, regenerating ...")
        import networkx as nx
        _N = split_data['metadata'].get('total_population', 10000)
        _m = split_data['metadata'].get('m', 2)
        network = {'graph': nx.barabasi_albert_graph(_N, _m, seed=42),
                   'N': _N, 'm': _m}

    augmented_split = {
        'network' : network,
        'metadata': deepcopy(split_data['metadata']),
    }

    # ── Augment each split ────────────────────────────────────────────────────
    flags = {'train': augment_train, 'val': augment_val, 'test': augment_test}

    for split_name, should_augment in flags.items():
        original_sims = split_data[split_name]['simulations']
        n_original    = len(original_sims)

        if should_augment:
            print(f"\n  {split_name}: {n_original} original simulations ...")
            augmented_sims = []

            for sim in tqdm(original_sims, desc=f"    Augmenting {split_name}"):
                augmented_sims.extend(augmenter.augment_simulation(sim))

            n_aug = len(augmented_sims)
            print(f"    {n_original} → {n_aug} samples  ({n_aug/n_original:.1f}x)")

            augmented_split[split_name] = {
                'simulations'   : augmented_sims,
                'indices'       : split_data[split_name]['indices'],
                'original_size' : n_original,
                'augmented_size': n_aug,
            }
        else:
            print(f"\n  {split_name}: not augmented  ({n_original} samples kept)")
            augmented_split[split_name] = deepcopy(split_data[split_name])

    # ── Record augmentation config in metadata ────────────────────────────────
    augmented_split['metadata']['augmentation'] = {
        'param_noise_std'       : augmenter.param_noise_std, # This is the standard deviation of multiplicative noise applied to parameters.
        'compartment_noise_std' : augmenter.compartment_noise_std,# Observed trajectory=latent deterministic curve+ϵ
        'n_param_augment'       : augmenter.n_param_augment, #How many synthetic copies per simulation via parameter perturbation. If 2 → each simulation generates 2 additional parameter-perturbed versions.
        'n_compartment_augment' : augmenter.n_compartment_augment,
        'multiplier'            : mult, #1+nparam​+ncomp​ , 2 param copies + 1 comp copy = 4x total
        'augmented_splits'      : flags, # i can defend this design choice by saying it provides a clear record of which splits were augmented and with what config, which is useful for future reference and reproducibility.  It also keeps all augmentation-related info in one place within the metadata, making it easier to track and understand the dataset's history.  This way, anyone using the dataset later can easily see the augmentation details without having to refer back to code or external documentation. I did not augment the validation and test splits, but I still want to record that fact explicitly in the metadata for clarity and transparency.
    } # This records which splits were augmented and with what config, for future reference.

    # Summary
    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    for split_name in ['train', 'val', 'test']:
        n = len(augmented_split[split_name]['simulations'])
        print(f"  {split_name:5s}: {n} simulations")

    return augmented_split


import pandas as pd

def export_split_to_csv(split_data, filepath, split_name="train"):
    """
    Export one split (train/val/test) to CSV.

    Each simulation becomes one row:
        tau, gamma, rho,
        S_t0, S_t1, ...
        I_t0, I_t1, ...
        R_t0, R_t1, ...
    """

    sims = split_data[split_name]['simulations']
    rows = []

    for sim in sims:
        params = sim['params']
        output = sim['output']

        row = {
            'tau': params['tau'],
            'gamma': params['gamma'],
            'rho': params['rho'],
        }

        # Add time-series values
        for i, val in enumerate(output['S']):
            row[f"S_t{i}"] = val
        for i, val in enumerate(output['I']):
            row[f"I_t{i}"] = val
        for i, val in enumerate(output['R']):
            row[f"R_t{i}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

    print(f"\nCSV saved {filepath}")
    print(f"Shape: {df.shape}")

# ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Augment 3-parameter SIR split dataset.\n"
            "Parameters: tau, gamma, rho  (no age structure)"
        )
    )
    parser.add_argument('--input',   type=str,
                        default='epidemic_data_age_adaptive_sobol_split.pkl',
                        help='Split pickle from step2_split_data.py')
    parser.add_argument('--output',  type=str,  default=None,
                        help='epidemic_data_age_adaptive_sobol_augmented.pkl')
    parser.add_argument('--param_noise',type=float, default=0.013,
                        help='Param noise std(default: 0.013)')
    parser.add_argument('--compartment_noise', type=float, default=0.01,
                        help='Compartment noise std (default: 0.01)')
    parser.add_argument('--n_param_aug',type=int, default=2,
                        help='Param-noised copies/sim(default: 2)')
    parser.add_argument('--n_comp_aug',type=int,default=1,
                        help='Compartment copies/sim (default: 1)')
    parser.add_argument('--augment_train', action='store_true', default=True)
    parser.add_argument('--augment_val',   action='store_true', default=False)
    parser.add_argument('--augment_test',  action='store_true', default=False)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("STEP 2A — DATA AUGMENTATION")
    print("3-Parameter SIR: tau, gamma, rho")
    print("=" * 70)

    # Load split dataset
    print(f"\nLoading: {args.input}")
    with open(args.input, 'rb') as f:
        split_data = pickle.load(f)

    print(f"  Train : {len(split_data['train']['simulations'])}")
    print(f"  Val   : {len(split_data['val']['simulations'])}")
    print(f"  Test  : {len(split_data['test']['simulations'])}")

    # Confirm parameter keys are correct
    sample_params = split_data['train']['simulations'][0]['params']
    print(f"\n  Param keys found : {list(sample_params.keys())}")
    for expected in ['tau', 'gamma', 'rho']:
        if expected not in sample_params:
            print(f"  WARNING: '{expected}' not in params — "
                  f"check step1 output matches utils_SIR.py")

    # Report replicate data
    sample_out = split_data['train']['simulations'][0]['output']
    if 'S_std' in sample_out:
        n_reps = sample_out.get('n_replicates', '?')
        print(f"  Replicate std fields present  (n_replicates={n_reps})")

    # Read param ranges from metadata (falls back to defaults if absent)
    param_ranges = split_data['metadata'].get('param_ranges', None)

    # Build augmenter
    augmenter = EpidemicAugmenterSIR(
        param_noise_std=args.param_noise,
        compartment_noise_std=args.compartment_noise,
        n_param_augment=args.n_param_aug,
        n_compartment_augment=args.n_comp_aug,
        param_ranges=param_ranges,
    )

    # Split level augmentation
    augmented = augment_split_dataset(
        split_data, augmenter,
        augment_train=args.augment_train,
        augment_val=args.augment_val,
        augment_test=args.augment_test,
    )

    # Save
    if args.output is None:
        stem        = Path(args.input).stem
        output_path = Path(args.input).parent / f"{stem}_augmented.pkl"
    else:
        output_path = Path(args.output)

    print(f"\nSaving: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(augmented, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  Saved  ({size_mb:.2f} MB)")


    csv_path = output_path.with_suffix(".csv")
    export_split_to_csv(augmented, csv_path, split_name="train")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nPipeline:")
    print(f"  1.  step1_generate_data_SIR3param.py")
    print(f"  2.  step2_split_data.py          ->  {Path(args.input).name}")
    print(f"  2A. step2A_augment_data_SIR3param.py  ->  {output_path.name}")
    print(f"  3.  python step3_train_SIR3param.py --input {output_path}")
    print(f"  4.  python step4_validate_SIR3param.py")
    print(f"  5.  python step5_test_SIR3param.py")
