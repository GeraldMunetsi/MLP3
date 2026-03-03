"""
STEP 2: Split Data - FIXED
Preserves network structure and handles replicate data
"""

import pickle
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
    #This is how data set will look like after splitting, the network is preserved as a dict and the simulations are split into train, val and test with their corresponding indices. The metadata is also preserved for reference.
    split_data = {
        'train': {'simulations': train_sims, 'indices': train_idx.tolist()},
        'val': {'simulations': val_sims, 'indices': val_idx.tolist()},
        'test': {'simulations': test_sims, 'indices': test_idx.tolist()},
        'network': network,
        'metadata': dataset['metadata']
    }
    
    return split_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="epidemic_data_age_adaptive_sobol.pkl")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("STEP 2: DATA SPLITTING (FIXED)")
    print("="*70)
    
    print(f"\nLoading: {args.input}")
    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)
    
    n_samples = len(dataset['simulations'])
    
    # Check replicates
    if 'S_std' in dataset['simulations'][0]['output']:
        n_reps = dataset['simulations'][0]['output'].get('n_replicates', '?')
        print(f"  ✓ {n_samples} samples ({n_reps} replicates each)")
    else:
        print(f"  ✓ {n_samples} samples")
    
    split_data = split_dataset(dataset, args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"\n  Train: {len(split_data['train']['simulations'])}")
    print(f"  Val:   {len(split_data['val']['simulations'])}")
    print(f"  Test:  {len(split_data['test']['simulations'])}")
    
    if args.output is None:
        output_path = Path(args.input).parent / (Path(args.input).stem + '_split.pkl')
    else:
        output_path = Path(args.output)
    
    print(f"\nSaving: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f" Saved ({output_path.stat().st_size / (1024**2):.2f} MB)")
    print("\n" + "="*70)
    print(" COMPLETE - Network preserved as dict")
    print("="*70)

#After splitting  i should gete something like 
# split_data = {
#     train:
#     val:
#     test:
#     network:
#     metadata:
# }


