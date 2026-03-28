import pickle
import numpy as np

with open('epidemic_data_age_adaptive_sobol_split.pkl','rb') as f:
    data = pickle.load(f)

K2K = 29.29
for split_name in ['train','val','test']:
    sims = data[split_name]['simulations']
    R0s  = [(s['params']['tau']/s['params']['gamma'])*K2K for s in sims]
    print(f"\n{split_name} ({len(R0s)} samples):")
    print(f"  R0 > 3:   {sum(r>3 for r in R0s)}  ({100*sum(r>3 for r in R0s)/len(R0s):.1f}%)")
    print(f"  R0 > 1.2: {sum(r>1.2 for r in R0s)}  ({100*sum(r>1.2 for r in R0s)/len(R0s):.1f}%)")
    print(f"  R0 < 0.8: {sum(r<0.8 for r in R0s)}  ({100*sum(r<0.8 for r in R0s)/len(R0s):.1f}%)")