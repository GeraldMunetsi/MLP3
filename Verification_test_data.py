import pickle
import numpy as np

with open('epidemic_data_age_adaptive_sobol_split.pkl','rb') as f:
    data = pickle.load(f)

K2K = 29.29
test_sims = data['test']['simulations']


R0s = [(s['params']['tau']/s['params']['gamma'])*K2K for s in test_sims]

print(f"Test set size: {len(R0s)}")
print(f"R0 < 0.8:    {sum(r<0.8 for r in R0s)}")
print(f"R0 0.8-1.2:  {sum(0.8<=r<=1.2 for r in R0s)}")
print(f"R0 > 1.2:    {sum(r>1.2 for r in R0s)}")
print(f"R0 > 3:      {sum(r>3   for r in R0s)}")

# Also check tmax
first = test_sims[0]
print(f"\nn_timepoints: {len(first['output']['t'])}")
print(f"tmax:         {max(first['output']['t']):.1f}")
print(f"Peak I in first 10 test sims:")
for s in test_sims[:100]:
    R0  = (s['params']['tau']/s['params']['gamma'])*K2K
    I   = s['output']['I']
    print(f"  R0={R0:.2f}  peak_I={max(I):.0f}  at_t={np.argmax(I)*0.5:.0f}  shape={'BELL' if np.argmax(I)>2 else 'DECAY'}")

