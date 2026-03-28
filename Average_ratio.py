import numpy as np
import networkx as nx
import EoN

N=10000
m=5

results_ratio=[]
results_k_avg=[]
results_k2_avg=[]

for i in range(1000) :
     G = nx.barabasi_albert_graph(N, m)
     degrees = np.array([d for _, d in G.degree()])
     k_avg  = degrees.mean()
     k2_avg = (degrees ** 2).mean()
     ratio= (degrees ** 2).mean()/ degrees.mean() #  <k²>/<k> for BA network
    
     results_k_avg.append(k_avg)
     results_k2_avg.append(k2_avg)
     results_ratio.append(ratio)

     mean_k_avg=np.mean(results_k_avg)
     mean_k2_avg=np.mean(results_k2_avg)
     mean_ratio=np.mean(results_ratio)


print(f"Mean of <k> over 1000 BA network: {k2_avg}")
print(f"Mean of <k²> over 1000 BA network: {k2_avg}")
print(f"Mean of <k²>/<k> over 1000 BA network: {ratio}")

