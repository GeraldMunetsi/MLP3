import numpy as np
import matplotlib.pyplot as plt

# # Define R0 range
# R0 = np.linspace(0, 2, 500)

# # Different sharpness values
# sharpness_values = [0.5, 1, 5]

# plt.figure(figsize=(8,5))

# for s in sharpness_values:
#     p = np.exp(-s * (R0 - 1)**2)
#     plt.plot(R0, p, label=f"sharpness={s}")

# plt.title("Target Density vs R0 for Different Sharpness")
# plt.xlabel("R0")
# plt.ylabel("p(R0) (unnormalized)")
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameter ranges
# tau = np.linspace(0.05, 2, 300)
# gamma = np.linspace(0.05, 2, 300)

# # Create 2D grid
# T, G = np.meshgrid(tau, gamma)

# # Compute R0
# R0 = T / G

# # Target density: sigma controls sharpness
# sigma = 0.1
# p = np.exp(-(R0 - 1)**2 / (2 * sigma**2))

# # Plot
# plt.figure(figsize=(7,6))
# cp = plt.contourf(T, G, p, levels=50, cmap='viridis')
# plt.colorbar(cp, label='Target density p(θ)')
# plt.xlabel('tau')
# plt.ylabel('gamma')
# plt.title('Target Density in Parameter Space Near R0=1')
# plt.show()


# Example: 10 samples along a 1D parameter
samples = np.linspace(0, 10, 10)
weights = np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.3, 0.1, 0.05, 0.025, 0.025])
weights = weights / weights.sum()

# Multinomial resampling
resampled_idx = np.random.choice(len(samples), size=len(samples), p=weights)
resampled_samples = samples[resampled_idx]

print("Original samples:", samples)
print("Resampled samples:", resampled_samples)

# Visualize
plt.figure(figsize=(8,3))
plt.scatter(samples, np.zeros_like(samples), label='original', s=100)
plt.scatter(resampled_samples, np.ones_like(resampled_samples), label='resampled', s=100)
plt.legend()
plt.title("Multinomial Resampling in Importance Sampling")
plt.yticks([0,1], ['Original', 'Resampled'])
plt.show()