# import numpy and generate a uniform sample from 0 to 10
import numpy as np
import matplotlib.pyplot as plt
x = np.random.uniform(0, 10, 500)

# make two panels side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# plot a histogram of the original data
axs[0].hist(x, bins=20, color='blue')

def weight(x):
    return 1 / (np.abs(x - 5) + 1)


w = weight(x)


resampled_x = np.random.choice(x, size=500, replace=True,
                               p=w / w.sum())

# plot a histogram of the resampled data
axs[1].hist(resampled_x, bins=20, color='orange')
# plot the weights as a line plot on the second panel
axs[1].plot(x, w, color='red')

plt.show()
