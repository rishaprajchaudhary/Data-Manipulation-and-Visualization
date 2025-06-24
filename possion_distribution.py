import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
num_obs = int(input("Enter the number of observations: "))
mean_lambda = float(input("Enter the mean (lambda) for Poisson distribution: "))
variates = poisson.rvs(mu=mean_lambda, size=num_obs)
hist_data, bin_edges = np.histogram(variates, bins=range(0, int(np.max(variates)) + 2))

plt.bar(bin_edges[:-1], hist_data, width=0.6,color= 'lightcoral', edgecolor='black')
plt.title(f'Histogram of Poisson Distribution with Mean {mean_lambda}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
