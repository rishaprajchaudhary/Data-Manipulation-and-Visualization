import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
num_obs = int(input("Enter the number of observations: "))
mean = float(input("Enter the mean (mu) for Normal distribution: "))
std_dev = float(input("Enter the standard deviation (sigma) for Normal distribution: "))
variates = norm.rvs(loc=mean, scale=std_dev, size=num_obs)
plt.hist(variates, bins=30, edgecolor='black', alpha=0.7, density=True)
x = np.linspace(min(variates), max(variates), 1000)
pdf = norm.pdf(x, loc=mean, scale=std_dev)
plt.plot(x, pdf, label='Normal Distribution Curve', color='red', lw=2)
plt.title(f'Histogram of Normal Distribution with Mean {mean} and Std Dev {std_dev}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
