import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
n = int(input("Enter number of trials: "))
p = float(input("Enter probability of success (0-1): "))

x = np.arange(0, n + 1)
pmf = binom.pmf(x, n, p)

plt.bar(x, pmf, width=0.6, color='lightgreen', edgecolor='black')

plt.xlabel('Number of Successes (x)')
plt.ylabel('P(X = x)')
plt.title(f'Binomial Distribution PMF (n={n}, p={p})')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
