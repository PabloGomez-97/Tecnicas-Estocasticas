import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

np.random.seed(42)

n = 10000       
alpha = 4.0       
xm = 1.0      

u = np.random.rand(n)

x = (1 - u)**(-1/alpha)

x_sorted = np.sort(x)               
cdf_empirica = np.arange(1, n+1) / n     

def pareto_cdf(x, alpha=alpha, xm=xm):
    return np.where(x < xm, 0, 1 - (xm / x)**alpha)

plt.figure(figsize=(8,6))
plt.step(x_sorted, cdf_empirica, label='CDF empírica (simulada)', where='post')
plt.plot(x_sorted, pareto_cdf(x_sorted), 'r-', label='CDF teórica Pareto(4,1)')
plt.xlim(1, x_sorted.max())
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Comparación de la CDF empírica y la CDF teórica Pareto(4,1)')
plt.legend()
plt.show()

D, p_value = kstest(x, lambda z: pareto_cdf(z, alpha, xm))

print(f"Estadístico KS: {D:.4f}")
print(f"p-value: {p_value:.4f}")
