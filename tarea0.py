import numpy as np
import matplotlib.pyplot as plt

# 1. Parámetros y tamaño de muestra
alpha = 4.0
n = 10000

# 2. Generar n valores uniformes en [0,1)
u = np.random.rand(n)

# 3. Aplicar la transformada inversa
#    F(y) = 1 - 1/(y^alpha), y >= 1
#    => y = (1 / (1 - u))^(1/alpha)
x = (1.0 / (1.0 - u))**(1.0 / alpha)

# 4. Construir la distribución empírica (ECDF)
#    Una forma sencilla es "ordenar" los datos
x_sorted = np.sort(x)
#    El valor i-ésimo del ECDF será i/n
ecdf = np.arange(1, n+1) / n

# 5. Crear puntos para la distribución teórica
y_vals = np.linspace(1, x_sorted[-1], 500)  # desde 1 hasta el máx. de x
F_teor = 1 - 1.0/(y_vals**alpha)

# 6. Graficar
plt.figure(figsize=(8,6))

# Gráfica empírica (ECDF)
plt.step(x_sorted, ecdf, where='post', label='Distribución empírica', color='blue')

# Gráfica teórica
plt.plot(y_vals, F_teor, 'r-', lw=2, label='Distribución teórica Pareto(α=4, x_m=1)')

plt.xlim([1, None])  # Empieza en 1, porque la Pareto(1) no tiene valores < 1
plt.ylim([0, 1.05])
plt.xlabel('y')
plt.ylabel('F(y)')
plt.title('Comparación: Distribución empírica vs. teórica Pareto(4,1)')
plt.legend()
plt.show()
