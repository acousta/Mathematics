# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gamma, beta

##### GAMMA GENERATOR
# Parámetros
# n = 10000  # Número de muestras
# a = 0  # Límite inferior de la distribución uniforme
# b = 1  # Límite superior de la distribución uniforme
# alpha = 10
# beta= 5
# gamma_sample_vec = np.zeros([n,1])
# for i in range(n):
#     # Generar las muestras
#     samples = np.random.uniform(a, b, alpha)
#     gamma_samples_aux = np.log(samples).sum()
#     gamma_sample = (-1/beta)*gamma_samples_aux
#     gamma_sample_vec[i]=gamma_sample


# # Generate Gamma PDF
# x = np.linspace(0, gamma_sample_vec.max(), 1000)
# gamma_pdf = gamma.pdf(x, a=alpha, scale=1/beta)


# # Graficar
# s = plt.figure(figsize=(8, 5))
# plt.hist(gamma_sample_vec, bins=100, color='skyblue', edgecolor='black', density=True)
# plt.title(f'Histograma de {n} muestras de una distribución uniforme [{a}, {b}]')
# plt.xlabel('Valor')
# plt.plot(x, gamma_pdf, color='blue', lw=2, label=f'Gamma PDF (alpha={alpha}, beta={beta})')
# plt.ylabel('Densidad')
# plt.grid(alpha=0.4)
# plt.show()

# print(s)

############# DIRICHLET GENERAGTOR (STICK BREAKING PROCESS)

# import numpy as np

# # Parámetros de la distribución Dirichlet
# alpha = [1, 1, 4, 9, 1, 5]  # Parámetros de la distribución
# n = len(alpha)

# # Precomputar las sumas acumulativas inversas de alpha
# alpha_cumsum = np.cumsum(alpha[::-1])[::-1]

# # Inicializar el vector de la muestra
# sample = np.zeros(n)
# stick = 1  # La longitud inicial del "palo"

# # Stick Breaking Process
# for k in range(n - 1):  # Solo necesitamos iterar hasta n-1
#     beta = np.random.beta(alpha[k], alpha_cumsum[k])
#     sample[k] = stick * beta
#     stick *= (1 - beta)  # Reducir el "palo" restante

# # La última parte del "palo" es asignada al último componente
# sample[-1] = stick

# # Mostrar el resultado
# print("Muestra generada:", sample)
# print("Suma de la muestra:", sum(sample))


########################


############# Codigo de estimacion bayesiana conjugada Binomial-Beta
# Supongamos que queremos conocer si una moneda tiene un sesgo hacia algun resultado 
# usando los resultados de la estadistica bayesiana

# Parametros: 
# a (int): Numero de aguilas
# b (int): Numero de intentos

# def resultadosMoneda(z,n):

#     # Crear un rango de valores para x entre 0 y 1
#     x = np.linspace(0, 1, 500)
#     params = [
#         (2,2, 'blue'),
#         (2+z, 2-z+n-1, 'green')
#     ]

#     # Graficar cada distribución en la misma figura
#     plt.figure(figsize=(8, 5))
#     for alpha, beta_param, color in params:
#         pdf = beta.pdf(x, alpha, beta_param)
#         plt.plot(x, pdf, color=color)

#     # Configuración de la gráfica
#     plt.title("Función de Densidad de Probabilidad - Distribuciones Beta")
#     plt.xlabel("x")
#     plt.ylabel("Densidad de Probabilidad")
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.show()

# resultadosMoneda(6,10)
# import numpy as np
# import matplotlib.pyplot as plt

# def stick_breaking_process(alpha, num_weights):
#     """
#     Implementa el Proceso de Dirichlet mediante el método de stick-breaking.

#     Args:
#     alpha (float): Parámetro de concentración.
#     num_weights (int): Número de pesos a generar.

#     Returns:
#     weights (np.array): Pesos generados por el método de stick-breaking.
#     """
#     # Generar fragmentos de la varilla
#     v = np.random.beta(1, alpha, size=num_weights)
    
#     # Calcular los pesos mediante el producto acumulado
#     remaining_stick = np.cumprod(1 - v[:-1])  # Longitud restante de la varilla
#     weights = np.concatenate(([v[0]], v[1:] * remaining_stick))
    
#     return weights

# # Parámetros
# alpha = 10.0  # Parámetro de concentración
# num_weights = 20  # Número de pesos a generar

# # Simulación del proceso
# weights = stick_breaking_process(alpha, num_weights)

# # Visualización de los pesos
# plt.figure(figsize=(8, 5))
# plt.bar(range(1, num_weights + 1), weights, color="skyblue", edgecolor="black")
# plt.title(f"Stick-Breaking Process (α={alpha})")
# plt.xlabel("Índice del peso")
# plt.ylabel("Peso")
# plt.grid(alpha=0.3)
# plt.show()


# #Dirichlet process con IGN
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, invgamma

# # Generar datos simulados (3 clusters para probar el modelo)
# np.random.seed(42)
# data = np.concatenate([
#     np.random.normal(0, 1, 100),
#     np.random.normal(5, 0.5, 100),
#     np.random.normal(-3, 0.8, 100)
# ])

# # Parámetros de la distribución base G0 (Normal-Inverse-Gamma)
# mu_0 = 0       # Media inicial
# lambda_0 = 1   # Precisión de la media
# alpha = 2.0    # Parámetro shape para la varianza
# beta = 1.0     # Parámetro scale para la varianza

# # Stick-Breaking Process para pesos π_k
# def stick_breaking(alpha, num_weights):
#     v = np.random.beta(1, alpha, size=num_weights)
#     remaining_stick = np.cumprod(1 - v[:-1])
#     weights = np.concatenate(([v[0]], v[1:] * remaining_stick))
#     return weights

# # Muestreo de θ_k = (μ_k, σ²_k) de G0 (NIG)
# def sample_nig(mu_0, lambda_0, alpha, beta, size=1):
#     sigma2 = invgamma.rvs(alpha, scale=beta, size=size)  # Muestreo de σ²
#     mu = norm.rvs(loc=mu_0, scale=np.sqrt(sigma2 / lambda_0), size=size)  # Muestreo de μ
#     return mu, sigma2

# # Modelo de mezcla infinita
# num_clusters = 10  # Aproximación finita
# weights = stick_breaking(2.0, num_clusters)  # Pesos del DP
# mus, sigmas2 = sample_nig(mu_0, lambda_0, alpha, beta, size=num_clusters)  # Parámetros de clusters

# # Calcular la mezcla de normales (FDP posterior)
# x_range = np.linspace(min(data) - 1, max(data) + 1, 1000)
# posterior_density = np.zeros_like(x_range)

# for k in range(num_clusters):
#     posterior_density += weights[k] * norm.pdf(x_range, loc=mus[k], scale=np.sqrt(sigmas2[k]))

# # Visualización
# plt.figure(figsize=(12, 6))

# # Histograma de datos reales
# plt.hist(data, bins=30, density=True, alpha=0.6, color='gray', edgecolor='black', label='Datos reales')

# # FDP posterior (mezcla de normales)
# plt.plot(x_range, posterior_density, color='red', lw=2, label='FDP posterior (mezcla de normales)')

# # Cada componente de la mezcla
# for k in range(num_clusters):
#     plt.plot(x_range, weights[k] * norm.pdf(x_range, loc=mus[k], scale=np.sqrt(sigmas2[k])),
#              linestyle='--', alpha=0.7, label=f'Componente {k+1}')

# plt.title("FDP posterior como mezcla de normales")
# plt.xlabel("x")
# plt.ylabel("Densidad")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()




####First MNDP
# import numpy as np
# from scipy.stats import invgamma, norm, dirichlet
# import matplotlib.pyplot as plt

# # Configuración de hiperparámetros para G0 (Normal-Inverse-Gamma)
# mu_0 = 0        # Media prior
# lambda_0 = 1    # Precisión prior para la media
# alpha_0 = 2.0   # Forma para la Inverse-Gamma
# beta_0 = 1.0    # Escala para la Inverse-Gamma

# # Datos simulados
# np.random.seed(42)
# n_data = 1000
# true_means = [-5, 0, 5]
# true_stds = [0.5, 1.0, 0.7]
# true_weights = [0.3, 0.4, 0.3]
# data = np.concatenate([
#     np.random.normal(m, s, size=int(w * n_data))
#     for m, s, w in zip(true_means, true_stds, true_weights)
# ])
# # Número máximo de clusters (truncación)
# K = 4
# alpha = 1/100  # Parámetro de concentración del DP

# # Inicialización de las asignaciones de clusters y pesos
# cluster_assignments = np.random.randint(0, K, size=n_data)
# cluster_counts = np.bincount(cluster_assignments, minlength=K)
# cluster_means = np.random.normal(mu_0, np.sqrt(beta_0 / lambda_0), size=K)
# cluster_vars = invgamma.rvs(alpha_0, scale=beta_0, size=K)
# pi = dirichlet.rvs(alpha=[alpha] * K)[0]

# # Muestreo Gibbs
# n_iter = 1000
# samples_pi = []

# for iteration in range(n_iter):
#     # Paso 1: Actualizar las asignaciones de clusters
#     for i in range(n_data):
#         probs = np.array([
#             pi[k] * norm.pdf(data[i], loc=cluster_means[k], scale=np.sqrt(cluster_vars[k]))
#             for k in range(K)
#         ])
#         probs /= probs.sum()  # Normalizar
#         cluster_assignments[i] = np.random.choice(K, p=probs)

#     # Paso 2: Actualizar los parámetros de los clusters
#     cluster_counts = np.bincount(cluster_assignments, minlength=K)
#     for k in range(K):
#         if cluster_counts[k] > 0:
#             # Media posterior
#             lambda_k = lambda_0 + cluster_counts[k]
#             mu_k = (lambda_0 * mu_0 + data[cluster_assignments == k].sum()) / lambda_k
#             # Varianza posterior
#             alpha_k = alpha_0 + cluster_counts[k] / 2
#             beta_k = beta_0 + 0.5 * np.sum((data[cluster_assignments == k] - mu_k)**2)
#             # Muestrear nuevos parámetros
#             cluster_means[k] = norm.rvs(loc=mu_k, scale=np.sqrt(1 / lambda_k))
#             cluster_vars[k] = invgamma.rvs(a=alpha_k, scale=beta_k)

#     # Paso 3: Actualizar los pesos pi usando el proceso de Dirichlet
#     pi = dirichlet.rvs(alpha=cluster_counts + alpha)[0]

#     # Almacenar muestras de pi
#     samples_pi.append(pi)

# # Convertir las muestras a un array para análisis
# samples_pi = np.array(samples_pi)

# # Visualización de los resultados
# plt.figure(figsize=(12, 6))

# # Pesos pi
# plt.subplot(1, 2, 1)
# for k in range(K):
#     plt.plot(samples_pi[:, k], label=f"Cluster {k+1}")
# plt.title("Evolución de los pesos pi")
# plt.xlabel("Iteraciones")
# plt.ylabel("Peso pi")
# plt.legend()

# # Histogramas de datos simulados y clusters asignados
# plt.subplot(1, 2, 2)
# plt.hist(data, bins=30, density=True, alpha=0.5, label="Datos")
# for k in range(K):
#     if cluster_counts[k] > 0:
#         x = np.linspace(data.min(), data.max(), 100)
#         y = norm.pdf(x, loc=cluster_means[k], scale=np.sqrt(cluster_vars[k]))
#         plt.plot(x, pi[k] * y, label=f"Cluster {k+1}")
# plt.title("Distribución estimada y componentes de mezcla")
# plt.legend()

# plt.tight_layout()
# plt.show()





# #####IMPLEMENTANDO el algorimto 2
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt

# # # Configuración inicial
# # np.random.seed(42)  # Para reproducibilidad
# # n = 100  # Número de observaciones
# # k = 3    # Número verdadero de componentes (desconocido en la práctica)

# # # Parámetros verdaderos (desconocidos en la práctica)
# # true_m = 0.0  # Media global
# # true_tau = 1.0  # Parámetro de escala
# # true_V = np.array([1.0, 2.0, 0.5])  # Varianzas de los componentes
# # true_mu = np.array([-3.0, 0.0, 3.0])  # Medias de los componentes

# # Generar datos sintéticos
# k=4 # Un numero inicial
# y = np.array([9.172, 9.350, 9.483, 9.558, 9.775, 10.227, 10.406, 16.084, 16.170, 18.419, 18.552, 18.600, 18.927, 19.052, 19.070, 19.330, 19.343, 19.349, 19.440, 19.473, 19.529, 19.541, 19.547, 19.663, 19.846, 19.856, 19.863, 19.914, 19.918, 19.973, 19.989, 20.166, 20.175, 20.179, 20.196, 20.215, 20.221, 20.415, 20.629, 20.795, 20.821, 20.846, 20.875, 20.986, 21.137, 21.492, 21.701, 21.814, 21.921, 21.960, 22.185, 22.209, 22.242, 22.249, 22.314, 22.374, 22.495, 22.746, 22.747, 22.888, 22.914, 23.206, 23.241, 23.263, 23.484, 23.538, 23.542, 23.666, 23.706, 23.711, 24.129, 24.285, 24.289, 24.366, 24.717, 24.990, 25.633, 26.960, 26.995, 32.065, 32.789, 34.279])
# z_true = np.random.choice(k, size=len(y))  # Asignaciones de componentes

# # Hiperparámetros del previo
# a = 0.0  # Media del previo para m
# A = 1.0  # Varianza del previo para m
# w = 1.0  # Forma del previo para tau^-1
# W = 1.0  # Escala del previo para tau^-1
# s = 4.0  # Forma del previo para V_j^-1
# S = 2.0  # Escala del previo para V_j^-1

# # Inicialización de parámetros
# m = 0.0  # Valor inicial para m
# tau = 1.0  # Valor inicial para tau
# mu = np.random.normal(0, 1, size=k)  # Valores iniciales para mu_j
# V = np.random.gamma(s/2, 2/S, size=k)  # Valores iniciales para V_j

# # Número de iteraciones del Gibbs sampler
# n_iter = 1000
# burn_in = 200  # Número de iteraciones para "quemar"

# # Almacenamiento de muestras
# samples_m = []
# samples_tau = []
# samples_mu = []
# samples_V = []

# # Algoritmo II: Gibbs sampler
# for it in range(n_iter):
#     # Paso 1': Muestrear m y tau
#     # Muestrear m
#     sum_mu_V = np.sum(mu / V) # Cálculos necesarios para la actualización
#     sum_inv_V = np.sum(1 / V)
#     m_mean = (a * A + tau * sum_mu_V) / (A + tau * sum_inv_V)
#     m_var = tau / (A + tau * sum_inv_V)
#     m = np.random.normal(m_mean, np.sqrt(m_var))
    
#     # Muestrear tau
#     K = np.sum((mu - m)**2 / V)
#     tau_shape = (w + k) / 2
#     tau_scale = (W + K) / 2
#     tau = 1 / np.random.gamma(tau_shape, 1 / tau_scale)
    
#     # Paso 2: Muestrear pi (mu_j y V_j)
#     for j in range(k):
#         # Muestrear V_j
#         y_j = y[z_true == j]
#         n_j = len(y_j)
#         S_j = S + np.sum((y_j - mu[j])**2)
#         V[j] = 1 / np.random.gamma((s + n_j) / 2, 2 / S_j)
        
#         # Muestrear mu_j
#         y_bar_j = np.mean(y_j) if n_j > 0 else 0
#         mu_var = tau * V[j] / (tau + n_j)
#         mu_mean = (m * tau + n_j * y_bar_j) / (tau + n_j)
#         mu[j] = np.random.normal(mu_mean, np.sqrt(mu_var))
    
#     # Almacenar muestras después del burn-in
#     if it >= burn_in:
#         samples_m.append(m)
#         samples_tau.append(tau)
#         samples_mu.append(mu.copy())
#         samples_V.append(V.copy())

# # Convertir a arrays de numpy
# samples_m = np.array(samples_m)
# samples_tau = np.array(samples_tau)
# samples_mu = np.array(samples_mu)
# samples_V = np.array(samples_V)

# # Resultados
# print("Media estimada de m:", np.mean(samples_m))
# print("Media estimada de tau:", np.mean(samples_tau))
# print("Medias estimadas de mu:", np.mean(samples_mu, axis=0))
# print("Varianzas estimadas de V:", np.mean(samples_V, axis=0))

# # Visualización de las muestras
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 2, 1)
# plt.hist(samples_m, bins=30, density=True, alpha=0.6, color='blue')
# plt.title("Distribución posterior de m")
# plt.subplot(2, 2, 2)
# plt.hist(samples_tau, bins=30, density=True, alpha=0.6, color='green')
# plt.title("Distribución posterior de tau")
# plt.subplot(2, 2, 3)
# for j in range(k):
#     plt.hist(samples_mu[:, j], bins=30, density=True, alpha=0.6, label=f"mu_{j+1}")
# plt.title("Distribución posterior de mu")
# plt.legend()
# plt.subplot(2, 2, 4)
# for j in range(k):
#     plt.hist(samples_V[:, j], bins=30, density=True, alpha=0.6, label=f"V_{j+1}")
# plt.title("Distribución posterior de V")
# plt.legend()
# plt.tight_layout()
# plt.show()



#####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv('McDonaldsMenuNutritionV2.csv')  # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo

# Crear histograma para Proteína con bins de ancho 5
plt.figure(figsize=(10, 5))
protein_bins = np.arange(0, df['Protein (g)'].max() + 1, 1)  # Bins de ancho 5
plt.hist(df['Protein (g)'], bins=protein_bins, color='blue', edgecolor='black')
plt.title('Histograma de Proteína (g)')
plt.xlabel('Proteína (g)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Crear histograma para Calorías con bins de ancho 50
plt.figure(figsize=(10, 5))
calories_bins = np.arange(0, df['Calories'].max() + 12, 12)  # Bins de ancho 50
plt.hist(df['Calories'], bins=calories_bins, color='orange', edgecolor='black')
plt.title('Histograma de Calorías')
plt.xlabel('Calorías')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()