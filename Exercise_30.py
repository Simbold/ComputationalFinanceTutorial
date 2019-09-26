import numpy as np
import matplotlib.pyplot as plt


def Sim_Paths_GeoBM(X0, mu, sigma, T, m):
    Dt = T/m
    X_exact = np.zeros(m+1, dtype=float)
    X_euler = np.zeros(m+1, dtype=float)
    X_milshtein = np.zeros(m+1, dtype=float)

    X_exact[0], X_euler[0], X_milshtein[0] = X0, X0, X0
    for i in range(1, m+1):
        DW = np.sqrt(Dt) * np.random.normal(loc=0, scale=1, size=1)
        X_exact[i] = X_exact[i-1] * np.exp((mu-0.5*sigma**2)*Dt + sigma*DW)
        X_euler[i] = X_euler[i-1]*(1 + mu*Dt + sigma*DW)
        X_milshtein[i] = X_milshtein[i-1] * (1 + mu*Dt + sigma*DW + 0.5 * (sigma**2)*((DW**2)-Dt))
    return [X_exact, X_euler, X_milshtein]

X0 = 100
mu = 0.1
sigma = 0.3
T = 1

m = np.array((10, 100, 1000, 10000))
plt.subplots(2, 2)
j=1
for i in m:
    result = Sim_Paths_GeoBM(X0, mu, sigma, T, i)
    X_exact = result[0]
    X_euler = result[1]
    X_milshtein = result[2]
    x = np.arange(0, T + T / i, T / i)
    plt.subplot(2, 2, j)
    j = j+1
    plt.plot(x, X_exact, linewidth=0.5)
    plt.plot(x, X_euler, linewidth=0.5)
    plt.plot(x, X_milshtein, linewidth=0.5)
    plt.legend(['Exact simulation', 'Euler approximation', 'Milshtein approximation'])


