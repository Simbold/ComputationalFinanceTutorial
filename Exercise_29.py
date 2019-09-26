import numpy as np
import matplotlib.pyplot as plt


def SimPath_Ito_Euler(X0, a, b, T, m, N):
    Dt = T/m
    paths = np.zeros((N, m+1), dtype=float)
    t = np.zeros(m + 1, dtype=float)
    paths[:, 0] = X0
    for i in range(1, m+1):
        t[i] = t[i-1] + Dt
        DW = np.sqrt(Dt) * np.random.normal(loc=0, scale=1, size=N)
        paths[:, i] = paths[:, i-1] + a(paths[:, i-1], t[i-1]) * Dt + b(paths[:, i-1], t[i-1]) * DW
    return paths

# test parameters
N = 10000
m = 1000
X0 = 0.3 ** 2
T = 2
# define a and b for the Heston model


def a(x, t):
    k = 0.3**2
    lamb = 2.5
    return (k-lamb*x)

def b(x, t):
    sigmat = 0.2
    return np.sqrt(x) * sigmat

paths = SimPath_Ito_Euler(X0, a, b, T, m, N)

x = np.arange(0, T+T/m, T/m)
y = paths[(4, 8, 15, 21), :]
plt.plot(x, y[0, :])
plt.plot(x, y[1, :])
plt.plot(x, y[2, :])
plt.plot(x, y[3, :])
plt.ylabel("Process value")
plt.xlabel("Time")
plt.title("Simulated paths of the stochastic process gamma(t) in the Heston model")
plt.show()

