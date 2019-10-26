import numpy as np
import matplotlib.pyplot as plt
from CompFinTutorial.Functions import BlackScholes


def tridig_invers(alpha, beta, gamma, b):
    # solution to Ax=b, where A is tridiagonal with values alpha, beta, gamma
    n = len(alpha)
    alpha_hat = np.zeros(n, dtype=float)
    b_hat = np.zeros(n, dtype=float)

    alpha_hat[0] = alpha[0]
    b_hat[0] = b[0]

    for i in range(1, n):
        alpha_hat[i] = alpha[i] - gamma[i-1]*beta[i-1]/alpha_hat[i-1]
        b_hat[i] = b[i] - gamma[i-1] * b_hat[i-1]  /alpha_hat[i-1]

    x = np.ones(n, dtype=float)
    x[-1] = b_hat[-1] / alpha_hat[-1]

    for i in range(n-2, -1, -1):
        x[i] = (b_hat[i] - beta[i]*x[i+1]) / alpha_hat[i]
    return x



def BS_EuCall_FiDi_IMP(r, sigma, a, b, m, nu_max, T, K):
    dx = (b-a) /m
    dt = sigma**2 * T / (2*nu_max)
    q = 2 * r / sigma ** 2
    lamb = dt / dx ** 2
    x = np.arange(0, m+1) * dx + a
    t = np.arange(0, nu_max+1) * dt
    w = np.ones(m + 1, dtype=float) * np.maximum(np.exp(0.5 * x * (q + 1)) - np.exp(0.5 * x * (q - 1)), 0)

    w[0] = 0

    alpha = np.ones(m -1, dtype=float) * (1+2*lamb)
    beta = np.ones(m -2, dtype=float) * (-lamb)
    #Acn = np.diag(alpha, 0) + np.diag(beta, 1) + np.diag(beta, -1)
    for i in range(1, nu_max+1):
        #boundary condition
        w[-1] = np.exp(0.5*(q+1) * b + (0.5*(q+1))**2 * t[i]) - np.exp(0.5*(q-1) * b + (0.5*(q-1))**2 * t[i])
        w[-2] = w[-2] + lamb*w[-1] #adjusting for boundary condition

        w[1:-1] = tridig_invers(alpha, beta, beta, w[1:-1])
    S = K * np.exp(x)
    V0 = K* w*np.exp(-(q-1)*x/2 - sigma**2 * T/2 * ((q-1)**2 /4 +q))
    return [V0, S]


r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 100

result = BS_EuCall_FiDi_IMP(r, sigma, a, b, m, nu_max, T, K)
V0_FD = result[0]
S0 = result[1]

V0_BS = np.zeros(len(S0), dtype=float)
j=0
for i in S0:
    V0_BS[j] = BlackScholes(i, T, K, sigma, r, t=0, Call=True)
    j = j+1


plt.plot(S0, V0_FD, linewidth=0.6, color="red")
plt.plot(S0, V0_BS, ".", linewidth=0.2, color="blue")
plt.legend(["Finite Differences method, implicit scheme", "Black-Scholes formula"])
plt.xlabel("Stock Price")
plt.ylabel("Option Price")