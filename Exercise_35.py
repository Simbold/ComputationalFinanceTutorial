import numpy as np
import matplotlib.pyplot as plt
from CompFinTutorial.Functions import CRRprice


# Brennon Schwartz Algortihm
def brennon_schwartz(alpha, beta, gamma, bs, g_nui):
    # Solution to Ax-b >= 0 ; x >= g and (Ax-b)'(x-g)=0 ; such that solution x satisfies x_i = g_i for i=1,...,k and x_i > g_i for i = k-1,...,n
    #  where A is tridiagonal with values alpha, beta, gamma
    n = len(alpha)
    alpha_hat = np.zeros(n, dtype=np.float64)
    b_hat = np.zeros(n, dtype=np.float64)

    alpha_hat[-1] = alpha[-1]
    b_hat[-1] = bs[-1]

    for i in range(n - 2, -1, -1):
        alpha_hat[i] = alpha[i] - beta[i] * gamma[i] / alpha_hat[i + 1]
        b_hat[i] = bs[i] - beta[i] * b_hat[i + 1] / alpha_hat[i + 1]

    x = np.zeros(n, dtype=np.float64)
    x[0] = np.maximum(b_hat[0] / alpha_hat[0], g_nui[0])
    for i in range(1, n):
        x[i] = np.maximum((b_hat[i] - gamma[i - 1] * x[i - 1]) / alpha_hat[i], g_nui[i])
    return x


def BS_AmPut_FiDi_CN(r, sigma, K, T, a, b, m, nu_max):
    dx = (b-a)/m
    dt = sigma**2 * T /(2*nu_max)
    xtilde = a + np.arange(0, m+1) * dx
    ttilde = np.arange(0, nu_max+1) * dt

    lamb = dt/dx**2
    q = 2*r/sigma**2

    def g(ttilde, xtilde):
        s = np.exp((q+1)**2 * ttilde/4) * np.maximum(np.exp(xtilde*(q-1)/2)-np.exp(xtilde*(q+1)/2), 0)
        return s

    # initial boundary condition for nu
    w = g(ttilde[0], xtilde)
    # the tridiagonal Matrix with alpha on diagonal and beta to the right and gamma to the left
    # here beta and gamma are equal and the Brennon Schwartz Algorithm is constructed such that the offdiagonal is shorter
    alpha = np.ones(m - 1, dtype=float) * (1 + lamb)
    beta = np.ones(m - 2, dtype=float) * (-0.5 * lamb)

    bs = np.zeros(m-1)
    for i in range(0, nu_max):
        g_nui = g(ttilde[i], xtilde)

        w[-1] = g_nui[-1]
        w[0] = g_nui[0]
        bs[1:-1] = w[2:-2] + 0.5*lamb*(w[1:-3] - 2*w[2:-2] + w[3:-1])

        bs[0] = w[1] + 0.5*lamb*(w[2] - 2*w[1] + g_nui[0] + g(ttilde[i+1], a))
        bs[-1] = w[-2] + 0.5*lamb*(g_nui[-1] - 2*w[-2] + w[-3] + g(ttilde[i+1], b))

        w[1:-1] = brennon_schwartz(alpha, beta, beta, bs, g_nui[1:-1])

    v0 = K*w*np.exp(-xtilde*(q-1)/2 - sigma**2 * T/2 *((q-1)**2 /4 + q))
    S = K*np.exp(xtilde)
    return [v0, S]
r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 100

[V0, S] = BS_AmPut_FiDi_CN(r, sigma, K, T, a, b, m, nu_max)


M = 500
crr_price = np.zeros(len(S))
for i in range(0, len(S)):
    crr_price[i] =CRRprice (S[i], K, r, sigma, T, m, american_exercise=True, option_type="put")

plt.plot(S, V0, linewidth=1)
plt.plot(S, crr_price, ".", linewidth=0.3)
plt.xlabel("Strike")
plt.ylabel("Price")
plt.title("American Put in the FiDi-CN scheme vs. the CRR-model")
plt.legend(["FiDi-CN", "CRR-model"])
