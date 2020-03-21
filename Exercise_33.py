import numpy as np
import matplotlib.pyplot as plt
from CompFinTutorial.Functions import BlackScholes

def BS_EuCall_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K):
    q = 2*r/sigma**2
    dx = (b-a)/m
    dt = sigma**2 * T /(2*nu_max)
    x = np.arange(0, m+1) * dx + a
    t = np.arange(0, nu_max+1) * dt
    lamb = dt / dx**2
    if lamb>=0.5:
        print("ERROR, Finite difference Scheme unstable: lambda >=0.5")
        return "NA"

    w = np.ones(m+1, dtype=float) * np.maximum(np.exp(0.5*x*(q+1)) - np.exp(0.5*x*(q-1)), 0) # for call

    w[0] = 0 # lower boundary stays always at zero since this is a call
    # question: which boundary condition in the 2 first "corners"?
    for i in range(1, nu_max + 1):
        t = i * dt
        # explicit fidi scheme
        w[1:-1] = lamb * w[0:m - 1] + (1 - 2 * lamb) * w[1:m] + lamb * w[2:m + 1]
        # boundary condition of the call
        w[-1] = np.exp(0.5 * (q + 1) * b + (0.5 * (q + 1)) ** 2 * t) - np.exp(
            0.5 * (q - 1) * b + (0.5 * (q - 1)) ** 2 * t)

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

result = BS_EuCall_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K)
V0_FD = result[0]
S0 = result[1]

V0_BS = np.zeros(len(S0), dtype=float)
j=0
for i in S0:
    V0_BS[j] = BlackScholes(i, T, K, sigma, r, t=0, Call=True)
    j = j+1


plt.plot(S0, V0_FD, linewidth=0.6, color="red")
plt.plot(S0, V0_BS, ".", linewidth=0.2, color="blue")
plt.legend(["Finite Differences method", "Black-Scholes formula"])
plt.xlabel("Stock Price")
plt.ylabel("Option Price")