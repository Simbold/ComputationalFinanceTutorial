import numpy as np
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt


# binomial model Put price using binomial distribution
def BinModEUPut(S0, r, sigma, T, K, M):
    dt = T / M
    b = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma ** 2) * dt))
    u = b + np.sqrt(b ** 2 - 1)
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    a = np.ceil((np.log(K / S0) - M * np.log(d)) / (np.log(u / d)))
    qtilde = q * u / np.exp(r * dt)

    V0 = K * np.exp(-r * T) * binom.cdf(a - 1, M, q) - S0 * binom.cdf(a - 1, M, qtilde)
    # print("The Value of the Option at time 0 is: " + str(V0))
    return V0


# simple BlackScholes
def BlackScholes(St, T, K, sigma, r, t=0, Call=True):
    d1 = (np.log(St / K) + (r + 0.5 * (sigma ** 2)) * (T - t)) / sigma * np.sqrt(T - t)
    d2 = d1 - sigma * np.sqrt(T - t)

    if Call == True:
        Vt = St * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    elif Call == False:
        Vt = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - St * norm.cdf(-d1)
    else:
        print("ERROR: Call must be either True or False")
    print("The option value at time t is: " + str(Vt))
    return Vt


S0 = 100
T = 1
K = 100
sigma = 0.2
r = 0.05

V = np.zeros(501)
for i in range(10, 501):
    V[i] = BinModEUPut(S0, r, sigma, T, K, M = i)

st = 100
t = 0
Call = False
bs_price = BlackScholes(St, T, K, sigma, r, t, Call)

# visualizing the convergence of binomial model to the Black Scholes model, note that this is for S0 = K
# for different Strike prices the convergence looks quite different!
x = np.arange(len(V))
y = np.repeat(bs_price, 501)
plt.plot(x[10:501], V[10:501], linewidth=0.5)
plt.plot(x, y, linewidth=0.8)
plt.xlabel("Number of steps")
plt.ylabel("put option price")
plt.title("Numerical convergence of put option prices in the binomial model")
plt.legend(["Price in the binomial model", "Price with BS-formula"])
plt.show()



