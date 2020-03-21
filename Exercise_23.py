import numpy as np
from scipy.stats import norm
from CompFinTutorial.Functions import BlackScholes


def Eu_Option_BS_MC(S0, r, sigma, K, T, N, payoff, alpha=0.05):
    z = np.random.normal(0, 1, N)
    paths = payoff(S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z))
    V0 = np.exp(-r * T) * np.mean(paths)
    var = np.var(paths, ddof=1)
    ci = [V0 - norm.isf(alpha/2) * np.sqrt(var / N), V0 + norm.isf(alpha/2) * np.sqrt(var / N)]
    epsilon = norm.isf(alpha/2) * np.sqrt(var / N)
    return [V0, ci, epsilon]

S0 = 100
r = 0.05
sigma = 0.2
T = 1
N = 100000
K = 90


def payoff(x):
    return np.maximum(x - K, 0)


result = Eu_Option_BS_MC(S0, r, sigma, K, T, N, payoff)
BSprice = BlackScholes(St=S0, T=T, K=K, sigma=sigma, r=r, t=0, Call=True)
print("Price of the European Call by use of plain Monte-Carlo simulation: " +
      str(np.round(result[0], 4)) + ", 95% confidence interval: " + str(np.round(result[1], 4)))
print("Exact price of the European Call by use of the BS-Formula: " + str(np.round(BSprice, 4)))