import numpy as np
from scipy.stats import norm
from CompFinTutorial.Functions import BlackScholes


def Eu_Option_BS_MC(S0, r, sigma, K, T, N, payoff, alpha=0.05):
    z = np.random.normal(0, 1, N)
    paths = np.exp(-r * T) * payoff(S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z))
    V0 = np.mean(paths)
    var = np.var(paths, ddof=1)
    ci = [V0 - norm.isf(alpha/2) * np.sqrt(var / N), V0 + norm.isf(alpha/2) * np.sqrt(var / N)]
    epsilon = norm.isf(alpha/2) * np.sqrt(var / N)
    return [V0, ci, epsilon]


def Eu_Option_BS_MC_AT(S0, r, sigma, K, T, N, payoff, alpha=0.05):
    z = np.random.normal(0, 1, N)
    paths = payoff(S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z))
    paths2 = payoff(S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * (-z)))

    V0 = 0.5 * np.mean(np.exp(-r * T) *(paths+paths2))
    var = np.var(paths+paths2, ddof=1)
    ci = [V0 - norm.isf(alpha/2) * np.sqrt(var / (4*N)), V0 + norm.isf(alpha/2) * np.sqrt(var /(4*N))]
    epsilon = norm.isf(alpha/2) * np.sqrt(var / (4*N))
    return [V0, ci, epsilon]

S0 = 100
r = 0.05
sigma = 0.2
T = 1
N = 3000000
K = 90


def payoff(x):
    return np.maximum(x - K, 0)


result = Eu_Option_BS_MC(S0, r, sigma, K, T, N, payoff)
result_AT = Eu_Option_BS_MC_AT(S0, r, sigma, K, T, N, payoff)
BSprice = BlackScholes(St=S0, T=T, K=K, sigma=sigma, r=r, t=0, Call=True)

print("Price of the European Call by use of plain Monte-Carlo simulation: " +
      str(np.round(result[0], 4)) + ", 95% confidence interval: " + str(np.round(result[1], 4)))
print("Price of the European Call by use of Monte-Carlo simulation with Antithetic Variable: " +
      str(np.round(result_AT[0], 4)) + ", 95% confidence interval: " + str(np.round(result_AT[1], 4)))
print("Exact price of the European Call by use of the BS-Formula: " + str(np.round(BSprice, 4)))