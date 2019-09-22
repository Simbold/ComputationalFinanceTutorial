import numpy as np
from CompFinTutorial.Functions import Eu_Option_BS_MC, BlackScholes


# self quanto call with monte carlo control variats using normal call as control
def BS_EuOption_MC_CV (S0, r, sigma, T, K, M):
    # Determine beta
    z = np.random.normal(0, 1, M)
    paths = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    x = paths * np.maximum(paths-K, 0) # payoff at T of self quanto call
    y = np.maximum(paths-K, 0) # payoff of call used as y = control variate
    Ey = BlackScholes(S0, T, K, sigma, r, t=0, Call=True) # expectation of call value is

    covar = np.mean((x - np.mean(x)) * (y - Ey*np.exp(r*T))) # making Ey undiscounted so it the same as x
    beta = covar / np.var(y, ddof=1)

    # now computing MC estimate of self quanto call
    z = np.random.normal(0, 1, M)
    paths = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    fX = paths * np.maximum(paths-K, 0)
    Y = beta * np.maximum(paths-K, 0)
    V0 = np.exp(-r*T) * np.mean(fX - Y) + beta * Ey
    epsilon = 1.96 * np.sqrt(np.var((fX - Y), ddof=1) / M)
    return [V0, epsilon, beta]


S0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 110
M = 100000

Vcv = BS_EuOption_MC_CV (S0, r, sigma, T, K, M)

def payoff(x):
    return x * np.maximum(x - K, 0)


Vmc = Eu_Option_BS_MC(S0, r, sigma, K, T, M, payoff, alpha=0.05)


print("Price of European call by use of Monte-Carlo simulation with control variate: " + str(round(Vcv[0], 4)) +
      ", radius of 95% confidence interval: " + str(round(Vcv[1], 4)) +
      "  Beta chosen in the estimation procedure:" + str(round(Vcv[2], 4)))

print("Price of European call by use of Monte-Carlo simulation without control variate: " + str(round(Vmc[0], 4)) +
      ", radius of 95% confidence interval: " + str(round(Vmc[2], 4)))

# as one can see in the output the variance is significantly smaller with the control variates method