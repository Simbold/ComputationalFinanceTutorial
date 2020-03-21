import numpy as np
from scipy.stats import norm

def EuCallHedge_BS_MC_FD(St, r, sigma, K, T, N, payoff):
    h = 0.000001
    z = np.random.normal(loc=0, scale=1, size=N)
    paths = np.exp(-r*T) / h * (payoff((St+h/2)*np.exp((r-0.5*sigma**2)*T + sigma *np.sqrt(T) * z)) -
                                payoff((St-h/2)*np.exp((r-0.5*sigma**2)*T + sigma *np.sqrt(T) * z)))
    Delta = np.mean(paths)
    epsilon = 1.96 * np.sqrt(np.var(paths, ddof=1) / N)
    ci = [Delta - epsilon, Delta + epsilon]
    return [Delta, ci, epsilon]


# explict black scholes Delta hedge formula from taking derivative of call BS formula
def BS_EuCallHedge(St, r, sigma, T, t, K):
    d1 = (np.log(St / K) + (r + 1 / 2 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    Delta = norm.cdf(d1)
    return Delta


St = 100
r = 0.05
sigma = 0.3
T = 1
N = 10000
K=80
t=0


def payoff(x):
    return np.maximum(x-K, 0)


Delta_MC_FD = EuCallHedge_BS_MC_FD(St, r, sigma, K, T, N, payoff)
Delta_BS = BS_EuCallHedge(St, r, sigma, T, t, K)

print("Hedge of European call by use of plain Monte-Carlo simulation (Finite Difference approach), with N=10000 simulations: " +
      str(round(Delta_MC_FD[0], 4)) + ", 95% confidence interval: " + str(np.round(Delta_MC_FD[1], 4)))
print("Exact Hedge of European call: " + str(round(Delta_BS, 4)))