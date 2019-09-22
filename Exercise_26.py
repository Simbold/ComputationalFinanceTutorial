import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from CompFinTutorial.Functions import BlackScholes, Eu_Option_BS_MC


# Monte Carlo Pricing using the importance smapling method ie useful for deep out of the money options
def BS_Eu_MC_IS(S0, r, sigma, K, T, mu, N, alpha, payoff):
    Y = np.random.normal(loc=mu, scale=1, size=N)
    paths = np.exp(-r*T - Y * mu + 0.5*mu**2) * payoff(S0 * np.exp((r-0.5*sigma**2)*T + sigma * np.sqrt(T)*Y))
    Vis = np.mean(paths)
    var = np.var(paths, ddof=1)
    ci = [Vis - norm.isf(alpha/2) * np.sqrt(var / N), Vis + norm.isf(alpha/2) * np.sqrt(var / N)]
    epsilon = norm.isf(alpha / 2) * np.sqrt(var / N)
    return [Vis, ci, epsilon]



S0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 220
N = 1000
alpha = 0.05
St = S0
def payoff(x):
    return np.maximum(x-K,0)

# there is good reason to believe mu = d is a good value
d =( np.log(K/S0) - (r-1/2*sigma**2)*T ) / ( sigma*np.sqrt(T) )
delta = abs(d)
mu = np.arange(d-delta*0.75, d + delta*0.75, 0.01)

result_MCis = BS_Eu_MC_IS(S0, r, sigma, K, T, d, N, alpha, payoff)
result_MC = Eu_Option_BS_MC(S0, r, sigma, K, T, N, payoff)
result_BS = BlackScholes(St, T, K, sigma, r, t=0, Call=True)

y = np.zeros((len(mu), 4))
for i in range(0, len(mu)):
    result = BS_Eu_MC_IS(S0, r, sigma, K, T, mu[i], N, alpha, payoff)
    y[i, 0] = result[0]
    y[i, 1] = result[2]
    result2 = Eu_Option_BS_MC(S0, r, sigma, K, T, N, payoff)
    y[i, 2] = result2[0]
    y[i, 3] = result2[2]
V0_BS = BlackScholes(St, T, K, sigma, r, t=0, Call=True)

plt.plot(mu, y[:, 0], linewidth=0.5, color="red")
plt.plot(mu,  np.repeat(V0_BS, len(mu)), linewidth=0.5, color="green")
plt.plot(mu,  y[:, 2], linewidth=0.3, color="blue")
plt.plot(mu, V0_BS + y[:, 1], linewidth=0.5, color="lightcoral")
plt.plot(mu,  V0_BS - y[:, 1], linewidth=0.5, color="lightcoral")
plt.plot(mu, V0_BS + np.repeat(np.mean(y[:, 3]), len(mu)), linewidth=0.3, color="lightblue")
plt.plot(mu, V0_BS - np.repeat(np.mean(y[:, 3]), len(mu)), linewidth=0.3, color="lightblue")
plt.legend(["Importance Sampling", "Black-Scholes price", "without importance sampling"])
plt.xlabel("mu")
plt.ylabel("Option price")
