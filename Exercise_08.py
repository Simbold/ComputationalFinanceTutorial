import numpy as np
import scipy.integrate as integrate

# Prcing by the Integration formula using numerical integration
def BS_Price_Int(S0, r, sigma, T, payoff):
    def integrand(x):
        y = 1 / np.sqrt(2 * np.pi) * payoff(S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * x)) * np.exp(
            -r * T) * np.exp(-x ** 2 / 2)
        return y

    V0 = integrate.quad(integrand, -np.inf, np.inf)[0]
    return V0


S0 = 100
r = 0.05
sigma = 0.2
T = 1
K = 100


# define the payoff of a call option
def payoff(x):
    y = np.maximum(x - K, 0)
    return y


result = BS_Price_Int(S0, r, sigma, T, payoff)
print("Price of a European Call option by use of the integration formula " + str(np.round(result, 4)))
