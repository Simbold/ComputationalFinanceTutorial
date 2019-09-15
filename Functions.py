import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

def CRRprice(S0, r, sigma, T, M, K, Am=True, Put=True):
    # calculate parameters u, d, q, dt
    dt = T/M
    b = 0.5*(np.exp(-r*dt) + np.exp((r + sigma**2) * dt))
    u = b + np.sqrt(b**2 - 1)
    d = 1/u
    q = (np.exp(r*dt) - d) / (u-d)

    # calculate the array of Stock Price evolution
    S = np.zeros((M+1, M+1))
    S[0,0] = S0
    # vectorized version is fastest in numpy
    for i in range(1, M+1):
            S[0:i, i] = S[0:i, i-1] * u
            S[i, i] = S[i-1, i-1] * d

    # calculate option payoff at maturity
    V = np.zeros((M+1, M+1))

    if Put == True:
        V[:, -1] = np.maximum(K - S[:, -1], 0)
    # calculate european or american price process
        if Am == False:
            for i in range(M, 0, -1):
                V[0:i, i-1] =  (q * V[0:i, i] + (1-q) * V[1:i+1, i]) * np.exp(-r*dt)
        elif Am == True:
         for i in range(M, -1, -1):
              for j in range(0, i):
                 V[j, i-1] = np.maximum((q * V[j, i] + (1-q) * V[j+1, i] )* np.exp(-r*dt), np.maximum(K - S[j, i-1], 0))
        else:
         print("ERROR: EU must be either True or False")
    # Finally the price of the Call, which is equivalent for American and European options
    elif Put == False:
        V[:, -1] = np.maximum(S[:, -1] - K, 0)
        for i in range(M, 0, -1):
            V[0:i, i - 1] = (q * V[0:i, i] + (1 - q) * V[1:i + 1, i]) * np.exp(-r * dt)
    else:
        print("ERROR: Call must be either True or False")

    V0 = V[0, 0]
    # output the Stock price process, fair option price process and the price at time 0
    return [V0, V, S]


def BlackScholes(St, T, K, sigma, r, t=0, Call=True):
    d1 = (np.log(St/K)+ (r+0.5*(sigma**2)) * (T-t)) / sigma * np.sqrt(T-t)
    d2 = d1 - sigma * np.sqrt(T-t)

    if Call == True:
        Vt = St * norm.cdf(d1) - K * np.exp(-r*(T-t)) * norm.cdf(d2)
    elif Call == False:
        Vt = K * np.exp(-r*(T-t)) * norm.cdf(-d2) - St * norm.cdf(-d1)
    else:
        print("ERROR: Call must be either True or False")
    return Vt


# Prcing by the Integration formula using numerical integration
def BS_Price_Int(S0, r, sigma, T, payoff):
    def integrand(x):
        y = 1 / np.sqrt(2 * np.pi) * payoff(S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * x)) * np.exp(
            -r * T) * np.exp(-x ** 2 / 2)
        return y

    V0 = integrate.quad(integrand, -np.inf, np.inf)[0]
    return V0