import numpy as np
from scipy.stats import norm
from CompFinTutorial.Functions import SimPath_Ito_Euler


def EUDownOutCall_BS(St, T, K, sigma, r, H, t=0):
    d1 = (np.log(St/K)+(r+1/2*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d11 = (np.log(H**2/(K*St))+(r+1/2*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = (np.log(St/K)+(r-1/2*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d22 = (np.log(H**2/(K*St))+(r-1/2*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    Vt = St*(norm.cdf(d1)-(H/St)**(1+2*r/sigma**2)*norm.cdf(d11)) - np.exp(-r*(T-t))*K*(norm.cdf(d2)-(H/St)**(-1+2*r/sigma**2)*norm.cdf(d22))
    return Vt


def DownOut_CRR(S0, r, sigma, T, M, K, H, Am=True, Put=True):
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
                V[0:i, i-1] = (q * V[0:i, i] + (1-q) * V[1:i+1, i]) * np.exp(-r*dt) * (S[0:i, i-1] > H)
        elif Am == True:
         for i in range(M, -1, -1):
              for j in range(0, i):
                 V[j, i-1] = np.maximum((q * V[j, i] + (1-q) * V[j+1, i])*np.exp(-r*dt)*(S[0:i, i-1] > H), np.maximum(K - S[j, i-1], 0))
        else:
         print("ERROR: EU must be either True or False")
    # Finally the price of the Call, which is equivalent for American and European options
    elif Put == False:
        V[:, -1] = np.maximum(S[:, -1] - K, 0)
        for i in range(M, 0, -1):
            V[0:i, i - 1] = (q * V[0:i, i] + (1 - q) * V[1:i + 1, i]) * np.exp(-r * dt)*(S[0:i, i-1] > H)
    else:
        print("ERROR: Call must be either True or False")

    V0 = V[0, 0]
    # output the Stock price process, fair option price process and the price at time 0
    return [V0, V, S]


def EU_DownOutCall_MC(r, paths, T, H, g):
    [N, m] = paths.shape
    ST = np.ones(N)
    for i in range(0, N):
        ST[i] = paths[i, -1] * (np.sum(paths[i, :] > H) == m)

    V0 = np.mean(np.exp(-r*T) * g(ST))
    return V0

# testing
def a(x, t):
    return x*r


def b(x, t):
    return x*sigma


def g(x):
    return np.maximum(x-K, 0)


S0 = 100
r = 0.05
sigma = 0.2
T = 1
K = 100
H = 90
M = 10000
t = 0
St = S0

N = 1000
m = 1000
nu0 = 100
paths = SimPath_Ito_Euler(S0, a, b, T, m, N)
V_MC = EU_DownOutCall_MC(r, paths, T, H, g)

V_CRR = DownOut_CRR(S0, r, sigma, T, M, K, H, Am=False, Put=False)

V_BS = EUDownOutCall_BS(St, T, K, sigma, r, H, t=0)

print("Price of a DownOutCall calculated with the explicit BS-formula: " + str(np.round(V_BS, 4)))
print("Price of a DownOutCall calculated with the Monte Carlo method: " + str(np.round(V_MC, 4)))
print("Price of a DownOutCall calculated with in the CRR model: " + str(np.round(V_CRR[0], 4)))
