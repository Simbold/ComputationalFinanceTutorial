import numpy as np
import matplotlib.pyplot as plt
from CompFinTutorial.Functions import BlackScholes, CRRprice


def CRRprice_altcond(S0, r, sigma, T, M, K, Am=True, Put=True):
    # calculate alternative parameters u, d, q, dt
    dt = T/M
    alpha = np.exp(r*dt)
    gamma = (K/ S0) ** (2/M)
    b = 0.5*(gamma * (alpha ** -1) + alpha * np.exp((sigma ** 2) * dt))
    u = b + np.sqrt(b**2 - gamma)
    d = gamma/u
    q = (alpha - d) / (u-d)

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
     # print("Price of the option at time 0 is: " + str(V0))
    # output the Stock price process, fair option price process and the price at time 0
    return [V0, V, S]


# a test parameters
S0, St = 100, 100
r = 0.03
sigma = 0.3
T = 1
M = 100
Call = True
Put = False
Am = False
t = 0


V0_altcond = np.zeros(201, dtype=float)
V0_CRR = np.zeros(201, dtype=float)
V0_BS = np.zeros(201, dtype=float)
for i in range(70, 200):
    V0_altcond[i] = CRRprice_altcond(S0, r, sigma, T, M , K = i, Am = Am, Put = Put)[0]
    V0_CRR[i] = CRRprice(S0, r, sigma, T, M, K=i, Am = False, Put= Put)[0]
    V0_BS[i] = BlackScholes(St, T, K=i, sigma=sigma, r=r, t=t, Call=Call)


error_altcond = V0_BS - V0_altcond
error_og = V0_BS - V0_CRR

x = np.arange(1, len(V0_altcond)+1)
plt.plot(x[70:201], error_altcond[70:201], linewidth = 0.5)
plt.plot(x[70:201], error_og[70:201], linewidth = 0.5)
plt.xlabel("Strike price")
plt.ylabel("Deveations from 'real' BS-Price")
plt.title("Alternatice Binomial Model (Price evolving around Strike Price")
plt.legend(["Error for alternative conditions", "Error for original conditions"])
plt.show()


