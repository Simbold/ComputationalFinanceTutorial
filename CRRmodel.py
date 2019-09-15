import numpy as np

#CRR Price mainly for american put options since trees are useful for an optimal stopping problem
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
     print("Price of the option at time 0 is: " + str(V0))
    # output the Stock price process, fair option price process and the price at time 0
     return [V0, V, S]

# a test example
S0 = 100
r = 0.06
sigma = 0.2
T = 2
M = 100
K = 120
Am = True
Put = True

CRRresults = CRRprice(S0, r, sigma, T, M, K, Am, Put)
price = CRRresults[0]
print("Price of the option at time 0 is: " + str(price))