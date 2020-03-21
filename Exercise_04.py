import numpy as np


def CRRprice (spot, strike, r, sigma, mt, m, american_exercise=True, option_type="put"):
    dt = mt / m
    b = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma ** 2) * dt))
    u = b + np.sqrt(b ** 2 - 1)
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    if option_type == "call":
        # generate tree state at maturity
        tree = np.zeros(m + 1)
        tree[0] = spot
        for j in range(1, m + 1):
            tree[j] = tree[j - 1] * d
            tree[0:j] = tree[0:j] * u

        # compute payoff to initialize value process
        v = np.maximum(tree - strike, 0)
        # compute option value
        for j in range(m, 0, -1):
            v = (q * v[0:j] + (1 - q) * v[1:j + 1]) * np.exp(-r * dt)
        v0 = v[0]

    elif (american_exercise == False) & (option_type == "put"):
        # generate tree state at maturity
        tree = np.zeros(m + 1)
        tree[0] = spot
        for j in range(1, m + 1):
            tree[j] = tree[j - 1] * d
            tree[0:j] = tree[0:j] * u
        # compute payoff to initialize value process
        v = np.maximum(strike - tree, 0)
        # compute option value
        for j in range(m, 0, -1):
            v = (q * v[0:j] + (1 - q) * v[1:j + 1]) * np.exp(-r * dt)
        v0 = v[0]

    elif (american_exercise == True) & (option_type == "put"):
        # generate entire tree
        full_tree = np.zeros((m + 1, m + 1))
        full_tree[0, 0] = spot
        for j in range(1, m + 1):
            full_tree[j, j] = full_tree[j - 1, j - 1] * d
            full_tree[0:j, j] = full_tree[0:j, j - 1] * u

        # compute payoff to initialize value process
        v = np.maximum(strike - full_tree[:, -1], 0)
        # compute option value
        for j in range(m, 0, -1):
            for i in range(0, j):
                v[i] = np.maximum((q*v[i]+(1-q)*v[i+1])*np.exp(-r*dt), np.maximum(strike-full_tree[i, j-1], 0))
            v = v[0:j]
        v0 = v[0]

    else:
        print("ERROR: variable: american must be True or False; variable: option_type must be 'call' or 'put'")
        return
    return v0


# a test example
spot = 100
r = 0.05
sigma = 0.3
mt = 1  # maturity time
m = 500  # number of steps
strike = 120
american_exercise = False

v0 = CRRprice(spot, strike, r, sigma, mt, m, american_exercise)

print("Price of the American put at time 0 is: " + str(v0))