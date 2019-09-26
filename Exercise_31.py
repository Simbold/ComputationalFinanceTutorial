import numpy as np
from CompFinTutorial.Functions import SimPath_Ito_Euler, Heston_PCall_Laplace


def Heston_EuOption_MC(S0, r, gamma, T, g):
    N, m = gamma.shape[0], gamma.shape[1]-1
    Dt = T/m

    drift = (np.ones((N, m), dtype=float)*r - gamma[:, 0:m] * Dt * 0.5) @ np.ones((m, 1))
    difusion = np.sqrt(Dt * (gamma[:, 0:m] * Dt) @ np.ones((m, 1)))
    ST = S0*np.exp(drift*Dt + difusion * np.random.normal(0, 1, N))
    V0 = np.exp(-r*T) * np.mean(np.apply_along_axis(g, axis=0, arr=ST))
    return V0

K = 90
p = 1.2
S0 = 100
r = 0.05
T = 1
X0 = 0.3**2
sigma_tilde = 0.2
lamb = 2.5
kappa = 0.3**2
N = 10000
m = 100

def a(x, t):
    return (kappa-lamb*x)

def b(x, t):
    return np.sqrt(x) * sigma_tilde
def g(x):
    return np.maximum(x**p - K, 0)

# generate gamma
gamma_paths = SimPath_Ito_Euler(X0, a, b, T, m, N)

V0_MC = Heston_EuOption_MC(S0, r, gamma_paths, T, g)


# comparing to the exact solution via Laplace transform
nu0 = X0
R = 1.5
V0_LP = Heston_PCall_Laplace(S0, r, nu0, kappa, lamb, sigma_tilde, T, K, R, p)
print("Price of the European power Call with strike price 90 and p=1.2 in the Heston model calculated by MC methods is: "+
      str(round(V0_MC, 5)) + ", compared to: " + str(round(V0_LP, 5)) + " when calculated via the laplace transform/characteristic function approach:")