import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def Heston_PCall_Laplace (S0, r, nu0, kappa, lamb, sigma_tilde,T, K, R):
    def integrand(u):
        def d(u):
            return np.sqrt(lamb ** 2+sigma_tilde ** 2 * ( 1j * u+u ** 2))
        # characteristic function for the Heston model
        def nu(u, x, nu0):
            y1 = np.exp( 1j * u * (x + r * (T)))
            y2 = ((np.exp(lamb* (T) / 2)) / (np.cosh(d(u) * (T) / 2)+ lamb * np.sinh(d(u) * (T) / 2) / d(u))) ** (2 * kappa / sigma_tilde ** 2)
            y3 = np.exp(-nu0 * (( 1j * u+u ** 2) * np.sinh(d(u) * (T) / 2) / d(u)) / (np.cosh(d(u) * (T) / 2)+ lamb * np.sinh(d(u) * (T) / 2) / d(u)))
            return y1*y2*y3

        def ftilde(z):
            return (K ** (1 - z / p) * p) / (z * (z - p)) # laplace transform for powercall payoff

        return np.real(ftilde(u*1j + R) * nu(u - 1j*R, np.log(S0), nu0))

    V0 = (np.exp(-r*T)/np.pi) * integrate.quad(integrand, 0, 100)[0] # cant use np.inf due to overflow, but 100 should be more than enough
    return V0


S0range = np.arange(50,151)
r=0.05
sigma_tilde=0.2
T=1
K=100
R=6
nu0 = 0.3**2
kappa=0.3**2
lamb=2.5
p=1
test = Heston_PCall_Laplace(50, r, nu0, kappa, lamb, sigma_tilde, T, K, R)
print(test)


V0 = np.zeros(101, dtype=float)
j=0
for i in S0range:
    V0[j] = Heston_PCall_Laplace (i, r, nu0, kappa, lamb, sigma_tilde,T, K, R)
    j = j+1

plt.plot(S0range, V0, linewidth = 0.5)
plt.xlabel("Strike K")
plt.ylabel("Price")
plt.legend(["Priceof the power call for p = 2"])
plt.show()