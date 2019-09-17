import numpy as np
import scipy.integrate as integrate
from CompFinTutorial.Functions import BlackScholes


def BS_Eu_Laplace(S0, r, sigma, T, K, R):
    def chi(u):
        return np.exp(1j * u * (np.log(S0) + r * T) - (1j * u + u ** 2) * sigma ** 2 / 2 * T)

    def ftilde(z):
        return (K ** (1 - z)) / (z * (z - 1))

    def integrand_Laplace(u):
        y = np.real(ftilde(R + 1j * u) * chi(u - 1j * R))
        return y

    V0 = np.exp(-r * T) / np.pi * integrate.quad(integrand_Laplace, 0, np.inf)[0]
    return V0




S0 = 100
r = 0.05
sigma = 0.2
T = 1
K = 100
R = 1.1

V0_Laplace = BS_Eu_Laplace(S0, r, sigma, T, K, R)


# Now comparing with the Black Scholes explicit formula:

V0_BS = BlackScholes(St=S0, T=T, K=K, sigma=sigma, r=r, t=0, Call=True)

print("The price of the call option using the Laplace transform is: " + str(np.round(V0_Laplace,
     5)) + ", the 'exact' price using the BS explicit formula is: " + str(np.round(V0_BS, 6)))


# when pricing EU Put option the only thing to be changed is the parameter R
# R < 0 must be set so that the integral of the laplace transform is finite
# for the call R > 1 is the condition

R = -1

V0_Laplace_put = BS_Eu_Laplace(S0, r, sigma, T, K, R)


# Again comparing with the Black Scholes explicit formula:

V0_BS_put = BlackScholes(St=S0, T=T, K=K, sigma=sigma, r=r, t=0, Call=False)


print("The price of the put option using the Laplace transform is: " + str(np.round(V0_Laplace_put,
     5)) + ", the 'exact' price using the BS explicit formula is: " + str(np.round(V0_BS_put, 6)))
