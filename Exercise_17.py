import numpy as np
import matplotlib.pyplot as plt
from CompFinTutorial.Functions import BlackScholes


def BS_EuCall_FFT (S0, r, sigma, T, t, R, N, M, kappa1):
    # first we define the g() function and its components ftilde (laplace transform payoff) and the BS characteristic function
    def ftilde0(z):
        return 1/((z-1)*z)

    def chi(u):
        return np.exp(1j * u * (np.log(S0) + r * T) - (1j * u + u ** 2) * sigma ** 2 / 2 * T)

    def g(u):
        return ftilde0(R+ 1j*u) * chi(u - 1j*R)
    # set Delta
    Delta = M/N
    # define vector x on which the DFT will be performed
    x = np.zeros(N, dtype=complex)
    for i in range(1, N):
        x[i-1] = g((i - 0.5)*Delta) * Delta * np.exp(-1j*(i-1)*Delta*kappa1)

    # perform DFT using the efficient FFT algorithm
    xhat = np.fft.fft(x)
    # compute vector kappa
    kappa_m = kappa1 + (np.arange(1, N+1) - 1 ) * 2 * np.pi / M

    Km = np.exp(kappa_m)
    # finally compute the option prices
    Vmkappa = (np.exp(-r*(T-t) + (1-R)*kappa_m))/np.pi * np.real(xhat*np.exp(-1j * Delta * kappa_m / 2))
    return [Km, Vmkappa]


# some test parameters
S0 = 100
r = 0.05
sigma = 0.2
T = 1
R = 1.1 # because we are pricing a call, for put R must be smaller zero
N = 2 ** 11
M = 50
t = 0
kappa1 = np.log(np.min(K))

# computing the prices for all the kappas
result = BS_EuCall_FFT(S0, r, sigma, T, t, R, N, M, kappa1)

# using simple interpolation to find prices corresponding to desired Strikes K:#
K = np.arange(80, 180+0.1, 0.1)
vm = np.interp(K, result[0], result[1])

Vm_bs = np.zeros(1001, dtype=float)
j=0
for i in K:
    Vm_bs[j] = BlackScholes(St=S0, T=T, K=i, sigma=sigma, r=r, t=0, Call=True)
    j=j+1


# plotting the results to compare fast fourier prices with BS prices
plt.plot(K, Vm_bs, linewidth = 0.5)
plt.plot(K, vm, linewidth = 0.5)
plt.xlabel("Strike K")
plt.ylabel("Price")
plt.legend(["BS-formula", "Fast Fourier algorithm"])
plt.show()