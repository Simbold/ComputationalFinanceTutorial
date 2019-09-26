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


def BS_EuCall_FFT (S0, r, sigma, T, K, t, R, N, M):
    # first we define the g() function and its components ftilde (laplace transform payoff) and the BS characteristic function
    def ftilde0(z):
        return 1/((z-1)*z)

    def chi(u):
        return np.exp(1j * u * (np.log(S0) + r * T) - (1j * u + u ** 2) * sigma ** 2 / 2 * T)

    def g(u):
        return ftilde0(R+ 1j*u) * chi(u - 1j*R)
    # set Delta
    Delta = M/N
    kappa1 = np.log(np.min(K))
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
    vt = np.interp(K, Km, Vmkappa)
    return vt


def Eu_Option_BS_MC(S0, r, sigma, K, T, N, payoff, alpha=0.05):
    z = np.random.normal(0, 1, N)
    paths = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    V0 = np.exp(-r * T) * np.mean(payoff(paths))
    var = np.var(payoff(paths), ddof=1)
    ci = [V0 - norm.isf(alpha/2) * np.sqrt(var / N), V0 + norm.isf(alpha/2) * np.sqrt(var / N)]
    epsilon = norm.isf(alpha/2) * np.sqrt(var / N)
    return [V0, ci, epsilon]

def SimPath_Ito_Euler(X0, a, b, T, m, N):
    Dt = T/m
    paths = np.zeros((N, m+1), dtype=float)
    t = np.zeros(m + 1, dtype=float)
    paths[:, 0] = X0
    for i in range(1, m+1):
        t[i] = t[i-1] + Dt
        DW = np.sqrt(Dt) * np.random.normal(loc=0, scale=1, size=N)
        paths[:, i] = paths[:, i-1] + a(paths[:, i-1], t[i-1]) * Dt + b(paths[:, i-1], t[i-1]) * DW
    return paths



def Heston_PCall_Laplace (S0, r, nu0, kappa, lamb, sigma_tilde,T, K, R, p):
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