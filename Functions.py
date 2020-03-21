import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate


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


def BlackScholes(St, T, K, sigma, r, t=0, Call=True):
    d1 = (np.log(St/K)+ (r+0.5*(sigma**2)) * (T-t)) / sigma * np.sqrt(T-t)
    d2 = d1 - sigma * np.sqrt(T-t)

    if Call == True:
        Vt = St * norm.cdf(d1) - K * np.exp(-r*(T-t)) * norm.cdf(d2)
    elif Call == False:
        Vt = K * np.exp(-r*(T-t)) * norm.cdf(-d2) - St * norm.cdf(-d1)
    else:
        print("ERROR: Call must be either True or False")
        return
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


def brennon_schwartz(alpha, beta, gamma, bs, g_nui):
    # Solution to Ax-b >= 0 ; x >= g and (Ax-b)'(x-g)=0 ; such that solution x satisfies x_i = g_i for i=1,...,k and x_i > g_i for i = k-1,...,n
    #  where A is tridiagonal with values alpha, beta, gamma
    n = len(alpha)
    alpha_hat = np.zeros(n, dtype=np.float64)
    b_hat = np.zeros(n, dtype=np.float64)

    alpha_hat[-1] = alpha[-1]
    b_hat[-1] = bs[-1]

    for i in range(n - 2, -1, -1):
        alpha_hat[i] = alpha[i] - beta[i] * gamma[i] / alpha_hat[i + 1]
        b_hat[i] = bs[i] - beta[i] * b_hat[i + 1] / alpha_hat[i + 1]

    x = np.zeros(n, dtype=np.float64)
    x[0] = np.maximum(b_hat[0] / alpha_hat[0], g_nui[0])
    for i in range(1, n):
        x[i] = np.maximum((b_hat[i] - gamma[i - 1] * x[i - 1]) / alpha_hat[i], g_nui[i])
    return x


def BS_AmPut_FiDi_CN(r, sigma, K, T, a, b, m, nu_max):
    dx = (b-a)/m
    dt = sigma**2 * T /(2*nu_max)
    xtilde = a + np.arange(0, m+1) * dx
    ttilde = np.arange(0, nu_max+1) * dt

    lamb = dt/dx**2
    q = 2*r/sigma**2

    def g(ttilde, xtilde):
        s = np.exp((q+1)**2 * ttilde/4) * np.maximum(np.exp(xtilde*(q-1)/2)-np.exp(xtilde*(q+1)/2), 0)
        return s

    # initial boundary condition for nu
    w = g(ttilde[0], xtilde)
    # the tridiagonal Matrix with alpha on diagonal and beta to the right and gamma to the left
    # here beta and gamma are equal and the Brennon Schwartz Algorithm is constructed such that the offdiagonal is shorter
    alpha = np.ones(m - 1, dtype=float) * (1 + lamb)
    beta = np.ones(m - 2, dtype=float) * (-0.5 * lamb)

    bs = np.zeros(m-1)
    for i in range(0, nu_max):
        g_nui = g(ttilde[i], xtilde)

        w[-1] = g_nui[-1]
        w[0] = g_nui[0]
        bs[1:-1] = w[2:-2] + 0.5*lamb*(w[1:-3] - 2*w[2:-2] + w[3:-1])

        bs[0] = w[1] + 0.5*lamb*(w[2] - 2*w[1] + g_nui[0] + g(ttilde[i+1], a))
        bs[-1] = w[-2] + 0.5*lamb*(g_nui[-1] - 2*w[-2] + w[-3] + g(ttilde[i+1], b))

        w[1:-1] = brennon_schwartz(alpha, beta, beta, bs, g_nui[1:-1])

    v0 = K*w*np.exp(-xtilde*(q-1)/2 - sigma**2 * T/2 *((q-1)**2 /4 + q))
    S = K*np.exp(xtilde)
    return [v0, S]
