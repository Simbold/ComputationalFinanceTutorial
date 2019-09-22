import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def TruncNorm_pdf(x, a, b, mu, sigma):
    # the pdf of the truncated normal distribution
    return (norm.pdf((x - mu) / sigma)) / (sigma * norm.cdf((b - mu) / sigma) - norm.cdf((a - mu) / sigma))

def Sample_TruncNormal_AR(a, b, mu, sigma, N):
    # calculate constant for acceptance rejection method which is f(x)/g(x) using x~unif(a,b)
    C = TruncNorm_pdf(x=mu, a=a, b=b, mu=mu, sigma=sigma) * (b-a)

    x = np.zeros(N, dtype=float)
    for i in range(0, N):
        success = False
        while success==False:
            U = np.random.random(size=2)
            Y = U[0] *(b-a) + a # transform unif(0,1) to unif(a,b)
            success = U[1] <= TruncNorm_pdf(x=Y, a=a, b=b, mu=mu, sigma=sigma) * (b-a) / C
        x[i] = Y
    return x



# the mu nd sigma refer to the parameters of the truncated normal as used in the formula of its pdf
a = -3
b = 2
mu = 0
sigma = 1
N = 20000
X = Sample_TruncNormal_AR(a, b, mu, sigma, N)

# plotting the results and comparing to true density:
plt.hist(X, bins=50, density=True)
x = np.arange(a, b+0.01, 0.01)
y = TruncNorm_pdf(x, a, b, mu, sigma)
plt.plot(x, y, color="red")
plt.ylabel("Density")
plt.xlabel("x")
plt.title("Illustration of the acceptance/rejection method to sample from the truncated normal distribution")
plt.legend(["True density", "Acceptance/rejection method"])
plt.show

