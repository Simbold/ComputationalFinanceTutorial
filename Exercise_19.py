import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from CompFinTutorial.Functions import BS_EuCall_FFT

# the data set must have strike prices in first column, maturities in the second and prices in the third column!!!!!!
def BS_EuCall_Calibrate (S0, r, data, sigma0, R, N, M):
    # defining function to be minimized
    def loss_function(x):
        T = np.unique(data[:,1])
        for j in T:
            ind = np.where(data[:, 1] == j)
            K = data[ind, 0] # set coresponding strikes
            v0 = BS_EuCall_FFT(S0, r, sigma=x, T=j, t=0, R=R, N=N, M=M, K=K) # BS_EuCall_FFT now returns prices coresponding to input strike K
            # calculate MSE
            mse = ((data[ind, 2] - v0)**2).mean()
        return mse
    # minimize loss function with respect to sigma
    optSigma = minimize(loss_function, sigma0, method="Powell")
    return optSigma




# import Data
prices = np.genfromtxt("Data/Dax_CallPrices_Eurex.csv", delimiter=";", skip_header=True, )

# set test parameters
S0 = 12658
r = 0.05
sigma0 = 0.3
R = 1.1
N = 2**11
M = 50

opt = BS_EuCall_Calibrate (S0, r, prices, sigma0, R, N, M)
print(opt)

sigma = opt["x"]

T = np.unique(prices[:,1])
for j in T:
    ind = np.where(prices[:, 1] == j)
    K = prices[ind, 0] # set coresponding strikes
    v0 = BS_EuCall_FFT(S0, r, sigma=sigma, T=j, t=0, R=R, N=N, M=M, K=K)
    obs = prices[ind, 2]
    plt.plot(K[0, :], v0[0, :], "-")
plt.xlabel("Strike")
plt.ylabel("Price")
Tr = np.round(T, 4)
plt.plot(prices[:, 0], prices[:, 2], "x", color="black")
plt.legend(["T="+str(Tr[0]), "T="+str(Tr[1]), "T="+str(Tr[2]), "T="+str(Tr[3]), "T="+str(Tr[4])])
plt.title("Calibrated prices vs observed prices BS-model")
plt.show()





