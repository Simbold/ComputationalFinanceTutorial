import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data/time_series_dax_2019.csv", delimiter=',')


def log_returns(data):
    lr = np.diff(np.log(data))
    return lr


# remove all na values from the data
data = data.dropna()

# calculate the log returns for the "close" prices of the Dax
lr = log_returns(data["Close"])

# plot the results
x = np.arange(len(lr))
plt.plot(x, lr, linewidth=0.5)
plt.xlabel("trading days")
plt.ylabel("log-return")
plt.title("log-return of DAX in the period 10.04.2017 to 09.04.2019")
plt.legend(["daily log-return"])
plt.show()

# compute annualized mean and standard deviation of the log-returns

annualized_mean = np.round(np.mean(lr) * 251, 6)  # assuming 251 trading days on average for the DAX
annualized_sd = np.round(np.std(lr, ddof=1) * np.sqrt(251), 6)

print("DAX log-returns: annualized mean = " + str(annualized_mean) + ", annualized standard deviation = " + str(
    annualized_sd) + "!")
