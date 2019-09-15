import numpy as np
import matplotlib.pyplot as plt


def cubic_regression(x, y):
    X = np.column_stack((x ** 3, x ** 2, x, np.ones(x.size)))
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    yhat = X @ beta
    return [beta, yhat]  # return a list


x = np.array([[0, 1, 2, 3, 4, 5, 6]]).T
y = np.array([[2, 1, 0, 1, 2, 4, 3]]).T

beta = cubic_regression(x, y)[0]  # the beta values are at position 0 of the list
yhat = cubic_regression(x, y)[1]

# make a plot, visualizing the regression fit
plt.plot(x, y, "o")
plt.plot(x, yhat)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cubic regression")
plt.legend(["Data points", "Fitted Values"])
plt.show()
