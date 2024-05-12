import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cost_and_gradient import compute_cost


x_train = np.array([1.0,2.0])
y_train = np.array([300.,500.])

x1_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y1_train = np.array([250, 300, 480, 430, 630, 730])

plt.scatter(x_train, y_train, marker='x', c='r')
plt.scatter(x1_train, y1_train, marker='o', c='b')

slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1_train, y1_train)

print(compute_cost(x_train, y_train, slope, intercept))
print(compute_cost(x1_train, y1_train, slope1, intercept1))

plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()


