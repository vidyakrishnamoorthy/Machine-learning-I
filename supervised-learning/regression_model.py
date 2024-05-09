import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 300., 345.])
y_train = np.array([300.0, 500.0, 121.,2.])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

print(f"x_train.shape: {x_train.shape}")
print(f"Number of training examples is: {x_train.shape[0]}")
print(f"Number of testing examples is: {len(y_train)}")

i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()