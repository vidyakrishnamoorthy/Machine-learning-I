import math, copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cost_and_gradient import compute_cost
from cost_and_gradient import compute_gradient
from cost_and_gradient import gradient_descent


x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)
cost = compute_cost(x_train, y_train, slope, intercept)
gradient = compute_gradient(x_train, y_train, slope, intercept)

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(f"cost: {cost}")
print(f"gradient: {gradient}")
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()


print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")


# some gradient descent settings
iterations = 10
tmp_alpha = 8.0e-1

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(f"cost: {cost}")
print(f"gradient: {gradient}")
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()