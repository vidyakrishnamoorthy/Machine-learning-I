import copy, math
import numpy as np
import matplotlib.pyplot as plt
from cost_and_gradient import predict_single_loop_multiple
from cost_and_gradient import compute_cost_multiple
from cost_and_gradient import predictdot_multiple
from cost_and_gradient import gradient_descent_multiple
from cost_and_gradient import compute_gradient_multiple


np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]) # multiple features
y_train = np.array([460, 232, 178])

print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)


b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

f_wb = predict_single_loop_multiple(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

f_wb = predictdot_multiple(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

cost = compute_cost_multiple(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

tmp_dj_db, tmp_dj_dw = compute_gradient_multiple(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

initial_w = np.zeros_like(w_init)
initial_b = 0.

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent_multiple(X_train, y_train, initial_w, initial_b,
                                                    compute_cost_multiple, compute_gradient_multiple,
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost');  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step');  ax2.set_xlabel('iteration step')
plt.show()
