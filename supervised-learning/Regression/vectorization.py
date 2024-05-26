import numpy as np
import time

a = np.zeros(4); print(f"{a}, {a.shape}, {a.dtype}")
a = np.zeros((4,)); print(f"{a}, {a.shape}, {a.dtype}")
a = np.random.random_sample(4); print(f"{a}, {a.shape}, {a.dtype}")

a = np.arange(4.); print(f"{a}, {a.shape}, {a.dtype}")
a = np.random.rand(4); print(f"{a}, {a.shape}, {a.dtype}")

a = np.arange(10); print(f"{a}, {a.shape}, {a.dtype}")
try:
    print(a[10])
except Exception as e:
    print(f"Errpr message: {e}")

a = np.arange(10); print(f"a = {a}")

c = a[2:7:1]; print("a[2:7:1] = ", c)

c = a[2:7:2]; print("a[2:7:2] = ", c)

c = a[3:]; print("a[3:] = ", c)

c = a[:3]; print("a[:3] = ", c)

c = a[:]; print("a[:] = ", c)

a = np.array([1,2,3,4]); print(f"a: {a}")

b = -a; print(f"b = -a: {b}")

b = np.sum(a); print(f"b = np.sum(a): {b}")

b = np.mean(a); print(f"b = np.mean(a): {b}")

b = a**2; print(f"b = a**2: {b}")

a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise {a} + {b}: {a + b}")

c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print(f"The error message you'll see is: {e}")

a = np.array([1, 2, 3, 4])
b = 5 * a
print(f"{b} = 5 * {a} : {b}")


