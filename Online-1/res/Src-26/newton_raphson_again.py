import numpy as np
import matplotlib.pyplot as plt

def f(coeffs, x):
    res = 0
    # for coeff in coeffs:
    #     res = res * x + coeff
    res = 1000 - 298*x + 3 * x**(2/3)
    return res

def derivative(coeffs, x, h = 0.001):
    return (f(coeffs, x+h)-f(coeffs, x))/h

def newton_raphson(coeffs, start, e_s):
    e_a = float('inf')
    x = start
    while e_a > e_s:
        x_next = x - f(coeffs,x)/derivative(coeffs, x)
        e_a = 100 * abs((x - x_next)/x)
        x = x_next
    return x

istr = input("Enter the coeffs: ")
istr = list(map(int, istr.split(" ")))
coeffs = istr.copy()

root = newton_raphson(coeffs, 0.1, 0.05)

x = np.arange(0, 100, 1)
y = 1000 - 298*x + 3 * x**(2/3)

plt.grid()
plt.plot(x, y)
plt.axhline(c='black')
plt.scatter(root, 0, c="red")
plt.show()


    