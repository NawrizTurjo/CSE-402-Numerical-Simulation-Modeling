import math
import matplotlib.pyplot as plt
import numpy as np

f = lambda x : -x**3-3*x**2+2

approx = []

def false_position(l,u,func,iter):
    for i in range(iter):
        r = (u*f(l) - l*f(u)) / (f(l)-f(u))
        approx.append(r)

        if f(l) * f(r) < 0:
            u = r
        elif f(l) * f(r) > 0:
            l = r
        else:
            break
    return r

a = 0
b = 50
divisions = 5

points = np.linspace(a,b,divisions)

index = -1

for i in range(len(points)-1):
    if f(points[i]) * f(points[i+1]) < 0:
        index = i
        break

if index == -1:
    print("points are not suitable")
    exit()

print(f"Chosen points: {points[index]}, {points[index+1]}")
soln = false_position(points[index], points[index+1], f, 100)
print(f"Soln: {soln}")

x_vals = np.linspace()

plt.plot(approx, f(np.array(approx)), color="red")
plt.show()