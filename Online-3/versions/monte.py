import matplotlib.pyplot as plt

def middle_square(seed,n):
    x=seed
    values=[]
    for i in range(n):
        x=x*x
        x_str="0"*(8-len(str(x)))+str(x)
        x_str=x_str[2:6]
        x=int(x_str)
        values.append(x)
    return values

def monte_carlo_pi(seed, n):
    values = middle_square(seed, 2 * n)

    inside = 0

    x_inside = []
    y_inside = []

    x_outside = []
    y_outside = []

    for i in range(n):

        # Convert two generated numbers to [0,1)
        x = values[ i] / 10000
        y = values[ i+1] / 10000

        # Check whether point lies inside quarter circle
        if x**2 + y**2 <= 1:
            inside += 1

            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    # Estimate pi
    pi_estimate = 4 * inside / n

    return pi_estimate, x_inside, y_inside, x_outside, y_outside


# Run simulation
n = 10000

pi_estimate, xi, yi, xo, yo = monte_carlo_pi(5731, n)

print("Number of points:", n)
print("Estimated pi:", pi_estimate)
print("Actual pi:", 3.141592653589793)
print("Error:", abs(pi_estimate - 3.141592653589793))

plt.figure(figsize=(7, 7))

plt.scatter(xi, yi, s=2, label="Inside")
plt.scatter(xo, yo, s=2, label="Outside")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Monte Carlo Estimation of Pi")

plt.legend()
plt.axis("equal")

plt.show()