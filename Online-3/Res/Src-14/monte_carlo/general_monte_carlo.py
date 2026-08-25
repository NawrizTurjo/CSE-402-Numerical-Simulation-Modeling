import random, math

def monte_carlo(g, a, b, n, seed=None):
    """
    Estimate integral of g(x) dx from a to b using n uniform samples.
    I = (b-a) * E[g(X)],  X ~ U(a,b)
    """
    if seed is not None:
        random.seed(seed)
    total = 0
    for _ in range(n):
        x = random.uniform(a, b)
        total += g(x)
    return (b - a) * total / n

# Example: integral of sin(x) from 0 to pi
# True answer = [-cos(x)] from 0 to pi = -cos(pi) + cos(0) = 1 + 1 = 2
result = monte_carlo(math.sin, 0, math.pi, n=10000, seed=1)
print(f"Estimated: {result:.4f}")   # close to 2.0
print(f"True value: 2.0")