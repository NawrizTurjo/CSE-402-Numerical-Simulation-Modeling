import math

def gen(seed, a, c, m, n):
    X = seed
    R_s = []
    for _ in range(n):
        X = (a*X + c) % m
        R_s.append(X/m)
    return R_s

Rs = gen(seed=27, a=17, c=43, m=100, n=5)

# Use the R values directly as U[0,1] numbers
print("Uniform [0,1]:", Rs)

# Transform to Exponential with mean β using inverse transform: X = -β·ln(R)
beta = 2.0
exponential = [-beta * math.log(r) for r in Rs]
print("Exponential(β=2):", exponential)

# Transform to uniform on [a,b]: X = a + (b-a)·R
uniform_ab = [5 + (10-5)*r for r in Rs]
print("Uniform[5,10]:", uniform_ab)