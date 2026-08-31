import math

def msws_generator_step(x,w,c,m):
    w = (w+c) % m
    x = (x*x + w) % m

    u = (x//(1<<16))/(1<<16)
    return x,w,u

def msws_generator(n_samples,x0=0,w0=0,c=0xb5ad4ec5,m=2**32):
    x = x0
    w = w0
    U = []

    for _ in range(n_samples):
        x,w,u = msws_generator_step(x,w,c,m)
        U.append(u)
    return U

# print(msws_generator(10))


N = 10000
U = msws_generator(4*N)

total = 0

for k in range(N):
    u1 = U[4*k]
    u2 = U[4*k+1]
    u3 = U[4*k+2]
    u4 = U[4*k+3]

    x1 = -10 * math.log(1-u1)
    x2 = -8 * math.log(1-u2)
    x3 = -7 * math.log(1-u3)
    x4 = -5 * math.log(1-u4)

    x34 = max(x3,x4)
    x134 = min(x34,x1)
    x1234 = max(x134,x2)

    y_k = x1234

    if y_k >= 7:
        total+=1

p_hat = total/N
print(p_hat)

    



