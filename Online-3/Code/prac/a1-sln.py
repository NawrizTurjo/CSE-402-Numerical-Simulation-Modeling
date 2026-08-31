import random
import math

def buffon_pi(n, L=1.0,D=2.0):
    hits = 0

    for i in range(n):
        x = (D/2.0) * random.random()
        theta = (math.pi/2.0) * random.random()

        condition = (L/2.0) * math.sin(theta)

        if x <= condition:
            hits+=1

    # print(hits)
    # print(n)

    p_hat = hits/n
    pi_est = (2.0 * L) / (D * p_hat)

    # print(p_hat)
    # print(pi_est)

    abs_error = abs(math.pi - pi_est)
    
    
    return hits, p_hat, pi_est, abs_error

N = [100,1000,10000,100000,1000000]

for n in N:
    print(f"=" * 60)
    print(f"N Drops: {n}")
    hits, p_hat, pi_est, abs_error = buffon_pi(n)
    print(f"Hits: {hits}")
    print(f"P_hat: {p_hat}")
    print(f"pi_est: {pi_est}")
    print(f"Abs error: {abs_error}")


