import random
import math
seed = 2500
def find_cycle(step, seed=2500):
    
    # step is a function

    seen_at = {}
    curr = seed
    iteration = 0

    while curr not in seen_at:
        seen_at[curr] = iteration
        curr = step(curr)
        iteration+=1

    first_seen_iteration = seen_at[curr]
    cycle_len = iteration - first_seen_iteration
    return curr,iteration,cycle_len

def bin_counts(uniforms, bins=10):
    counts = [0] * bins
    for value in uniforms:
        index = min(
            bins-1, int(value*bins)
        ) # atmost bins-1 porjonto hobe
        counts[index] +=1
    return counts

def lcg_step(x, a=21, c=1, m=10000):
    return (a * x + c) % m

def lcg(n,seed=2500, a=21, c=1, m=10000):
    """Generate n successive integer states X1..Xn."""
    values = []
    x = seed
    for _ in range(n):
        x = lcg_step(x, a, c, m)
        values.append(x)
    return values

def lcg_uniforms(n, seed=2500, a=21, c=1, m=10000):
    """Same sequence as lcg(), rescaled to floats in [0, 1)."""
    return [x / m for x in lcg(n, seed, a, c, m)]

def find_lcg_cycle(seed=2500, a=21, c=1, m=10000):
    return find_cycle(
        step=lambda x: lcg_step(x,a,c,m),
        seed=seed
    )

def lcg_next(a=21, c=1, m=10000):
    """
    Generate the next LCG value using the global seed,
    then update the global seed.
    """
    global seed

    seed = lcg_step(seed, a, c, m)
    val = seed / m

    return val

# ==========================================================

def buffon_pi(n, L=1.0,D=2.0):
    hits = 0

    samples = lcg_uniforms(n=2*n)

    # x_samples = samples[:n]
    # theta_samples = samples[n:]

    for i in range(n):
        # x = (D/2.0) * random.random()
        # theta = (math.pi/2.0) * random.random()
        
        # x = (D/2.0) * x_samples[i]
        # theta = (math.pi/2.0) * theta_samples[i]

        x = (D/2.0) * lcg_next()
        theta = (math.pi/2.0) * lcg_next()
        

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


