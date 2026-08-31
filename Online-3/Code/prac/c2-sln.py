def find_cycle(step, seed):
    
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


def lcg_step(x, a, c, m):
    return (a * x + c) % m

def lcg(n, seed=1, a=5, c=0, m=65536):
    """Generate n successive integer states X1..Xn."""
    values = []
    x = seed
    for _ in range(n):
        x = lcg_step(x, a, c, m)
        values.append(x)
    return values

def lcg_uniforms(n, seed=1, a=5, c=0, m=65536):
    """Same sequence as lcg(), rescaled to floats in [0, 1)."""
    return [x / m for x in lcg(n, seed,a,c,m)]

def find_lcg_cycle(seed=1, a=5, c=0, m=65536):
    return find_cycle(
        step=lambda x: lcg_step(x,a,c,m),
        seed=seed
    )

from scipy import stats
import math

def ks_statistic(sample):
    """Manual D+, D-, D computation (matches the slide's table layout)."""
    n = len(sample)
    s = sorted(sample)
    d_plus = max(i / n - s[i - 1] for i in range(1, n + 1))
    d_minus = max(s[i - 1] - (i - 1) / n for i in range(1, n + 1))
    return d_plus, d_minus, max(d_plus, d_minus)


def ks_uniform_test(sample, alpha=0.05):
    """
    Full K-S uniformity test. Uses scipy's `ksone` distribution for the
    critical value / p-value -- exact for any N and alpha (no table needed).
    """
    d_plus, d_minus, d = ks_statistic(sample)
    n = len(sample)
    d_critical = 1.36/ math.sqrt(n)
    p_value = 1 - stats.ksone.cdf(d, n)
    # print(f"d_critical: {d_critical}")
    # print(f"p_value: {p_value}")
    # print(f"D value: {d}")
    decision = "Reject H0" if d > d_critical else "Do not reject H0"
    return {
        "D+": d_plus,
        "D-": d_minus,
        "D": d,
        "critical_value": d_critical,
        "p_value": p_value,
        "decision": decision,
    }

n = 100000
samples = lcg_uniforms(n)
print(ks_uniform_test(samples))

from math import gcd

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def check_max_period_conditions(a=5, m=65536, seed=1,c=0):
    """
    Checks which theoretical max-period case applies and whether its
    conditions are satisfied. Returns (case_description, theoretical_max_period,
    conditions_satisfied, details_string). Falls back to "measure it yourself
    with find_lcg_cycle" when m doesn't match a known closed-form case
    (e.g. m composite and not a power of 2).
    """
    is_power_of_two = m > 0 and (m & (m - 1)) == 0

    if is_power_of_two and c != 0:
        cond1 = gcd(c, m) == 1
        cond2 = (a - 1) % 4 == 0
        return ("Case 1: m=2^b, mixed (c!=0)", m, cond1 and cond2,
                f"gcd(c,m)==1: {cond1}, (a-1)%4==0: {cond2}")

    if is_power_of_two and c == 0:
        cond1 = seed % 2 == 1
        cond2 = a % 8 in (3, 5)
        return ("Case 2: m=2^b, multiplicative (c=0)", m // 4, cond1 and cond2,
                f"seed is odd: {cond1}, a%8 in {{3,5}}: {cond2}")

    if c == 0 and is_prime(m):
        # a is a primitive root mod m  <=>  order of a divides (m-1) and equals it exactly.
        order, val = 1, a % m
        while val != 1 and order < m:
            val = (val * a) % m
            order += 1
        return ("Case 3: m prime, multiplicative (c=0)", m - 1, order == m - 1,
                f"multiplicative order of a mod m = {order}")

    return ("General case (no closed form here)", None, None,
            "use find_lcg_cycle(seed, a, c, m) to measure the period directly")

print(check_max_period_conditions())
first_repeat, iteration, cycle_len = find_lcg_cycle()
print(f"Seed: 1, First Repeated Value: {first_repeat}, Iteration: {iteration}, Cycle Length: {cycle_len}")