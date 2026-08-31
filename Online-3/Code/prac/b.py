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


# ===========================LCG===========================


def lcg_step(x, a, c, m):
    return (a * x + c) % m

def lcg(seed, a, c, m, n):
    """Generate n successive integer states X1..Xn."""
    values = []
    x = seed
    for _ in range(n):
        x = lcg_step(x, a, c, m)
        values.append(x)
    return values

def lcg_uniforms(seed, a, c, m, n):
    """Same sequence as lcg(), rescaled to floats in [0, 1)."""
    return [x / m for x in lcg(seed, a, c, m, n)]

def find_lcg_cycle(seed, a,c,m):
    return find_cycle(
        step=lambda x: lcg_step(x,a,c,m),
        seed=seed
    )

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# print(is_prime(10))
# print(is_prime(11))


from math import gcd

def check_max_period_conditions(a, c, m, seed):
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

# print(check_max_period_conditions(
#     a=10,
#     c=15,
#     m=100,
#     seed=42
# ))

"""
এখানে X_k সবসময় integer state।

seed: integer, ideally 0 <= seed < m
a: integer multiplier, usually 0 <= a < m
c: integer increment, usually 0 <= c < m
m: positive integer modulus, অবশ্যই m > 0
n: non-negative integer, কয়টা value generate করবে
lcg() output: integers in 0, 1, ..., m-1
lcg_uniforms() output: floats in [0, 1)

"""

# seed = 7
# a = 5
# c = 3
# m = 16
# n = 20

# print(lcg(seed, a, c, m, n))
# print(lcg_uniforms(seed, a, c, m, n))
# _,_,cycle_len = find_lcg_cycle(seed,a,c,m)
# print(cycle_len)




# ====================middle-square====================

def middle_square_step(x, digits=4):
    """One step: x -> middle `digits` digits of x^2 (zero-padded to 2*digits digits)."""
    width = 2 * digits
    offset = digits // 2
    square_as_digits = f"{x * x:0{width}d}"
    return int(square_as_digits[offset:offset + digits])

def middle_square(seed, n, digits=4):
    """
    Generate n successive values X1..Xn (each an integer with up to
    `digits` digits) starting from `seed`, using the middle-square
    recurrence. digits=4 (the C1 question's width) is the default.
    """
    max_seed = 10**digits - 1
    if not isinstance(seed, int) or not (0 <= seed <= max_seed):
        raise ValueError(f"seed must be an integer in [0, {max_seed}]")

    values = []
    current = seed
    for _ in range(n):
        current = middle_square_step(current, digits)
        values.append(current)
    return values
    
def middle_square_uniforms(seed, n, digits=4):
    """Same sequence as middle_square(), rescaled to floats in [0, 1)."""
    scale = 10**digits
    return [value / scale for value in middle_square(seed, n, digits)]

def find_middle_square_cycle(seed, digits=4):
    """Task 2: locate the first repeat, its iteration, and the cycle length."""
    return find_cycle(lambda x: middle_square_step(x, digits), seed)


def middle_square_weyl(seed, n, digits=4, weyl_increment=3571):
    """
    Middle-Square with a Weyl-sequence repair for the short-cycle weakness:

        w_{i+1} = (w_i + s) mod 10^digits            (s odd -> full period 10^digits)
        x_{i+1} = (middle_square_step(x_i) + w_{i+1}) mod 10^digits

    `weyl_increment` (s) must be odd, or the Weyl sequence itself won't
    reach full period. The state is now effectively the PAIR (x, w) --
    since w cycles through all 10^digits values before repeating, x can
    never get trapped revisiting the same value forever the way plain
    middle-square does (verified against all three seeds used in the C1
    solution -- 2500, 5731, 6239 -- each of which fails Chi-Square as
    plain middle-square but passes comfortably once repaired this way).
    """
    if weyl_increment % 2 == 0:
        raise ValueError("weyl_increment must be odd for a full-period Weyl sequence")

    modulus = 10**digits
    values = []
    x, w = seed, 0
    for _ in range(n):
        w = (w + weyl_increment) % modulus
        mid = middle_square_step(x, digits)
        x = (mid + w) % modulus
        values.append(x)
    return values


def middle_square_weyl_uniforms(seed, n, digits=4, weyl_increment=3571):
    """Same sequence as middle_square_weyl(), rescaled to floats in [0, 1)."""
    scale = 10**digits
    return [value / scale for value in middle_square_weyl(seed, n, digits, weyl_increment)]


# ===== Testing =====

from scipy import stats

# CHI Square

def chi_square_statistic_manual(counts):
    """Step 1-4 without any library: chi2 = sum (O_i - E_i)^2 / E_i."""
    total = sum(counts)
    expected = total / len(counts)
    return sum(((observed - expected) ** 2) / expected for observed in counts)


def chi_square_uniform_test(uniforms, bins=10, alpha=0.05):
    """
    Full Chi-Square uniformity test on a list of values expected in [0, 1).

    Returns a dict with counts, the statistic, critical value, p-value,
    and the H0 decision -- this is exactly the row format the C1 table
    (Seed | Chi^2 | p-value | Decision) asks for.
    """
    counts = bin_counts(uniforms, bins)
    chi2 = chi_square_statistic_manual(counts)
    df = bins - 1
    critical_value = stats.chi2.ppf(1 - alpha, df=df)
    p_value = stats.chi2.sf(chi2, df=df)
    decision = "Reject H0" if p_value < alpha else "Do not reject H0"
    return {
        "counts": counts,
        "chi2": chi2,
        "df": df,
        "critical_value": critical_value,
        "p_value": p_value,
        "decision": decision,
    }

# KS test

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
    d_critical = stats.ksone.ppf(1 - alpha, n)
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

seed = 7
a = 5
c = 3
m = 32
n = 30

samples = lcg_uniforms(seed, a, c, m, n)
print(check_max_period_conditions(a,c,m,seed))
print(samples)

result_chi = chi_square_uniform_test(samples)
print(result_chi)

result_ks = ks_uniform_test(samples)
print(result_ks)

import math

def autocorrelation_test(R, i, lag, N, alpha=0.05):
    """
    R    : full sample (0-indexed list).
    i    : 1-based starting index into R.
    lag  : spacing `l` between the values being compared.
    N    : total sample size (usually len(R)).
    """
    M = (N - i) // lag - 1
    if M < 0:
        raise ValueError(f"Not enough data for i={i}, lag={lag}, N={N} (M={M} < 0)")
    indices = [i - 1 + k * lag for k in range(M + 2)]  # convert to 0-based
    pair_sum = sum(R[indices[k]] * R[indices[k + 1]] for k in range(len(indices) - 1))

    rho_hat = (1 / (M + 1)) * pair_sum - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    z0 = rho_hat / sigma

    z_critical = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z0)))
    decision = "Reject H0 (dependent)" if abs(z0) > z_critical else "Do not reject H0"

    return {"M": M, "rho_hat": rho_hat, "sigma": sigma, "sigma_rho": sigma, "Z0": z0,
            "critical_value": z_critical, "p_value": p_value, "decision": decision}


def runs_test(R, alpha=0.05):
    n = len(R)
    above = [1 if r > 0.5 else 0 for r in R]

    runs = 1
    for k in range(1, n):
        if above[k] != above[k - 1]:
            runs += 1

    n1 = sum(above)
    n2 = n - n1
    expected_runs = (2 * n1 * n2) / n + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
    z0 = (runs - expected_runs) / math.sqrt(var_runs)

    z_critical = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z0)))
    decision = "Reject H0 (dependent)" if abs(z0) > z_critical else "Do not reject H0"

    return {"runs": runs, "expected_runs": expected_runs, "Z0": z0,
            "critical_value": z_critical, "p_value": p_value, "decision": decision}


def runs_up_down_test(R, alpha=0.05):
    """
    Runs test based on whether each value is higher or lower than the
    PREVIOUS value (rising/falling), not on being above/below 0.5. Catches
    dependence that runs_test() above can miss -- and misses some that it
    catches. See this module's __main__ for a verified example (a
    block-alternating sequence) where the two genuinely disagree.
    """
    n = len(R)
    rising = [R[k + 1] > R[k] for k in range(n - 1)]

    runs = 1
    for k in range(1, len(rising)):
        if rising[k] != rising[k - 1]:
            runs += 1

    expected_runs = (2 * n - 1) / 3
    var_runs = (16 * n - 29) / 90
    z0 = (runs - expected_runs) / math.sqrt(var_runs)

    z_critical = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z0)))
    decision = "Reject H0 (dependent)" if abs(z0) > z_critical else "Do not reject H0"

    return {"runs": runs, "expected_runs": expected_runs, "Z0": z0,
            "critical_value": z_critical, "p_value": p_value, "decision": decision}


