"""
CSE 401: Random Number Generation (Simple Implementation)
Linear Congruential Generator (LCG) & Maximum Period Rules
"""

from math import gcd

# ==============================================================================
# 1. Linear Congruential Generator (LCG)
# ==============================================================================
def lcg_step(x, a, c, m):
    """
    Computes one step of LCG:
        X_{i+1} = (a * X_i + c) mod m
    """
    return (a * x + c) % m


def generate_lcg_integers(seed, a, c, m, n):
    """
    Generates n random integers X_1, ..., X_n.
    """
    values = []
    x = seed
    for _ in range(n):
        x = lcg_step(x, a, c, m)
        values.append(x)
    return values


def generate_lcg_uniforms(seed, a, c, m, n):
    """
    Generates n uniform random numbers R_i = X_i / m in [0, 1).
    """
    integers = generate_lcg_integers(seed, a, c, m, n)
    return [x / m for x in integers]


# ==============================================================================
# 2. Period Detection
# ==============================================================================
def find_lcg_period(seed, a, c, m):
    """
    Finds the period (cycle length) of an LCG starting from a given seed.
    """
    seen = {}
    x = seed
    step = 0
    while x not in seen:
        seen[x] = step
        x = lcg_step(x, a, c, m)
        step += 1
    
    first_seen_step = seen[x]
    cycle_length = step - first_seen_step
    return cycle_length


# ==============================================================================
# 3. Checking Maximum Period Conditions (From Slides)
# ==============================================================================
def check_max_period_conditions(a, c, m, seed):
    """
    Checks the 3 theoretical cases for maximum period:
      Case 1: m = 2^b and c != 0 (Mixed LCG) -> Max Period = m
              Conditions: gcd(c, m) == 1 and a = 1 + 4k (i.e. (a-1)%4 == 0)
      Case 2: m = 2^b and c == 0 (Multiplicative LCG) -> Max Period = m / 4
              Conditions: seed is odd and a = 3 + 8k or 5 + 8k
      Case 3: m is prime and c == 0 -> Max Period = m - 1
              Conditions: a is a primitive root modulo m
    """
    is_power_of_two = (m > 0) and (m & (m - 1)) == 0

    if is_power_of_two and c != 0:
        cond1 = (gcd(c, m) == 1)
        cond2 = ((a - 1) % 4 == 0)
        satisfied = cond1 and cond2
        return "Case 1: m=2^b, c!=0", m, satisfied, f"gcd(c,m)=1: {cond1}, a=1+4k: {cond2}"

    elif is_power_of_two and c == 0:
        cond1 = (seed % 2 == 1)
        cond2 = (a % 8 == 3 or a % 8 == 5)
        satisfied = cond1 and cond2
        return "Case 2: m=2^b, c=0", m // 4, satisfied, f"X0 is odd: {cond1}, a=3+8k or 5+8k: {cond2}"

    else:
        return "General / Prime Case", None, None, "Check specific theoretical conditions"


# ==============================================================================
# Main: Slide Examples Demonstration
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Mixed LCG (X0 = 27, a = 17, c = 43, m = 100)")
    print("=" * 70)
    x = 27
    for i in range(1, 5):
        next_x = lcg_step(x, a=17, c=43, m=100)
        print(f"X{i} = (17 * {x} + 43) mod 100 = {next_x:2d}  ->  R{i} = {next_x / 100:.2f}")
        x = next_x
    print(f"Measured Period = {find_lcg_period(27, 17, 43, 100)}")

    print("\n" + "=" * 70)
    print("Example 2: Multiplicative LCG (X_{i+1} = 13 * X_i mod 64)")
    print("=" * 70)
    for s in (1, 2, 3, 4):
        seq = generate_lcg_integers(s, a=13, c=0, m=64, n=8)
        p = find_lcg_period(s, a=13, c=0, m=64)
        print(f"Seed {s:2d} -> Period: {p:2d} | Sequence: {seq}")
    
    case, max_p, ok, details = check_max_period_conditions(13, 0, 64, seed=1)
    print(f"\nCondition Check for Seed 1: {case} -> Max Period = {max_p} (Satisfied: {ok})")
    print(f"Details: {details}")

    print("\n" + "=" * 70)
    print("Example 4: Standard Lehmer Generator (a = 16807, c = 0, m = 2^31 - 1)")
    print("=" * 70)
    lehmer_m = 2**31 - 1
    lehmer_a = 16807
    lehmer_sample = generate_lcg_integers(seed=123457, a=lehmer_a, c=0, m=lehmer_m, n=3)
    for i, val in enumerate(lehmer_sample, 1):
        print(f"X{i} = {val:10d}  ->  R{i} = {val / lehmer_m:.6f}")
    print(f"Theoretical Maximum Period = m - 1 = {lehmer_m - 1}")
