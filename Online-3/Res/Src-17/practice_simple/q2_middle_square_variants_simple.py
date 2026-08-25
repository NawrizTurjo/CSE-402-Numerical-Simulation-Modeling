"""
Practice Q2: Middle-Square Variants (d-Digit Scaling & Weyl Sequence Repair)
Simple, Student-Friendly Implementation
"""

from scipy.stats import chisquare

# ==============================================================================
# Task 1: Generalized d-Digit Middle-Square
# ==============================================================================
def middle_square_d(seed, n, d=4):
    """
    Generates n values from a d-digit seed using the Middle-Square method.
    """
    results = []
    x = seed
    for _ in range(n):
        # 1. Square current number
        sq = x * x
        # 2. Pad to 2*d digits
        sq_str = str(sq).zfill(2 * d)
        # 3. Extract middle d digits
        start_idx = d // 2
        middle_str = sq_str[start_idx : start_idx + d]
        # 4. Convert to int
        x = int(middle_str)
        results.append(x / (10 ** d))
    return results


# ==============================================================================
# Task 2: Seed Survey (Why scaling digits doesn't fix it)
# ==============================================================================
def get_run_info(seed, d):
    seen = {}
    x = seed
    step = 0
    while x not in seen:
        seen[x] = step
        sq_str = str(x * x).zfill(2 * d)
        x = int(sq_str[d // 2 : d // 2 + d])
        step += 1
    return {"tail": seen[x], "cycle_length": step - seen[x], "collapses_to_zero": (x == 0)}


def survey_seeds(d, sample_step=1):
    lo = 10 ** (d - 1)
    hi = 10 ** d
    tails = []
    zero_count = 0
    max_tail = 0
    max_seed = 0
    
    for s in range(lo, hi, sample_step):
        info = get_run_info(s, d)
        tails.append(info["tail"])
        if info["collapses_to_zero"]:
            zero_count += 1
        if info["tail"] > max_tail:
            max_tail = info["tail"]
            max_seed = s
            
    return {
        "d": d,
        "total_seeds": len(tails),
        "avg_run": sum(tails) / len(tails),
        "max_run": max_tail,
        "max_seed": max_seed,
        "zero_percent": (zero_count / len(tails)) * 100
    }


# ==============================================================================
# Task 3: Middle-Square Weyl Sequence Repair
# ==============================================================================
def middle_square_weyl(seed, n, d=4, s=3571):
    """
    Middle-Square with Weyl sequence:
        w_{i+1} = (w_i + s) mod 10^d   (s is odd)
        x_{i+1} = (middle(x_i^2) + w_{i+1}) mod 10^d
    """
    modulus = 10 ** d
    results = []
    x = seed
    w = 0
    
    for _ in range(n):
        w = (w + s) % modulus
        # Middle square of x
        sq_str = str(x * x).zfill(2 * d)
        mid_val = int(sq_str[d // 2 : d // 2 + d])
        x = (mid_val + w) % modulus
        results.append(x / modulus)
        
    return results


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TASK 1: d-Digit Middle Square Demo")
    print("=" * 70)
    for d, s in ((2, 57), (4, 5731), (6, 675248)):
        sample = middle_square_d(s, n=5, d=d)
        print(f"d = {d}, seed = {s} -> First 5 numbers: {[round(v, 4) for v in sample]}")

    print("\n" + "=" * 70)
    print("TASK 2: Survey of d-Digit Seeds (2, 4, 6 digits)")
    print("=" * 70)
    print(f"{'d':<4} | {'Seeds Checked':<14} | {'Avg Run':<10} | {'Max Run':<10} | {'Dies at 0000'}")
    print("-" * 65)
    for d, step in ((2, 1), (4, 1), (6, 10)):  # sample step 10 for d=6 speed
        surv = survey_seeds(d, step)
        print(f"{surv['d']:<4} | {surv['total_seeds']:<14} | {surv['avg_run']:<10.1f} | {surv['max_run']:<10} | {surv['zero_percent']:.1f}%")

    print("\n" + "=" * 70)
    print("TASK 3: Fixing with Middle-Square Weyl Sequence")
    print("=" * 70)
    # Seed 3792 failed completely in plain middle square (Chi^2 = 9000)
    weyl_sample_3792 = middle_square_weyl(seed=3792, n=1000, d=4, s=3571)
    
    # Test uniformity with Chi-Square
    observed = [0] * 10
    for r in weyl_sample_3792:
        observed[min(int(r * 10), 9)] += 1
    chi2, p_val = chisquare(observed)
    decision = "Reject H0" if p_val < 0.05 else "Do not reject H0"
    
    print(f"Seed 3792 with Weyl Repair (N = 1000):")
    print(f"  Observed bin counts: {observed}")
    print(f"  Chi^2 = {chi2:.2f}, p-value = {p_val:.4f} -> Decision: {decision}")

    print("\n" + "=" * 70)
    print("TASK 4: Reflection Summary")
    print("=" * 70)
    print("1. Why increasing d cannot fix plain Middle Square:")
    print("   The state is only a single d-digit number (10^d possible states).")
    print("   Because squaring is not a 1-to-1 bijection, points quickly funnel into short cycles")
    print("   (bounded by the birthday bound ~sqrt(10^d)), and 0 is an absorbing trap.")
    print("2. Why Weyl Sequence fixes it:")
    print("   Adding an odd constant s gives a full-period Weyl sequence (period 10^d).")
    print("   The new state is the pair (x, w), meaning no state can get stuck at 0.")
