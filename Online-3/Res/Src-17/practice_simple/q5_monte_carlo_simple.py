"""
Practice Q5: Monte Carlo Estimation & Error Analysis
Simple, Student-Friendly Implementation
"""

import math

def get_uniforms(n, seed=123457):
    x = seed
    a, m = 16807, 2**31 - 1
    for _ in range(n):
        x = (a * x) % m
        yield x / m


# ==============================================================================
# Task 1 & 3: Sample-Mean Monte Carlo Integration
# ==============================================================================
def mc_integrate(func, a, b, n, seed=123457):
    rng = get_uniforms(n, seed)
    width = b - a
    
    total = 0.0
    total_sq = 0.0
    for _ in range(n):
        x = a + width * next(rng)
        y = width * func(x)
        total += y
        total_sq += y * y
        
    mean = total / n
    var = (total_sq - n * (mean**2)) / (n - 1)
    std_dev = math.sqrt(max(0.0, var))
    se = std_dev / math.sqrt(n)
    
    return {
        "estimate": mean,
        "std_dev": std_dev,
        "std_error": se,
        "ci": (mean - 1.96 * se, mean + 1.96 * se)
    }


# ==============================================================================
# Task 2: Hit-or-Miss Pi Estimation & 1/sqrt(n) Law
# ==============================================================================
def mc_estimate_pi(n, seed=123457):
    rng = get_uniforms(2 * n, seed)
    hits = 0
    for _ in range(n):
        x = next(rng)
        y = next(rng)
        if x * x + y * y <= 1.0:
            hits += 1
            
    p = hits / n
    pi_est = 4.0 * p
    se = 4.0 * math.sqrt(max(0.0, p * (1 - p)) / n)
    return pi_est, se, abs(pi_est - math.pi)


# ==============================================================================
# Task 4: Simpson's Rule Comparison
# ==============================================================================
def simpson_1d(func, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = func(a) + func(b)
    for i in range(1, n):
        coef = 4 if i % 2 != 0 else 2
        total += coef * func(a + i * h)
    return total * h / 3.0


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TASKS 1 & 3: Sample-Mean Integration with 95% Confidence Intervals")
    print("=" * 70)
    
    integrals = [
        ("integral_0^pi sin(x) dx", math.sin, 0.0, math.pi, 2.0),
        ("integral_0^1 4/(1+x^2) dx", lambda x: 4.0 / (1.0 + x*x), 0.0, 1.0, math.pi),
        ("integral_0^1 e^(-x^2/2) dx", lambda x: math.exp(-x*x / 2.0), 0.0, 1.0, math.sqrt(math.pi / 2.0) * math.erf(1.0 / math.sqrt(2.0)))
    ]
    
    for label, fn, a, b, exact in integrals:
        res = mc_integrate(fn, a, b, n=100000)
        lo, hi = res["ci"]
        # Required n for half-width <= 0.001: n >= (1.96 * s / 0.001)^2
        n_needed = math.ceil((1.96 * res["std_dev"] / 0.001) ** 2)
        print(f"• {label}:")
        print(f"    Estimate: {res['estimate']:.5f} (Exact = {exact:.5f}) | Error = {res['estimate'] - exact:.5f}")
        print(f"    95% CI: [{lo:.5f}, {hi:.5f}]")
        print(f"    Samples needed for +/-0.001 half-width: n >= {n_needed:,}\n")

    print("=" * 70)
    print("TASK 2: Pi Hit-or-Miss and the 1/sqrt(n) Error Law")
    print("=" * 70)
    print(f"{'n':<10} | {'Pi Hat':<12} | {'|Error|':<10} | {'|Error| * sqrt(n)'}")
    print("-" * 55)
    for n in (100, 1000, 10000, 100000, 1000000):
        est, se, err = mc_estimate_pi(n)
        print(f"{n:<10} | {est:<12.5f} | {err:<10.5f} | {err * math.sqrt(n):.3f}")
    print("\nNote: |Error| * sqrt(n) is approximately constant (~0.5 - 1.0), confirming")
    print("that quadrupling n roughly halves the error (1/sqrt(n) convergence rate).")

    print("\n" + "=" * 70)
    print("TASK 4: Monte Carlo vs Simpson's Rule (Why Dimension Matters)")
    print("=" * 70)
    print("In 1-Dimension for integral_0^pi sin(x) dx:")
    s_err = abs(simpson_1d(math.sin, 0, math.pi, n=1000) - 2.0)
    m_err = abs(mc_integrate(math.sin, 0, math.pi, n=1000)["estimate"] - 2.0)
    print(f"  Simpson (n=1000): Error = {s_err:.2e} (O(n^-4) convergence)")
    print(f"  Monte Carlo (n=1000): Error = {m_err:.2e} (O(n^-1/2) convergence)")
    print("\nWhy ever use Monte Carlo?")
    print("  In d-dimensions, Simpson's rule requires n^d grid points (O(n^(-4/d))),")
    print("  which becomes impossible for d > 4. Monte Carlo stays O(n^-1/2) for ANY dimension!")
