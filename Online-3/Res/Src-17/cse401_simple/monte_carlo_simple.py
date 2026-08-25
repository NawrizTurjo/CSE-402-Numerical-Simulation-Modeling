"""
CSE 401: Monte Carlo Simulation (Simple Implementation)
Includes:
  1. Monte Carlo Integration (Sample-Mean Method)
  2. Hit-or-Miss Estimation of Pi
  3. Error & Confidence Interval Analysis
"""

import math

# Simple standard LCG (Lehmer Generator)
def simple_lcg(seed=123457):
    x = seed
    a = 16807
    m = 2**31 - 1
    while True:
        x = (a * x) % m
        yield x / m

# ==============================================================================
# 1. Monte Carlo Integration (Sample-Mean Estimator)
# ==============================================================================
def monte_carlo_integrate(func, a, b, n, seed=123457):
    """
    Estimates the integral I = integral_a^b g(x) dx using n random samples:
        I ~= (b - a) * (1 / n) * sum( g(X_i) ), where X_i ~ Uniform(a, b)
    """
    rng = simple_lcg(seed)
    width = b - a
    
    total = 0.0
    total_sq = 0.0
    
    for _ in range(n):
        r = next(rng)
        x = a + width * r
        y = width * func(x)
        total += y
        total_sq += y * y
        
    mean = total / n
    variance = (total_sq - n * (mean ** 2)) / (n - 1) if n > 1 else 0.0
    std_dev = math.sqrt(max(0.0, variance))
    std_error = std_dev / math.sqrt(n)
    
    # 95% Confidence Interval: mean +/- 1.96 * SE
    ci_lower = mean - 1.96 * std_error
    ci_upper = mean + 1.96 * std_error
    
    return {
        "estimate": mean,
        "std_dev": std_dev,
        "std_error": std_error,
        "confidence_interval": (ci_lower, ci_upper)
    }


# ==============================================================================
# 2. Hit-or-Miss Estimation of Pi
# ==============================================================================
def estimate_pi(n, seed=123457):
    """
    Estimates pi by generating (X, Y) in unit square [0, 1]x[0, 1]
    and checking if X^2 + Y^2 <= 1.
        pi ~= 4 * (hits / n)
    """
    rng = simple_lcg(seed)
    hits = 0
    
    for _ in range(n):
        x = next(rng)
        y = next(rng)
        if x * x + y * y <= 1.0:
            hits += 1
            
    p_hat = hits / n
    pi_estimate = 4.0 * p_hat
    
    std_error = 4.0 * math.sqrt(p_hat * (1.0 - p_hat) / n)
    ci_lower = pi_estimate - 1.96 * std_error
    ci_upper = pi_estimate + 1.96 * std_error
    
    return {
        "estimate": pi_estimate,
        "hits": hits,
        "std_error": std_error,
        "confidence_interval": (ci_lower, ci_upper)
    }


# ==============================================================================
# Main: Verification & Demos
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("1. Monte Carlo Integration: integral_0^pi sin(x) dx (Exact = 2)")
    print("=" * 70)
    # Worked example with fixed points from slides
    slide_samples = [0.52, 1.19, 2.83, 0.21, 1.77, 2.50, 0.89, 1.53, 2.97, 2.11]
    heights = [math.sin(x) for x in slide_samples]
    avg_height = sum(heights) / len(heights)
    slide_est = math.pi * avg_height
    print(f"Slide Samples: {slide_samples}")
    print(f"Average Height = {avg_height:.3f}")
    print(f"Ybar(10) = pi * {avg_height:.3f} = {slide_est:.3f} (True = 2.0, Error = {abs(slide_est - 2):.3f})\n")

    print(f"{'n':<10} | {'Estimate':<10} | {'Error':<10} | {'95% Confidence Interval'}")
    print("-" * 60)
    for n in (10, 100, 1000, 10000, 100000):
        res = monte_carlo_integrate(math.sin, 0.0, math.pi, n)
        lo, hi = res["confidence_interval"]
        err = res["estimate"] - 2.0
        print(f"{n:<10} | {res['estimate']:<10.4f} | {err:<10.4f} | [{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 70)
    print("2. Hit-or-Miss Estimation of Pi (Exact = 3.14159...)")
    print("=" * 70)
    print(f"{'n':<10} | {'Pi Estimate':<12} | {'|Error|':<10} | {'95% Confidence Interval'}")
    print("-" * 60)
    for n in (100, 1000, 10000, 100000, 1000000):
        res_pi = estimate_pi(n)
        lo, hi = res_pi["confidence_interval"]
        err = abs(res_pi["estimate"] - math.pi)
        print(f"{n:<10} | {res_pi['estimate']:<12.5f} | {err:<10.5f} | [{lo:.5f}, {hi:.5f}]")
