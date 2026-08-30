"""
Practice R2: RANDU -- The Generator That Fooled Everyone
(see ../../../Practice/PRACTICE_QUESTIONS.md).

Run standalone:
    python -m solutions.practice_generated.r2_randu_independence
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from rng.lcg import lcg, lcg_uniforms
from testing.chi_square_test import chi_square_uniform_test
from testing.independence_test import autocorrelation_test

SEED, A, C, M = 1, 65539, 0, 2**31


def task1_generate():
    values = lcg(SEED, A, C, M, 20)
    print("TASK 1: first 20 RANDU values (a=65539, c=0, m=2^31, seed=1)")
    print(values)
    print()


def task2_chi_square():
    uniforms = lcg_uniforms(SEED, A, C, M, 1000)
    result = chi_square_uniform_test(uniforms)
    print("TASK 2: Chi-Square uniformity test, N=1000, 10 bins")
    print(f"  chi2={result['chi2']:.4f}  p={result['p_value']:.6g}  -> {result['decision']}")
    print()
    return uniforms


def task3_autocorrelation(uniforms):
    result = autocorrelation_test(uniforms, i=1, lag=1, N=len(uniforms))
    print("TASK 3: Autocorrelation test, lag=1")
    print(f"  rho_hat={result['rho_hat']:.4f}  Z0={result['Z0']:.4f}  -> {result['decision']}")
    print()


def task4_hyperplane_check():
    print("TASK 4: Verify the hyperplane identity 9*X_i - 6*X_{i+1} + X_{i+2} == 0 (mod m)")
    X = lcg(SEED, A, C, M, 10)
    all_zero = True
    for i in range(len(X) - 2):
        residual = (9 * X[i] - 6 * X[i + 1] + X[i + 2]) % M
        print(f"  triple starting at index {i}: residual = {residual}")
        all_zero = all_zero and residual == 0
    print(f"  All residuals zero: {all_zero}")
    print()


def task5_reflect():
    print("TASK 5: Reflection")
    print(
        "RANDU passes BOTH the Chi-Square uniformity test AND the lag-1\n"
        "autocorrelation independence test -- by every check used so far, it\n"
        "looks like a fine generator. Yet Task 4 shows every consecutive TRIPLE\n"
        "(X_i, X_{i+1}, X_{i+2}) satisfies an exact linear identity mod m, which\n"
        "means all triples fall on just 15 parallel planes instead of filling\n"
        "3-D space uniformly. Neither the 1-D Chi-Square test nor the pairwise\n"
        "(lag-1) autocorrelation test can see this -- it only shows up once you\n"
        "look at 3 (or more) values at a time. Lesson: passing the standard 1-D/\n"
        "pairwise battery of tests is necessary but NOT sufficient; higher-\n"
        "dimensional structure can still be badly broken. (RANDU was IBM's\n"
        "default generator through the 1960s-70s and caused real, silently\n"
        "wrong simulation results before this defect was widely understood.)"
    )


if __name__ == "__main__":
    task1_generate()
    uniforms = task2_chi_square()
    task3_autocorrelation(uniforms)
    task4_hyperplane_check()
    task5_reflect()
