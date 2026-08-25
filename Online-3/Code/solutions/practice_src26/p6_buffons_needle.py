"""
Practice 6 (Res/Src-26/practice-problems.md): Buffon's Needle.

This is the IDENTICAL problem to the A1 online-exam question (same L=0.5,
plank width=1, n=100000) -- proof that studying solutions/a1_buffons_needle.py
once covers both. Nothing new is implemented here; this just re-runs it.

Run standalone:
    python -m solutions.practice_src26.p6_buffons_needle
"""

from solutions.a1_buffons_needle import buffons_needle_pi

if __name__ == "__main__":
    result = buffons_needle_pi(n=100000, seed=1)
    print(f"P(needle crosses a line) = {result['p_cross']:.4f}")
    print(f"pi estimate              = {result['pi_estimate']:.4f}")
