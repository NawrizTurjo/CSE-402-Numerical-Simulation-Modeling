"""
Practice 6 (Res/Src-26/practice-problems.md): Buffon's Needle.

This is the IDENTICAL problem to the A1 online-exam question (L=1.0,
spacing D=2.0, n=100000) -- proof that studying solutions/a1_buffons_needle.py
once covers both.

Run standalone:
    python -m solutions.practice_src26.p6_buffons_needle
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from solutions.a1_buffons_needle import buffon_needle_pi

if __name__ == "__main__":
    result = buffon_needle_pi(n_drops=100000, seed=1)
    print(f"P(needle crosses a line) = {result['p_hat']:.5f}")
    print(f"pi estimate              = {result['pi_estimate']:.6f}")
    print(f"Absolute error           = {result['abs_error']:.6f}")
