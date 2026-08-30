"""
NumPy syntax cheat sheet -- pure syntax reference, no simulation logic.
Everything in Code/ itself is written in plain Python (no numpy
dependency) so it works with nothing but the standard library, but numpy
is handy for quick exploratory work at the terminal during the exam.

Run standalone (prints every section):
    python -m cheatsheets.numpy_syntax
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np


def creating_arrays():
    print("Creating arrays")
    print("np.array([1,2,3,4])   :", np.array([1, 2, 3, 4]))
    print("np.zeros(5)           :", np.zeros(5))
    print("np.arange(1, 10, 2)   :", np.arange(1, 10, 2))
    print("np.linspace(0, 1, 6)  :", np.linspace(0, 1, 6))  # 6 equally-spaced points, endpoints included


def array_properties():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print("\nArray properties")
    print("shape:", a.shape, " ndim:", a.ndim, " size:", a.size, " dtype:", a.dtype)


def indexing_and_slicing():
    a = np.array([10, 20, 30, 40, 50])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    print("\nIndexing and slicing")
    print("a[1:4]:", a[1:4])
    print("matrix[0, 1]:", matrix[0, 1], " first row:", matrix[0, :], " second col:", matrix[:, 1])


def vector_operations():
    a, b = np.array([1, 2, 3]), np.array([10, 20, 30])
    print("\nVector operations (all element-wise, no loop needed)")
    print("a + b:", a + b, " a * b:", a * b, " a ** 2:", a ** 2, " a + 5:", a + 5)


def statistics_basics():
    a = np.array([4, 7, 1, 9, 2])
    print("\nStatistics")
    print(f"mean={np.mean(a)}  median={np.median(a)}  var={np.var(a)}  std={np.std(a)}")


def random_numbers():
    """np.random is a DIFFERENT generator from Python's random module --
    seeding one does not seed the other."""
    rng = np.random.default_rng(seed=1)   # modern API (preferred over np.random.seed(...))
    print("\nRandom numbers (np.random.default_rng, modern API)")
    print("5x U[0,1):", rng.random(5))
    print("5x uniform(5,10):", rng.uniform(5, 10, 5))
    print("5x integers in [1,7):", rng.integers(1, 7, 5))
    print("5x normal(mean=0,std=1):", rng.normal(0, 1, 5))


def boolean_filtering():
    a = np.array([10, 25, 30, 45, 50])
    print("\nBoolean filtering (no explicit loop needed)")
    print("a > 30:", a > 30)
    print("a[a > 30]:", a[a > 30])


def useful_for_this_course():
    """The two numpy calls that would actually save typing in this
    course's problems: histogram binning (chi_square_test.py's manual
    bin_counts loop, done in one call) and cumulative sum (building
    arrival times from interarrival times, simulation/single_server_queue.py's
    manual running-total loop, done in one call)."""
    values = np.array([0.23, 0.81, 0.44, 0.05, 0.93])
    interarrival = np.array([0.4, 1.2, 0.5, 1.7, 0.2])

    print("\nUseful for this course")
    counts, edges = np.histogram(values, bins=10, range=(0, 1))
    print("bin counts (replaces the manual binning loop):", counts)
    print("cumulative sum (replaces the manual running-total loop):", np.cumsum(interarrival))
    print("  -> that cumsum IS the arrival_times list simulate_ssq() builds internally")


def main():
    creating_arrays()
    array_properties()
    indexing_and_slicing()
    vector_operations()
    statistics_basics()
    random_numbers()
    boolean_filtering()
    useful_for_this_course()


if __name__ == "__main__":
    main()
