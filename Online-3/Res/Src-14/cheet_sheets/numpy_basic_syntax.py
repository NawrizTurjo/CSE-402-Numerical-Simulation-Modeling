"""
NumPy basic syntax cheat sheet.

Run:
    python cheet_sheets/numpy_basic_syntax.py
"""

import numpy as np


def creating_arrays():
    a = np.array([1, 2, 3, 4])
    zeros = np.zeros(5)
    ones = np.ones(5)
    sequence = np.arange(1, 10, 2)
    equally_spaced = np.linspace(0, 1, 6)

    print("Creating arrays")
    print("array:", a)
    print("zeros:", zeros)
    print("ones:", ones)
    print("arange(1, 10, 2):", sequence)
    print("linspace(0, 1, 6):", equally_spaced)


def array_properties():
    a = np.array([[1, 2, 3], [4, 5, 6]])

    print("\nArray properties")
    print("array:\n", a)
    print("shape:", a.shape)
    print("number of dimensions:", a.ndim)
    print("size:", a.size)
    print("data type:", a.dtype)


def indexing_and_slicing():
    a = np.array([10, 20, 30, 40, 50])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    print("\nIndexing and slicing")
    print("a:", a)
    print("a[0]:", a[0])
    print("a[-1]:", a[-1])
    print("a[1:4]:", a[1:4])

    print("matrix:\n", matrix)
    print("matrix[0, 1]:", matrix[0, 1])
    print("first row:", matrix[0, :])
    print("second column:", matrix[:, 1])


def vector_operations():
    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])

    print("\nVector operations")
    print("a:", a)
    print("b:", b)
    print("a + b:", a + b)
    print("b - a:", b - a)
    print("a * b:", a * b)
    print("b / a:", b / a)
    print("a ** 2:", a ** 2)
    print("a + 5:", a + 5)


def statistics_basics():
    a = np.array([4, 7, 1, 9, 2])

    print("\nStatistics")
    print("a:", a)
    print("mean:", np.mean(a))
    print("median:", np.median(a))
    print("variance:", np.var(a))
    print("standard deviation:", np.std(a))
    print("min:", np.min(a))
    print("max:", np.max(a))
    print("sum:", np.sum(a))


def random_numbers():
    np.random.seed(1)

    print("\nRandom numbers")
    print("one U[0,1):", np.random.random())
    print("five U[0,1):", np.random.random(5))
    print("uniform 5 to 10:", np.random.uniform(5, 10, 5))
    print("integers 1 to 6:", np.random.randint(1, 7, 5))
    print("normal mean 0 std 1:", np.random.normal(0, 1, 5))


def reshaping():
    a = np.arange(1, 13)
    matrix = a.reshape(3, 4)

    print("\nReshaping")
    print("original:", a)
    print("reshape 3x4:\n", matrix)
    print("flatten:", matrix.flatten())


def boolean_filtering():
    a = np.array([10, 25, 30, 45, 50])

    print("\nBoolean filtering")
    print("a:", a)
    print("a > 30:", a > 30)
    print("values greater than 30:", a[a > 30])
    print("values divisible by 10:", a[a % 10 == 0])


def matrix_operations():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("\nMatrix operations")
    print("A:\n", A)
    print("B:\n", B)
    print("A + B:\n", A + B)
    print("element-wise A * B:\n", A * B)
    print("matrix multiplication A @ B:\n", A @ B)
    print("transpose A.T:\n", A.T)


def useful_for_simulation():
    values = np.array([0.23, 0.81, 0.44, 0.05, 0.93])
    bins = 10
    counts, edges = np.histogram(values, bins=bins, range=(0, 1))

    print("\nUseful for simulation/random-number tests")
    print("values:", values)
    print("bin counts:", counts)
    print("bin edges:", edges)
    print("sorted values:", np.sort(values))
    print("cumulative sum:", np.cumsum(values))


def main():
    creating_arrays()
    array_properties()
    indexing_and_slicing()
    vector_operations()
    statistics_basics()
    random_numbers()
    reshaping()
    boolean_filtering()
    matrix_operations()
    useful_for_simulation()


if __name__ == "__main__":
    main()
