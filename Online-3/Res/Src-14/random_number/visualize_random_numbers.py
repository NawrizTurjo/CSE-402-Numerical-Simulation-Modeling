import random


def python_random_generator(seed, n):
    random.seed(seed)
    return [random.random() for _ in range(n)]


def middle_square(seed, n):
    values = []
    x = seed
    for _ in range(n):
        x = int(str(x * x).zfill(8)[2:6])
        values.append(x / 10000)
    return values


def visualize_random_numbers(numbers, title="Random Numbers", bins=10):
    """
    Visualize random numbers in three ways:
    1. Sequence plot: value against generation order.
    2. Histogram: frequency in equal bins.
    3. Scatter plot: R_i against R_(i+1), useful for spotting patterns.
    """
    import matplotlib.pyplot as plt

    x_axis = list(range(1, len(numbers) + 1))

    plt.figure(figsize=(12, 4))
    plt.plot(x_axis, numbers, marker="o", markersize=2, linewidth=1)
    plt.title(title + " - Sequence Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Random number")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(numbers, bins=bins, range=(0, 1), edgecolor="black")
    plt.title(title + " - Histogram")
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(numbers[:-1], numbers[1:], s=10)
    plt.title(title + " - Consecutive Pair Scatter")
    plt.xlabel("R_i")
    plt.ylabel("R_(i+1)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def compare_python_and_middle_square(seed=5731, n=1000):
    python_numbers = python_random_generator(seed, n)
    middle_square_numbers = middle_square(seed, n)

    visualize_random_numbers(
        python_numbers,
        title=f"Python random.random(), seed={seed}, n={n}",
    )
    visualize_random_numbers(
        middle_square_numbers,
        title=f"Middle-Square, seed={seed}, n={n}",
    )


if __name__ == "__main__":
    compare_python_and_middle_square(seed=5731, n=1000)
