"""Easy serial test and runs test for random numbers from 0 to 1."""

import math

text = input("Enter random numbers: ").replace(",", " ")
numbers = list(map(float, text.split()))


# ---------------- SERIAL TEST ----------------
# Make non-overlapping pairs: (R1, R2), (R3, R4), ...
pairs = list(zip(numbers[::2], numbers[1::2]))
counts = [0, 0, 0, 0]

for x, y in pairs:
    position = (x >= 0.5) * 2 + (y >= 0.5)
    counts[position] += 1

expected = len(pairs) / 4
chi_square = sum((count - expected) ** 2 / expected for count in counts)

print("\nSERIAL TEST")
print("Frequencies:", counts)
print("Chi-square:", round(chi_square, 4))

if chi_square < 7.815:  # alpha = 0.05 and df = 3
    print("Result: Passed")
else:
    print("Result: Failed")


# ---------------- RUNS TEST ----------------
# A means below 0.5 and B means 0.5 or above.
symbols = ["A" if number < 0.5 else "B" for number in numbers]
runs = 1

for i in range(1, len(symbols)):
    if symbols[i] != symbols[i - 1]:
        runs += 1

n1 = symbols.count("A")
n2 = symbols.count("B")
mean = (2 * n1 * n2) / (n1 + n2) + 1
variance = (
    2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
    / ((n1 + n2) ** 2 * (n1 + n2 - 1))
)
z = (runs - mean) / math.sqrt(variance)

print("\nRUNS TEST")
print("Sequence:", "".join(symbols))
print("Number of runs:", runs)
print("Z value:", round(z, 4))

if abs(z) < 1.96:  # alpha = 0.05
    print("Result: Passed")
else:
    print("Result: Failed")
