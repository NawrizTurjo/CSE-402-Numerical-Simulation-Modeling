"""Linear Congruential Generator (LCG) using user input."""

values = input("Enter seed, a, c, m, n: ").replace(",", " ").split()
seed, a, c, m, n = map(int, values)

random_numbers = []
x = seed

for i in range(n):
    x = (a * x + c) % m
    random_numbers.append(x)

print("LCG values:", random_numbers)
print("Values from 0 to 1:", [round(x / m, 4) for x in random_numbers])
