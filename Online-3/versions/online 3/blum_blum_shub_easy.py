"""Easy Blum Blum Shub random-number generator."""

p = 11
q = 19
m = p * q
x = 3  # seed must not share a factor with m
bits = []

for i in range(20):
    x = (x * x) % m
    bits.append(x % 2)

print("Blum Blum Shub bits:")
print(bits)
