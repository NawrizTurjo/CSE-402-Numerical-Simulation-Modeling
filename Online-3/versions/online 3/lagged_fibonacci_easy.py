"""Easy additive Lagged Fibonacci Generator."""

values = [12, 25, 33, 47, 58, 64, 71]
j = 3
k = 7
m = 100

for i in range(10):
    new_value = (values[-j] + values[-k]) % m
    values.append(new_value)

generated = values[7:]

print("Lagged Fibonacci values:")
print(generated)
print("Values from 0 to 1:")
print([x / m for x in generated])
