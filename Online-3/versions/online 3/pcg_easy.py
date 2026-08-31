"""Small PCG random-number generator."""

state = 42
mask = (1 << 64) - 1
numbers = []

for i in range(10):
    old_state = state
    state = (old_state * 6364136223846793005 + 1442695040888963407) & mask

    x = ((old_state >> 18) ^ old_state) >> 27
    rotation = old_state >> 59
    value = ((x >> rotation) | (x << ((-rotation) & 31))) & 0xFFFFFFFF
    numbers.append(value / 2**32)

print("PCG values:")
print(numbers)
