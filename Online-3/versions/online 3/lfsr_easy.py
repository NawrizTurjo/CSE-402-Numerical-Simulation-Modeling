"""Easy 4-bit Linear Feedback Shift Register (LFSR)."""

state = 0b1011
numbers = []

for i in range(15):
    # XOR the last two selected bits to make a feedback bit.
    feedback = ((state >> 0) ^ (state >> 1)) & 1
    state = (state >> 1) | (feedback << 3)
    numbers.append(state)

print("LFSR values:")
print(numbers)
