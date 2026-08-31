"""Mersenne Twister using Python's random module."""

import random

generator = random.Random(12345)  # fixed seed

numbers = []
for i in range(10):
    numbers.append(generator.random())

print("Mersenne Twister values:")
print(numbers)
