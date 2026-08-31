"""Check the three maximum-period conditions for congruential generators."""

import math


def is_power_of_two(number):
    """Return True for 2, 4, 8, 16, 32, and so on."""
    while number > 1 and number % 2 == 0:
        number = number // 2

    return number == 1


def is_prime(number):
    """Return True if number is prime."""
    if number < 2:
        return False

    for divisor in range(2, number):
        if number % divisor == 0:
            return False

    return True


def generate_lcg(seed, a, c, m, n):
    """Generate n integer values using an LCG."""
    values = []
    x = seed

    for i in range(n):
        x = (a * x + c) % m
        values.append(x)

    return values


def find_actual_period(seed, a, c, m):
    """Generate values until one repeats and return the cycle length."""
    generated = [seed]
    x = seed

    while True:
        x = (a * x + c) % m

        if x in generated:
            first_position = generated.index(x)
            return len(generated) - first_position

        generated.append(x)


def smallest_power_equal_to_one(a, m):
    """Find the smallest k for which a^k mod m equals 1."""
    value = 1

    for k in range(1, m):
        value = (value * a) % m

        if value == 1:
            return k

    return None


def check_maximum_period(seed, a, c, m):
    print("\nParameters: X0 =", seed, ", a =", a, ", c =", c, ", m =", m)

    # CASE 1: m = 2^b and c is not zero.
    if is_power_of_two(m) and c != 0:
        relatively_prime = math.gcd(c, m) == 1
        correct_a = (a - 1) % 4 == 0

        print("Case 1: Mixed congruential generator")
        print("c and m are relatively prime:", relatively_prime)
        print("a has the form 1 + 4k:", correct_a)

        if relatively_prime and correct_a:
            print("Maximum-period conditions are satisfied.")
            print("Expected maximum period:", m)
        else:
            print("Maximum-period conditions are not satisfied.")

    # CASE 2: m = 2^b and c is zero.
    elif is_power_of_two(m) and c == 0:
        odd_seed = seed % 2 == 1
        correct_a = a % 8 == 3 or a % 8 == 5

        print("Case 2: Multiplicative congruential generator")
        print("X0 is odd:", odd_seed)
        print("a has the form 3 + 8k or 5 + 8k:", correct_a)

        if odd_seed and correct_a:
            print("Maximum-period conditions are satisfied.")
            print("Expected maximum period:", m // 4)
        else:
            print("Maximum-period conditions are not satisfied.")

    # CASE 3: m is prime and c is zero.
    elif is_prime(m) and c == 0:
        smallest_k = smallest_power_equal_to_one(a, m)

        print("Case 3: Prime modulus multiplicative generator")
        print("Smallest k for which a^k mod m = 1:", smallest_k)

        if seed != 0 and smallest_k == m - 1:
            print("Maximum-period conditions are satisfied.")
            print("Expected maximum period:", m - 1)
        else:
            print("Maximum-period conditions are not satisfied.")

    else:
        print("The parameters do not belong to any of the three cases.")

    print("Actual period:", find_actual_period(seed, a, c, m))
    print("First values:", generate_lcg(seed, a, c, m, 10))


print("CASE 1 EXAMPLE")
# gcd(3, 16) = 1 and 5 = 1 + 4(1), so maximum period = 16.
check_maximum_period(seed=7, a=5, c=3, m=16)

print("\nCASE 2 EXAMPLE")
# X0 is odd and 5 = 5 + 8(0), so maximum period = 16/4 = 4.
check_maximum_period(seed=7, a=5, c=0, m=16)

print("\nCASE 3 EXAMPLE")
# 3 is a primitive root modulo the prime number 7, so period = 7 - 1 = 6.
check_maximum_period(seed=1, a=3, c=0, m=7)
