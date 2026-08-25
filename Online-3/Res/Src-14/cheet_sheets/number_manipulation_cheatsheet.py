"""
Number manipulation tricks for simulation / random number generation exams.

Run:
    python3 number_manipulation_cheatsheet.py
"""


def middle_digits_by_string(number, width=8, start=2, end=6):
    """Pad with leading zeros and take digits [start:end]."""
    text = f"{number:0{width}d}"
    return text[start:end]


def left_pad_zeros(number, width):
    """Add zeros before the number: 591361 -> 00591361."""
    return f"{number:0{width}d}"


def right_pad_zeros(number, width):
    """Add zeros after the number: 591361 -> 59136100."""
    return str(number).ljust(width, "0")


def middle_digits_by_arithmetic(number, width=8, middle_width=4):
    """Extract middle digits without converting to string."""
    remove_right = (width - middle_width) // 2
    divisor = 10**remove_right
    modulus = 10**middle_width
    return (number // divisor) % modulus


def split_digits(number, width=None):
    """Return digits as a list. width pads with leading zeros if provided."""
    if width is None:
        return [int(ch) for ch in str(number)]
    return [int(ch) for ch in f"{number:0{width}d}"]


def join_digits(digits):
    """Turn [5, 9, 1, 3] into 5913."""
    return int("".join(str(digit) for digit in digits))


def reverse_number(number):
    return int(str(number)[::-1])


def count_digits(number):
    return len(str(abs(number)))


def last_k_digits(number, k):
    return number % (10**k)


def first_k_digits(number, k):
    text = str(abs(number))
    return int(text[:k])


def remove_last_k_digits(number, k):
    return number // (10**k)


def kth_digit_from_right(number, k):
    """k=1 means ones digit, k=2 means tens digit, etc."""
    return (number // (10 ** (k - 1))) % 10


def kth_digit_from_left(number, k):
    """k=1 means first digit from left."""
    text = str(abs(number))
    return int(text[k - 1])


def is_four_digit_seed(seed):
    return isinstance(seed, int) and 1000 <= seed <= 9999


def normalize_four_digit_seed(seed):
    """Useful when a generated value like 769 should be displayed as 0769."""
    return f"{seed:04d}"


def middle_square_one_step(seed):
    square = seed * seed
    square_as_8_digits = f"{square:08d}"
    next_value = int(square_as_8_digits[2:6])
    return square, square_as_8_digits, next_value


def demo_middle_square_steps():
    seeds = [5731, 8443, 769, 2500]

    print("Middle-square step demo")
    print("| Seed | Square | 8-digit square | Middle four |")
    print("|---:|---:|:---:|---:|")
    for seed in seeds:
        square, padded, next_value = middle_square_one_step(seed)
        print(f"| {seed:04d} | {square} | {padded} | {next_value:04d} |")


def demo_common_tricks():
    number = 32844361
    small_square = 769 * 769

    print("\nCommon digit tricks")
    print("Number:", number)
    print("Middle four by string:", middle_digits_by_string(number))
    print("Middle four by arithmetic:", middle_digits_by_arithmetic(number))
    print("Digits:", split_digits(number))
    print("Join [8, 4, 4, 3]:", join_digits([8, 4, 4, 3]))
    print("Reverse:", reverse_number(number))
    print("Digit count:", count_digits(number))
    print("First 3 digits:", first_k_digits(number, 3))
    print("Last 3 digits:", last_k_digits(number, 3))
    print("Remove last 3 digits:", remove_last_k_digits(number, 3))
    print("3rd digit from right:", kth_digit_from_right(number, 3))
    print("3rd digit from left:", kth_digit_from_left(number, 3))

    print("\nLeading-zero example")
    print("769^2 normally:", small_square)
    print("769^2 left padded to 8 digits:", left_pad_zeros(small_square, 8))
    print("769^2 right padded to 8 digits:", right_pad_zeros(small_square, 8))
    print("Middle four:", middle_digits_by_string(small_square))
    print("Generated value as integer:", int(middle_digits_by_string(small_square)))
    print("Generated value displayed as 4 digits:", normalize_four_digit_seed(769))


if __name__ == "__main__":
    demo_middle_square_steps()
    demo_common_tricks()
