"""
Python basic syntax cheat sheet.

Run:
    python cheet_sheets/python_basic_syntax.py
"""


def variables_and_types():
    x = 10              # int
    y = 3.5             # float
    name = "Alice"      # string
    passed = True       # bool

    print("Variables and types")
    print(x, type(x))
    print(y, type(y))
    print(name, type(name))
    print(passed, type(passed))


def input_and_output():
    # user_input = input("Enter a number: ")
    # number = int(user_input)
    # print("You entered:", number)

    value = 42
    print("\nInput/output")
    print("Normal print:", value)
    print(f"Formatted print: value = {value}")


def arithmetic():
    a = 10
    b = 3

    print("\nArithmetic")
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)       # float division
    print("a // b =", a // b)     # integer division
    print("a % b =", a % b)       # remainder
    print("a ** b =", a ** b)     # power


def conditions():
    mark = 75

    print("\nConditions")
    if mark >= 80:
        grade = "A+"
    elif mark >= 70:
        grade = "A"
    else:
        grade = "Below A"

    print("Grade:", grade)


def loops():
    print("\nLoops")

    print("for loop:")
    for i in range(5):
        print(i, end=" ")
    print()

    print("while loop:")
    count = 0
    while count < 5:
        print(count, end=" ")
        count += 1
    print()


def lists():
    numbers = [10, 20, 30, 40]

    print("\nLists")
    print("numbers:", numbers)
    print("first item:", numbers[0])
    print("last item:", numbers[-1])
    print("slice [1:3]:", numbers[1:3])

    numbers.append(50)
    print("after append:", numbers)

    numbers[0] = 99
    print("after update:", numbers)

    print("loop over list:")
    for value in numbers:
        print(value, end=" ")
    print()


def list_comprehension():
    squares = [x * x for x in range(1, 6)]
    evens = [x for x in range(10) if x % 2 == 0]

    print("\nList comprehension")
    print("squares:", squares)
    print("evens:", evens)


def tuples_sets_dictionaries():
    point = (3, 4)
    unique_numbers = {1, 2, 2, 3}
    student = {"id": 101, "name": "Alice", "mark": 85}

    print("\nTuple, set, dictionary")
    print("tuple:", point)
    print("set:", unique_numbers)
    print("dictionary:", student)
    print("student name:", student["name"])

    student["mark"] = 90
    print("updated dictionary:", student)


def functions():
    def add(a, b):
        return a + b

    def square(x):
        return x * x

    print("\nFunctions")
    print("add(2, 3):", add(2, 3))
    print("square(5):", square(5))


def strings():
    text = "simulation"

    print("\nStrings")
    print("text:", text)
    print("length:", len(text))
    print("first char:", text[0])
    print("slice [0:4]:", text[0:4])
    print("uppercase:", text.upper())
    print("contains 'sim':", "sim" in text)

    number = 769
    print("left pad to 4 digits:", f"{number:04d}")
    print("left pad to 8 digits:", f"{number:08d}")


def useful_builtins():
    numbers = [4, 7, 1, 9, 2]

    print("\nUseful built-ins")
    print("len:", len(numbers))
    print("sum:", sum(numbers))
    print("min:", min(numbers))
    print("max:", max(numbers))
    print("sorted:", sorted(numbers))
    print("enumerate:")
    for index, value in enumerate(numbers):
        print(index, value)


def file_pattern():
    print("\nFile read/write pattern")
    print("with open('file.txt', 'r') as f:")
    print("    data = f.read()")
    print("with open('file.txt', 'w') as f:")
    print("    f.write('hello')")


def main():
    variables_and_types()
    input_and_output()
    arithmetic()
    conditions()
    loops()
    lists()
    list_comprehension()
    tuples_sets_dictionaries()
    functions()
    strings()
    useful_builtins()
    file_pattern()


if __name__ == "__main__":
    main()
