"""
Python syntax cheat sheet -- pure syntax reference, no simulation logic.
One function per topic, each runnable/printable on its own so you can
jog your memory on exactly the piece you forgot without reading the rest.

Run standalone (prints every section):
    python -m cheatsheets.python_syntax
"""


def variables_and_types():
    x = 10              # int
    y = 3.5              # float
    name = "Alice"       # string
    passed = True        # bool

    print("Variables and types")
    print(x, type(x))
    print(y, type(y))
    print(name, type(name))
    print(passed, type(passed))


def arithmetic():
    a, b = 10, 3

    print("\nArithmetic")
    print("a / b  =", a / b)     # float division
    print("a // b =", a // b)    # integer (floor) division
    print("a % b  =", a % b)     # remainder
    print("a ** b =", a ** b)    # power


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

    print("enumerate (index + value together):")
    for index, value in enumerate([10, 20, 30]):
        print(f"  {index}: {value}")


def lists_and_comprehensions():
    numbers = [10, 20, 30, 40]

    print("\nLists")
    print("slice [1:3]:", numbers[1:3])
    numbers.append(50)
    print("after append:", numbers)

    squares = [x * x for x in range(1, 6)]
    evens = [x for x in range(10) if x % 2 == 0]
    print("list comprehension squares:", squares)
    print("list comprehension evens:", evens)


def tuples_sets_dictionaries():
    point = (3, 4)
    unique_numbers = {1, 2, 2, 3}   # duplicates collapse -> {1, 2, 3}
    student = {"id": 101, "name": "Alice", "mark": 85}

    print("\nTuple, set, dictionary")
    print("tuple:", point)
    print("set (dedups automatically):", unique_numbers)
    print("dict:", student, "-> student['name']:", student["name"])
    student["mark"] = 90  # update in place
    print("dict.get with default (no KeyError):", student.get("age", "not set"))


def string_formatting():
    """The formatting patterns that show up constantly in this course's
    code: zero-padding for Middle-Square digit extraction, and
    fixed-decimal printing for performance measures/statistics."""
    number = 769
    pi = 3.14159265

    print("\nString formatting")
    print("zero-pad to 4 digits :", f"{number:04d}")   # "0769"
    print("zero-pad to 8 digits :", f"{number*number:08d}")
    print("fixed 4 decimals     :", f"{pi:.4f}")         # "3.1416"
    print("scientific notation  :", f"{0.0000123:.4g}")
    print("thousands separator  :", f"{1234567:,}")


def functions_and_lambdas():
    def add(a, b):
        return a + b

    square = lambda x: x * x   # anonymous one-liner function, common as a trial_fn/target

    print("\nFunctions and lambdas")
    print("add(2, 3):", add(2, 3))
    print("square(5):", square(5))


def useful_builtins():
    numbers = [4, 7, 1, 9, 2]

    print("\nUseful built-ins")
    print("sum/min/max/sorted:", sum(numbers), min(numbers), max(numbers), sorted(numbers))
    print("zip (pair up two lists):", list(zip([1, 2, 3], ["a", "b", "c"])))


def heapq_basics():
    """
    Every DES engine in this course (simulation/single_server_queue.py)
    uses heapq as a min-priority-queue for the event list: the earliest
    (smallest) event time is always popped first. This is the one
    stdlib module worth knowing cold for the DES section.
    """
    import heapq

    print("\nheapq (min-priority-queue) basics")
    events = []
    heapq.heappush(events, (1.5, "A"))   # push a (priority, payload) tuple
    heapq.heappush(events, (0.4, "A"))
    heapq.heappush(events, (4.0, "D"))
    print("heap internal list (NOT fully sorted, only heap-ordered):", events)

    print("popping in priority order:")
    while events:
        time, kind = heapq.heappop(events)   # always returns the smallest tuple
        print(f"  time={time}  kind={kind}")

    print("heapq.heapify(list) turns an existing list into a heap in place, O(n)")


def file_io_pattern():
    print("\nFile read/write pattern (rarely needed live, but here if asked)")
    print("with open('file.txt', 'r') as f:")
    print("    data = f.read()")
    print("with open('file.txt', 'w') as f:")
    print("    f.write('hello')")


def main():
    variables_and_types()
    arithmetic()
    loops()
    lists_and_comprehensions()
    tuples_sets_dictionaries()
    string_formatting()
    functions_and_lambdas()
    useful_builtins()
    heapq_basics()
    file_io_pattern()


if __name__ == "__main__":
    main()
