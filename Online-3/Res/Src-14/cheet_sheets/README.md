# Cheat Sheets

## Purpose

This folder contains short, runnable cheat sheets for exam preparation.

The goal is not to build full programs. The goal is to remember small syntax patterns quickly.

## Files

### `python_basic_syntax.py`

Covers basic Python syntax:

```text
variables
types
print
arithmetic
conditions
loops
lists
dictionaries
functions
strings
file read/write pattern
```

Useful examples:

```python
for i in range(5):
    print(i)
```

```python
numbers = [10, 20, 30]
numbers.append(40)
```

```python
f"{number:08d}"
```

### `numpy_basic_syntax.py`

Covers common NumPy syntax:

```text
array creation
indexing
slicing
vector operations
statistics
random numbers
reshaping
boolean filtering
matrix multiplication
histogram/bin counts
```

Useful examples:

```python
a = np.array([1, 2, 3])
a + 5
a * a
```

```python
np.mean(a)
np.var(a)
np.std(a)
```

```python
counts, edges = np.histogram(values, bins=10, range=(0, 1))
```

### `number_manipulation_cheatsheet.py`

Covers number and digit tricks:

```text
leading zero padding
right zero padding
middle digit extraction
first k digits
last k digits
reverse number
split digits
join digits
```

Important for Middle-Square:

```python
text = f"{number:08d}"
middle = text[2:6]
```

### `random_module_cheatsheet.py`

Covers Python's `random` module:

```text
random.seed
random.random
random.uniform
random.randint
random.choice
random.shuffle
random.expovariate
random.gauss
```

It also compares Python random numbers with custom generators.

### `middle_square_solution.py`

Contains a complete Middle-Square assignment solution:

```text
generate values
find cycle
run Chi-Square test
print result table
write reflection
```

### `scipy.py`

Use this file for remembering SciPy-related syntax if needed.

## Run

```bash
python cheet_sheets/python_basic_syntax.py
python cheet_sheets/numpy_basic_syntax.py
python cheet_sheets/number_manipulation_cheatsheet.py
python cheet_sheets/random_module_cheatsheet.py
```
