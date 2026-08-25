# Random Numbers

## Basic Idea

Random-number generation means producing numbers that look random. Most simulation problems use numbers in:

```text
[0, 1)
```

These values are often written as:

```text
R1, R2, R3, ...
```

Important properties:

```text
1. Uniformity: values should be spread evenly over [0, 1).
2. Independence: one value should not predict the next value.
3. Long period: the sequence should not repeat too early.
```

## Linear Congruential Generator

A common generator is:

```text
X_n = (a X_(n-1) + c) mod m
R_n = X_n / m
```

Here:

```text
X = integer state
a = multiplier
c = increment
m = modulus
R = normalized random number in [0, 1)
```

## Middle-Square Generator

Steps:

```text
1. Start with a 4-digit seed.
2. Square it.
3. Pad the square to 8 digits.
4. Take the middle 4 digits.
5. Use that as the next value.
```

Python pattern:

```python
s = str(x * x).zfill(8)
x = int(s[2:6])
```

Example:

```text
5731^2 = 32844361
middle four digits = 8443
```

Weakness:

```text
Middle-Square can fall into short cycles.
```

Example:

```text
2500 -> 2500 -> 2500 -> ...
```

## Chi-Square Test

The Chi-Square test checks whether numbers are uniformly distributed.

For 10 bins and 1000 values:

```text
Expected count in each bin = 1000 / 10 = 100
```

Formula:

```text
Chi^2 = sum((Observed - Expected)^2 / Expected)
```

Decision:

```text
If p < 0.05, reject H0.
Otherwise, fail to reject H0.
```

## Files

### `rng_generator.py`

Implements an LCG:

```python
X = (a*X + c) % m
R_s.append(X/m)
```

It also transforms uniform random numbers into:

```text
Exponential distribution
Uniform[a, b]
```

### `chi_square_test.py`

Contains Chi-Square testing logic. It counts values in bins and compares observed counts with expected counts.

Important pattern:

```python
counts = [0] * bins
idx = min(bins - 1, int(r * bins))
counts[idx] += 1
```

### `chi_square_scipy.py`

Uses Python's `random` module to generate random numbers, then uses SciPy for Chi-Square critical value and p-value.

Important code:

```python
random.seed(seed)
random.random()
```

### `visualize_random_numbers.py`

Creates visual plots:

```text
1. Sequence plot
2. Histogram
3. R_i vs R_(i+1) scatter plot
```

These plots help detect obvious patterns.

### `onlines/c.py`

Complete Middle-Square assignment solution:

```text
Task 1: generate 100 values
Task 2: find problematic seed
Task 3: Chi-Square test
Task 4: reflection
```

## Run

```bash
python random_number/onlines/c.py
python random_number/visualize_random_numbers.py
```
