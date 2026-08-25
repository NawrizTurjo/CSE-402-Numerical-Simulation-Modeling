# Independence Tests

## Basic Idea

Uniformity alone is not enough for a random-number generator.

Example:

```text
0.1, 0.2, 0.3, 0.4, ...
```

These values may look spread out, but they are predictable. A good random-number generator should produce values that are approximately independent.

Null hypothesis:

```text
H0: The random numbers are independent.
```

Common decision rule for Z-tests:

```text
If |Z0| > 1.96, reject H0 at alpha = 0.05.
Otherwise, fail to reject H0.
```

## Runs Test

The runs test checks whether numbers move above and below a reference point randomly.

In this code, the reference point is:

```text
0.5
```

Each number is labeled:

```python
1 if r > 0.5 else 0
```

A run is a continuous sequence of the same label.

Example:

```text
1 1 0 0 1
```

Runs:

```text
11, 00, 1
```

So the number of runs is 3.

## Autocorrelation Test

The autocorrelation test checks whether values are related to later values.

It compares:

```text
R_i with R_(i+l)
```

where:

```text
i = starting index
l = lag
N = sample size
```

If values are independent, autocorrelation should be close to zero.

## Files

### `run_test.py`

This file implements the runs test.

Important code:

```python
above = [1 if r > 0.5 else 0 for r in R]
```

This converts random numbers into above/below labels.

Then:

```python
if above[k] != above[k-1]:
    runs += 1
```

This counts a new run whenever the label changes.

### `auto_correlation_test.py`

This file implements the autocorrelation test.

Important code:

```python
idx = [i - 1 + k * l for k in range(M + 2)]
summ = sum(R[idx[k]] * R[idx[k+1]] for k in range(len(idx)-1))
```

This selects values separated by lag `l` and multiplies neighboring lagged values.

Then:

```python
Z0 = rho_hat / sigma
```

The decision is made using `|Z0| > 1.96`.

## Run

```bash
python Independence/run_test.py
python Independence/auto_correlation_test.py
```
