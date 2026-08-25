# Kolmogorov-Smirnov Test

## Basic Idea

The Kolmogorov-Smirnov test, or K-S test, checks whether a sample follows a given distribution.

In this folder, the test is used to check whether random numbers are uniformly distributed over:

```text
[0, 1)
```

The null hypothesis is:

```text
H0: The sample is uniformly distributed.
```

The test compares:

```text
empirical CDF of the sample
```

with:

```text
theoretical CDF of U[0, 1)
```

## Formula

First sort the sample:

```python
s = sorted(sample)
```

Then calculate:

```text
D+ = max(i/N - R_i)
D- = max(R_i - (i-1)/N)
D  = max(D+, D-)
```

Decision:

```text
If D > D_critical, reject H0.
Otherwise, fail to reject H0.
```

## Files

### `ks_test.py`

This file implements the K-S test manually.

Important code:

```python
D_plus = max(i/N - s[i-1] for i in range(1, N+1))
D_minus = max(s[i-1] - (i-1)/N for i in range(1, N+1))
D = max(D_plus, D_minus)
```

Then it uses either a hardcoded table value or a large-sample approximation:

```python
D_critical = c[alpha] / math.sqrt(N)
```

### `using_scipy.py`

This file does the same test but uses SciPy to get the critical value and p-value.

Important code:

```python
D_critical = stats.ksone.ppf(1 - alpha, N)
p_value = 1 - stats.ksone.cdf(D, N)
```

This is useful if the exam asks you to compare your manual result with Python libraries.

## Run

```bash
python ks_test/using_scipy.py
```
