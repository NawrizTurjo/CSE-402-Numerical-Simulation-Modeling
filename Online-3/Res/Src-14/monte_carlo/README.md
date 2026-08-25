# Monte Carlo Methods

## Basic Idea

Monte Carlo methods use random sampling to estimate numerical values.

They are useful when:

```text
1. Exact calculation is hard.
2. A simulation is easier than a formula.
3. We can approximate an answer using many random trials.
```

More samples usually give a better estimate.

## Monte Carlo Integration

To estimate:

```text
Integral of g(x) from a to b
```

Use:

```text
Integral ≈ (b - a) * average(g(X))
```

where:

```text
X is sampled uniformly from [a, b].
```

In code:

```python
x = random.uniform(a, b)
total += g(x)
return (b - a) * total / n
```

Here:

```text
g = function being integrated
a = lower limit
b = upper limit
n = number of random samples
```

## Estimating Pi

Generate random points inside the unit square:

```text
0 <= x <= 1
0 <= y <= 1
```

Count points inside the quarter circle:

```text
x^2 + y^2 <= 1
```

Then:

```text
pi ≈ 4 * hits / n
```

## Files

### `general_monte_carlo.py`

This file estimates a definite integral.

Example:

```python
result = monte_carlo(math.sin, 0, math.pi, n=10000, seed=1)
```

This estimates:

```text
Integral of sin(x) from 0 to pi
```

The true value is:

```text
2
```

### `estimate_pi.py`

This file estimates pi using hit-or-miss Monte Carlo.

Important code:

```python
x = random.random()
y = random.random()

if x**2 + y**2 <= 1:
    hits += 1
```

Finally:

```python
return 4 * hits / n
```

## Run

```bash
python monte_carlo/general_monte_carlo.py
python monte_carlo/estimate_pi.py
```
