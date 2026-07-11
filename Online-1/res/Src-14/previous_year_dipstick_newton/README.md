# Previous Year Problem: Measure Dipstick Height

Tank diameter:

```text
8 ft
```

So radius:

```text
r = 4 ft
```

Given spherical cap volume formula:

```text
V = pi*h^2*(3r - h) / 3
```

Given:

```text
V = 5 ft^3
```

Substitute `r = 4`:

```text
5 = pi*h^2*(3*4 - h) / 3
```

So:

```text
5 = pi*h^2*(12 - h) / 3
```

Bring everything to one side:

```text
f(h) = pi*h^2*(12 - h)/3 - 5
```

We need to solve:

```text
f(h) = 0
```

## Newton-Raphson Formula

```text
h(i+1) = h(i) - f(h(i)) / f'(h(i))
```

Here:

```text
f(h) = pi*h^2*(12 - h)/3 - 5
```

Expand if needed:

```text
f(h) = (pi/3)*(12h^2 - h^3) - 5
```

Derivative:

```text
f'(h) = pi*(8h - h^2)
```

## Initial Guess

Since oil height must be between `0` and tank diameter `8`:

```text
0 <= h <= 8
```

Check:

```text
f(0) = -5
f(1) = 6.519173
```

Since:

```text
f(0)*f(1) < 0
```

the root lies between `0` and `1`.

So choose:

```text
h0 = 0.5
```

## Iteration Table

Using error tolerance `0.05%`:

```text
iter      h estimate          f(h)        ea (%)
1           0.668858      0.308474     25.245676
2           0.648833      0.004205      3.086243
3           0.648552      0.000001      0.043267
```

Since:

```text
0.043267% <= 0.05%
```

stop.

## Final Answer

```text
h = 0.648552 ft
```

So the dipstick wet height should be approximately:

```text
0.6486 ft
```

## Files

- `dipstick_newton_solution.py`: Python solution with graph and table.
- `dipstick_newton_solution.m`: MATLAB/Octave solution.
- `graph.png`: generated after running the Python script.

