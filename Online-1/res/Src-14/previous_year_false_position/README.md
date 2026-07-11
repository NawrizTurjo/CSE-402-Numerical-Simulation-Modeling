# Previous Year Problem: False Position Method

Given equation:

```text
x^3 - x - 1 = 0
```

So,

```text
f(x) = x^3 - x - 1
```

## Initial Guess

From the graph, the curve crosses the x-axis between `x = 1` and `x = 2`.

Also:

```text
f(1) = 1^3 - 1 - 1 = -1
f(2) = 2^3 - 2 - 1 = 5
```

Since:

```text
f(1) * f(2) < 0
```

there is a root between `1` and `2`.

So choose:

```text
x_l = 1
x_u = 2
```

## False Position Formula

```text
x_r = (x_u*f(x_l) - x_l*f(x_u)) / (f(x_l) - f(x_u))
```

Approximate relative error:

```text
ea = abs((new x_r - old x_r) / new x_r) * 100
```

## Iteration Table

Using `x_l = 1`, `x_u = 2`, and error tolerance `0.001%`:

```text
iter       x_l       x_u       x_r       f(x_r)      ea(%)
1      1.00000   2.00000   1.16667   -0.57870       ---
2      1.16667   2.00000   1.25311   -0.28536   6.89845
3      1.25311   2.00000   1.29344   -0.12954   3.11769
4      1.29344   2.00000   1.31128   -0.05659   1.36078
5      1.31128   2.00000   1.31899   -0.02430   0.58435
6      1.31899   2.00000   1.32228   -0.01036   0.24913
7      1.32228   2.00000   1.32368   -0.00440   0.10588
8      1.32368   2.00000   1.32428   -0.00187   0.04494
9      1.32428   2.00000   1.32453   -0.00079   0.01907
10     1.32453   2.00000   1.32464   -0.00034   0.00809
11     1.32464   2.00000   1.32468   -0.00014   0.00343
12     1.32468   2.00000   1.32470   -0.00006   0.00145
13     1.32470   2.00000   1.32471   -0.00003   0.00062
```

Since `0.00062% <= 0.001%`, stop.

## Final Answer

```text
The real root is x = 1.324712 approximately.
```

This cubic has only one real root because its graph crosses the x-axis only
once. Therefore, all real roots:

```text
x = 1.324712
```

## Files

- `false_position_solution.py`: Python solution with graph, sign interval scan, and table.
- `false_position_solution.m`: MATLAB/Octave solution.
- `graph.png`: generated after running the Python script.

