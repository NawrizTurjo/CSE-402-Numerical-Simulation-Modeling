# Previous Year Problem: Break-Even Point by Newton-Raphson

Cost function:

```text
C(q) = 1000 + 2q + 3q^(2/3)
```

Selling price is `300` per gram, so revenue is:

```text
R(q) = 300q
```

At break-even point:

```text
C(q) = R(q)
```

So,

```text
1000 + 2q + 3q^(2/3) = 300q
```

Bring everything to one side:

```text
f(q) = 1000 - 298q + 3q^(2/3)
```

Now solve:

```text
f(q) = 0
```

## Newton-Raphson Formula

```text
x(i+1) = x(i) - f(x(i)) / f'(x(i))
```

Here,

```text
f(x)  = 1000 - 298x + 3x^(2/3)
f'(x) = -298 + 2x^(-1/3)
```

Stopping condition:

```text
absolute relative approximate error <= 0.05%
```

where,

```text
ea = abs((new value - old value) / new value) * 100
```

## Initial Guess

From the graph, the curve crosses the x-axis between `x = 3` and `x = 4`.
Also:

```text
f(3) > 0
f(4) < 0
```

So the root lies between `3` and `4`. I choose:

```text
x0 = 4
```

## Iteration Table

Using `x0 = 4`:

```text
Iteration     x estimate       f(x)              ea (%)
1             3.378444        -0.021824         18.397688
2             3.378371        -0.000000          0.002177
```

Since `0.002177% <= 0.05%`, stop.

## Final Answer

```text
Break-even quantity = 3.378371 grams per day
```

So the firm should produce approximately:

```text
3.3784 grams per day
```

to have neither profit nor loss.

## Files

- `break_even_newton.py`: Python solution with graph and table.
- `break_even_newton.m`: MATLAB/Octave solution with graph and table.
- `graph.png`: generated after running the Python script.

