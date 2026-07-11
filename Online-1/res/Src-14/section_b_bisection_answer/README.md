# Section B: Graph, Sign Change Intervals, and Bisection

Question function:

```text
f(x) = 0.6*ln(x+1) - sth*sin(1.7*x) - 0.08*x^2 - 0.08
```

Here `sth` is treated as a parameter. In the scripts, change this line if your
exam gives a different value:

```text
sth = 1.0
```

## What the code does

1. Plots `f(x)` for `0 <= x <= 10`.
2. Uses step size `h = 0.1` to find all sign-change intervals.
3. Applies the bisection method to each sign-change interval.
4. Prints all roots.
5. Prints a separate bisection iteration table for each root.

## Important exam answer for question 6

If we run bisection only once on the full interval `[0, 10]`, the problem is:

- Bisection needs `f(a)*f(b) < 0`.
- For the full interval, `f(0)` and `f(10)` may have the same sign.
- Even if several roots exist inside `[0, 10]`, the endpoint signs can still be
  the same.
- So bisection may not start at all.
- Also, one bisection run can find only one root, not all roots.

That is why we first scan using step size `0.1`, find smaller sign-change
intervals, and then apply bisection separately on each interval.

## Sample result when `sth = 1.0`

Sign-change intervals:

```text
[1.6, 1.7]
[3.5, 3.6]
```

Approximate roots:

```text
x = 1.677495
x = 3.581715
```

## Files

- `section_b_answer.py`: Python answer.
- `section_b_answer.m`: MATLAB/Octave answer.
- `graph.png`: created after running the Python script.
