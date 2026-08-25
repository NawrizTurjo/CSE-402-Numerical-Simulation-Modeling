# Metropolis-Hastings

## Basic Idea

Metropolis-Hastings is a Markov Chain Monte Carlo method.

It generates samples from a target distribution, even when the normalizing constant is unknown.

Goal:

```text
Generate samples from a distribution proportional to exp(log_target(x)).
```

## Algorithm

Steps:

```text
1. Start from an initial value x.
2. Propose a new value near x.
3. Compute the acceptance ratio.
4. Accept or reject the proposed value.
5. Repeat many times.
```

If the proposal is rejected, the chain stays at the current value.

## Why Log Probability?

The code uses `log_target(x)` instead of the normal probability density.

This is common because:

```text
1. Log values avoid underflow for tiny probabilities.
2. Multiplication becomes addition.
3. The unknown normalizing constant cancels out.
```

## File

### `metropolis_hashtigs.py`

Main function:

```python
def metropolis_hastings(log_target, x0, n_samples, proposal_std=1.0, seed=None):
```

Parameters:

```text
log_target   = log of target density, ignoring constants
x0           = starting value
n_samples    = number of samples to generate
proposal_std = standard deviation of proposal distribution
seed         = optional random seed
```

Important code:

```python
x_proposed = x + random.gauss(0, proposal_std)
```

This proposes a nearby value using a Gaussian random step.

Then:

```python
log_alpha = log_target(x_proposed) - log_target(x)
```

This computes the log acceptance ratio.

Accept/reject step:

```python
if math.log(random.random()) < log_alpha:
    x = x_proposed
```

If accepted, move to the proposed value. Otherwise, stay at the old value.

## Examples in the Code

The file includes examples for:

```text
1. Standard normal N(0, 1)
2. Shifted normal N(3, 0.5^2)
3. Bimodal distribution
```

## Run

```bash
python metropolis_hashtings/metropolis_hashtigs.py
```
