# A1: Buffon's Needle Experiment

[got the exact question]

---

### Page 1

## 1. Problem Statement

You are required to estimate the mathematical constant $\pi$ using a Monte Carlo Simulation of the classical Buffon's Needle Experiment. A needle of length $\mathbf{L} = \mathbf{1.0}$ is dropped randomly onto a flat surface ruled with parallel lines spaced at a uniform distance $\mathbf{D} = \mathbf{2.0}$ apart $(\mathbf{L} \leq \mathbf{D})$.

**Given System Parameters:**
- Needle Length: $\mathbf{L} = \mathbf{1.0}$
- Spacing between Parallel Lines: $\mathbf{D} = \mathbf{2.0}$
- Condition: $\mathbf{L} \leq \mathbf{D}$ (Needle cannot cross more than one line simultaneously)

## 2. Mathematical Modeling & Random Variables

Each random drop of the needle is uniquely defined by two independent continuous uniform random variables:

1. **Needle Center Position ($X$):** The perpendicular distance from the midpoint (center) of the needle to the nearest parallel line.
   - **Distribution:** $\mathbf{X} \sim \mathbf{U}(0, \mathbf{D} / 2) = \mathbf{U}(0, \mathbf{1.0})$
   - **Generation:** Using uniform random number $\mathbf{R}_{1} \in [0, 1)$:
     $$\mathbf{X} = (\mathbf{D} / 2) \times \mathbf{R}_{1} = \mathbf{1.0} \times \mathbf{R}_{1}$$

2. **Needle Orientation Angle ($\theta$):** The acute angle formed between the needle and the parallel lines.
   - **Distribution:** $\theta \sim \mathbf{U}(0, \pi / 2)$
   - **Generation:** Using uniform random number $\mathbf{R}_{2} \in [0, 1)$:
     $$\theta = (\pi / 2) \times \mathbf{R}_{2}$$

## 3. Hit (Line Crossing) Condition

A needle crosses (hits) a line if and only if the center distance $\mathbf{X}$ is less than or equal to the perpendicular projection of half the needle length:

- **Hit Condition:** $\mathbf{X} \leq (\mathbf{L} / 2) \times \sin(\theta) \implies \mathbf{X} \leq \mathbf{0.5} \times \sin(\theta)$

## 4. Theoretical Probability & $\pi$ Estimator

The theoretical crossing probability $\mathbf{P}$ is calculated by integrating the hit region over the sample space:

$$\mathbf{P} = \frac{\int_{0}^{\pi / 2} (\mathbf{L} / 2) \sin(\theta) \, \mathrm{d}\theta}{(\mathbf{D} / 2)(\pi / 2)} = \frac{2\mathbf{L}}{\pi \mathbf{D}}$$

For $\mathbf{L} = \mathbf{1.0}$ and $\mathbf{D} = \mathbf{2.0}$, the exact crossing probability is:

$$\mathbf{P} = \frac{2(1)}{2\pi} = \frac{1}{\pi} \approx 0.31831$$

In a simulation of $\mathbf{N}$ total drops with $\mathbf{H}$ total hits (crossings):

- **Empirical Crossing Probability:** $\mathbf{P} \equiv \mathbf{H} / \mathbf{N}$
- **Monte Carlo Estimator for $\pi$:**
  $$\pi \equiv \frac{2\mathbf{L}}{\mathbf{D} \times \mathbf{P}} = \frac{2 \times \mathbf{L} \times \mathbf{N}}{\mathbf{D} \times \mathbf{H}}$$
  For $\mathbf{L} = 1$, $\mathbf{D} = 2$:
  $$\pi \equiv \frac{\mathbf{N}}{\mathbf{H}} = \frac{1}{\mathbf{P}}$$

## 5. Implementation Requirements

Write a Python program that performs the following:

1. Implement the simulation function `buffon_needle_pi(n_drops, L=1.0, D=2.0)`.
2. For each drop, generates $x = (D / 2.0) \times \text{random.random()}$ and $\theta = (\pi / 2.0) \times \text{random.random()}$.
3. Tests condition $x \leq (L / 2.0) \times \sin(\theta)$ and counts total hits $H$.
4. Computes $\hat{p} = \text{hits} / n\_drops$ and $\text{pi\_estimate} = (2.0 \times L) / (D \times \hat{p})$.
5. Executes the simulation for $N \in \{100, 1000, 10000, 100000, 1000000\}$ and displays the formatted results.

---

### Page 2

## 6. Expected Sample Output

| N (Drops) | Hits (H) | p_hat (H/N) | Pi Estimate | Absolute Error |
| :--- | :--- | :--- | :--- | :--- |
| 100 | 320 | 0.32000 | 3.125000 | 0.016593 |
| 1,000 | 315 | 0.31500 | 3.174603 | 0.033011 |
| 10,000 | 3,184 | 0.31840 | 3.140703 | 0.000889 |
| 100,000 | 31,812 | 0.31812 | 3.143468 | 0.001875 |
| 1,000,000 | 318,310 | 0.31831 | 3.141592 | 0.000001 |

---

# C1: Investigating the Middle-Square Random Number Generator

In this assignment, you will implement a pseudo-random number generator using the Middle-Square Method, investigate its weaknesses, and statistically evaluate the generated numbers using a Chi-Square goodness-of-fit test. Do not use Python's `random` module.

## Task 1: Implement the Middle-Square Generator

Start with a 4-digit integer seed $X_0$.

For every iteration:
1. Square the current value.
2. Represent the result as an 8-digit number, adding leading zeros if necessary.
3. Extract the middle four digits.
4. Use these four digits as the next value.

**For example:**
- $\text{Seed} = 5731$
- $5731^2 = 32844361 \implies \text{Middle four digits} = 8443$
- $8443^2 = 71284249 \implies \text{Middle four digits} = 2842$

Implement the following function:

```python
def middle_square(seed, n):
    # your implementation
```

Generate at least 100 values.

## Task 2: Investigate Edge Cases

The Middle-Square method has an important weakness. Find at least one seed that causes the generator to repeatedly generate the same value.

For the problematic seed, write:
- **Seed:**
- **First repeated value:**
- **Iteration at which it repeats:**
- **Cycle length:**

## Task 3: Test Whether the Numbers Are Uniform

Generating numbers that appear random is not sufficient. Use a Chi-Square goodness-of-fit test to investigate whether your generated numbers are approximately uniformly distributed between 0 and 1.

Divide the interval $[0, 1)$ into 10 equal-sized bins:
- **Bin 1:** $[0.0, 0.1)$
- **Bin 2:** $[0.1, 0.2)$
- ...
- **Bin 10:** $[0.9, 1.0)$

Generate 1,000 random numbers and count how many values fall into each bin.
Under the assumption of a uniform distribution, each bin is expected to contain:

$$E = \frac{1000}{10} = 100 \text{ values}$$

You may use Python's statistical libraries, such as SciPy, to perform the test.
Use a significance level of $\alpha = 0.05$. Therefore, your decision rule is:
- If $p < 0.05$, reject $H_0$; otherwise, do not reject.

**Null Hypothesis:**
- $H_0$: The generated numbers are uniformly distributed over $[0, 1)$.

Perform the Chi-Square test for the following three different seeds. For each seed, generate 1,000 values, perform the Chi-Square test, and report your results as shown below:

| Seed | Chi^2 | p-value | Decision |
| :--- | :--- | :--- | :--- |
| 5731 | | | |
| 6239 | | | |
| Problematic Seed | | | |

## Task 4: Reflect

Does passing the Chi-Square test necessarily mean that the generator is a good random-number generator? Explain briefly.

---

# C2: When the Generator Fails the Test: Hidden Period Collapse and the K-S Test

### Page 1

## Background

Consider the multiplicative congruential generator (MCG):

$$X_{n + 1} = (a \cdot X_{n}) \bmod m, \qquad U_{n} = \frac{X_{n}}{m}, \qquad \text{with } m = 2^{16} = 65536, \; a = 5, \; X_{0} = 1$$

Note that $c = 0$ here, so the Hull-Dobell full-period conditions do not apply. A generator can still look perfectly uniform on paper while being structurally broken underneath.

## Task 1

Implement the following function. It should return a list of $U_{n}$ values.

```python
def lcg_generator(n_samples, m=65536, a=5, X0=1):
    # your implementation
```

## Task 2

Perform a one-sample, two-sided Kolmogorov-Smirnov (K-S) test to evaluate whether the generated stream adheres to:
- $H_{0}: U_{n} \sim \text{Uniform}(0, 1)$ vs. $H_{1}: U_{n} \nsim \text{Uniform}(0, 1)$

### Steps:

1. **Generate Stream:** Produce $N = 100,000$ values $(U_{1}, U_{2}, \ldots, U_{N})$ using `lcg_generator`.
2. **Sort Variates:** Sort the sequence in ascending order such that $U_{(1)} \leq U_{(2)} \leq \dots \leq U_{(N)}$.
3. **Compute Empirical CDF:** Compute the two-sided K-S statistic $D_{N}$:
   $$D_{N} = \max_{1 \leq i \leq N} \left\{ \max \left( \frac{i}{N} - U_{(i)}, \; U_{(i)} - \frac{i - 1}{N} \right) \right\}$$
4. **Evaluate Critical Threshold:** Calculate the large-sample critical value at significance level $\alpha = 0.05$:
   $$D_{\text{crit}} = \frac{1.36}{\sqrt{N}}$$
5. **Decision Rule:** Compare $D_{N}$ against $D_{\text{crit}}$ and explicitly state whether to accept or reject $H_{0}$.

## Task 3

You should find that the test in Task 2 fails to reject $H_{0}$; the marginal distribution appears to be uniform. Answer the following questions:

(a) Calculate the period of the MCG programmatically (in code). Does your result match the theoretically expected value for this MCG?

(b) Given that the period is $p$ and $p \ll N$, explain concretely what happens to the sequence $U_{n}$ for $n > p$, and why this behavior does not manifest as a violation of uniformity in a one-sample K-S test.
