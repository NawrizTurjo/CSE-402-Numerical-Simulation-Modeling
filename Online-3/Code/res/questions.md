# Online-3 Exam Questions Reference

---

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

# B2: Reliability Analysis of a Stochastic Network via Weyl-Sequence Simulation

### Page 1

## Background

The Middle-Square Weyl Sequence (MSWS) generator keeps two running 32-bit states: a value $x$ (the "middle-square" state) and a value $w$ (the "Weyl" state), both taken modulo $2^{32}$. Starting from $x_0 = 0$ and $w_0 = 0$, each new variate is produced by the following update, using the fixed increment constant $c = \text{0xb5ad4ec5}$:

**One generator step (produces the next state $x_{n+1}$ from $x_n, w_n$):**
1. Advance the Weyl state:
   $$w_{n+1} = (w_n + c) \bmod 2^{32}$$
2. Square-and-add:
   $$x_{n+1} = (x_n^2 + w_{n+1}) \bmod 2^{32}$$
3. Extract the output bits by discarding the low 16 bits of $x_{n+1}$:
   $$U_n = \frac{\lfloor x_{n+1} / 2^{16} \rfloor}{2^{16}} \in [0, 1)$$
   *(This is equivalent to the bit-shift $x_{n+1} \gg 16$ described by right-shifting; you may implement it with plain integer division, no bitwise operators required.)*

## Task 1

Implement the Middle-Square Weyl Sequence generator described above, with initial seed $x_0 = 0$ and initial Weyl state $w_0 = 0$. It should return a list of $N$ variates $U_0, \ldots, U_{N-1}$.

```python
def msws_generator(n_samples, x0=0, w0=0, c=0xb5ad4ec5, m=2**32):
    # your implementation
```

## Task 2: Stochastic Network Simulation

Consider a 4-component stochastic network trying to send a signal from node A to node B.

Each component $i \in \{1, 2, 3, 4\}$ fails independently after time $X_i \sim \text{Exponential}(\text{mean} = \beta_i)$ in days:
$$\beta_1 = 10, \quad \beta_2 = 8, \quad \beta_3 = 7, \quad \beta_4 = 5$$

Build a Monte Carlo simulation of $N = 10,000$ replications to estimate the probability $p$ that the overall system survival time $Y \ge 7$ days ($\theta = 7$), using a random number assignment scheme from the Middle-Square Weyl Sequence generator.

**Inverse-CDF Formula:** To convert a uniform random variate $U \sim \text{Uniform}(0, 1)$ to an exponentially distributed lifetime $X \sim \text{Exponential}(\text{mean} = \beta)$, use:
$$X = F^{-1}(U) = -\beta \ln(1 - U)$$

### Steps:

1. **Generate Stream:** Produce a $U(n)$ stream, long enough to satisfy the scheme in the next step ($4 \times 10,000 = 40,000$ numbers).
2. **Stream Assignment:** Select uniform variates $U(n)$ for the four component lifetimes of each replication $k \in \{0, \ldots, 9999\}$ under the following scheme:
   $$U_1 = U(4k), \quad U_2 = U(4k + 1), \quad U_3 = U(4k + 2), \quad U_4 = U(4k + 3)$$
3. **Variate Transformation:** Convert the assigned uniform variates $U_1, U_2, U_3, U_4$ into component lifetimes $X_1, X_2, X_3, X_4$ using the inverse-CDF formula:
   $$X_i = -\beta_i \ln(1 - U_i), \quad \text{for } i \in \{1, 2, 3, 4\}$$
4. **Simulate Network:** For any two subsystem lifetimes $T_a$ and $T_b$, the survival time of their series and parallel combinations are given by:
   $$T_{\text{series}}(T_a, T_b) = \min(T_a, T_b), \qquad T_{\text{parallel}}(T_a, T_b) = \max(T_a, T_b)$$
   Using only these two combination rules, work out and implement the correct sequence of operations that computes the overall system survival time $Y_k$ for the topology, applied to the four component lifetimes $X_1, X_2, X_3, X_4$ of replication $k$.
   - *Parallel-of-Series branches:* $Y_k = \max(\min(X_1, X_2), \min(X_3, X_4))$
   - *Series-of-Parallel stages:* $Y_k = \min(\max(X_1, X_2), \max(X_3, X_4))$
   - *Series with Parallel Subsystem:* $Y_k = \min(X_1, \max(X_2, \min(X_3, X_4)))$
5. **Calculate Reliability:** Compute the estimated probability of survival past $\theta = 7$ days:
   $$\hat{p} = \frac{1}{N} \sum_{k=0}^{N-1} \mathbb{I}(Y_k \ge 7)$$

## Task 3: Redundancy Optimization & Reflection

The company maintains this network but can afford to buy exactly one spare component, identical in reliability to one of components 1, 2, 3, or 4, and install it as a redundant unit in parallel with that one chosen component (leaving the rest of the topology unchanged).

Based on the structure of the network in Task 2, not on any new simulation or formula derivation, decide which single component (1, 2, 3, or 4) the spare should be added to in order to improve the 7-day survival probability $p$ the most, and justify your choice.

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
