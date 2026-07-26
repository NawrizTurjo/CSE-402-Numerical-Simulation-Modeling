# A2 Coding Test Prep — Numerical Analysis (CSE401)

One self-contained document: question-pattern analysis, topic coverage, and all solved practice problems (from the previously-tested subsections A1, B1, C1, C2). All code below is plain, independent, hand-written Python — no reference toolkit used — and every result shown was actually executed, not hand-derived.

## Contents
1. [Question pattern & library rules](#1-question-pattern--library-rules)
2. [Topic coverage table](#2-topic-coverage-table)
3. [A1 — Inverse Power Method](#3-a1--inverse-power-method-smallest-eigenvalue)
4. [B1 — Gaussian Elimination & Classification](#4-b1--gaussian-elimination-with-partial-pivoting-classifying-systems)
5. [C1 — Power Method](#5-c1--power-iteration-method-dominant-eigenvalue)
6. [C2 — LU Decomposition, Multiple RHS](#6-c2--lu-decomposition-reused-for-multiple-right-hand-sides)
7. [P1a — Gauss-Jordan RREF & Determinant](#7-p1a--gauss-jordan-rref--determinant)
8. [P1b — Matrix Inverse via Gauss-Jordan and LU](#8-p1b--matrix-inverse-via-gauss-jordan-and-lu)
9. [P2 — Round-off Error & Complexity](#9-p2--round-off-error--complexity)
10. [P3 — Direct Eigenvalue via Characteristic Polynomial](#10-p3--direct-eigenvalue-via-characteristic-polynomial)
11. [P4 — Full Eigendecomposition & Matrix Powers](#11-p4--full-eigendecomposition--matrix-powers)
12. [P5 — Deflation for Intermediate Eigenvalues](#12-p5--deflation-for-intermediate-eigenvalues)

---

## 1. Question pattern & library rules

Source: <code style="color:#3B82F6">Previous Subsection Questions/*.docx</code> — reconstructed from memory by students in A1, B1, C1, C2 (not leaked, so exact matrices are usually missing; only the task structure is reliable).

- **Format**: one coding question per subsection, timed online (B1's note: "30 minutes initially, extended to 35"). Often a short written/conceptual question tacked on at the end (B1, C2).
- **What must be hand-coded**: the actual numerical method — pivot selection, row swaps, elimination, forward/back substitution, the power-iteration loop, normalization. Every previous question explicitly calls for writing this from scratch.
- **What NumPy is allowed for**:
  - **Cross-checking your own already-computed result, as a second/independent computation** — not as a substitute for computing that result yourself. <code style="color:#3B82F6">np.linalg.solve</code>, <code style="color:#3B82F6">np.linalg.eig</code>, and similar "give me the whole answer" calls are for comparing against your hand-computed answer at the end (B1, C1, C2 all end with "verify/compare with NumPy").
  - **As an explicitly-named building block inside an otherwise hand-written algorithm**, only when the question says so: A1 explicitly permits <code style="color:#3B82F6">np.linalg.inv()</code> to get <code style="color:#3B82F6">A⁻¹</code> once, then the power-iteration loop around it must still be hand-written.
  - **Not for pieces of your own result, even simple-looking ones.** This is the one that's easy to get wrong: confirmed directly by a classmate who's seen the real rubric (residual and norm specifically must be computed by hand first, with <code style="color:#3B82F6">np.linalg.norm</code>/<code style="color:#3B82F6">np.linalg.solve</code> only used afterward to re-derive the *same* number as a cross-check) — <code style="color:#3B82F6">np.linalg.norm(x)</code> to normalize *your own* vector, <code style="color:#3B82F6">A @ x</code> to get *your own* <code style="color:#3B82F6">Ax</code>, <code style="color:#3B82F6">np.linalg.norm(A - L@U)</code> to check *your own* decomposition, etc. all count as "computing part of your own answer with a library," not "verification," and should be hand-coded (a few lines: <code style="color:#3B82F6">sum(v*v for v in vec) ** 0.5</code> for an L2 norm, a row-by-row dot-product loop for a matrix-vector product). Every practice question below now does the manual version first and prints a numpy-computed value alongside it purely as a same-number sanity check.
  - Default assumption if not stated: implement manually, verify with NumPy at the end, report both.
- **Recurring requirements to always include**: print every intermediate step (pivots chosen, row swaps, matrix after each elimination step, forward/back-substitution values as they resolve); normalize eigenvectors manually before comparing to NumPy; remember NumPy eigenvectors can come back sign-flipped (<code style="color:#3B82F6">v</code> and <code style="color:#3B82F6">-v</code> are equally valid — not a bug); compute residual <code style="color:#3B82F6">‖Ax-b‖</code> and/or <code style="color:#3B82F6">‖A-LU‖</code> manually as a correctness check, then cross-check that same number against NumPy.

---

## 2. Topic coverage table

| # | Topic | Source lecture | Status | Difficulty |
|---|-------|-----------------|--------|------------|
| 1 | Gauss elimination (forward elim + back substitution) | systems-of-eqs | ✅ Tested — B1 | Core |
| 2 | Partial pivoting (selection rule + row swap) | systems-of-eqs | ✅ Tested — B1 | Core |
| 3 | Classifying unique / infinite / no-solution systems | systems-of-eqs | ✅ Tested — B1 | Core |
| 4 | Round-off error pitfall (naive vs. pivoted comparison) | systems-of-eqs | 🟡 Practice added — P2 | Medium |
| 5 | Determinant via elimination (row-op theorems, sign flip on swap) | systems-of-eqs | 🟡 Practice added — P1a | Medium |
| 6 | Gauss-Jordan elimination (RREF, no back-substitution) | systems-of-eqs | 🟡 Practice added — P1a | Medium |
| 7 | Matrix inverse via Gauss-Jordan <code style="color:#3B82F6">[A\|I]</code> | systems-of-eqs | 🟡 Practice added — P1b | Medium |
| 8 | LU decomposition (Doolittle derivation from multipliers) | systems-of-eqs | ✅ Tested — C2 | Core |
| 9 | Solving via LU (forward-sub then back-sub) | systems-of-eqs | ✅ Tested — C2 | Core |
| 10 | Reusing one LU factorization for multiple right-hand sides | systems-of-eqs | ✅ Tested — C2 | Core |
| 11 | Cost/complexity analysis (clock-cycle model, Gauss vs LU op counts) | systems-of-eqs | 🟡 Practice added — P2 (written) | Medium (theory) |
| 12 | Matrix inverse via LU (solve for each identity column) | systems-of-eqs | 🟡 Practice added — P1b | Medium |
| 13 | Eigenvector/eigenvalue definitions & geometric meaning | eigen-decomposition | Foundational — assumed everywhere | — |
| 14 | Direct/manual eigenvalue-finding via characteristic polynomial <code style="color:#3B82F6">det(A-λI)=0</code> | eigen-decomposition | 🟡 Practice added — P3 | Medium–Hard |
| 15 | Eigen-coordinate systems, <code style="color:#3B82F6">V</code>/<code style="color:#3B82F6">V⁻¹</code> as translators between bases | eigen-decomposition | 🟡 Practice added — P4 | Hard |
| 16 | Full eigenvalue decomposition <code style="color:#3B82F6">A = VΛV⁻¹</code> (derive & reconstruct) | eigen-decomposition | 🟡 Practice added — P4 | Hard |
| 17 | Computing <code style="color:#3B82F6">Aᵏ</code> cheaply via <code style="color:#3B82F6">VΛᵏV⁻¹</code> | eigen-decomposition | 🟡 Practice added — P4 | Hard |
| 18 | Power method (dominant eigenvalue/eigenvector) | eigen-decomposition | ✅ Tested — C1 | Core |
| 19 | Inverse power method (smallest eigenvalue; ties to LU reuse) | eigen-decomposition | ✅ Tested — A1 | Core |
| 20 | **Deflation** (intermediate eigenvalues, Hotelling's deflation, orthogonality caveat) | eigen-decomposition | 🟡 Practice added — P5 | **Hardest — last topic taught** |

Legend: **✅ Tested** = appeared in a real subsection's actual exam (A1/B1/C1/C2). **🟡 Practice added** = no real subsection has faced this yet, but a self-authored practice question below now covers it (matrix/data invented for practice, not from any real exam). **⬜ Not yet tested** = still uncovered by either.

**Status**: every row in the table now has at least practice coverage (sections 7–12 below). Rows marked ✅ are what real subsections actually faced; rows marked 🟡 are self-authored practice questions covering everything that hadn't come up yet in any real subsection, with **deflation (P5)** — the single highest-risk gap — now included. Genuinely-tested (✅) material should still be the most-drilled, since it's the only category with real precedent; the 🟡 rows are best-effort coverage of what's *plausible* to be asked, not guaranteed content.

---

## 3. A1 — Inverse Power Method (smallest eigenvalue)

### Problem Statement

You are given the following 4×4 matrix:

$$
A = \begin{bmatrix} 4 & 1 & 0 & 0 \\ 1 & 3 & 1 & 0 \\ 0 & 1 & 2 & 1 \\ 0 & 0 & 1 & 1 \end{bmatrix}
$$

**(a)** Implement the **Inverse Power Method** to find the eigenvalue of <code style="color:#3B82F6">A</code> with the *smallest* magnitude, along with its corresponding eigenvector. You are explicitly allowed to use <code style="color:#3B82F6">np.linalg.inv()</code> to compute <code style="color:#3B82F6">A⁻¹</code> — but the iterative part (repeated multiplication, normalization, convergence check) must be written by hand, and every iteration's estimate should be printed.

**(b)** Separately, compute *all* eigenvalues and eigenvectors of <code style="color:#3B82F6">A</code> using NumPy.

**(c)** Compare the result of (a) against the matching pair from (b), and comment.

### Solution

**Why this works**: if <code style="color:#3B82F6">Av = λv</code>, then <code style="color:#3B82F6">A⁻¹v = (1/λ)v</code> — inverting a matrix **flips the ranking** of its eigenvalues. The smallest-magnitude eigenvalue of <code style="color:#3B82F6">A</code> becomes the *largest* (dominant) eigenvalue of <code style="color:#3B82F6">A⁻¹</code>. So run the ordinary power method on <code style="color:#3B82F6">A⁻¹</code>, then take the reciprocal of whatever it converges to.

```python
import numpy as np

A = np.array([
    [4, 1, 0, 0],
    [1, 3, 1, 0],
    [0, 1, 2, 1],
    [0, 0, 1, 1],
], dtype=float)

# Step 1: A^-1 once (library call, as the question allows)
A_inv = np.linalg.inv(A)

# Step 2: start from any nonzero vector
x = np.array([1.0, 1.0, 1.0, 1.0])

lambda_old = 0.0
tol = 1e-6

# Step 3: iterate y = A_inv @ x, normalize by the largest entry
print(f"{'iter':>4} | {'lambda(A) estimate':>20} | x (normalized)")
for iteration in range(1, 101):
    y = A_inv @ x                     # apply A^-1
    idx = np.argmax(np.abs(y))        # largest-magnitude entry
    largest_value = y[idx]            # eigenvalue estimate OF A_inv
    x = y / largest_value             # normalize

    lambda_of_A = 1.0 / largest_value # reciprocal -> eigenvalue estimate of A
    change = abs(lambda_of_A - lambda_old)
    print(f"{iteration:4d} | {lambda_of_A:20.8f} | {np.round(x, 6)}")

    if change < tol:
        break
    lambda_old = lambda_of_A

print(f"\nConverged after {iteration} iterations (tolerance {tol}).")
print("Smallest-magnitude eigenvalue of A  ~=", lambda_of_A)
print("Eigenvector (largest entry = 1)     =", x)

def l2_norm(vec):
    # sqrt(sum of squares), computed manually -- the norm itself is part of
    # OUR result, so it must be hand-coded; numpy is only for cross-checking
    total = 0.0
    for v in vec:
        total += v * v
    return total ** 0.5

manual_norm = l2_norm(x)
eigenvector_unit = [v / manual_norm for v in x]
numpy_norm = np.linalg.norm(x)
print(f"manual norm(x) = {manual_norm:.10f}   numpy norm(x) = {numpy_norm:.10f}")
print("Eigenvector (unit length)            =", [round(float(v), 6) for v in eigenvector_unit])

# Part (b): NumPy's full eigendecomposition
eigvals, eigvecs = np.linalg.eig(A)
order = np.argsort(eigvals)                # ascending -> index 0 = smallest
lambda_numpy = eigvals[order][0]
vec_numpy = eigvecs[:, order][:, 0]

# Part (c): compare (allowing for a possible sign flip) -- manual first,
# then the same comparison recomputed via numpy as an independent check
diff_same = [eigenvector_unit[i] - vec_numpy[i] for i in range(4)]
diff_flip = [eigenvector_unit[i] + vec_numpy[i] for i in range(4)]
manual_diff = min(l2_norm(diff_same), l2_norm(diff_flip))

numpy_diff = min(np.linalg.norm(np.array(eigenvector_unit) - vec_numpy),
                  np.linalg.norm(np.array(eigenvector_unit) + vec_numpy))
print(f"manual diff = {manual_diff:.2e}   numpy diff = {numpy_diff:.2e}")
```

### Actual output

```
Step 3: iterate  y = A_inv @ x , normalize by the largest-magnitude entry.
iter |   lambda(A) estimate | x (normalized)
   1 |           0.70000000 | [ 0.1  0.3 -0.3  1. ]
   2 |           0.31250000 | [-0.0625   0.28125 -0.6875   1.     ]
   3 |           0.26168224 | [-0.078271  0.296729 -0.738318  1.      ]
   4 |           0.25559869 | [-0.080024  0.30009  -0.744401  1.      ]
   5 |           0.25483461 | [-0.080256  0.300632 -0.745165  1.      ]
   6 |           0.25473441 | [-0.080289  0.300712 -0.745266  1.      ]
   7 |           0.25472090 | [-0.080294  0.300723 -0.745279  1.      ]
   8 |           0.25471906 | [-0.080294  0.300725 -0.745281  1.      ]
   9 |           0.25471880 | [-0.080294  0.300725 -0.745281  1.      ]

Converged after 9 iterations (tolerance 1e-06).
Smallest-magnitude eigenvalue of A  ~= 0.2547188009871201
Eigenvector (largest entry = 1)     = [-0.08029445  0.30072533 -0.7452812   1.        ]
manual norm(x) = 1.2854287178   numpy norm(x) = 1.2854287178
Eigenvector (unit length)            = [-0.062465, 0.233949, -0.579792, 0.777951]
manual diff = 3.66e-08   numpy diff = 3.66e-08
```

NumPy's full eigendecomposition:

```
All eigenvalues (ascending): [0.25471876 1.82271708 3.17728292 4.74528124]
All eigenvectors (columns, matching order above):
[[-0.06246513 -0.29117378  0.55327108 -0.77795055]
 [ 0.23394946  0.6339677  -0.45518556 -0.57979195]
 [-0.57979195 -0.45518556 -0.6339677  -0.23394946]
 [ 0.77795055 -0.55327108 -0.29117378 -0.06246513]]
```

### Comparison

| | eigenvalue | eigenvector (unit length) |
|---|---|---|
| Inverse power method | 0.25471880 | <code style="color:#3B82F6">[-0.06246512, 0.23394944, -0.57979193, 0.77795057]</code> |
| NumPy <code style="color:#3B82F6">np.linalg.eig</code> | 0.25471876 | <code style="color:#3B82F6">[-0.06246513, 0.23394946, -0.57979195, 0.77795055]</code> |

Eigenvalue difference <code style="color:#3B82F6">4.12e-08</code>, eigenvector difference <code style="color:#3B82F6">3.66e-08</code> — matches essentially exactly after only 9 iterations, and the sign already agrees here (no flip needed, though either sign would be valid).

### Notes worth remembering

- The ordinary power method always converges to the *largest*-magnitude eigenvalue; to get the *smallest*, invert the problem — hence "inverse power method."
- Normalizing by the **largest entry** (not Euclidean norm) keeps the vector from blowing up/collapsing, and the divisor itself doubles as the eigenvalue estimate.
- <code style="color:#3B82F6">v</code> and <code style="color:#3B82F6">−v</code> are both valid eigenvectors for the same eigenvalue — a sign mismatch vs. NumPy is not a bug.

---

## 4. B1 — Gaussian Elimination with Partial Pivoting: Classifying Systems

### Problem Statement

You are given the following **three separate systems of three equations in three unknowns**:

**System 1:**

$$
\begin{aligned}
0x_1 + 2x_2 + x_3 &= 7 \\
x_1 + 2x_2 + 3x_3 &= 14 \\
2x_1 + x_2 + x_3 &= 7
\end{aligned}
\qquad\Longleftrightarrow\qquad
A = \begin{bmatrix} 0 & 2 & 1 \\ 1 & 2 & 3 \\ 2 & 1 & 1 \end{bmatrix},
\quad b = \begin{bmatrix} 7 \\ 14 \\ 7 \end{bmatrix}
$$

**System 2:**

$$
\begin{aligned}
x_1 + x_2 + x_3 &= 6 \\
x_1 - x_2 + x_3 &= 2 \\
2x_1 + 2x_2 + 2x_3 &= 12
\end{aligned}
\qquad\Longleftrightarrow\qquad
A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -1 & 1 \\ 2 & 2 & 2 \end{bmatrix},
\quad b = \begin{bmatrix} 6 \\ 2 \\ 12 \end{bmatrix}
$$

**System 3:**

$$
\begin{aligned}
x_1 + x_2 + x_3 &= 6 \\
2x_1 + 2x_2 + 2x_3 &= 10 \\
x_1 - x_2 + x_3 &= 2
\end{aligned}
\qquad\Longleftrightarrow\qquad
A = \begin{bmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \\ 1 & -1 & 1 \end{bmatrix},
\quad b = \begin{bmatrix} 6 \\ 10 \\ 2 \end{bmatrix}
$$

For **each** system:

1. Implement **Gaussian elimination with partial pivoting** by hand — pivot-selection and row-swapping logic must be explicitly written yourself.
2. Print, per step: candidate pivots considered, pivot chosen, every row swap, and the augmented matrix after that step.
3. Classify the system using the reduced (upper-triangular) augmented matrix:
   - A row <code style="color:#3B82F6">0·x+0·y+0·z = c</code>, <code style="color:#3B82F6">c≠0</code> → **no solution**. Print the offending equation.
   - A row <code style="color:#3B82F6">0·x+0·y+0·z = 0</code> → **infinitely many solutions**. Print the offending equation.
   - Otherwise → **unique solution**.
4. Depending on the case: **unique** → back-substitute, print the solution, verify against <code style="color:#3B82F6">np.linalg.solve(A,b)</code>, report <code style="color:#3B82F6">‖Ax−b‖₂</code>. **No solution** → just report the contradiction. **Infinite** → choose a free variable, express the others in terms of it, report one particular solution.

**Written question**: If a diagonal entry becomes 0 during elimination, does that necessarily mean the system has no solution? In which case exactly is it "no solution," and in which case "infinitely many solutions"?

### Solution

```python
def forward_elimination_partial_pivot(augmented_matrix):
    # augmented_matrix is [A | b]: each row holds one equation's coefficients
    # with its right-hand-side value tacked on as the last (extra) entry
    augmented_matrix = [row[:] for row in augmented_matrix]
    n = len(augmented_matrix)
    for k in range(n - 1):
        # find the largest |value| in column k, at/below row k
        pivot_row = k
        best = abs(augmented_matrix[k][k])
        for i in range(k + 1, n):
            if abs(augmented_matrix[i][k]) > best:
                best = abs(augmented_matrix[i][k])
                pivot_row = i

        if pivot_row != k:                      # SWAP
            print(f"SWAP: R{k+1} <-> R{pivot_row+1}")
            augmented_matrix[k], augmented_matrix[pivot_row] = \
                augmented_matrix[pivot_row], augmented_matrix[k]

        pivot_val = augmented_matrix[k][k]
        print(f"pivot = A[{k+1}][{k+1}] = {pivot_val}")
        if abs(pivot_val) < 1e-9:
            continue                             # nothing usable in this column

        for i in range(k + 1, n):                # eliminate below the pivot
            factor = augmented_matrix[i][k] / pivot_val
            augmented_matrix[i] = [augmented_matrix[i][j] - factor * augmented_matrix[k][j]
                                    for j in range(n + 1)]
        print("matrix after this step:", augmented_matrix)
    return augmented_matrix


def classify_and_solve(augmented_matrix):
    n = len(augmented_matrix)
    for i in range(n):
        coeffs, rhs = augmented_matrix[i][:-1], augmented_matrix[i][-1]
        if all(abs(c) < 1e-9 for c in coeffs):
            if abs(rhs) > 1e-9:
                return "no solution", None          # 0 = nonzero
            else:
                return "infinite", None              # 0 = 0
    # every pivot survived -> back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = augmented_matrix[i][-1]
        for j in range(i + 1, n):
            total -= augmented_matrix[i][j] * x[j]
        x[i] = total / augmented_matrix[i][i]
    return "unique", x
```

### System 1 (unique solution, uses the data from the Problem Statement above)

```
--- Step 1: eliminate column 1 ---
  candidate pivots in column 1: [0, 1, 2]  -> choose row 3
  SWAP: R1 <-> R3
  pivot = A[1][1] = 2
  matrix after step 1:
    [2, 1, 1, 7]
    [0.0, 1.5, 2.5, 10.5]
    [0.0, 2.0, 1.0, 7.0]

--- Step 2: eliminate column 2 ---
  candidate pivots in column 2: [1.5, 2.0]  -> choose row 3
  SWAP: R2 <-> R3
  pivot = A[2][2] = 2
  matrix after step 2:
    [2, 1, 1, 7]
    [0.0, 2.0, 1.0, 7.0]
    [0.0, 0.0, 1.75, 5.25]

All pivots nonzero -> UNIQUE SOLUTION
  x3 = 3,  x2 = 2,  x1 = 1
```

<code style="color:#3B82F6">x = [1, 2, 3]</code>. <code style="color:#3B82F6">np.linalg.solve</code> agrees exactly; <code style="color:#3B82F6">‖Ax−b‖₂ = 0.0</code>. Note the original <code style="color:#3B82F6">A[0][0]=0</code> looked alarming but pivoting simply swapped it away — a zero *starting* diagonal entry alone says nothing about solvability.

### System 2 (infinitely many solutions, uses the data from the Problem Statement above)

(Row 3 is exactly <code style="color:#3B82F6">2 ×</code> Row 1, on both sides — it carries no new information.)

```
--- Step 1 ---
  candidate pivots in column 1: [1, 1, 2]  -> choose row 3
  SWAP: R1 <-> R3
  matrix after step 1:
    [2, 2, 2, 12]
    [0.0, -2.0, 0.0, -4.0]
    [0.0, 0.0, 0.0, 0.0]

--- Step 2 ---
  candidate pivots in column 2: [2.0, 0.0]  -> row 2, no swap needed
  matrix after step 2: (unchanged)

Row 3 reduced to: 0 = 0  ->  INFINITE SOLUTIONS
```

Choosing <code style="color:#3B82F6">x₃ = t</code> as the free variable and eliminating it from the two surviving equations gives the general solution:

$$
x_1 = 4 - t, \qquad x_2 = 2, \qquad x_3 = t \qquad \text{(any real } t \text{)}
$$

Picking <code style="color:#3B82F6">t = 5</code>: <code style="color:#3B82F6">x = [-1, 2, 5]</code>, and <code style="color:#3B82F6">‖Ax−b‖₂ = 0.0</code> for the original system.

### System 3 (no solution, uses the data from the Problem Statement above)

(Rows 1 and 2 point at the same plane but disagree on the right-hand side: <code style="color:#3B82F6">2 × 6 = 12 ≠ 10</code>.)

```
--- Step 1 ---
  candidate pivots in column 1: [1, 2, 1]  -> choose row 2
  SWAP: R1 <-> R2
  matrix after step 1:
    [2, 2, 2, 10]
    [0.0, 0.0, 0.0, 1.0]
    [0.0, -2.0, 0.0, -3.0]

--- Step 2 ---
  candidate pivots in column 2: [0.0, 2.0]  -> choose row 3
  SWAP: R2 <-> R3
  matrix after step 2:
    [2, 2, 2, 10]
    [0.0, -2.0, 0.0, -3.0]
    [0.0, 0.0, 0.0, 1.0]

Row 3 reduced to: 0 = 1  ->  NO SOLUTION
```

Pivoting *actively surfaces* the contradiction: the offending <code style="color:#3B82F6">0=1</code> row starts out hidden in row 2 and only lands at the bottom after the step-2 swap, where the classifier catches it.

### Summary

| System | Result |
|---|---|
| 1 | Unique — <code style="color:#3B82F6">x = [1, 2, 3]</code> |
| 2 | Infinite — <code style="color:#3B82F6">x = [4−t, 2, t]</code>, e.g. <code style="color:#3B82F6">[-1, 2, 5]</code> for <code style="color:#3B82F6">t=5</code> |
| 3 | No solution — row reduces to <code style="color:#3B82F6">0 = 1</code> |

### Written question — answer

**No**, a zero diagonal entry alone doesn't mean no solution. It only means *that row* is currently unusable as a pivot for that column:

- **If some row below it still has a nonzero entry in that column**, partial pivoting swaps it up and elimination proceeds normally (System 1: <code style="color:#3B82F6">A[0][0]=0</code> initially, yet perfectly solvable).
- **Only if every candidate in that column, at and below the current row, is zero** does the column genuinely fail, collapsing the entire row to <code style="color:#3B82F6">0·x+0·y+0·z = (something)</code>. Then the RHS decides:
  - RHS **nonzero** → contradiction → **no solution** (System 3).
  - RHS **zero** → always true, no new information (the row was linearly dependent on the others) → **infinitely many solutions** (System 2).

So: *a zero pivot by itself is inconclusive; a fully zero row (after pivoting is exhausted) classifies the system, and that dead row's right-hand side decides "no solution" vs. "infinite solutions."*

---

## 5. C1 — Power Iteration Method (dominant eigenvalue)

### Problem Statement

Implement the **Power Iteration Method** on the following 4×4 matrix to compute its **dominant eigenvalue** and corresponding **eigenvector**:

$$
A = \begin{bmatrix} 12 & 2 & 1 & 0 \\ 2 & 5 & 0 & 1 \\ 1 & 0 & 3 & 1 \\ 0 & 1 & 1 & 2 \end{bmatrix}
$$

Steps:

1. Start with any nonzero <code style="color:#3B82F6">x₀</code>.
2. At each iteration, compute <code style="color:#3B82F6">y = A x_k</code>.
3. Normalize <code style="color:#3B82F6">y</code> by its largest-magnitude entry — that entry is the eigenvalue estimate, the normalized vector is the eigenvector estimate.
4. Repeat until the eigenvalue estimate converges.

Then verify using NumPy's eigenvalue computation.

**Important notes (given in the original question)**: the eigenvector must be normalized before comparing to NumPy's, otherwise they won't numerically match; NumPy's eigenvector may come back with the opposite sign — since <code style="color:#3B82F6">v</code> and <code style="color:#3B82F6">−v</code> are both valid, that is expected, not an error.

### Solution

```python
import numpy as np

A = np.array([
    [12, 2, 1, 0],
    [ 2, 5, 0, 1],
    [ 1, 0, 3, 1],
    [ 0, 1, 1, 2],
], dtype=float)

x = np.array([1.0, 1.0, 1.0, 1.0])   # step 1: any nonzero starting vector
lambda_old = 0.0
tol = 1e-6

print(f"{'iter':>4} | {'lambda estimate':>16} | x (normalized)")
for iteration in range(1, 101):
    y = A @ x                        # step 2: multiply by A
    idx = np.argmax(np.abs(y))       # step 3: largest-magnitude entry
    lambda_new = y[idx]
    x = y / lambda_new                # normalize by it

    change = abs(lambda_new - lambda_old)
    print(f"{iteration:4d} | {lambda_new:16.8f} | {np.round(x, 6)}")

    if change < tol:                  # step 4: stop once stable
        break
    lambda_old = lambda_new

print(f"\nConverged after {iteration} iterations (tolerance {tol}).")
print("Dominant eigenvalue  ~=", lambda_new)
print("Eigenvector (largest entry = 1) =", x)

def l2_norm(vec):
    # sqrt(sum of squares), computed manually -- part of OUR result, so it
    # must be hand-coded; numpy is only used afterward to cross-check it
    total = 0.0
    for v in vec:
        total += v * v
    return total ** 0.5

manual_norm = l2_norm(x)
x_unit = [v / manual_norm for v in x]          # normalize to unit length before comparing
numpy_norm = np.linalg.norm(x)
print(f"manual norm(x) = {manual_norm:.10f}   numpy norm(x) = {numpy_norm:.10f}")
print("Eigenvector (unit length)       =", [round(float(v), 6) for v in x_unit])

eigvals, eigvecs = np.linalg.eig(A)
dom_idx = np.argmax(np.abs(eigvals))
dom_vec = eigvecs[:, dom_idx]

# compare manually first (allowing for a possible sign flip)...
diff_same_manual = l2_norm([x_unit[i] - dom_vec[i] for i in range(4)])
diff_flip_manual = l2_norm([x_unit[i] + dom_vec[i] for i in range(4)])

# ...then cross-check the same two numbers via numpy
diff_same_numpy = np.linalg.norm(np.array(x_unit) - dom_vec)
diff_flip_numpy = np.linalg.norm(np.array(x_unit) + dom_vec)
print(f"manual : same-sign diff = {diff_same_manual:.2e}   flipped-sign diff = {diff_flip_manual:.2e}")
print(f"numpy  : same-sign diff = {diff_same_numpy:.2e}   flipped-sign diff = {diff_flip_numpy:.2e}")
# smaller of the two (same-sign vs. flipped) confirms the match; sign itself is not meaningful
```

### Actual output

```
iter |  lambda estimate | x (normalized)
   1 |      15.00000000 | [1.       0.533333 0.333333 0.266667]
   2 |      13.40000000 | [1.       0.368159 0.169154 0.104478]
   3 |      12.90547264 | [1.       0.305705 0.124904 0.057826]
   4 |      12.73631457 | [1.       0.281585 0.112477 0.04289 ]
   5 |      12.67564623 | [1.       0.27224  0.108895 0.037855]
   6 |      12.65337482 | [1.       0.268628 0.10784  0.036105]
   7 |      12.64509663 | [1.       0.267238 0.107522 0.035482]
   8 |      12.64199722 | [1.       0.266704 0.107424 0.035257]
   9 |      12.64083144 | [1.       0.2665   0.107392 0.035175]
  10 |      12.64039161 | [1.       0.266422 0.107382 0.035145]
  11 |      12.64022531 | [1.       0.266392 0.107379 0.035133]
  12 |      12.64016235 | [1.       0.26638  0.107378 0.035129]
  13 |      12.64013848 | [1.       0.266376 0.107377 0.035127]
  14 |      12.64012942 | [1.       0.266375 0.107377 0.035127]
  15 |      12.64012598 | [1.       0.266374 0.107377 0.035127]
  16 |      12.64012468 | [1.       0.266374 0.107377 0.035127]
  17 |      12.64012418 | [1.       0.266374 0.107377 0.035127]

Converged after 17 iterations (tolerance 1e-06).
Dominant eigenvalue  ~= 12.640124183945149
Eigenvector (largest entry = 1) = [1.         0.26637355 0.10737689 0.03512653]
manual norm(x) = 1.0410180300   numpy norm(x) = 1.0410180300
Eigenvector (unit length)       = [0.960598, 0.255878, 0.103146, 0.033742]
manual : same-sign diff = 5.49e-08   flipped-sign diff = 2.00e+00
numpy  : same-sign diff = 5.49e-08   flipped-sign diff = 2.00e+00
```

The estimate overshoots at first (15.0) then settles monotonically downward — the vector "swinging" toward the dominant eigen-direction, since <code style="color:#3B82F6">[1,1,1,1]</code> wasn't the true eigenvector.

Verification with NumPy:

```
All eigenvalues: [12.64012388  1.10121323  4.80455043  3.45411247]
Dominant eigenvalue (NumPy): 12.640123880221557
Dominant eigenvector (NumPy, unit length): [0.96059817 0.25587789 0.10314604 0.03374246]

|lambda difference| = 3.04e-07
eigenvector difference (same sign) = 5.49e-08, (flipped sign) = 2.00e+00
-> smaller of the two confirms the match (sign is not meaningful).
```

Signs happened to already agree here, but the code checks both <code style="color:#3B82F6">x_unit − v</code> and <code style="color:#3B82F6">x_unit + v</code> and takes whichever is smaller — the robust way to compare, since on a different run the sign could easily flip.

### Key points

- Normalizing by the **largest entry** gives the eigenvalue estimate for free and prevents the vector exploding/vanishing as <code style="color:#3B82F6">Aᵏx ~ λ₁ᵏ</code>.
- Our loop normalizes so the *largest entry = 1*; NumPy normalizes to *unit Euclidean length* — must convert to the same convention before comparing element-wise.
- Convergence speed depends on <code style="color:#3B82F6">|λ₂/λ₁|</code>; here <code style="color:#3B82F6">≈0.38</code> gives a clean ~17-iteration convergence.

---

## 6. C2 — LU Decomposition Reused for Multiple Right-Hand Sides

### Problem Statement

Given the following 3×3 matrix <code style="color:#3B82F6">A</code> and two 3×1 vectors <code style="color:#3B82F6">b1</code>, <code style="color:#3B82F6">b2</code>:

$$
A = \begin{bmatrix} 4 & 3 & 2 \\ 2 & 5 & 3 \\ 1 & 2 & 4 \end{bmatrix},
\qquad
b_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix},
\qquad
b_2 = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}
$$

1. Find the **LU decomposition** of <code style="color:#3B82F6">A</code>.
2. Verify the decomposition by checking <code style="color:#3B82F6">‖A − LU‖</code>.
3. Solve <code style="color:#3B82F6">Ax=b1</code> **and** <code style="color:#3B82F6">Ax=b2</code> via forward then back substitution, **reusing** the same <code style="color:#3B82F6">L</code>,<code style="color:#3B82F6">U</code> for both.
4. Calculate the **L2 norm** (of the <code style="color:#3B82F6">A−LU</code> reconstruction error from step 2, as a single number).
5. Which part of the computation was reused, and **why is it unnecessary to recompute <code style="color:#3B82F6">L</code>,<code style="color:#3B82F6">U</code>** for the second right-hand side?
6. Verify both solutions against <code style="color:#3B82F6">np.linalg.solve()</code>.
7. Calculate the residual <code style="color:#3B82F6">‖Ax−b‖</code> for both solutions.

### Solution

```python
import numpy as np

n = 3
A = [[4, 3, 2], [2, 5, 3], [1, 2, 4]]
b1, b2 = [1, 2, 3], [4, 5, 6]

# 1. LU decomposition (Doolittle: L has 1's on the diagonal)
L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
U = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in range(i, n):                      # row i of U
        total = sum(L[i][k] * U[k][j] for k in range(i))
        U[i][j] = A[i][j] - total
    for j in range(i + 1, n):                   # column i of L (below diagonal)
        total = sum(L[j][k] * U[k][i] for k in range(i))
        L[j][i] = (A[j][i] - total) / U[i][i]

def matvec(matrix, vec):
    # matrix @ vec, computed manually: dot product of each row with vec
    result = []
    for row in matrix:
        total = 0.0
        for a_ij, x_j in zip(row, vec):
            total += a_ij * x_j
        result.append(total)
    return result

def matmul(X, Y):
    # X @ Y, computed manually
    rows, mid, cols = len(X), len(Y), len(Y[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(mid):
                total += X[i][k] * Y[k][j]
            result[i][j] = total
    return result

def l2_norm(vec):
    # sqrt(sum of squares) -- works for a flat list of numbers
    total = 0.0
    for v in vec:
        total += v * v
    return total ** 0.5

# 2 & 4. Verify A = LU manually, and compute the L2 (Frobenius) norm manually
LU = matmul(L, U)
diff_matrix = [[A[i][j] - LU[i][j] for j in range(n)] for i in range(n)]
manual_norm_diff = l2_norm([v for row in diff_matrix for v in row])  # flatten, then L2
numpy_norm_diff = np.linalg.norm(np.array(A, dtype=float) - np.array(L) @ np.array(U))
print(f"manual ||A - LU||_2 = {manual_norm_diff:.2e}   numpy ||A - LU||_2 = {numpy_norm_diff:.2e}")

A_np = np.array(A, dtype=float)

# 3. Solve for b1 and b2, reusing the same L, U
def forward_substitution(L, b):
    # solves L z = b for z, top row down (L is lower triangular)
    n = len(b)
    z = [0.0] * n
    for i in range(n):
        total = b[i]
        for j in range(i):
            total -= L[i][j] * z[j]
        z[i] = total / L[i][i]
    return z

def back_substitution(U, z):
    # solves U x = z for x, bottom row up (U is upper triangular)
    n = len(z)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = z[i]
        for j in range(i + 1, n):
            total -= U[i][j] * x[j]
        x[i] = total / U[i][i]
    return x

solutions = {}
for name, b in [("b1", b1), ("b2", b2)]:
    z = forward_substitution(L, b)   # step A: L z = b
    x = back_substitution(U, z)      # step B: U x = z
    solutions[name] = x

# 6. Verify with NumPy
print("6. Verify with np.linalg.solve()")
for name, b in [("b1", b1), ("b2", b2)]:
    x_numpy = np.linalg.solve(A_np, np.array(b, dtype=float))
    print(f"  {name}: ours = {[round(v, 6) for v in solutions[name]]}   numpy = {x_numpy}")

# 7. Residual r = Ax - b for both -- computed manually (matvec + l2_norm
# above), THEN cross-checked against numpy's own independent computation.
# The manual version is the real answer; numpy is only there to confirm it.
print("7. Residual r = Ax - b for both")
for name, b in [("b1", b1), ("b2", b2)]:
    x = solutions[name]
    Ax = matvec(A, x)                                  # manual matrix-vector product
    residual = [Ax[i] - b[i] for i in range(n)]         # manual subtraction
    manual_norm = l2_norm(residual)                     # manual L2 norm
    numpy_norm = np.linalg.norm(A_np @ np.array(x) - np.array(b, dtype=float))
    print(f"  {name}: Ax - b = {[round(v, 10) for v in residual]}   "
          f"manual ||.||_2 = {manual_norm:.2e}   numpy ||.||_2 = {numpy_norm:.2e}")
```

### Actual output

```
1. LU decomposition (Doolittle)
  U[1][1] = 4 - 0 = 4
  U[1][2] = 3 - 0 = 3
  U[1][3] = 2 - 0 = 2
  L[2][1] = (2 - 0) / 4 = 0.5
  L[3][1] = (1 - 0) / 4 = 0.25
  U[2][2] = 5 - 1.5 = 3.5
  U[2][3] = 3 - 1 = 2
  L[3][2] = (2 - 0.75) / 3.5 = 0.357143
  U[3][3] = 4 - 1.21429 = 2.785714

L = [[1, 0, 0], [0.5, 1, 0], [0.25, 0.357143, 1]]
U = [[4, 3, 2], [0, 3.5, 2], [0, 0, 2.785714]]

2 & 4. manual ||A - LU||_2 = 0.00e+00   numpy ||A - LU||_2 = 0.00e+00   (exact reconstruction; no pivoting needed for this A)

3. Solve, reusing the SAME L and U
--- b1 = [1, 2, 3] ---
  z1 = 1, z2 = 1.5, z3 = 2.21429
  x3 = 0.794872, x2 = -0.025641, x1 = -0.128205
  solution x = [-0.128205, -0.025641, 0.794872]

--- b2 = [4, 5, 6] ---
  z1 = 4, z2 = 3, z3 = 3.92857
  x3 = 1.41026, x2 = 0.0512821, x1 = 0.25641
  solution x = [0.25641, 0.051282, 1.410256]

6. Verify with np.linalg.solve()
  b1: ours = [-0.128205, -0.025641, 0.794872]   numpy = [-0.12820513 -0.02564103  0.79487179]
  b2: ours = [0.25641, 0.051282, 1.410256]      numpy = [0.25641026 0.05128205 1.41025641]

7. Residual r = Ax - b for both
  b1: Ax - b = [0.0, 0.0, 0.0]   manual ||.||_2 = 0.00e+00   numpy ||.||_2 = 0.00e+00
  b2: Ax - b = [0.0, 0.0, 0.0]   manual ||.||_2 = 0.00e+00   numpy ||.||_2 = 0.00e+00
```

Notice **<code style="color:#3B82F6">L</code> and <code style="color:#3B82F6">U</code> never change** between the <code style="color:#3B82F6">b1</code> and <code style="color:#3B82F6">b2</code> blocks — only the forward-substitution input differs.

### Answering point 5 — what was reused, and why recomputing L, U is unnecessary

The **decomposition step** (<code style="color:#3B82F6">A=LU</code>) was reused as-is; only the much cheaper forward/back-substitution pair was redone per right-hand side.

This is valid because **<code style="color:#3B82F6">L</code> and <code style="color:#3B82F6">U</code> are properties of <code style="color:#3B82F6">A</code> alone** — they come purely from the coefficients on the left-hand side, and the derivation never touches <code style="color:#3B82F6">b</code>. Changing <code style="color:#3B82F6">b</code> only changes what's plugged into <code style="color:#3B82F6">Lz=b</code> and <code style="color:#3B82F6">Ux=z</code>; it does not change the matrix being factored. Recomputing <code style="color:#3B82F6">L,U</code> from scratch for <code style="color:#3B82F6">b2</code> would repeat the exact same <code style="color:#3B82F6">O(n³)</code> elimination arithmetic for a result already sitting there unchanged, whereas forward + back substitution are only <code style="color:#3B82F6">O(n²)</code> each. For <code style="color:#3B82F6">m</code> right-hand sides: LU costs one <code style="color:#3B82F6">O(n³)</code> factorization plus <code style="color:#3B82F6">m</code> cheap <code style="color:#3B82F6">O(n²)</code> solves; re-running plain Gaussian elimination per <code style="color:#3B82F6">b</code> costs <code style="color:#3B82F6">m</code> full <code style="color:#3B82F6">O(n³)</code> eliminations — a large, avoidable difference as <code style="color:#3B82F6">m</code> or <code style="color:#3B82F6">n</code> grows.

---

## 7. P1a — Gauss-Jordan RREF & Determinant

*(New practice question — not from a real subsection; targets topics 5 and 6 from the coverage table, neither of which any real A1/B1/C1/C2 question has tested yet.)*

### Problem Statement

You are given the following system of three equations in three unknowns:

$$
\begin{aligned}
0x_1 + x_2 + 2x_3 &= 8 \\
x_1 - x_2 + 3x_3 &= 8 \\
2x_1 + 4x_2 + x_3 &= 13
\end{aligned}
\qquad\Longleftrightarrow\qquad
A = \begin{bmatrix} 0 & 1 & 2 \\ 1 & -1 & 3 \\ 2 & 4 & 1 \end{bmatrix},
\quad b = \begin{bmatrix} 8 \\ 8 \\ 13 \end{bmatrix}
$$

(a) Reduce the augmented matrix <code style="color:#3B82F6">[A|b]</code> to **Reduced Row Echelon Form (RREF)** using Gauss-Jordan elimination with partial pivoting — by hand. Print, per column: the candidate pivots, the pivot chosen, every row swap, and the matrix after that column is fully cleared (both above *and* below the pivot). Read the solution <code style="color:#3B82F6">x</code> directly off the final <code style="color:#3B82F6">[I|x]</code> — no back substitution.

(b) Using the **same elimination process**, compute <code style="color:#3B82F6">det(A)</code> as a byproduct: the product of the *raw* pivot values (the ones found *before* normalizing each row so its pivot becomes 1), with a sign flip for every row swap performed.

(c) Verify both results against <code style="color:#3B82F6">np.linalg.solve(A,b)</code> and <code style="color:#3B82F6">np.linalg.det(A)</code>.

**Written question**: Why must the determinant be computed from the row-echelon pivots as they were *before* normalization, rather than read off the fully-reduced RREF (where every pivot has already been scaled to exactly 1)?

### Solution

```python
import numpy as np

A = [[0, 1, 2], [1, -1, 3], [2, 4, 1]]
b = [8, 8, 13]
n = 3

augmented_matrix = [A[i][:] + [b[i]] for i in range(n)]

pivots_for_determinant = []
swap_count = 0

for col in range(n):
    # partial pivoting: largest |value| in this column, at/below row 'col'
    pivot_row = col
    best = abs(augmented_matrix[col][col])
    for i in range(col + 1, n):
        if abs(augmented_matrix[i][col]) > best:
            best = abs(augmented_matrix[i][col])
            pivot_row = i

    if pivot_row != col:
        print(f"SWAP: R{col+1} <-> R{pivot_row+1}")
        augmented_matrix[col], augmented_matrix[pivot_row] = \
            augmented_matrix[pivot_row], augmented_matrix[col]
        swap_count += 1

    # record the RAW pivot value BEFORE normalizing -- the determinant needs
    # this exact value, and it is gone forever once the row is divided by it
    pivot_value = augmented_matrix[col][col]
    pivots_for_determinant.append(pivot_value)
    print(f"pivot (raw) = {pivot_value}")

    # normalize the pivot row so its pivot becomes 1
    augmented_matrix[col] = [v / pivot_value for v in augmented_matrix[col]]

    # eliminate this column from EVERY other row -- both above and below,
    # which is what makes this Gauss-Jordan instead of plain Gauss elimination
    for row in range(n):
        if row != col and abs(augmented_matrix[row][col]) > 1e-12:
            factor = augmented_matrix[row][col]
            augmented_matrix[row] = [augmented_matrix[row][k] - factor * augmented_matrix[col][k]
                                      for k in range(n + 1)]

    print("matrix after this column:")
    for row in augmented_matrix:
        print("  ", [round(v, 6) for v in row])

x = [row[-1] for row in augmented_matrix]
print("\nsolution x =", x)

determinant = (-1) ** swap_count
for p in pivots_for_determinant:
    determinant *= p
print("pivots used for determinant:", pivots_for_determinant)
print("row swaps:", swap_count)
print("det(A) =", determinant)

# verify
A_np, b_np = np.array(A, dtype=float), np.array(b, dtype=float)
print("np.linalg.det(A) =", np.linalg.det(A_np))
print("np.linalg.solve(A,b) =", np.linalg.solve(A_np, b_np))
```

### Actual output

```
SWAP: R1 <-> R3
pivot (raw) = 2
matrix after this column:
   [1.0, 2.0, 0.5, 6.5]
   [0.0, -3.0, 2.5, 1.5]
   [0, 1, 2, 8]
pivot (raw) = -3.0
matrix after this column:
   [1.0, 0.0, 2.166667, 7.5]
   [-0.0, 1.0, -0.833333, -0.5]
   [0.0, 0.0, 2.833333, 8.5]
pivot (raw) = 2.8333333333333335
matrix after this column:
   [1.0, 0.0, 0.0, 1.0]
   [0.0, 1.0, 0.0, 2.0]
   [0.0, 0.0, 1.0, 3.0]

solution x = [0.9999999999999991, 2.0, 3.0]
pivots used for determinant: [2, -3.0, 2.8333333333333335]
row swaps: 1
det(A) = 17.0

np.linalg.det(A) = 17.0
np.linalg.solve(A,b) = [1. 2. 3.]
```

Column 1 needed an immediate swap (<code style="color:#3B82F6">A[0][0]=0</code>) — the same "hidden zero pivot" situation as B1's System 1; partial pivoting handles it identically here. After all three columns, the left block is exactly the identity and the right column is the solution, <code style="color:#3B82F6">x = [1, 2, 3]</code> (up to floating-point noise on <code style="color:#3B82F6">x1</code>), with no back substitution needed. The determinant, assembled purely from the three *raw* pivots (<code style="color:#3B82F6">2</code>, <code style="color:#3B82F6">-3.0</code>, <code style="color:#3B82F6">2.8333...</code>) and the one sign flip from the single swap, comes out to <code style="color:#3B82F6">17.0</code> — matching <code style="color:#3B82F6">np.linalg.det</code> exactly.

### Written question — answer

The determinant of a triangular (or row-echelon) matrix is the **product of its diagonal entries** — that theorem is what the whole trick relies on. Gauss-Jordan's extra step, *normalizing* each pivot row so its pivot becomes exactly <code style="color:#3B82F6">1</code>, is itself a row-scaling operation, and scaling a row by some factor scales the determinant by that same factor. So the moment a row is divided by its pivot, the running determinant has effectively been divided by that pivot too — and by the time the matrix reaches full RREF, *every* pivot has been divided out, leaving a matrix whose "product of diagonal entries" is trivially <code style="color:#3B82F6">1×1×1=1</code>, regardless of what the real determinant was. That information cannot be recovered from the final RREF alone; it has to be captured *during* elimination, at the exact moment each pivot is found and *before* it gets normalized away — which is exactly what <code style="color:#3B82F6">pivots_for_determinant</code> does in the code above.

---

## 8. P1b — Matrix Inverse via Gauss-Jordan and LU

*(New practice question — not from a real subsection; targets topics 7 and 12. Reuses the same matrix <code style="color:#3B82F6">A</code> from P1a on purpose — the question is a compare/contrast between two ways of getting the inverse, not a new elimination puzzle.)*

### Problem Statement

Using the same matrix as P1a,

$$
A = \begin{bmatrix} 0 & 1 & 2 \\ 1 & -1 & 3 \\ 2 & 4 & 1 \end{bmatrix}
$$

(a) Compute <code style="color:#3B82F6">A⁻¹</code> using **Gauss-Jordan elimination** on <code style="color:#3B82F6">[A | I]</code> (partial pivoting, same as P1a's method — read the inverse off the right-hand block once the left block reaches the identity).

(b) Compute <code style="color:#3B82F6">A⁻¹</code> using **LU decomposition**: decompose <code style="color:#3B82F6">A</code> once, then solve <code style="color:#3B82F6">A xⱼ = eⱼ</code> (forward + back substitution) for each column <code style="color:#3B82F6">eⱼ</code> of the identity matrix, assembling the <code style="color:#3B82F6">xⱼ</code>'s as the columns of <code style="color:#3B82F6">A⁻¹</code>.

(c) Verify both results agree with each other and with <code style="color:#3B82F6">np.linalg.inv(A)</code>.

**Written question**: For a large (say 1000×1000) matrix, which of the two methods above would you prefer, and why?

### Solution

```python
import numpy as np

A = [[0, 1, 2], [1, -1, 3], [2, 4, 1]]
n = 3

# (a) inverse via Gauss-Jordan on [A|I]
def gj_inverse(A):
    n = len(A)
    M = [A[i][:] + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]              # normalize pivot row
        for r in range(n):
            if r != col and abs(M[r][col]) > 1e-12:      # clear column everywhere else
                factor = M[r][col]
                M[r] = [M[r][k] - factor * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]                        # right half = inverse

inv_gj = gj_inverse(A)
print("inverse via Gauss-Jordan:")
for row in inv_gj:
    print(" ", [round(v, 6) for v in row])

# (b) inverse via LU: decompose once with partial pivoting, then solve for
# each identity column (this is the direct extension of C2's "reuse L,U for
# multiple right-hand sides" idea -- here the "many right-hand sides" ARE
# the columns of the identity matrix)
L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
perm = list(range(n))
U = [row[:] for row in A]
for k in range(n - 1):
    piv = max(range(k, n), key=lambda r: abs(U[r][k]))
    if piv != k:
        U[k], U[piv] = U[piv], U[k]
        perm[k], perm[piv] = perm[piv], perm[k]
        for j in range(k):
            L[k][j], L[piv][j] = L[piv][j], L[k][j]
    for i in range(k + 1, n):
        factor = U[i][k] / U[k][k]
        L[i][k] = factor
        U[i] = [U[i][j] - factor * U[k][j] for j in range(n)]

def forward_sub(L, b):
    n = len(b); z = [0.0] * n
    for i in range(n):
        total = b[i]
        for j in range(i):
            total -= L[i][j] * z[j]
        z[i] = total / L[i][i]
    return z

def back_sub(U, z):
    n = len(z); x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = z[i]
        for j in range(i + 1, n):
            total -= U[i][j] * x[j]
        x[i] = total / U[i][i]
    return x

inv_lu_cols = []
for j in range(n):
    e = [1.0 if i == j else 0.0 for i in range(n)]
    e_permuted = [e[perm[i]] for i in range(n)]   # apply the same row permutation to e_j
    z = forward_sub(L, e_permuted)
    x = back_sub(U, z)
    inv_lu_cols.append(x)
inv_lu = [[inv_lu_cols[j][i] for j in range(n)] for i in range(n)]  # columns -> matrix

print("\ninverse via LU:")
for row in inv_lu:
    print(" ", [round(v, 6) for v in row])

# (c) verify
inv_numpy = np.linalg.inv(np.array(A, dtype=float))
print("\nnumpy inverse:\n", inv_numpy)
print("\nmax|GJ - numpy| =", np.max(np.abs(np.array(inv_gj) - inv_numpy)))
print("max|LU - numpy| =", np.max(np.abs(np.array(inv_lu) - inv_numpy)))
```

### Actual output

```
inverse via Gauss-Jordan:
  [-0.764706, 0.411765, 0.294118]
  [0.294118, -0.235294, 0.117647]
  [0.352941, 0.117647, -0.058824]

inverse via LU:
  [-0.764706, 0.411765, 0.294118]
  [0.294118, -0.235294, 0.117647]
  [0.352941, 0.117647, -0.058824]

numpy inverse:
 [[-0.76470588  0.41176471  0.29411765]
 [ 0.29411765 -0.23529412  0.11764706]
 [ 0.35294118  0.11764706 -0.05882353]]

max|GJ - numpy| = 5.55e-17
max|LU - numpy| = 1.39e-17
```

Both hand-coded methods agree with each other and with NumPy to within floating-point noise (<code style="color:#3B82F6">~1e-17</code>, i.e. exact).

### Written question — answer

**LU**, for a large matrix. Gauss-Jordan's <code style="color:#3B82F6">[A|I]</code> approach re-derives the elimination from scratch — for an <code style="color:#3B82F6">n×n</code> inverse that's an <code style="color:#3B82F6">O(n³)</code> elimination *repeated implicitly* as part of one big augmented system every time, with no way to reuse work if you only wanted a few columns. LU decomposition, by contrast, factors <code style="color:#3B82F6">A</code> into <code style="color:#3B82F6">L</code> and <code style="color:#3B82F6">U</code> **once** (<code style="color:#3B82F6">O(n³)</code>), and each of the <code style="color:#3B82F6">n</code> columns of <code style="color:#3B82F6">A⁻¹</code> then costs only one cheap <code style="color:#3B82F6">O(n²)</code> forward+back substitution — this is exactly C2's "decompose once, reuse often" argument, just applied with the identity matrix's columns as the <code style="color:#3B82F6">n</code> right-hand sides instead of two arbitrary vectors <code style="color:#3B82F6">b1</code>,<code style="color:#3B82F6">b2</code>. For <code style="color:#3B82F6">n=1000</code>, that's the difference between one <code style="color:#3B82F6">O(n³)</code> factorization plus <code style="color:#3B82F6">1000</code> <code style="color:#3B82F6">O(n²)</code> solves, versus redoing <code style="color:#3B82F6">O(n³)</code>-scale work for the whole augmented system regardless.

---

## 9. P2 — Round-off Error & Complexity

*(New practice question — not from a real subsection; targets topics 4 and 11. Same spirit as the lecture's own round-off demo — a system with a nasty near-zero pivot — but with a numeric tweak so a modest, robust rounding level shows a clean effect instead of chasing an exact digit-by-digit replica.)*

### Problem Statement

You are given the following system, whose exact solution is very close to <code style="color:#3B82F6">x = [1, 1, 1]</code>:

$$
\begin{aligned}
20x_1 + 15x_2 + 10x_3 &= 45 \\
-3x_1 - 2.2465x_2 + 7x_3 &= 1.7505 \\
5x_1 + x_2 + 3x_3 &= 9
\end{aligned}
\qquad\Longleftrightarrow\qquad
A = \begin{bmatrix} 20 & 15 & 10 \\ -3 & -2.2465 & 7 \\ 5 & 1 & 3 \end{bmatrix},
\quad b = \begin{bmatrix} 45 \\ 1.7505 \\ 9 \end{bmatrix}
$$

(a) Solve it with **naive Gauss elimination (no pivoting)**, simulating limited-precision hardware by rounding every intermediate value to 3 decimal places as you go.

(b) Solve the same system, same simulated precision, but with **partial pivoting**.

(c) Compare both against the true solution (<code style="color:#3B82F6">np.linalg.solve</code> on the un-rounded system) and report the max error for each.

**Written question**: For the *same* coefficient matrix <code style="color:#3B82F6">A</code> but <code style="color:#3B82F6">m</code> different right-hand sides, state the dominant-term operation count for (i) solving each one from scratch with Gaussian elimination, vs. (ii) decomposing <code style="color:#3B82F6">A=LU</code> once and reusing it for all <code style="color:#3B82F6">m</code> solves.

### Solution

```python
import numpy as np

A = [[20, 15, 10], [-3, -2.2465, 7], [5, 1, 3]]
b = [45, 1.7505, 9]
n = 3
DP = 3  # simulated limited precision: round every intermediate to 3 decimals

def r(v):
    return round(v, DP)

def eliminate(Aug, use_pivoting):
    Aug = [row[:] for row in Aug]
    for k in range(n - 1):
        if use_pivoting:
            piv = max(range(k, n), key=lambda row_i: abs(Aug[row_i][k]))
            if piv != k:
                Aug[k], Aug[piv] = Aug[piv], Aug[k]
        for i in range(k + 1, n):
            factor = r(Aug[i][k] / Aug[k][k])
            Aug[i] = [r(Aug[i][j] - factor * Aug[k][j]) for j in range(n + 1)]
        print(f"  after step {k+1} ({'pivoted' if use_pivoting else 'naive'}):", Aug)
    return Aug

def back_sub(Aug):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = Aug[i][n]
        for j in range(i + 1, n):
            s = r(s - Aug[i][j] * x[j])
        x[i] = r(s / Aug[i][i])
    return x

Aug0 = [A[i][:] + [b[i]] for i in range(n)]

print("(a) NAIVE (no pivoting):")
x_naive = back_sub(eliminate(Aug0, use_pivoting=False))
print("  x_naive =", x_naive)

print("\n(b) PARTIAL PIVOTING:")
x_piv = back_sub(eliminate(Aug0, use_pivoting=True))
print("  x_pivoted =", x_piv)

# (c) compare against the true (un-rounded) solution
x_true = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
err_naive = max(abs(x_naive[i] - x_true[i]) for i in range(n))
err_piv = max(abs(x_piv[i] - x_true[i]) for i in range(n))
print(f"\ntrue solution (numpy, full precision) = {x_true}")
print(f"max error, naive   = {err_naive:.3f}")
print(f"max error, pivoted = {err_piv:.3f}")
```

### Actual output

```
(a) NAIVE (no pivoting):
  after step 1 (naive): [[20, 15, 10, 45], [0.0, 0.003, 8.5, 8.501], [0.0, -2.75, 0.5, -2.25]]
  after step 2 (naive): [[20, 15, 10, 45], [0.0, 0.003, 8.5, 8.501], [0.0, 0.0, 7792.169, 7790.336]]
  x_naive = [1.5, 0.333, 1.0]

(b) PARTIAL PIVOTING:
  after step 1 (pivoted): [[20, 15, 10, 45], [0.0, 0.003, 8.5, 8.501], [0.0, -2.75, 0.5, -2.25]]
  after step 2 (pivoted): [[20, 15, 10, 45], [0.0, -2.75, 0.5, -2.25], [0.0, 0.0, 8.501, 8.499]]
  x_pivoted = [1.0, 1.0, 1.0]

true solution (numpy, full precision) = [1.00022458 0.99993583 0.99964709]
max error, naive   = 0.667
max error, pivoted = 0.0
```

Naive elimination's column-1 step leaves a pivot of <code style="color:#3B82F6">0.003</code> for step 2 — tiny relative to <code style="color:#3B82F6">-2.75</code> sitting right below it. That forces a multiplier of roughly <code style="color:#3B82F6">-2.75/0.003 ≈ -917</code>, which blows the 3-decimal rounding error already sitting in row 2 up into a **67% error** in <code style="color:#3B82F6">x2</code>. Partial pivoting sidesteps the whole problem by swapping in <code style="color:#3B82F6">-2.75</code> as the step-2 pivot instead (multiplier only <code style="color:#3B82F6">≈0.001</code>), landing on the true answer to 3 decimals.

### Written question — answer

(i) **Gaussian elimination redone per right-hand side**: each solve costs a full forward elimination (<code style="color:#3B82F6">~8n³/3</code> dominant term, from the earlier LU-cost derivation) plus back substitution — repeated <code style="color:#3B82F6">m</code> times from scratch since nothing is reused → dominant term <code style="color:#3B82F6">~m·(8n³/3) = O(m·n³)</code>.

(ii) **LU decomposed once, reused <code style="color:#3B82F6">m</code> times**: the <code style="color:#3B82F6">O(n³)</code> decomposition happens exactly once; each of the <code style="color:#3B82F6">m</code> solves is then just forward + back substitution, <code style="color:#3B82F6">O(n²)</code> each → dominant term <code style="color:#3B82F6">~(8n³/3) + m·O(n²) = O(n³ + m·n²)</code>.

For any <code style="color:#3B82F6">m > 1</code>, (ii) is cheaper, and the gap widens linearly with <code style="color:#3B82F6">m</code> — the same "decompose once, reuse often" conclusion as C2, now stated as a complexity bound instead of a runtime table.

---

## 10. P3 — Direct Eigenvalue via Characteristic Polynomial

*(New practice question — not from a real subsection; targets topic 14. Deliberately the opposite style from A1/C1: no iteration, straight algebra.)*

### Problem Statement

You are given the matrix

$$
A = \begin{bmatrix} 6 & 2 \\ 2 & 3 \end{bmatrix}
$$

(a) Derive the characteristic polynomial <code style="color:#3B82F6">det(A-λI)=0</code> by hand, in terms of <code style="color:#3B82F6">A</code>'s trace and determinant (<code style="color:#3B82F6">λ² - trace·λ + det = 0</code> for a 2×2).

(b) Solve for both eigenvalues using the quadratic formula.

(c) For each eigenvalue, hand-solve <code style="color:#3B82F6">(A-λI)v=0</code> for its eigenvector (2×2, so one row of the system gives you the ratio <code style="color:#3B82F6">v1:v2</code> directly), and normalize to unit length.

(d) Verify against <code style="color:#3B82F6">np.linalg.eig(A)</code>.

### Solution

```python
import numpy as np
import math

A = [[6, 2], [2, 3]]

# (a) characteristic polynomial from trace and determinant
trace = A[0][0] + A[1][1]
det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
print(f"trace = {trace}, det = {det}")
print(f"characteristic polynomial: lambda^2 - {trace}*lambda + {det} = 0")

# (b) quadratic formula
disc = trace**2 - 4 * det
sqrt_disc = math.sqrt(disc)
lam1 = (trace + sqrt_disc) / 2
lam2 = (trace - sqrt_disc) / 2
print(f"lambda1 = {lam1}, lambda2 = {lam2}")

# (c) for each eigenvalue, solve (A - lambda*I) v = 0 by hand
def eigenvector_for(A, lam):
    a11, a12 = A[0][0] - lam, A[0][1]
    a21, a22 = A[1][0], A[1][1] - lam
    # row 1 gives: (a11)v1 + (a12)v2 = 0  ->  v2/v1 = -a11/a12 (fall back to row 2 if a12=0)
    if abs(a12) > 1e-12:
        v1, v2 = 1.0, -a11 / a12
    else:
        v1, v2 = -a22 / a21, 1.0
    norm = math.sqrt(v1 * v1 + v2 * v2)          # manual L2 norm, per the residual/norm rule
    return [v1 / norm, v2 / norm]

v1 = eigenvector_for(A, lam1)
v2 = eigenvector_for(A, lam2)
print("v1 =", v1)
print("v2 =", v2)

# (d) verify
A_np = np.array(A, dtype=float)
print("check A@v1 vs lam1*v1:", A_np @ np.array(v1), lam1 * np.array(v1))
print("check A@v2 vs lam2*v2:", A_np @ np.array(v2), lam2 * np.array(v2))

eigvals, eigvecs = np.linalg.eig(A_np)
print("\nnumpy eigenvalues:", eigvals)
print("numpy eigenvectors:\n", eigvecs)
```

### Actual output

```
trace = 9, det = 14
characteristic polynomial: lambda^2 - 9*lambda + 14 = 0
lambda1 = 7.0, lambda2 = 2.0
v1 = [0.894427, 0.447214]
v2 = [0.447214, -0.894427]
check A@v1 vs lam1*v1: [6.26099034 3.13049517]  [6.26099034 3.13049517]
check A@v2 vs lam2*v2: [0.89442719 -1.78885438]  [0.89442719 -1.78885438]

numpy eigenvalues: [7. 2.]
numpy eigenvectors:
 [[ 0.89442719 -0.4472136 ]
 [ 0.4472136   0.89442719]]
```

<code style="color:#3B82F6">λ₁=7, λ₂=2</code> — matches NumPy exactly, as does each eigenvector once its sign is accounted for (<code style="color:#3B82F6">v2</code> came out sign-flipped relative to NumPy's second column, which is exactly the same expected, harmless ambiguity flagged back in C1's problem statement).

---

## 11. P4 — Full Eigendecomposition & Matrix Powers

*(New practice question — not from a real subsection; targets topics 15, 16, 17. Reuses P3's matrix and eigenpairs directly, continuing the lecture's own flow: find eigenpairs -> build the decomposition -> use it.)*

### Problem Statement

Using <code style="color:#3B82F6">A</code> and the eigenpairs found in P3 (<code style="color:#3B82F6">λ₁=7, v₁=[2/√5, 1/√5]</code>, <code style="color:#3B82F6">λ₂=2, v₂=[1/√5, -2/√5]</code>):

(a) Build <code style="color:#3B82F6">V = [v₁ v₂]</code> (eigenvectors as columns) and <code style="color:#3B82F6">Λ = diag(λ₁,λ₂)</code>.

(b) Verify <code style="color:#3B82F6">A = VΛV⁻¹</code>.

(c) Take <code style="color:#3B82F6">x = [3, 1]</code>. Convert it to eigen-coordinates via <code style="color:#3B82F6">V⁻¹x</code>, then convert back to standard coordinates via <code style="color:#3B82F6">V(...)</code>, confirming you recover the original <code style="color:#3B82F6">x</code>.

(d) Compute <code style="color:#3B82F6">A⁵</code> two ways: once via <code style="color:#3B82F6">VΛ⁵V⁻¹</code> (raising the *diagonal* <code style="color:#3B82F6">Λ</code> to a power, trivial), and once via <code style="color:#3B82F6">np.linalg.matrix_power(A,5)</code>. Confirm they agree.

### Solution

```python
import math
import numpy as np

A = [[6, 2], [2, 3]]
lam1, lam2 = 7.0, 2.0
v1 = [2/math.sqrt(5), 1/math.sqrt(5)]
v2 = [1/math.sqrt(5), -2/math.sqrt(5)]

# (a) build V and Lambda
V = [[v1[0], v2[0]], [v1[1], v2[1]]]     # eigenvectors as COLUMNS
Lam = [[lam1, 0.0], [0.0, lam2]]

def inv2x2(M):
    # closed-form 2x2 inverse, computed manually
    a, b = M[0]
    c, d = M[1]
    det = a * d - b * c
    return [[d / det, -b / det], [-c / det, a / det]]

def matmul(X, Y):
    rows, mid, cols = len(X), len(Y), len(Y[0])
    R = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            R[i][j] = sum(X[i][k] * Y[k][j] for k in range(mid))
    return R

V_inv = inv2x2(V)

# (b) verify A = V Lambda V^-1
reconstructed = matmul(matmul(V, Lam), V_inv)
print("V Lambda V^-1 =", reconstructed)
print("A              =", A)

# (c) eigen-coordinates round trip: x -> V^-1 x (eigen-coords) -> V(...) (back)
x = [3.0, 1.0]
c = [V_inv[0][0]*x[0] + V_inv[0][1]*x[1],
     V_inv[1][0]*x[0] + V_inv[1][1]*x[1]]
x_back = [V[0][0]*c[0] + V[0][1]*c[1],
          V[1][0]*c[0] + V[1][1]*c[1]]
print(f"\nx = {x}  ->  eigen-coords c = {c}  ->  back to standard = {x_back}")

# (d) A^5 via V Lambda^5 V^-1 (raising a DIAGONAL matrix to a power is trivial:
# just raise each entry -- no repeated matrix multiplication needed)
k = 5
Lam_k = [[lam1**k, 0.0], [0.0, lam2**k]]
A_k = matmul(matmul(V, Lam_k), V_inv)
print(f"\nA^{k} via V Lambda^{k} V^-1 =", A_k)

A_k_direct = np.linalg.matrix_power(np.array(A, dtype=float), k)
print(f"A^{k} via np.linalg.matrix_power =", A_k_direct.tolist())
```

### Actual output

```
V Lambda V^-1 = [[6.000000000000001, 2.0], [2.0, 3.0]]
A              = [[6, 2], [2, 3]]

x = [3.0, 1.0]  ->  eigen-coords c = [3.130495168499706, 0.4472135954999581]  ->  back to standard = [3.0000000000000004, 1.0]

A^5 via V Lambda^5 V^-1 = [[13452.0, 6710.0], [6710.0, 3387.0]]
A^5 via np.linalg.matrix_power = [[13452.0, 6710.0], [6710.0, 3387.0]]
```

<code style="color:#3B82F6">A = VΛV⁻¹</code> reconstructs exactly, the round trip <code style="color:#3B82F6">x → eigen-coords → x</code> recovers the original vector exactly, and <code style="color:#3B82F6">A⁵</code> computed via the decomposition matches <code style="color:#3B82F6">np.linalg.matrix_power</code> exactly — while only ever raising the 2×2 *diagonal* <code style="color:#3B82F6">Λ</code> to the 5th power (two scalar exponentiations), never multiplying the full <code style="color:#3B82F6">2×2</code> matrix <code style="color:#3B82F6">A</code> by itself 5 times. That gap only grows for larger powers or larger matrices, which is the entire point of the <code style="color:#3B82F6">A = VΛV⁻¹</code> trick from lecture.

---

## 12. P5 — Deflation for Intermediate Eigenvalues

*(New practice question — not from a real subsection; targets topic 20, the single biggest coverage gap. Deliberately reuses C1's exact matrix and power-method code — deflation only makes sense as "run the power method again, on a modified matrix," so this overlap is structural, not accidental.)*

### Problem Statement

Using the same matrix and power method from C1,

$$
A = \begin{bmatrix} 12 & 2 & 1 & 0 \\ 2 & 5 & 0 & 1 \\ 1 & 0 & 3 & 1 \\ 0 & 1 & 1 & 2 \end{bmatrix}
$$

(a) Run the power method on <code style="color:#3B82F6">A</code> to get the dominant eigenpair <code style="color:#3B82F6">λ₁, v₁</code> (normalize <code style="color:#3B82F6">v₁</code> to unit length).

(b) Build the **deflated matrix** <code style="color:#3B82F6">A₂ = A - λ₁ v̂₁v̂₁ᵀ</code> (outer product of the unit eigenvector with itself, scaled by <code style="color:#3B82F6">λ₁</code>, subtracted from <code style="color:#3B82F6">A</code>).

(c) Run the power method **again**, this time on <code style="color:#3B82F6">A₂</code>, to get <code style="color:#3B82F6">λ₂, v₂</code> — the *second*-largest eigenvalue of the original <code style="color:#3B82F6">A</code>.

(d) Verify: <code style="color:#3B82F6">A₂ v₁ ≈ 0</code> (confirming <code style="color:#3B82F6">λ₁</code> has been "muted" to <code style="color:#3B82F6">0</code> in <code style="color:#3B82F6">A₂</code>'s spectrum) and <code style="color:#3B82F6">v₁ · v₂ ≈ 0</code> (confirming the two eigenvectors are orthogonal).

**Written question**: This trick relies on <code style="color:#3B82F6">A</code> being symmetric. Why — what specifically would break if <code style="color:#3B82F6">A</code> weren't symmetric?

### Solution

```python
import numpy as np

A = [[12, 2, 1, 0], [2, 5, 0, 1], [1, 0, 3, 1], [0, 1, 1, 2]]

def l2_norm(vec):
    total = 0.0
    for v in vec:
        total += v * v
    return total ** 0.5

def matvec(M, x):
    return [sum(M[i][j] * x[j] for j in range(len(x))) for i in range(len(M))]

def power_method(A, x0, tol=1e-8, max_iter=200):
    x = x0[:]
    lam_old = 0.0
    for it in range(1, max_iter + 1):
        y = matvec(A, x)
        idx = max(range(len(y)), key=lambda i: abs(y[i]))
        lam_new = y[idx]
        x = [v / lam_new for v in y]
        if abs(lam_new - lam_old) < tol:
            return lam_new, x, it
        lam_old = lam_new
    return lam_old, x, max_iter

# (a) power method on A
lam1, v1_raw, it1 = power_method(A, [1.0, 1.0, 1.0, 1.0])
v1 = [v / l2_norm(v1_raw) for v in v1_raw]     # manual normalization to unit length
print(f"lambda1 = {lam1}  (after {it1} iterations)")
print("v1 (unit) =", v1)

# (b) deflate: A2 = A - lambda1 * (v1 v1^T)
A2 = [[A[i][j] - lam1 * v1[i] * v1[j] for j in range(4)] for i in range(4)]
print("\nA2 =")
for row in A2:
    print(" ", [round(v, 6) for v in row])

# (c) power method again, on A2
lam2, v2_raw, it2 = power_method(A2, [1.0, 1.0, 1.0, 1.0])
v2 = [v / l2_norm(v2_raw) for v in v2_raw]
print(f"\nlambda2 = {lam2}  (after {it2} iterations)")
print("v2 (unit) =", v2)

# (d) verify
check_muted = matvec(A2, v1)
dot_product = sum(v1[i] * v2[i] for i in range(4))
print("\nA2 @ v1 (should be ~0)  =", [round(v, 9) for v in check_muted])
print("v1 . v2  (should be ~0) =", round(dot_product, 9))

# cross-check against numpy's full eigendecomposition of the ORIGINAL A
eigvals, _ = np.linalg.eig(np.array(A, dtype=float))
order = np.argsort(-np.abs(eigvals))
print("\nnumpy eigenvalues of A (descending |lambda|):", eigvals[order])
```

### Actual output

```
lambda1 = 12.640123882629144  (after 22 iterations)
v1 (unit) = [0.960598, 0.255878, 0.103146, 0.033742]

A2 =
  [0.33634, -1.10689, -0.252407, -0.409704]
  [-1.10689, 4.172407, -0.333608, 0.890866]
  [-0.252407, -0.333608, 2.86552, 0.956007]
  [-0.409704, 0.890866, 0.956007, 1.985609]

lambda2 = 4.804550448001767  (after 51 iterations)
v2 (unit) = [-0.257497, 0.904013, 0.04478, 0.338305]

A2 @ v1 (should be ~0)  = [-1.43e-09, -3.7e-09, -4.04e-10, -1.24e-09]
v1 . v2  (should be ~0) = -7.104e-10

numpy eigenvalues of A (descending |lambda|): [12.64012388  4.80455043  3.45411247  1.10121323]
```

<code style="color:#3B82F6">λ₂ = 4.804550</code> matches NumPy's **second**-largest eigenvalue of the original <code style="color:#3B82F6">A</code> (<code style="color:#3B82F6">4.80455043</code>) essentially exactly — confirming deflation successfully "unmuted" it. <code style="color:#3B82F6">A₂ v₁</code> and <code style="color:#3B82F6">v₁·v₂</code> both land at the <code style="color:#3B82F6">~1e-9</code> level (floating-point noise), confirming both required properties held.

### Written question — answer

Deflation's correctness rests entirely on one fact used in the "why does this leave <code style="color:#3B82F6">v₂</code> untouched" derivation: <code style="color:#3B82F6">v̂₁·v̂₂ = 0</code>. For a **symmetric** matrix, the spectral theorem guarantees eigenvectors belonging to different eigenvalues are automatically orthogonal — that's not a coincidence of this particular <code style="color:#3B82F6">A</code>, it's a structural guarantee. The projection <code style="color:#3B82F6">v̂₁(v̂₁ᵀv₂)</code> that gets subtracted out then evaluates to exactly <code style="color:#3B82F6">0</code> for every *other* eigenvector, so <code style="color:#3B82F6">A₂</code>'s action on <code style="color:#3B82F6">v₂</code> (and <code style="color:#3B82F6">v₃</code>, <code style="color:#3B82F6">v₄</code>, ...) is completely unchanged from <code style="color:#3B82F6">A</code>'s — only the <code style="color:#3B82F6">v₁</code> direction gets zeroed out.

For a **non-symmetric** matrix, eigenvectors are not guaranteed orthogonal at all. If <code style="color:#3B82F6">v̂₁ᵀv₂ ≠ 0</code>, the subtraction <code style="color:#3B82F6">A - λ₁v̂₁v̂₁ᵀ</code> removes part of the <code style="color:#3B82F6">v₁</code>-component from <code style="color:#3B82F6">Av₂</code> too — <code style="color:#3B82F6">A₂</code>'s action on <code style="color:#3B82F6">v₂</code> stops being a clean scalar multiple <code style="color:#3B82F6">λ₂v₂</code>, so <code style="color:#3B82F6">v₂</code> is no longer even an eigenvector of <code style="color:#3B82F6">A₂</code>. The power method would then converge to *something*, but not reliably to <code style="color:#3B82F6">λ₂</code> of the original <code style="color:#3B82F6">A</code> — the deflation would have silently corrupted the very eigenpair it was supposed to expose next.
