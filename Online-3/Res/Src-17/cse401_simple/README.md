# CSE 401: Numerical Analysis, Simulation & Modeling Sessional (Simple Implementations)

This folder contains clean, beginner-friendly, and student-oriented Python implementations of the key simulation algorithms covered in the slide decks.

---

## 📁 Files Overview

| File | Core Algorithms Covered |
| :--- | :--- |
| **`rng_simple.py`** | • Linear Congruential Method (Mixed & Multiplicative)<br>• Period detection loop<br>• Maximum period condition checker (Cases 1, 2, 3)<br>• Slide Examples 1–4 |
| **`rng_tests_simple.py`** | • Kolmogorov-Smirnov (K-S) Uniformity Test<br>• Chi-Square Frequency Test<br>• Autocorrelation Independence Test<br>• Full table traces & Slide Examples 6–8 |
| **`monte_carlo_simple.py`** | • Monte Carlo Sample-Mean Integration: $\bar{Y} = (b-a)\frac{1}{n}\sum g(X_i)$<br>• Hit-or-Miss estimation of $\pi$<br>• 95% Confidence Intervals & $1/\sqrt{n}$ Error convergence |
| **`des_single_server_simple.py`** | • Next-Event Time-Advance Single-Server Queue<br>• Trace output, Average delay $d(n)$, Time-average in queue $q(n)$, Utilization $u(n)$<br>• Verification with 6-delay slide example |
| **`middle_square_simple.py`** | • 4-digit Middle-Square RNG<br>• Degenerate seed analysis & cycles<br>• Chi-square uniformity testing |

---

## 🚀 How to Run

Run any file directly in terminal:
```bash
python cse401_simple/rng_simple.py
python cse401_simple/rng_tests_simple.py
python cse401_simple/monte_carlo_simple.py
python cse401_simple/des_single_server_simple.py
python cse401_simple/middle_square_simple.py
```
