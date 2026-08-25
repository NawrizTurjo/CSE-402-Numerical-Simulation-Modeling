# Practice Questions (Simple Implementations)

Clean, simplified, exam-ready Python implementations for all 6 practice problems.

---

## 📁 Files & Topics Overview

| File | Topic & Exam Key Concepts |
| :--- | :--- |
| **`q1_lcg_investigation_simple.py`** | • LCG period calculations (e.g. period 4 for $17X+43 \pmod{100}$)<br>• Why odd seeds give maximum period $m/4$ for multiplicative LCG<br>• IBM's **RANDU** 3D hyperplane defect ($9X_i - 6X_{i+1} + X_{i+2} = 0$) |
| **`q2_middle_square_variants_simple.py`** | • $d$-digit Middle-Square scaling survey ($d=2, 4, 6$)<br>• Why increasing digits cannot fix it (birthday bound $\sim O(\sqrt{10^d})$)<br>• **Weyl Sequence repair** ($w_{i+1} = w_i + s$) |
| **`q3_test_battery_simple.py`** | • 4 streams ($A$: good LCG, $B$: ramp, $C$: middle-square, $D$: $\sqrt{\text{LCG}}$)<br>• Pass/Fail matrix for Chi-Square, K-S, Autocorrelation (lags 1, 4, 5), and Runs tests<br>• Demonstrates why testing both Uniformity AND Independence is required |
| **`q4_inverse_transform_simple.py`** | • Inverse CDF: Uniform, Exponential, Weibull, Triangular, Discrete<br>• Verification by moments (mean and variance)<br>• **Equiprobable bins** for Chi-Square test of non-uniform variates |
| **`q5_monte_carlo_simple.py`** | • Sample-mean integration with 95% Confidence Intervals<br>• $\pi$ Hit-or-Miss estimation & $1/\sqrt{n}$ error convergence law<br>• Monte Carlo vs Simpson's rule comparison in 1D vs high dimensions |
| **`q6_queue_simulation_simple.py`** | • Single-server barber shop next-event DES<br>• Full event trace table<br>• Performance metrics $d(n), q(n), u(n)$<br>• Balking / waiting-room limit $k=2$<br>• Why area accumulators must update BEFORE state variables |

---

## 🚀 How to Run

```bash
python practice_simple/q1_lcg_investigation_simple.py
python practice_simple/q2_middle_square_variants_simple.py
python practice_simple/q3_test_battery_simple.py
python practice_simple/q4_inverse_transform_simple.py
python practice_simple/q5_monte_carlo_simple.py
python practice_simple/q6_queue_simulation_simple.py
```
