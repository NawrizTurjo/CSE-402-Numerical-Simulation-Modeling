"""
Linear Congruential Method (LCM) for generating (pseudo)random numbers

Recurrence:
    X_(i+1) = (a * X_i + c) mod m
    R_i     = X_i / m
"""

# ---------------------------------------------------------------------
# PARAMETERS (defined at the start, as requested)
# ---------------------------------------------------------------------
X0 = 27          # seed (initial value X_0)
a  = 17         # multiplier
c  = 43         # increment
m  = 100        # modulus
n  = 10         # how many random numbers to generate

# ---------------------------------------------------------------------
# GENERATE THE SEQUENCE & DETECT PERIOD
# ---------------------------------------------------------------------
X = X0
sequence = []
seen = {}
period_info = None

print(f"{'i':>3} | {'X_i':>6} | {'R_i = X_i/m':>12}")
print("-" * 28)

for i in range(n):
    R = X / m
    print(f"{i:3d} | {X:6d} | {R:12.4f}")
    sequence.append(X)

    if period_info is None and X in seen:
        start_idx = seen[X]
        period_length = i - start_idx
        period_info = {
            "period": period_length,
            "start_idx": start_idx,
            "repeat_idx": i,
            "value": X,
            "cycle": sequence[start_idx:i]
        }

    if X not in seen:
        seen[X] = i

    X = (a * X + c) % m

# ---------------------------------------------------------------------
# PERIOD REPORT
# ---------------------------------------------------------------------
print("-" * 28)
if period_info:
    print(f"Period of the sequence : {period_info['period']}")
    print(f"Repeating cycle        : {period_info['cycle']}")
    print(f"First repetition       : X_{period_info['repeat_idx']} = X_{period_info['start_idx']} = {period_info['value']}")
else:
    print(f"No period found within the generated sequence of {n} numbers.")