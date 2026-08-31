from scipy import stats

def find_cycle(step, seed):
    
    # step is a function

    seen_at = {}
    curr = seed
    iteration = 0

    while curr not in seen_at:
        seen_at[curr] = iteration
        curr = step(curr)
        iteration+=1

    first_seen_iteration = seen_at[curr]
    cycle_len = iteration - first_seen_iteration
    return curr,iteration,cycle_len

def bin_counts(uniforms, bins=10):
    counts = [0] * bins
    for value in uniforms:
        index = min(
            bins-1, int(value*bins)
        ) # atmost bins-1 porjonto hobe
        counts[index] +=1
    return counts

def chi_square_statistic_manual(counts):
    """Step 1-4 without any library: chi2 = sum (O_i - E_i)^2 / E_i."""
    total = sum(counts)
    expected = total / len(counts)
    return sum(((observed - expected) ** 2) / expected for observed in counts)


def chi_square_uniform_test(uniforms, bins=10, alpha=0.05):
    """
    Full Chi-Square uniformity test on a list of values expected in [0, 1).

    Returns a dict with counts, the statistic, critical value, p-value,
    and the H0 decision -- this is exactly the row format the C1 table
    (Seed | Chi^2 | p-value | Decision) asks for.
    """
    counts = bin_counts(uniforms, bins)
    chi2 = chi_square_statistic_manual(counts)
    df = bins - 1
    critical_value = stats.chi2.ppf(1 - alpha, df=df)
    p_value = stats.chi2.sf(chi2, df=df)
    decision = "Reject H0" if p_value < alpha else "Do not reject H0"
    return {
        "counts": counts,
        "chi2": chi2,
        "df": df,
        "critical_value": critical_value,
        "p_value": p_value,
        "decision": decision,
    }

def middle_square_step(x):
    offset = 2
    total_digits = 4
    square = x*x
    str_sqr = str(square)
    while len(str_sqr)<8:
        str_sqr="0"+str_sqr
    # print(str_sqr)
    str_x = str_sqr[offset:offset+total_digits]
    # print(str_x)
    x = int(str_x)
    # print(x)
    return x

def middle_square_unifrom(seed,n):
    return [x / 10000 for x in middle_square(seed,n)]

def middle_square(seed, n):
    x = seed
    values = []

    for _ in range(n):
        # square = x*x
        # str_sqr = str(square)
        # while len(str_sqr)<8:
        #     str_sqr="0"+str_sqr
        # # print(str_sqr)
        # str_x = str_sqr[offset:offset+total_digits]
        # # print(str_x)
        # x = int(str_x)
        # # print(x)

        x = middle_square_step(x)
        values.append(x)
    return values

values = middle_square(5731,5)
print(values)

def find_ms_cycle(seed):
    return find_cycle(
        lambda x: middle_square_step(x), seed
    )

# print(find_ms_cycle(5731))

# samples = middle_square_unifrom(5731,20)
# print(samples)
# print(chi_square_uniform_test(samples))
problematic_seed = 1000
seeds = [5731, 6239, problematic_seed]


first_repeat, iteration, cycle_len = find_ms_cycle(seed=problematic_seed)
print(f"Problematic Seed: {problematic_seed}, First Repeated Value: {first_repeat}, Iteration: {iteration}, Cycle Length: {cycle_len}")
n = 1000
for seed in seeds:
    samples = middle_square_unifrom(seed, n)
    print(f" Seed: {seed}")
    print(chi_square_uniform_test(samples))


