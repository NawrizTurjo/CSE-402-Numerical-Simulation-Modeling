import scipy.stats as stats

def middle_square(seed,n):
    x=seed
    values=[]
    for i in range(n):
        x=x*x
        x_str="0"*(8-len(str(x)))+str(x)
        x_str=x_str[2:6]
        x=int(x_str)
        values.append(x)
    return values

values=middle_square(5731,10)

print(values[:10])

def find_cycle(seed,iter=1000):
    x=seed
    seen={}
    for i in range(iter+1):
        if x in seen:
            first_seen=seen[x]
            cycle_length=i-first_seen
            return {
                "seed": seed,
                "first_seen": first_seen,
                "cycle_length": cycle_length,
                "repeat_value": x
            }
        seen[x]=i
        x=x*x
        x_str="0"*(8-len(str(x)))+str(x)
        x_str=x_str[2:6]
        x=int(x_str)
    return None

def chi_square_test(seed,n=1000,bins=10):
    bin_counts=[0]*bins
    values=middle_square(seed,n)
    for v in values:
        i=v/10000
        #print(i)
        bin_index=int(i*bins)
        if bin_index==bins:
            bin_index=bins-1
        bin_counts[bin_index]+=1
    expected=[100]*10
    chi_square,p_value=stats.chisquare(bin_counts,expected)
    if p_value < 0.05:
        decision = "Reject H0"
    else:
        decision = "Do not reject H0"

    return bin_counts, chi_square, p_value, decision

seeds = [5731, 6239, 1]

for seed in seeds:
    observed, chi_square, p_value, decision = chi_square_test(seed)

    print("\nSeed:", seed)
    print("Observed frequencies:", observed)
    print("Chi-square:", chi_square)
    print("p-value:", p_value)
    print("Decision:", decision)


import matplotlib.pyplot as plt

seeds = [5731, 6239, 1]

for seed in seeds:
    values = middle_square(seed, 1000)
    numbers = [x / 10000 for x in values]

    plt.figure(figsize=(8, 5))

    plt.hist(
        numbers,
        bins=10,
        range=(0, 1),
        edgecolor="black"
    )

    plt.xlabel("Generated value")
    plt.ylabel("Frequency")
    plt.title(f"Middle-Square Generator - Seed {seed}")

    plt.show()