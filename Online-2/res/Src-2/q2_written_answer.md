# Online Assessment - Written Question Answer

**Question:** If a diagonal entry becomes 0, is it necessary that this situation results in no solution? In which case will it be no solution/inf solution exactly?

**Answer:**

No, a diagonal entry becoming 0 does **not** necessarily result in "no solution." 

During the forward elimination process, a 0 on the diagonal simply means that the matrix might be singular (if all elements below it in the same column are also 0) or that a row swap is required (which partial pivoting resolves). 

If, after all possible pivoting and row operations are completed, you are left with a row where all the coefficients on the left-hand side are zero (i.e., the diagonal and everything else in that row is 0), the resulting state depends entirely on the right-hand side constant of that row:

1. **No Solution (Inconsistent System):**
   If the entire row of coefficients is 0, but the corresponding right-hand side constant $c$ is non-zero, you end up with an equation like:
   $0x + 0y + 0z = c \quad \text{(where } c \neq 0\text{)}$
   This is mathematically impossible, so the system has **no solution**.

2. **Infinite Solutions (Dependent System):**
   If the entire row of coefficients is 0, and the corresponding right-hand side constant is *also* 0, you end up with an equation like:
   $0x + 0y + 0z = 0$
   This is a tautology (always true). It implies that you have a free variable, meaning there are not enough independent equations to lock down a single point in space. This results in **infinite solutions**.
