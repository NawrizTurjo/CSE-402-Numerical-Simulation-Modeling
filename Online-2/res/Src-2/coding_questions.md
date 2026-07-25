# Numerical Analysis: System of Equations Coding Questions

Based on the topics covered in the lecture (Gauss Elimination, Partial Pivoting, Determinant Calculation, Gauss-Jordan Elimination, LU Decomposition, and Matrix Inversion using LU), here are some comprehensive coding questions designed to test your understanding of these algorithms.

### Question 1: Robust System Solver (Gauss Elimination with Partial Pivoting)
**Context:** Naïve Gauss elimination can fail due to division by zero or suffer from severe round-off errors when pivoting on very small numbers.
**Task:** Write a program to solve a system of $n$ linear equations $Ax = b$ using Gauss Elimination with **Partial Pivoting**.
* **Input:** An integer $n$, followed by an $n \times n$ matrix $A$ and an $n \times 1$ vector $b$.
* **Output:** The solution vector $x$.
* **Requirement:** At each step $k$, your code must search the column $k$ (from row $k$ to $n$) for the absolute largest value to use as the pivot. If a row swap is necessary, print a message indicating which rows were swapped.

### Question 2: The "Two-Step Dance" (LU Decomposition)
**Context:** The true power of LU decomposition is "Decompose once, reuse often." 
**Task:** Write a program that implements LU decomposition and uses it to solve a system.
* **Input:** An $n \times n$ matrix $A$ and an $n \times 1$ vector $b$.
* **Output:** 
  1. Print the Lower Triangular matrix $L$ (which contains the multipliers and $1$s on the diagonal).
  2. Print the Upper Triangular matrix $U$ (the result of forward elimination on $A$).
  3. Solve $Lz = b$ (Forward Substitution) and print the intermediate vector $z$.
  4. Solve $Ux = z$ (Backward Substitution) and print the final solution vector $x$.
* **Note:** You can assume the input matrix does not require pivoting (Naïve LU) for simplicity, or you can implement PLU decomposition for an extra challenge.

### Question 3: Matrix Inversion via LU Decomposition
**Context:** Finding the inverse of a matrix $A$ is equivalent to solving $Ax = e_j$ for every column of the identity matrix. LU decomposition makes this highly efficient.
**Task:** Write a program to compute the inverse of an $n \times n$ matrix $A$ using LU decomposition.
* **Input:** An $n \times n$ matrix $A$.
* **Algorithm Steps:**
  1. Decompose $A$ into $L$ and $U$.
  2. For each column $e_j$ of an $n \times n$ identity matrix $I$:
     - Solve $Lz = e_j$ (Forward Substitution)
     - Solve $Ux_j = z$ (Backward Substitution)
  3. Assemble the vectors $x_j$ as the columns of the inverted matrix $A^{-1}$.
* **Output:** The inverted matrix $A^{-1}$.

### Question 4: Determinants for Free
**Context:** Gauss elimination inherently computes the determinant of a matrix. The determinant of an upper triangular matrix is just the product of its diagonals. However, row swaps flip the sign of the determinant.
**Task:** Write a program that calculates the determinant of an $n \times n$ matrix using Gauss Elimination with Partial Pivoting.
* **Input:** An $n \times n$ matrix $A$.
* **Output:** The determinant of $A$.
* **Requirement:** Keep a counter of how many row swaps occur during the partial pivoting phase. Let this count be $S$. Calculate the determinant as: 
  $\text{det}(A) = (-1)^S \times (u_{11} \times u_{22} \times \dots \times u_{nn})$, where $u_{ii}$ are the diagonal elements of the resulting upper triangular matrix.

### Question 5: Gauss-Jordan Elimination
**Context:** Gauss-Jordan goes one step further than Gauss Elimination by driving the coefficient matrix all the way to the identity matrix, avoiding the need for back-substitution.
**Task:** Write a program to solve $Ax = b$ using Gauss-Jordan Elimination.
* **Input:** An $n \times n$ matrix $A$ and an $n \times 1$ vector $b$.
* **Requirement:**
  1. Normalize each row by its pivot so the pivot becomes $1$.
  2. When eliminating an unknown, eliminate it from *every other* equation (both below and above the pivot).
* **Output:** The augmented matrix after each column is cleared, and the final solution vector $x$.
