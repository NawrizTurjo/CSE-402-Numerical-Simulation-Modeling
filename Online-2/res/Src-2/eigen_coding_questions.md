# Numerical Analysis: Eigenvalue Decomposition Coding Questions

Based on the topics covered in the lecture (Power Method, Inverse Power Method, Deflation, and Eigenvalue Decomposition), here are several coding questions designed to test your understanding of these algorithms.

### Question 1: Finding the Dominant Eigenvalue (The Power Method)
**Context:** The Power Method allows us to find the largest-magnitude eigenvalue without expanding determinants. By repeatedly multiplying a matrix by a vector and normalizing it, the vector converges to the dominant eigenvector.
**Task:** Write a program to find the dominant eigenvalue $\lambda_1$ and its corresponding eigenvector $v_1$ of a matrix $A$ using the Power Method.
* **Input:** An integer $n$, followed by an $n \times n$ matrix $A$, an $n \times 1$ initial guess vector $x_0$, and a tolerance value $\epsilon$ (e.g., $0.0001$).
* **Requirement:** 
  1. Multiply $y_{k+1} = A x_k$.
  2. Find the entry with the largest absolute value in $y_{k+1}$. This is your current eigenvalue estimate.
  3. Divide $y_{k+1}$ by this estimate to get the normalized vector $x_{k+1}$.
  4. Repeat until the absolute difference between consecutive eigenvalue estimates is less than $\epsilon$.
* **Output:** The dominant eigenvalue and its corresponding normalized eigenvector.

### Question 2: The Smallest Eigenvalue (Inverse Power Method)
**Context:** If $\lambda$ is an eigenvalue of $A$, then $1/\lambda$ is an eigenvalue of $A^{-1}$. The Inverse Power Method leverages this to find the smallest eigenvalue of $A$.
**Task:** Write a program to find the smallest eigenvalue of a matrix $A$ using the Inverse Power Method.
* **Input:** An integer $n$, an $n \times n$ matrix $A$, an initial guess vector $x_0$, and a tolerance $\epsilon$.
* **Requirement:** 
  1. In each iteration, instead of computing $A^{-1} x_k$ directly, solve the linear system $A y_{k+1} = x_k$ (you can use your Gauss Elimination or LU Decomposition code from the previous assignment).
  2. Normalize $y_{k+1}$ by its largest-magnitude entry to get $x_{k+1}$.
  3. The inverse of this normalizing factor is your current estimate for the smallest eigenvalue $\lambda_{min}$.
* **Output:** The smallest eigenvalue $\lambda_{min}$.

### Question 3: Peeling off Eigenvalues (Hotelling's Deflation)
**Context:** The Power Method only finds the loudest (dominant) eigenvalue. To find intermediate eigenvalues, we must "mute" the dominant one by subtracting its contribution from the matrix.
**Task:** Write a program to find the second largest eigenvalue $\lambda_2$ of a symmetric matrix $A$.
* **Input:** An $n \times n$ symmetric matrix $A$, its dominant eigenvalue $\lambda_1$, and its corresponding eigenvector $v_1$.
* **Requirement:**
  1. Normalize the eigenvector $v_1$ to unit length to obtain $\hat{v}_1 = \frac{v_1}{\sqrt{v_1^T v_1}}$.
  2. Compute the deflated matrix: $A_2 = A - \lambda_1 \hat{v}_1 \hat{v}_1^T$.
  3. Run the standard Power Method (from Question 1) on the deflated matrix $A_2$ to find its dominant eigenvalue, which corresponds to the second largest eigenvalue of $A$, $\lambda_2$.
* **Output:** The deflated matrix $A_2$ and the second largest eigenvalue $\lambda_2$.

### Question 4: Translating Between Languages (Eigenvalue Decomposition)
**Context:** A matrix $A$ can be decomposed as $A = V \Lambda V^{-1}$, where $V$ is the matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues.
**Task:** Write a program to verify the eigenvalue decomposition for a $2 \times 2$ matrix.
* **Input:** A $2 \times 2$ matrix $A$, its two eigenvalues $\lambda_1, \lambda_2$, and its two eigenvectors $v_1, v_2$.
* **Requirement:**
  1. Construct the matrix $V$ by putting $v_1$ and $v_2$ as columns.
  2. Construct the diagonal matrix $\Lambda$ with $\lambda_1$ and $\lambda_2$ on the diagonal.
  3. Calculate the inverse matrix $V^{-1}$ using the explicit $2 \times 2$ inverse formula.
  4. Multiply $V \times \Lambda \times V^{-1}$ to reconstruct the matrix.
* **Output:** Print $V$, $\Lambda$, $V^{-1}$, and the reconstructed matrix $A$. Check if the reconstructed matrix matches the original matrix $A$.
