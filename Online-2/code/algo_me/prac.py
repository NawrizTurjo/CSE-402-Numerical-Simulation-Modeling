import numpy as np

def plu_decompose_trj(A,hand_written=True):

    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n-1):

        pivot = k

        if hand_written:
            for i in range(k+1, n):
                if abs(U[i,k])>abs(U[pivot,k]):
                    pivot = i
        else:
            pivot = k+np.argmax(np.abs(U[k:, k]))
        
        if pivot!=k:

            if hand_written:
                # swap for U and P, row k and row pivot
                for j in range(n):

                    U[k,j], U[pivot,j] = U[pivot,j], U[k,j]
                    P[k,j], P[pivot,j] = P[pivot,j], P[k,j]
                if k>0:
                    # swap the already stored multipliers before the kth ""column""
                    for j in range(k):
                        L[k,j], L[pivot,j] = L[pivot,j], L[k,j]
            else:
                U[[k,pivot]] = U[[pivot,k]]
                P[[k,pivot]] = P[[pivot,k]]

                if k>0:
                    L[[k,pivot], :k] = L[[pivot,k],:k]
            
        for i in range(k+1,n):
            m = U[i,k]/U[k,k]
            L[i,k] = m

            if hand_written:
                for j in range(k,n): # <- columns actually
                    U[i,j] -= m*U[k,j]
            else:
                U[i,k:] -= m*U[k,k:]
            
    return P,L,U

def for_sub(L,P,b,hand_written=True):
    Pb = P@b
    n = len(Pb)
    z = np.zeros(n)

    for i in range(n):
        if hand_written:
            total_sum = 0.0
            for j in range(i):
                total_sum += L[i,j]*z[j]
            z[i] = (Pb[i] - total_sum)/L[i,i]
        else:
            z[i] = (Pb[i]-L[i,:i]@z[:i])/L[i,i]
    return z



def back_sub(U,z,hand_written=True):
    n = len(z)
    x = np.zeros(n)

    for i in range (n-1,-1,-1):
        if hand_written:
            total_sum = 0.0
            for j in range(i+1,n):
                total_sum += U[i,j]*x[j]
            x[i] = (z[i]-total_sum)/U[i,i]
        else:
            x[i] = (z[i]-U[i,i+1:]@x[i+1:])/U[i,i]

    
    return x








import matplotlib.pyplot as plt

def plot_convergence(eval_history, evec_history, title="Power Method Convergence", save_path="convergence_plot.png", show_plot=False):
    """
    Plots the iteration-by-iteration convergence of:
    1. Eigenvalues
    2. Eigenvector components (v1, v2, ..., vn)
    """
    eval_history = np.array(eval_history)
    evec_history = np.array(evec_history)  # shape (iterations, n)
    iters = np.arange(1, len(eval_history) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # --- Subplot 1: Eigenvalue Convergence ---
    ax1.plot(iters, eval_history, 'o-', color='b', linewidth=2, label=r'$\lambda$ Estimate')
    ax1.set_title("Eigenvalue Convergence", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Eigenvalue Estimate")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # --- Subplot 2: Eigenvector Components Convergence ---
    n_dim = evec_history.shape[1]
    for comp in range(n_dim):
        comp_vals = evec_history[:, comp]
        ax2.plot(iters, comp_vals, 'o--', label=f'v[{comp+1}]')
        
    ax2.set_title("Eigenvector Components Convergence", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Vector Component Value")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\n[Plot Saved] Convergence plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# normal one

def power_iteration_trj( A, x0, tol=1e-9, max_iter=1000, hand_written= True ):
    x = x0.astype(float).copy()
    prev_est = None
    eval_history = []
    evec_history = []

    for _ in range(max_iter):

        y = A @ x

        if hand_written:
            curr_est = y[0]
            for i in range(1,len(y)):
                if abs(y[i])>abs(curr_est):
                    curr_est = y[i]
        else:
            curr_est = y[np.argmax(np.abs(y))]

        eval_history.append(curr_est)

        if abs(curr_est) < tol:
            raise ValueError("Zero eigenvalue estimate encountered.")
        
        if prev_est is not None:
            ea = abs((curr_est-prev_est)/curr_est)*100
        else:
            ea = None
        
        prev_est = curr_est
        
        x_new = y/curr_est

        # Track normalized eigenvector estimate
        if hand_written:
            norm_sq = 0.0
            for val in x_new:
                norm_sq += val * val
            evec_history.append(x_new / (norm_sq ** 0.5))
        else:
            evec_history.append(x_new / np.linalg.norm(x_new))

        
        if hand_written:
            max_diff = 0.0
            for i in range(len(x)):
                diff = abs(x[i]-x_new[i])
                if diff > max_diff:
                    max_diff = diff
        else:
            max_diff = np.max(np.abs(x-x_new))
        
        x = x_new

        if max_diff < tol:
            break

    # outside loop
    if hand_written:
        norm_sq = 0.0
        for val in x:
            norm_sq+=val*val
        x = x/(norm_sq**0.5)
    else:
        x = x / np.linalg.norm(x)
    
    return curr_est, x, ea, eval_history, evec_history

def deflate( A, v, lam, hand_written=True):


    if hand_written:
        # normalize v
        n = len(v)
        norm_sq = 0.0
        for val in v:
            norm_sq+=val*val
        norm = norm_sq**0.5
        if norm < 1e-10:
            raise ValueError('Division by zero')
        
        v_hat = v/norm

        A_next = A.copy()

        for i in range(n):
            for j in range(n):
                A_next[i,j]-=lam*v_hat[i]*v_hat[j]

    else:
        v_hat = v / np.linalg.norm(v)
        A_next = A - lam*np.outer(v_hat,v_hat)

    return A_next


def find_all_eignes( A, x0, tol=1e-9, max_iter=1000, hand_written=True):
    n = A.shape[0]
    A_work = A.astype(float).copy()

    evals, evecs = [], []
    eval_histories, evec_histories = [], []

    for i in range(n):
        curr_lam, curr_vec, _, eval_hist, evec_hist = power_iteration_trj(
            A=A_work, x0=x0, tol=tol, max_iter=max_iter, hand_written=hand_written
        )
        evals.append(curr_lam)
        evecs.append(curr_vec)
        eval_histories.append(eval_hist)
        evec_histories.append(evec_hist)

        print(f"eigenpair {i+1}: lambda={curr_lam:.6f}  v={np.round(curr_vec,5)}")

        A_work = deflate(A=A_work, v=curr_vec, lam=curr_lam, hand_written=hand_written)
    
    evals = np.array(evals)
    evecs = np.column_stack(evecs)

    return evals, evecs, eval_histories, evec_histories



A = np.array([
    [2,1],
    [1,2]
], dtype=float)

x0 = np.array([1,2], dtype=float)

evals, evecs, eval_histories, evec_histories = find_all_eignes(
    A,
    x0,
    hand_written=True
)

print("Eigenvalues:", np.round(evals, 6))
print("Eigenvectors:\n", np.round(evecs, 6))

print("\nHistories:")
for i, h in enumerate(eval_histories):
    print(f"Eigenpair {i+1} Eigenvalues:", np.round(h, 6))

# Plot convergence for Eigenpair 1
plot_convergence(eval_histories[0], evec_histories[0], title="Dominant Eigenpair Convergence", save_path="dominant_eigenpair_convergence.png")


# --- 2. Test PLU Decomposition ---

A = np.array(
    [
        [20, 15, 10],
        [-3, -2.249, 7],
        [5, 1, 3]
    ], dtype=float
)
b = np.array(
    [45, 1.751, 9], dtype=float
)


print("\n" + "=" * 45)
print("PLU DECOMPOSITION & SOLVER")
print("=" * 45)
P, L, U = plu_decompose_trj(A, hand_written=True)
print("P =\n", np.round(P, 6))
print("L =\n", np.round(L, 6))
print("U =\n", np.round(U, 6))
print("Reconstruction check (P @ A == L @ U):", np.allclose(P @ A, L @ U))
z = for_sub(L, P, b, hand_written=True)
x_sol = back_sub(U, z, hand_written=True)
print(f"\nSolving Ax = b for b = {b}:")
print("z (Forward Sub): ", np.round(z, 6))
print("x (Backward Sub):", np.round(x_sol, 6))
print("NumPy solve:     ", np.round(np.linalg.solve(A, b), 6))
print("=" * 45)


A = np.array(
    [
        [2,1],
        [1,2]
    ], dtype=float
)

b = np.array(
    [
        1,2
    ], dtype=float
)

A_inv = np.linalg.inv(A)

eigen_val, eigen_vec, ea, eval_h, evec_h = power_iteration_trj(A_inv, b, hand_written=False)
print(np.round(eigen_val))
min_eigne_val = 1 / eigen_val
print(np.round(min_eigne_val))

lambdas, _ = np.linalg.eig(A)
print(f'Min EigenValue (numpy): {np.min(np.abs(lambdas)):.6f}')


def inverse_power_iter_trj(A, x0, tol=1e-9, max_iter=1000, hand_written=True):
    A_inv = np.linalg.inv(A)
    eigen_val, eigen_vec, ea, eval_hist, evec_hist = power_iteration_trj(A_inv, x0, tol=tol, max_iter=max_iter, hand_written=hand_written)
    eigen_val = 1 / eigen_val
    return eigen_val, eigen_vec, ea, eval_hist, evec_hist


def power_iteration_plu_generalized(A, x0, tol=1e-9, max_iter=1000, hand_written=True, is_inverse=True):
    x = x0.astype(float).copy()
    prev_est = None
    eval_history = []
    evec_history = []

    if is_inverse:
        # 1. Compute PLU decomposition once
        P, L, U = plu_decompose_trj(A, hand_written=hand_written)

    for _ in range(max_iter):
        if is_inverse:
            # 2. Solve A @ y = x using forward and backward substitution (y = A^-1 @ x)
            z = for_sub(L, P, x, hand_written=hand_written)
            y = back_sub(U, z, hand_written=hand_written)
        else:
            # otherwise normal operations as always
            y = A @ x

        if hand_written:
            curr_est = y[0]
            for i in range(1, len(y)):
                if abs(y[i]) > abs(curr_est):
                    curr_est = y[i]
        else:
            curr_est = y[np.argmax(np.abs(y))]

        eval_val = (1.0 / curr_est) if is_inverse else curr_est
        eval_history.append(eval_val)

        if abs(curr_est) < tol:
            raise ValueError("Zero eigenvalue estimate encountered.")
        
        if prev_est is not None:
            ea = abs((curr_est - prev_est) / curr_est) * 100
        else:
            ea = None
        
        prev_est = curr_est
        
        x_new = y / curr_est

        # Track normalized eigenvector estimate
        if hand_written:
            norm_sq = 0.0
            for val in x_new:
                norm_sq += val * val
            evec_history.append(x_new / (norm_sq ** 0.5))
        else:
            evec_history.append(x_new / np.linalg.norm(x_new))

        if hand_written:
            max_diff = 0.0
            for i in range(len(x)):
                diff = abs(x[i] - x_new[i])
                if diff > max_diff:
                    max_diff = diff
        else:
            max_diff = np.max(np.abs(x - x_new))
        
        x = x_new

        if max_diff < tol:
            break

    # Normalize final eigenvector
    if hand_written:
        norm_sq = 0.0
        for val in x:
            norm_sq += val * val
        x = x / (norm_sq ** 0.5)
    else:
        x = x / np.linalg.norm(x)
    
    if is_inverse:
        curr_est = 1.0 / curr_est

    return curr_est, x, ea, eval_history, evec_history


# --- Test Inverse Power Methods ---
print("\n" + "=" * 45)
print("INVERSE POWER METHOD (PLU vs Explicit Inverse)")
print("=" * 45)

A = np.array([[4, 1, 0, 0],
              [1, 3, 1, 0],
              [0, 1, 2, 1],
              [0, 0, 1, 1]], dtype=float)
x0 = np.array([1, 2, -5, 7], dtype=float)  

eval_plu, evec_plu, _, eval_h_plu, evec_h_plu = power_iteration_plu_generalized(A, x0, hand_written=False, is_inverse=True)
eval_inv, evec_inv, _, eval_h_inv, evec_h_inv = inverse_power_iter_trj(A, x0, hand_written=True)

print("Smallest Eigenvalue (via PLU):           ", round(eval_plu, 6))
print("Smallest Eigenvalue (via Explicit Inv):  ", round(eval_inv, 6))
print("Smallest Eigenvalue (np.linalg.eig):     ", round(float(np.min(np.abs(np.linalg.eig(A)[0]))), 6))
print("Eigenvector (via PLU):", np.round(evec_plu, 6))
print("=" * 45)

# Plot convergence for Inverse Power Method
plot_convergence(eval_h_plu, evec_h_plu, title="Inverse Power Method PLU Convergence", save_path="inverse_power_plu_convergence.png")