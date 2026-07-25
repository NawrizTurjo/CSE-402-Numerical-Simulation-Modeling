"""
Gauss-Jordan wasn't among the 3 literally-transcribed past exam questions
(res/Online Questions.txt), but it's explicitly named in the A2 syllabus and
worked step-by-step in Slides/3_system_of_eqs.txt (slides 33-39) -- included
here as the closest available "real" reference, matching the slide's own
worked numbers exactly so you can cross-check against the lecture.
"""
import numpy as np


def gauss_jordan(A, b, verbose=True):
    n = len(b)
    Aug = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        Aug[k] = Aug[k] / Aug[k, k]          # normalize pivot row to 1
        for i in range(n):
            if i != k:
                Aug[i] -= Aug[i, k] * Aug[k]  # eliminate from every other row
        if verbose:
            print(f"after clearing column {k}:\n{np.round(Aug, 6)}\n")

    return Aug[:, -1]


if __name__ == "__main__":
    # exact system from the slides:
    #   3x1  - 0.1x2 - 0.2x3 = 7.85
    #   0.1x1 + 7x2  - 0.3x3 = -19.3
    #   0.3x1 - 0.2x2 + 10x3 = 71.4
    A = np.array([[3, -0.1, -0.2],
                  [0.1, 7, -0.3],
                  [0.3, -0.2, 10]], dtype=float)
    b = np.array([7.85, -19.3, 71.4], dtype=float)

    x = gauss_jordan(A, b)
    print("solution x =", np.round(x, 4))
    print("expected (from slides): [3, -2.5, 7]")
    print("np.linalg.solve check :", np.round(np.linalg.solve(A, b), 4))
    print("||Ax - b||_2          :", np.linalg.norm(A @ x - b))
