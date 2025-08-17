import numpy as np
from numpy.typing import NDArray

# %% Jacobi Iteration


def jacobi(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    p: NDArray[np.float64],
    max_itr: int = 25,
    tol: float = 1e-4
) -> NDArray[np.float64]:
    """
    Parameters:
        A       : (n x n) non-singular coefficient matrix
        b       : (n x 1) column vector
        p       : (n x 1) column vector of initial guess
        max_itr : maximum number of iterations (default = 25)
        tol     : tolerance for convergence (default = 1e-4)

    Return:
        x       : (n x 1) Jacobi approximation to the solution of Ax = b
    """

    # validate input
    m, n = A.shape
    if m != n:
        # jacobi method only works with square coefficient matrices
        raise ValueError("coefficient matrix 'A' should be a square matrix")

    # get the number of equations (rows in b)
    n = b.shape[0]

    # allocate vectors for:
    x = np.zeros((n, 1))    # current iteration solution
    # previous iteration solution (for convergence check)
    ref = np.zeros((n, 1))

    # --- Jacobi Iteration ---
    # the formula for each variable x_j:
    #           x_j^(k+1) = ( b_j - sum_{i != j} a_ji * p_i ) / a_jj
    # where 'p' is the previous iteration's solution vector

    for i in range(1, max_itr + 1):
        # loop through each equation j
        for j in range(n):
            # create list of indices except j
            indices = list(range(0, j)) + list(range(j + 1, n))

            # compute numerator: b[j] - sum(a[j,k] * p[k] for k != j)
            # then divide by diagonal element A[j,j] to isolate x_j
            x[j, 0] = b[j, 0] / A[j, j] - \
                (np.dot(A[j, indices], p[indices, 0]) / A[j, j])

        # update guess vector for next iteration
        p = x.copy()

        # show current iteration results
        print(f"{i}:\t" + " ".join([f"{val:.6f}" for val in p[:, 0]]))

        # convergence check: if all changes from previous solution are within
        # tolerance, stop
        if np.all(np.abs(ref - x) < tol):
            break

        # update reference vector for next iteration's comparison
        ref = x.copy()

    return x


if __name__ == "__main__":
    A = np.array([
        [10., -1.,  2.,  0.],
        [-1., 11., -1.,  3.],
        [2., -1., 10., -1.],
        [0.,  3., -1.,  8.]])
    b = np.array([[6.], [25.], [-11.], [15.]])
    p = np.zeros((4, 1))

    x = jacobi(A, b, p, max_itr=50, tol=1e-5)
    print("\nFinal approximation of x is:\n", x)


# %% Gauss-Seidel Iteration

def gaussSeidel(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    p: NDArray[np.float64],
    max_itr: int = 25,
    tol: float = 1e-4
) -> NDArray[np.float64]:
    """
    Parameters:
        A        : (n x n) coefficient matrix
        b        : (n x 1) right-hand side column vector
        p        : (n x 1) initial guess column vector
        max_itr  : maximum number of iterations (default = 25)
        tol      : tolerance for convergence of 'p' (default = 0.0001)

    Return:
        x        : (n x 1) Gauss-Seidel approximation to solution of Ax = b
    """

    # ensure A is square (n equations, n unknowns)
    if A.shape[0] != A.shape[1]:
        raise ValueError("coefficient matrix 'A' should be square.")

    # number of unknowns
    n = len(b)

    # allocate memory for:
    x = np.zeros((n, 1))    # current iteration solution vector
    # previous iteration solution vector (for convergence check)
    ref = np.zeros((n, 1))

    # outer loop for iterations
    for i in range(1, max_itr + 1):
        # loop through each equation j
        for j in range(n):
            # get all column indices except the current variable j
            idx = [i for i in range(n) if i != j]

            # --- Gauss-Seidel Iteration---
            # the formula for each variable x_j:
            #       x_j^(new) = ( b_j - Σ_{k≠j} A[j,k] * x_k ) / A[j,j]
            # here, we use updated values immediately for x_k when available
            # (p already contains updated ones)
            x[j, 0] = (b[j, 0] - np.dot(A[j, idx], p[idx]).item()) / A[j, j]

            # store the updated value directly into the guess vector 'p'
            p[j, 0] = x[j, 0]

        # print current iteration and the updated guess vector
        print(f"{i}:\t" + " ".join([f"{val:.6f}" for val in p[:, 0]]))

        # convergence check: using infinity norm (max absolute difference
        # between iterations)
        if np.linalg.norm(ref - x, ord=np.inf) < tol:
            break

        # update the reference vector for the next convergence check
        ref = x.copy()

    return x


if __name__ == "__main__":
    A = np.array([[4, 1, 2],
                  [3, 5, 1],
                  [1, 1, 3]], dtype=float)
    b = np.array([[4], [7], [3]], dtype=float)
    p = np.zeros((3, 1))

    x = gaussSeidel(A, b, p, max_itr=10, tol=1e-5)
    print("\nFinal solution x:\n", x)
