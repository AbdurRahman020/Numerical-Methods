import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# %% Back-Substitution


def backSub(A: NDArray[np.float64], b: NDArray[np.float64]
            ) -> NDArray[np.float64]:
    """
    Parameters:
        A : (n x n) upper triangular
        b : (n x 1) column vector

    Return:
        x : (n x 1) column vector solution of Ax = b
    """

    m, n = A.shape

    if m != n:
        raise ValueError("The input matrix A isn't a square matrix")

    x = np.zeros((n, 1))

    x[n-1, 0] = b[n-1, 0] / A[n-1, n-1]

    for i in range(n-2, -1, -1):
        x[i, 0] = (b[i, 0] - np.dot(A[i, i+1:n], x[i+1:n, 0])) / A[i, i]

    return x


if __name__ == '__main__':
    A = np.array([[2, -1, 3],
                  [0, 4, 2],
                  [0, 0, 5]], dtype=float)
    b = np.array([[5],
                  [6],
                  [10]], dtype=float)

    print("x = \n", backSub(A, b))

# %% Forward Substitution


def forSub(A: NDArray[np.float64], b: NDArray[np.float64]
           ) -> NDArray[np.float64]:
    """
    Parameters:
        A : (n x n) lower triangular matrix
        b : (n x 1) column vector

    Return:
        y : (n x 1) column vector solution of Ax = b
    """

    m, n = A.shape

    if m != n:
        raise ValueError("The input matrix A isn't a square matrix.")

    y = np.zeros((n, 1))

    y[0, 0] = b[0, 0] / A[0, 0]

    for i in range(1, n):
        y[i, 0] = (b[i, 0] - np.dot(A[i, :i], y[:i, 0])) / A[i, i]

    return y


if __name__ == '__main__':
    A = np.array([[2, 0, 0],
                  [3, 5, 0],
                  [1, -2, 4]], dtype=float)
    b = np.array([[4],
                  [7],
                  [3]], dtype=float)

    print("y = \n", forSub(A, b))

# %% Cramer's Rule


def cramer(A: NDArray[np.float64], b: NDArray[np.float64]
           ) -> NDArray[np.float64]:
    """
    Parameters:
        A : (n x n) coefficient matrix
        b : (n x 1) source vector

    Return:
        x : (n x 1) column vector solution of Ax = b
    """

    n = len(b)

    # initialize solution column vector
    x = np.zeros((n, 1))

    # store original matrix
    Aold = A.copy()

    # determinant of original matrix
    d = np.linalg.det(A)

    # check if A is non-singular
    if not np.isclose(d, 0):
        for i in range(n):
            # replace i-th column of A with b
            A[:, i] = b[:, 0]
            # compute Cramer's rule
            x[i, 0] = np.linalg.det(A) / d
            # restore original A
            A = Aold.copy()
    else:
        raise ValueError("'A' is a singular matrix.")

    return x


if __name__ == '__main__':
    A = np.array([[2, -1, 5],
                  [3, 2, 2],
                  [1, 3, 3]], dtype=float)
    b = np.array([[8],
                  [14],
                  [14]], dtype=float)

    print("x = \n", cramer(A, b))

# %% Gauss-Jordan Elimination Method


def gaussJordan(A: NDArray[np.float64], b: NDArray[np.float64]
                ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        A : (n x n) coefficient matrix
        b : (n x 1) column vector

    Returns:
        Ab : (n x n+1) augmented matrix in reduced row echelon form
        x  : (n x 1) column vector solution of Ax = b
    """

    n = len(b)

    A = A.astype(float)
    b = b.astype(float)

    # form augmented matrix Ab = [A | b]
    Ab = np.hstack((A, b))

    r, c = Ab.shape

    # check if matrix A is singular
    if np.linalg.det(A) == 0:
        raise ValueError("A is a singular matrix")

    # pivot check: swap rows if diagonal entry is zero
    for j in range(r - 1):
        if Ab[j, j] == 0:
            for k in range(j + 1, r):
                if Ab[k, j] != 0:
                    # swap rows
                    Ab[[j, k]] = Ab[[k, j]]
                    break

    # gauss-Jordan elimination
    for j in range(n):
        # make pivot = 1
        Ab[j, :] = Ab[j, :] / Ab[j, j]
        for i in range(n):
            if i != j:
                Ab[i, :] = Ab[i, :] - Ab[i, j] * Ab[j, :]

    # extract solution (is a column vector)
    x = Ab[:, c - 1:c]
    return Ab, x


if __name__ == '__main__':
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=float)
    b = np.array([[8],
                  [-11],
                  [-3]], dtype=float)

    Ab, x = gaussJordan(A, b)
    print("Reduced Row Echelon Form (Ab):\n", Ab)
    print("Solution x:\n", x)

# %% Gauss-Elimination Method


def gaussElimn(A: NDArray[np.float64], b: NDArray[np.float64]
               ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        A : (n x n) coefficient matrix
        b : (n x 1) column vector

    Returns:
        U : (n x n) upper triangular matrix
        x : (n x 1) column vector solution of Ax = b
    """

    n = len(b)

    x = np.zeros((n, 1))

    # augment A with b
    Ab = np.hstack((A.astype(float), b.astype(float)))

    # check if A is singular
    if np.linalg.det(A) == 0:
        raise ValueError("A is a singular matrix")

    # forward elimination
    for j in range(n - 1):
        for i in range(j + 1, n):
            m = Ab[i, j] / Ab[j, j]
            Ab[i, :] = Ab[i, :] - m * Ab[j, :]

    # back substitution
    x[n - 1, 0] = Ab[n - 1, n] / Ab[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        x[k, 0] = (Ab[k, n] - np.dot(Ab[k, k + 1:n], x[k + 1:n, 0])) / Ab[k, k]

    # extract upper triangular matrix U
    U = Ab[:, :n]

    return U, x


if __name__ == '__main__':
    A = np.array([[2, -1, 1],
                  [3, 3, 9],
                  [3, 3, 5]], dtype=float)
    b = np.array([[2],
                  [-1],
                  [4]], dtype=float)

    U, x = gaussElimn(A, b)
    print("U = \n", U)
    print("x = \n", x)

# %% Cholesky's Method
# (manual LU decomposition via LLᵗ for symmetric positive definite matrix)


def cholesky(A: NDArray[np.float64], b: NDArray[np.float64]
             ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                        NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        A : (n x n) coefficient matrix (positive definite)
        b : (n x 1) source vector

    Returns:
        L : lower triangular matrix
        U : upper triangular matrix (transpose of L)
        y : solution of Ly = b (forward substitution)
        x : solution of Ux = y (backward substitution)
    """

    n = len(b)

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    # first element
    L[0, 0] = np.sqrt(A[0, 0])
    U[0, 0] = L[0, 0]

    # fill first row/column of L and U
    for i in range(1, n):
        L[i, 0] = A[i, 0] / L[0, 0]
        U[0, i] = A[0, i] / L[0, 0]

    # main loop to fill L and U
    for i in range(1, n):
        for j in range(i, n):
            if i == j:
                L[j, i] = np.sqrt(A[j, i] - np.dot(L[j, :i], U[:i, i]))
                U[j, i] = L[j, i]
            else:
                L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / L[i, i]

        for k in range(i+1, n):
            U[i, k] = (A[i, k] - np.dot(L[i, :i], U[:i, k])) / L[i, i]

    # forward substitution to solve Ly = b
    y[0, 0] = b[0, 0] / L[0, 0]
    for i in range(1, n):
        y[i, 0] = (b[i, 0] - np.dot(L[i, :i], y[:i, 0])) / L[i, i]

    # backward substitution to solve Ux = y
    x[n-1, 0] = y[n-1, 0] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i, 0] = (y[i, 0] - np.dot(U[i, i+1:n], x[i+1:n, 0])) / U[i, i]

    return L, U, y, x


if __name__ == '__main__':
    A = np.array([[25, 15, -5],
                  [15, 18,  0],
                  [-5, 0, 11]], dtype=float)
    b = np.array([[35],
                  [33],
                  [6]], dtype=float)

    L, U, y, x = cholesky(A, b)
    print("L = \n", L)
    print("\nU = \n", U)
    print("\ny = \n", y)
    print("\nx = \n", x)

# %% Crout's Method (LU decomposition)


def crout(A: NDArray[np.float64], b: NDArray[np.float64]
          ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                     NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        A : (n x n) coefficient matrix
        b : (n x 1) source vector

    Returns:
        L : (n x n) lower triangular matrix
        U : (n x n) upper triangular matrix (unit diagonal)
        y : solution of Ly = b (forward substitution)
        x : solution of Ux = y (backward substitution)
    """

    n = len(b)

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    # set diagonal of U to 1
    for i in range(n):
        U[i, i] = 1

    # first column of L and first row of U
    L[:, 0] = A[:, 0]
    U[0, :] = A[0, :] / L[0, 0]

    # fill L and U using Crout's method
    for i in range(1, n):
        for j in range(i, n):
            L[j, i] = A[j, i] - np.dot(L[j, :i], U[:i, i])
        for k in range(i + 1, n):
            U[i, k] = (A[i, k] - np.dot(L[i, :i], U[:i, k])) / L[i, i]

    # forward substitution: solve Ly = b
    y[0, 0] = b[0, 0] / L[0, 0]
    for i in range(1, n):
        y[i, 0] = (b[i, 0] - np.dot(L[i, :i], y[:i, 0])) / L[i, i]

    # backward substitution: solve Ux = y
    x[n - 1, 0] = y[n - 1, 0] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i, 0] = (y[i, 0] - np.dot(U[i, i + 1:n], x[i + 1:n, 0])) / U[i, i]

    return L, U, y, x


if __name__ == '__main__':
    A = np.array([[3, -7, -2],
                  [-3, 5, 1],
                  [6, -4, 0]], dtype=float)
    b = np.array([[1],
                  [2],
                  [3]], dtype=float)

    L, U, y, x = crout(A, b)
    print("L = \n", L)
    print("\nU = \n", U)
    print("\ny = \n", y)
    print("\nx = \n", x)

# %% Doolittle's LU Decomposition Method


def doolittle(A: NDArray[np.float64], b: NDArray[np.float64]
              ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                         NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        A : (n x n) coefficient matrix
        b : (n x 1) column vector

    Returns:
        L : (n x n) lower triangular matrix (unit diagonal)
        U : (n x n) upper triangular matrix
        y : solution of Ly = b (forward substitution)
        x : solution of Ux = y (backward substitution)
    """

    n = b.shape[0]

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    # Set diagonal of L to 1
    for i in range(n):
        L[i, i] = 1

    # first row of U, first column of L
    U[0, :] = A[0, :]
    L[:, 0] = A[:, 0] / U[0, 0]

    # LU decomposition with pivoting check
    for i in range(1, n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        for k in range(i + 1, n):
            if U[i, i] != 0:
                L[k, i] = (A[k, i] - np.dot(L[k, :i], U[:i, i])) / U[i, i]
            else:
                raise ValueError(
                    f"Pivot element U[{i},{i}] is zero. LU decomposition failed.")

    # forward substitution: Ly = b
    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # backward substitution: Ux = y
    x[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return L, U, y, x


if __name__ == '__main__':
    A = np.array([[2, -1, 1],
                  [3, 3, 9],
                  [3, 3, 5]], dtype=float)
    b = np.array([[2],
                  [-1],
                  [4]], dtype=float)

    L, U, y, x = doolittle(A, b)
    print("L = \n", L)
    print("U = \n", U)
    print("y = \n", y)
    print("x = \n", x)

# %% Gram-Schmidt Method


def gramSchmidt(V: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Parameters:
        V : A (m x n) matrix, where each column is a vector of dimension 'm'

    Return:
        Q : (m x n) orthonormal matrix
    """

    m, n = V.shape

    Q = np.zeros((m, n))

    for i in range(n):
        vi = V[:, i].copy()

        for j in range(i):
            qj = Q[:, j]
            vi -= np.dot(qj, vi) * qj

        norm = np.linalg.norm(vi)

        if norm < 1e-10:
            raise ValueError(
                f"Vector {i} is linearly dependent or too close to zero.")

        Q[:, i] = vi / norm

    return Q


if __name__ == '__main__':
    A = np.array([[1, 1, 0, 2],
                  [1, 0, 1, -1],
                  [1, 0, 0, 1],
                  [1, 1, 2, 1]], dtype=float).T

    Q = gramSchmidt(A)
    print("Orthonormal basis (as columns of Q):\n", Q)
    print("\nNorms of columns:", np.linalg.norm(Q, axis=0))
    print("QᵀQ (should be identity, approx):\n", np.dot(Q.T, Q))

# %% QR Decomposition


def qrDecomposition(A: NDArray[np.float64]
                    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        A : A real-valued (m x n) matrix

    Returns:
        Q : (m x n) orthonormal matrix
        R : (n x n) upper triangular matrix
    """

    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # take the j-th column of A
        v = A[:, j].copy()

        # subtract projections onto previously computed Q columns
        for i in range(j):
            # compute projection scalar
            R[i, j] = np.dot(Q[:, i], A[:, j])
            # remove component in direction of Q[:, i]
            v -= R[i, j] * Q[:, i]

        # normalize v to get orthonormal vector
        R[j, j] = np.linalg.norm(v)

        if R[j, j] == 0:
            raise ValueError("Matrix has linearly dependent columns")

        # store normalized vector in Q
        Q[:, j] = v / R[j, j]

    return Q, R


if __name__ == '__main__':
    A = np.array([[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [0, 1, 0, 2],
                  [2, -1, 1, 1]], dtype=float)

    Q, R = qrDecomposition(A)
    print("Matrix A:\n", A)
    print("\nMatrix Q (orthonormal columns):\n", Q)
    print("\nMatrix R (upper triangular):\n", R)
    print("\nQ @ R:\n", Q @ R)
