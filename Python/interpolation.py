import numpy as np
from numpy.typing import ArrayLike

# %% Lagrange's Interpolating Polynomial


def lagranIntpl(x: ArrayLike, y: ArrayLike, x_intrcpt: float) -> float:
    """
    Parameters :
        x         : [x0, x1, ..., xn]
        y         : [y0, y1, ..., yn]
        x_intrcpt : value at which interpolation is calculated

    Return:
        y_intrcpt : interpolated y-value at x_intrcpt
    """

    # convert input lists/arrays into NumPy arrays (float type for accuracy)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # number of data points
    n = len(x)

    # ensure 'x' and 'y' arrays have the same length
    if len(y) != n:
        raise ValueError("x and y must be of same length")

    # initialize the result of interpolation
    y_intrcpt = 0.0

    # loop over each term in the Lagrange polynomial
    for i in range(n):
        # start with y[i] as the base value for this term
        product = y[i]

        # multiply by Lagrange basis polynomial factors
        for j in range(n):
            if i != j:  # skip division by zero for the current index
                product = product * (x_intrcpt - x[j]) / (x[i] - x[j])

        # add the computed term to the total sum
        y_intrcpt = y_intrcpt + product

    # return the final interpolated value
    return y_intrcpt


if __name__ == '__main__':
    x_vals = [0, 1, 2]
    y_vals = [1, 3, 2]
    x_interp = 1.5

    print(f"Interpolated value at x = {x_interp} is "
          "{lagranIntpl(x_vals, y_vals, x_interp)}")


# %% Lagrange's Interpolating Polynomial


def lagranPoly(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters :
        x : [x0, x1, ..., xn]
        y : [y0, y1, ..., yn]

    Return:
        C : coefficients of degree 'n-1' polynomial (highest degree first)
        L : n-by-n matrix where each row contains the coefficients of the
            corresponding Lagrange basis polynomial L_i(x)
    """

    # convert input lists/arrays into NumPy arrays (float type for accuracy)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # number of data points
    n = len(x)

    # ensure 'x' and 'y' arrays have the same length
    if len(y) != n:
        raise ValueError("x and y must be of same length")

    # initialize matrix to store coefficients of Lagrange basis polynomials
    L = np.zeros((n, n))

    # loop over each lagrange basis polynomial L_i(x)
    for i in range(n):
        # start with polynomial "1"
        V = np.array([1.0])

        # multiply factors (x - x_j) / (x_i - x_j) for j ≠ i
        for j in range(n):
            if i != j:
                # np.convolve multiplies polynomials
                V = np.convolve(V, [1.0, -x[j]]) / (x[i] - x[j])

        # store the coefficients of L_i(x) in row i
        L[i, :] = V

    # multiply y-values with Lagrange basis polynomial coefficients to get
    # the final interpolating polynomial coefficients
    C = y @ L

    # return the polynomial coefficients and lagrange basis polynomials
    return C, L


if __name__ == '__main__':
    x_points = [1, 2, 5]
    y_points = [1, 4, 16]

    C, L = lagranPoly(x_points, y_points)
    print("Polynomial coefficients (C):", C)
    print("Lagrange coefficient polynomials (L):\n", L)

    # create a polynomial function from coefficients
    p = np.poly1d(C)
    print(f"\nPolynomial as function: p(t) = \n{p}")

    # test interpolation at x = 3
    test_x = 3
    print(f"p({test_x}) =", p(test_x))


# %% Newton's Interpolating Polynomial


def newtIntpl(x: ArrayLike, y: ArrayLike, x_intrcpt: float) -> float:
    """
    Parameters:
        x          : [x0, x1, ..., xn]
        y          : [y0, y1, ..., yn]
        x_intrcpt  : value at which interpolation is calculated

    Return:
        y_intrcpt  : interpolated y-value at x_intrcpt
    """

    # convert input lists/arrays into NumPy arrays (float type for accuracy)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # number of known data points
    n = len(x)

    # ensure 'x' and 'y' arrays are of the same length
    if len(y) != n:
        raise ValueError("x and y must be of same length")

    # create an n×n zero matrix for the divided difference table
    DD = np.zeros((n, n))

    # first column (order 0 differences) is just the y-values
    DD[:, 0] = y

    # fill the divided difference table column-by-column
    for i in range(1, n):       # i → order of difference
        for j in range(i, n):       # j → row index in the table
            # formula: f[x_j, ..., x_{j-i}] = (upper - lower) / (x_j - x_{j-i})
            DD[j, i] = (DD[j, i - 1] - DD[j - 1, i - 1]) / (x[j] - x[j - i])

    # start with the first term of the Newton polynomial
    xt = 1.0               # running product term (x - x0)(x - x1)...
    y_intrcpt = DD[0, 0]   # initial value = f[x0]

    # build the polynomial value at x_intrcpt
    for i in range(n - 1):
        # multiply next factor
        xt *= (x_intrcpt - x[i])
        # add next term's contribution
        y_intrcpt += DD[i + 1, i + 1] * xt

    # return the interpolated value
    return y_intrcpt


if __name__ == '__main__':
    x_vals = [1, 4, 6, 5]
    y_vals = [0, 1.386294, 1.791760, 1.609438]

    x_intercept = 2

    y_intercept = newtIntpl(x_vals, y_vals, x_intercept)
    print(f"Interpolated value at x = {x_intercept}: {y_intercept}")

# %% Newton's Interpolating Polynomial


def newtPoly(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        x : [x0, x1, ..., xn]
        y : [y0, y1, ..., yn]

    Returns:
        C  : Polynomial coefficients of degree 'n' (highest power first)
        DD : Full divided difference table
    """

    # convert input lists/arrays into NumPy arrays (float type for accuracy)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # number of known data points
    n = len(x)

    # ensure 'x' and 'y' arrays match in length
    if len(y) != n:
        raise ValueError("x and y must be of same length")

    # create the divided difference table
    DD = np.zeros((n, n))
    # first column is just y-values
    DD[:, 0] = y

    # fill in higher-order divided differences
    for i in range(1, n):           # i → order of the difference
        for j in range(i, n):       # j → row index
            DD[j, i] = (DD[j, i - 1] - DD[j - 1, i - 1]) / (x[j] - x[j - i])

    # build the polynomial in expanded form
    # start with the last coefficient (highest-order term in Newton form)
    C = np.array([DD[n - 1, n - 1]])  # initially just the top-most coefficient

    # work backwards to expand (multiply out) the newton basis form
    for k in range(n - 2, -1, -1):
        # multiply existing polynomial by (x - x_k)
        C = np.convolve(C, np.poly1d([1, -x[k]]))
        # add the divided difference coefficient for this term
        C[-1] += DD[k, k]

    # return the polynomial coefficients and divided difference table
    return np.array(C, dtype=float), DD


if __name__ == '__main__':
    x = [1, 2, 4]
    y = [1, 4, 16]

    C, DD = newtPoly(x, y)

    print("Polynomial Coefficients:", C)
    print("Divided Difference Table:\n", DD)
