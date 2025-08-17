from numpy.typing import ArrayLike
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

# %% Least-Squares Polynomial Fit


def lstSqPoly(x: ArrayLike, y: ArrayLike, n: int) -> np.ndarray:
    """
    Parameters:
        x : array of x-values (e.g., [x0, x1, ..., xm])
        y : array of y-values (e.g., [y0, y1, ..., ym])
        n : degree of the polynomial to fit (highest degree first)

    Returns:
        np.ndarray: column vector (n+1, 1) of polynomial coefficients,
                    with the highest degree first
    """
    # ensure that x and y have the same length
    m = len(x)
    if len(y) != m:
        raise ValueError("input arrays x and y must have the same length")

    # create the Vandermonde matrix for the polynomial fit
    F = np.vander(x, n+1, increasing=True)

    # solve the normal equations: (F.T * F) * C = F.T * y
    A = F.T @ F  # matrix multiplication: F.T * F
    B = F.T @ y  # matrix multiplication: F.T * y

    # use np.linalg.solve to compute the coefficients C
    try:
        C = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("the matrix system is singular")

    # return coefficients as a column vector, with the highest degree first
    return np.flipud(C.reshape(-1, 1))


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2.3, 2.9, 4.1, 4.8, 6.2])
    degree = 4

    print("Polynomial Coefficients (highest degree first):\n",
          lstSqPoly(x, y, degree))

# %% Linear Regression Curve Fitting


def linRegr(x: ArrayLike, y: ArrayLike) -> Tuple[float, float, float]:
    """
    Parameters:
        x : array of x-values (e.g., [x0, x1, ..., xn])
        y : array of y-values (e.g., [y0, y1, ..., yn])

    Returns:
        a tuple containing:
            - b  : slope of the best-fit line
            - a  : intercept of the best-fit line
            - r2 : coefficient of determination (R²), indicating the goodness
                   of fit
   """

    # convert input to numpy arrays and ensure they are column vectors
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    n = len(x)

    # check if lengths of x and y match
    if len(y) != n:
        raise ValueError("x and y must have the same length")

    # calculate necessary summations for the linear regression formula
    sx = np.sum(x)           # sum of x-values
    sy = np.sum(y)           # sum of y-values
    sx2 = np.sum(x * x)      # sum of x^2
    sy2 = np.sum(y * y)      # sum of y^2
    sxy = np.sum(x * y)      # sum of x * y

    # compute the slope 'b' and intercept 'a' using the normal equations
    b = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
    a = (sy - b * sx) / n

    # compute the coefficient of determination 'R²'
    r2 = ((n * sxy - sx * sy) / (np.sqrt(n * sx2 - sx ** 2) *
                                 np.sqrt(n * sy2 - sy ** 2))) ** 2

    # generate x-values for the best-fit line
    xp = np.linspace(np.min(x), np.max(x), 2)
    yp = b * xp + a

    # plot the data and best-fit line
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(xp, yp, color='red', label='Best Fit Line')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression: Best Fit Line')
    plt.legend()
    plt.show()

    return b.item(), a.item(), r2.item()


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [2.2, 2.8, 3.6, 4.5, 5.1]

    b, a, r2 = linRegr(x, y)
    print(f"Slope (b): {b}")
    print(f"Intercept (a): {a}")
    print(f"R²: {r2}")
