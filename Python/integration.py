import numpy as np
from typing import Callable
from numpy.typing import NDArray

# %% Composite Boole's Rule


def boole(func: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Parameters:
        func : function to integrate
        a, b : lower and upper bounds
        n    : number of intervals (must be multiple of 4)

    Returns:
        I    : integral estimate
    """

    # ensure integration bounds are valid
    if a > b:
        raise ValueError("b must be greater than a")

    # boole's rule requires number of intervals to be multiple of 4
    if n % 4 != 0:
        raise ValueError("n must be a multiple of 4")

    # arrays to store x-values and corresponding function y-values
    x = np.zeros(n - 1)
    y = np.zeros(n - 1)

    # accumulators for the different weighted sums in boole's rule
    sum_ = 0.0       # for terms where i % 4 == 0
    sum_2 = 0.0      # for terms where i % 4 == 2
    sum_1_3 = 0.0    # for terms where i % 4 == 1 or 3

    # step size
    h = (b - a) / n

    # loop through internal points (excluding a and b)
    for i in range(1, n):
        # compute the x-value for this point
        x[i - 1] = a + i * h
        # evaluate function at this x-value
        y[i - 1] = func(x[i - 1])

        # distribute the term into the correct sum based on position
        if i % 4 == 1 or i % 4 == 3:
            sum_1_3 += y[i - 1]   # weight 32
        elif i % 4 == 2:
            sum_2 += y[i - 1]     # weight 12
        else:
            sum_ += y[i - 1]      # weight 14

    # apply formula:
    I = (2 * h / 45) * (7 * (func(a) + func(b)) +
                        32 * sum_1_3 + 12 * sum_2 + 14 * sum_)

    return I


if __name__ == "__main__":
    def f(x): return np.exp(-x**2)

    result = boole(f, 0, 1, 8)
    print("Approximate integral:", result)

# %% Composite Simpson's 1/3 Rule


def simpson_one_third(func: Callable[[float], float], a: float,
                      b: float, n: int) -> float:
    """
    Parameters:
        func : function to integrate
        a, b : lower and upper bounds of integration
        n    : number of subintervals (must be even)

    Returns:
        I    : approximate integral using Simpson's 1/3 Rule
    """

    # ceck that lower bound is less than upper bound
    if a > b:
        raise ValueError("'b' must be greater than 'a'")

    # simpson's 1/3 rule requires an even number of intervals
    if n % 2 == 1:
        raise ValueError("'n' must be even")

    # step size
    h = (b - a) / n

    # arrays to store x-values and corresponding function y-values
    x = np.zeros(n - 1)
    y = np.zeros(n - 1)

    # accumulators for sums
    sum_odd = 0.0   # sum of y-values at odd-indexed interior points
    sum_even = 0.0  # sum of y-values at even-indexed interior points

    # loop through all interior points
    for i in range(1, n):
        # compute x_i (excluding the endpoints a and b)
        x[i - 1] = a + i * h
        # evaluate the function at x_i
        y[i - 1] = func(x[i - 1])

        # distribute the term into the correct sum based on position
        if i % 2 == 1:
            sum_odd += y[i - 1]     # weight 4
        else:
            sum_even += y[i - 1]    # weight 2

    # apply formula:
    I = (h / 3) * (func(a) + func(b) + 4 * sum_odd + 2 * sum_even)

    return I


if __name__ == "__main__":
    def f(x): return np.sin(x)

    result = simpson_one_third(f, 0, np.pi, 10)
    print(f"Approximate integral: {result}")

# %% Composite Simpson's 3/8 Rule


def simpson3by8(func: Callable[[float], float], a: float,
                b: float, n: int) -> float:
    """
    Parameters:
        func : function to integrate
        a, b : lower and upper bounds of integration
        n    : number of subintervals (must be a multiple of 3)

    Returns:
        I    : approximate integral using Simpson's 3/8 Rule
    """

    # ensure lower bound is less than upper bound
    if a > b:
        raise ValueError("b must be greater than a")

    # simpson's 3/8 rule requires number of intervals to be a multiple of 3
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3")

    # arrays to store x-values and corresponding function y-values
    x = np.zeros(n - 1)
    y = np.zeros(n - 1)

    # accumulators for weighted sums
    sum_3 = 0.0       # for points where i % 3 == 0
    sum_other = 0.0   # for all other interior points

    # step size
    h = (b - a) / n

    # loop over all interior points
    for i in range(1, n):
        # compute x_i (excluding the endpoints a and b)
        x[i - 1] = a + i * h
        # evaluate the function at x_i
        y[i - 1] = func(x[i - 1])

        # distribute the term into the correct sum based on position
        if i % 3 == 0:
            sum_3 += y[i - 1]       # weight 2
        else:
            sum_other += y[i - 1]   # weight 3

    # apply formula:
    I = (3 * h / 8) * (func(a) + func(b) + 2 * sum_3 + 3 * sum_other)

    return I


if __name__ == '__main__':
    def f(x): return np.exp(-x**2)

    result = simpson3by8(f, 0, 1, 6)
    print("Approximate integral:", result)


# %% Unequally Spaced Trapezoidal Rule Quadrature

def trapuneq(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Parameters:
        x : independent variable values (must be monotonically ascending)
        y : dependent variable values

    Returns:
        I : Integral estimate using the trapezoidal rule for unequal intervals
    """

    # convert input to NumPy arrays with float dtype
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # number of data points
    n = len(x)

    # must have at least two points to form a trapezoid
    if n < 2:
        raise ValueError("At least 2 input arguments required.")

    # check that x-values are strictly non-decreasing (monotonically ascending)
    if np.any(np.diff(x) < 0):
        raise ValueError("x is not monotonically ascending.")

    # check that x and y arrays are the same length
    if n != len(y):
        raise ValueError("x and y must be of same length.")

    # initialize integral sum
    I = 0.0

    # loop through each interval and apply trapezoidal area formula
    # area of each trapezoid = 0.5 * base * (sum of parallel sides)
    # here, base = (x[k+1] - x[k]), parallel sides = y[k] and y[k+1]
    for k in range(n - 1):
        I += 0.5 * (x[k+1] - x[k]) * (y[k] + y[k+1])

    return I


if __name__ == '__main__':
    x = [0, 1, 1.5, 2]
    y = [0, 1, 2, 1.5]

    result = trapuneq(x, y)
    print("Approximate integral:", result)

# %% Composite Trapezoidal Rule


def trpzds(func: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Parameters:
        func : function to integrate
        a, b : lower and upper bounds
        n    : number of intervals (must be even)

    Returns:
        I    : integral estimate using the trapezoidal rule
    """

    # ensure integration limits are valid
    if a > b:
        raise ValueError("b must be greater than a")

    # 'n' must be even
    if n % 2 == 1:
        raise ValueError("n must be even")

    # arrays to store x-values and corresponding function y-values
    x = np.zeros(n - 1)
    y = np.zeros(n - 1)

    # accumulator for the sum of interior function values
    total_sum = 0.0

    # step size
    h = (b - a) / n

    # loop over interior points (excluding a and b)
    for i in range(1, n):
        # compute x_i (excluding the endpoints a and b)
        x[i - 1] = a + i * h
        # evaluate the function at x_i
        y[i - 1] = func(x[i - 1])
        # add this value to the total sum
        total_sum += y[i - 1]

    # apply formula:
    I = h / 2 * (func(a) + 2 * total_sum + func(b))

    return I


if __name__ == '__main__':
    def f(x): return np.sin(x)

    result = trpzds(f, 0, np.pi, 10)
    print("Approximate integral:", result)


# %% Composite Weddle's Rule


def weddle(func: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Parameters:
        func : function to integrate
        a, b : lower and upper bounds of integration
        n    : number of intervals (must be a multiple of 6)

    Returns:
        I    : integral estimate using Weddle's rule
    """

    # ensure integration limits are valid
    if a > b:
        raise ValueError("b must be greater than a")

    # weddle's rule requires the number of subintervals to be a multiple of 6
    if n % 6 != 0:
        raise ValueError("n must be multiple of 6")

    # arrays for storing intermediate x-values and their function values
    x = np.zeros(n - 1)
    y = np.zeros(n - 1)

    # accumulators for sums corresponding to different coefficients in Weddle's rule
    sum_1_5 = 0      # for terms where i mod 6 == 1 or 5
    sum_mid = 0      # for terms where i mod 6 == 2 or 4
    sum_3 = 0      # for terms where i mod 6 == 3
    sum_6 = 0      # for terms where i mod 6 == 0

    # step size
    h = (b - a) / n

    # loop over interior points only (exclude a and b)
    for i in range(1, n):
        # compute x_i (excluding the endpoints a and b)
        x[i - 1] = a + i * h
        # evaluate the function at x_i
        y[i - 1] = func(x[i - 1])

        # classify and add the term to the appropriate sum based on its position
        if i % 6 == 1 or i % 6 == 5:
            sum_1_5 += y[i - 1]  # weight 5
        elif i % 6 == 0:
            sum_6 += y[i - 1]    # weight 2
        elif i % 6 == 3:
            sum_3 += y[i - 1]    # weight 6
        else:
            sum_mid += y[i - 1]  # weight 1

    # apply formula:
    I = (3 * h / 10) * (func(a) + sum_mid + 2 *
                        sum_6 + 6 * sum_3 + 5 * sum_1_5 + func(b))

    return I


if __name__ == '__main__':
    def f(x): return np.sin(x)

    result = weddle(f, 0, np.pi, 12)

# %% Gauss-Legendre Integration/Quadrature


def gausslegend(f: Callable[[float], float], a: float, b: float, g: int) -> float:
    """
    Parameters:
        f : function to integrate
        a, b : lower and upper bounds of integration
        g : number of Gauss–Legendre points (allowed values: 2, 3, 4)

    Returns:
        I : approximate integral using the g-point Gauss–Legendre rule
    """

    # select Gauss–Legendre nodes (x) and weights (w) based on g
    if g == 2:
        w = np.array([1.0, 1.0])
        x = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    elif g == 3:
        w = np.array([5/9, 8/9, 5/9])
        x = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    elif g == 4:
        w = np.array([
            0.347854845137454, 0.347854845137454,
            0.652145154862546, 0.652145154862546
        ])
        x = np.array([
            -0.861136311594053,  0.861136311594053,
            -0.339981043584856,  0.339981043584856
        ])
    else:
        # if g is not in the set {2, 3, 4}, integration is not supported
        raise ValueError("'g' is out of range, can't calculate integral.")

    # transform nodes from the standard interval [-1, 1] to [a, b]
    t = ((a + b) / 2) - ((b - a) / 2) * x

    # apply formula:
    I = ((b - a) / 2) * np.sum(w * f(t))

    return I


if __name__ == '__main__':
    def f(x): return np.exp(-x**2)

    result = gausslegend(f, 0, 1, 3)
    print("Approximate integral:", result)
