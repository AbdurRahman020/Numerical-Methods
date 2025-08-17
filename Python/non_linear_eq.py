import numpy as np
from typing import Callable

# %% Bisection Method


def bisection(func: Callable[[float], float],
              xl: float,
              xu: float,
              n: int = 25,
              delta: float = 1e-4
              ) -> float:
    """
    Parameters:
        func   : the function to find the root of
        xl, xu : left and right interval endpoints
        n      : number of iterations (default = 25)
        delta  : tolerance for x0 (default = 0.0001)

    Return:
        xr : float, approximation to the zero of func
    """

    # check if the initial interval is valid (signs must be opposite)
    if func(xl) * func(xu) >= 0:
        raise ValueError("No root between given interval.")

    # perform the bisection loop for a maximum of 'n' iterations
    for i in range(1, n+1):
        # find the midpoint of the current interval
        xr = (xl + xu) / 2.0

        # stop early if the interval is smaller than the tolerance 'delta'
        if abs(xr - xu) < delta or abs(xr - xl) < delta:
            break

        # display the iteration number, midpoint, and function value at midpoint
        print(f"{i}\t\txr = {xr:.6f}\t f(xr) = {func(xr):.6f}")

        # determine which subinterval contains the root
        if func(xl) * func(xr) < 0:
            # root lies between xl and xr → update xu
            xu = xr
        elif func(xu) * func(xr) < 0:
            # root lies between xr and xu → update xl
            xl = xr

    # return the final midpoint as the approximate root
    return xr


if __name__ == '__main__':
    def f(x): return np.cos(x) - x

    root_approx = bisection(f, 0, 1)
    print("\nApproximate root:", root_approx)

# %% Fixed-Point Iteration


def fixPtIt(
    func: Callable[[float], float],
    x0: float,
    n: int = 25,
    delta: float = 1e-4
) -> float:
    """
    Parameters:
        func  : the function to find the root of
        x0    : initial guess
        n     : number of iterations (default = 25)
        delta : tolerance for x0 (default = 0.0001)

    Return:
        xr : approximation to zero of 'func'
    """

    # ensure at least the function and initial guess are provided
    if func is None or x0 is None:
        raise ValueError("At least 2 input arguments are required.")

    # iterate for at most 'n' iterations
    for i in range(1, n + 1):
        # apply the fixed-point iteration formula: x_{r} = g(x_{0})
        xr = func(x0)

        # display the iteration number and the new approximation
        print(f"{i}\txr = {xr:.10f}")

        # check stopping condition: if the change is smaller than 'delta'
        if abs(xr - x0) < delta:
            break

        # update x0 for the next iteration
        x0 = xr

    # return the final approximation of the root
    return xr


if __name__ == "__main__":
    def g(x): return np.cos(x)

    root_approx = fixPtIt(g, x0=0.5)
    print("\nApproximate root:", root_approx)

# %% Newton-Raphson Iteration


def nwtRaph(
    func: Callable[[float], float],
    dfunc: Callable[[float], float],
    x0: float,
    n: int = 25,
    delta: float = 1e-4
) -> float:
    """
    Parameters:
        func  : the function to find the root of
        dfunc : derivative of func
        x0    : initial approximation to zero of 'func'
        n     : number of iterations (default = 25)
        delta : tolerance for x0 (default = 0.0001)

    Return:
        xr : approximation to zero of 'func'
    """

    # ensure at least the function and initial guess are provided
    if func is None or dfunc is None or x0 is None:
        raise ValueError("At least 3 input arguments are required.")

    # check that the derivative is not zero at the initial guess
    if dfunc(x0) != 0:
        # perform at most 'n' iterations
        for i in range(1, n+1):
            # newton-raphson formula: xr = x0 - f(x0) / f'(x0)
            xr = x0 - func(x0) / dfunc(x0)

            # display current iteration number, xr, and f(xr)
            print(f"{i}\txr = {xr:.6f}\t f(xr) = {func(xr):.8f}")

            # stopping criterion: if the change is less than 'delta'
            if abs(xr - x0) < delta:
                break

            # check if derivative is zero at the new approximation,
            # to avoid division by zero
            if dfunc(xr) == 0:
                raise ZeroDivisionError("Newton-Raphson method has failed.")

            # update guess for the next iteration
            x0 = xr
    else:
        # derivative was zero at the starting point → can't proceed
        raise ZeroDivisionError("Newton-Raphson method has failed.")

    # return the final approximation of the root
    return xr


if __name__ == '__main__':
    def f(x): return -0.1*x**4 - 0.15*x**3 - 0.5*x**2 - 0.25*x + 1.2
    def f_prime(x): return -0.4*x**3 - 0.45*x**2 - 1.0*x - 0.25

    root = nwtRaph(f, f_prime, x0=1.0)
    print("\nRoot found:", root)

# %% Regula-Falsi (False Position) Method


def regulaFalsi(
    func: Callable[[float], float],
    xl: float,
    xu: float,
    n: int = 25,
    delta: float = 1e-4
) -> float:
    """
    Parameters:
        func  : the function to find the root of
        xl, xu: left and right end points of the interval [xl, xu]
        n     : maximum number of iterations (default = 25)
        delta : tolerance for |f(xr)| (default = 0.0001)

    Return:
        xr : approximation to zero of 'func'
    """

    # validate input arguments for bounds
    if xl is None or xu is None:
        raise ValueError("At least 3 input arguments are required.")

    # calculate function values at the endpoints
    yl = func(xl)
    yu = func(xu)

    # check that the function changes sign over the interval [xl, xu]
    if yl * yu < 0 and xl < xu:
        # perform at most 'n' iterations
        for i in range(1, n+1):
            # compute new estimate using false position formula
            xr = (xl * yu - xu * yl) / (yu - yl)
            # evaluate function at new estimate
            yr = func(xr)

            # display iteration number, xr, and f(xr)
            print(f"{i}\txr = {xr:.6f}\t f(xr) = {yr:.6f}")

            # stopping criterion: |f(xr)| is within the tolerance
            if abs(yr) < delta:
                break

            # determine which subinterval to keep for the next iteration
            if yl * yr < 0:
                # root lies between 'xl' and 'xr' → move upper bound
                xu = xr
                yu = yr
            elif yu * yr < 0:
                # root lies between 'xr' and 'xu' → move lower bound
                xl = xr
                yl = yr
    else:
        # invalid bounds or no sign change detected
        raise ValueError(
            "Invalid initial guesses: function must change sign over [xl, xu]")

    # return final approximation
    return xr


if __name__ == '__main__':
    def f(x): return x**3 - x - 2

    root = regulaFalsi(f, 1, 2)
    print("\nApproximate root:", root)

# %% Secant Method


def secant(
    func: Callable[[float], float],
    xl: float,
    xu: float,
    n: int = 25,
    delta: float = 1e-4
) -> float:
    """
    Parameters:
        func   : the function to find the root of
        xl, xu : two initial guesses for the root
        n      : maximum number of iterations (default = 25)
        delta  : tolerance for change in xr between iterations (default = 0.0001)

    Return:
        xr : approximation to zero of 'func'
    """

    # iterate up to 'n' times
    for i in range(1, n+1):
        # apply secant method formula to estimate the new root
        xr = xu - func(xu) * (xu - xl) / (func(xu) - func(xl))

        # display current iteration number, estimate, and function value
        print(f"{i}\tx2 = {xr:.8f}\tf(xr) = {func(xr):.8f}")

        # stopping criterion: if change in successive 'xr' is within the tolerance
        if abs(xr - xu) < delta:
            break

        # update the two most recent guesses for the next iteration
        xl, xu = xu, xr

    # return the final Approximate root
    return xr


if __name__ == '__main__':
    def f(x): return x**3 - x - 2

    root = secant(f, 1, 2)
    print("\nApproximate root:", root)
