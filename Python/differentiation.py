import numpy as np
from numpy.typing import NDArray
from typing import Callable

# %% Backward finite-difference formulae

def bkwdfnitdiff(
    f: Callable[[float], float],
    x: float,
    h: float,
    d: int
) -> NDArray[np.float64]:
    """
    Parameters:
        f : The function whose derivative is to be calculated
        x : The point at which the derivative is calculated
        h : The step size
        d : The derivative order (1 to 4)

    Returns:
        A numpy array containing two estimates of the derivative,
        the second value is usually more accurate
    """
    
    # check the derivative order and compute accordingly
    if d == 1:
        # first derivative estimates (1st and 2nd-order)
        bd1 = (f(x) - f(x - h)) / h
        bd2 = (f(x - 2*h) - 4*f(x - h) + 3*f(x)) / (2*h)
    elif d == 2:
        # second derivative estimates (1st and 2nd-order)
        bd1 = (f(x - 2*h) - 2*f(x - h) + f(x)) / h**2
        bd2 = (-f(x - 3*h) + 4*f(x - 2*h) - 5*f(x - h) + 2*f(x)) / h**2
    elif d == 3:
        # third derivative estimates (1st and 2nd-order)
        bd1 = (-f(x - 3*h) + 3*f(x - 2*h) - 3*f(x - h) + f(x)) / h**3
        bd2 = (3*f(x - 4*h) - 14*f(x - 3*h) + 24*f(x - 2*h) 
               - 18*f(x - h) + 5*f(x)) / (2*h**3)
    elif d == 4:
        # fourth derivative estimates (1st and 2nd-order)
        bd1 = (f(x - 4*h) - 4*f(x - 3*h) + 6*f(x - 2*h) 
               - 4*f(x - h) + f(x)) / h**4
        bd2 = (-2*f(x - 5*h) + 11*f(x - 4*h) - 24*f(x - 3*h) 
               + 26*f(x - 2*h) - 14*f(x - h) + 3*f(x)) / h**4
    else:
        raise ValueError("Derivative order 'd' must be between 1 and 4.")

    # return both derivative estimates in a numpy array
    return np.array([bd1, bd2])


if __name__ == "__main__":
    f = lambda x: -0.1*x**4 - 0.15*x**3 - 0.5*x**2 - 0.25*x + 1.2
    
    result = bkwdfnitdiff(f, 0.5, 0.25, 1)
    print("Derivative estimates:", result)

# %% Forward finite-difference formulae

def frwdfnitdiff(
    f: Callable[[float], float],
    x: float,
    h: float,
    d: int
) -> NDArray[np.float64]:
    """
    Parameters:
        f : the function whose derivative is to be calculated
        x : the point at which the derivative is calculated
        h : the step size
        d : the derivative order (1 to 4)

    Returns:
        A numpy array containing two estimates of the derivative,
        the second value is typically more accurate
    """
    
    # check the derivative order and compute accordingly
    if d == 1:
        # first derivative estimates (1st and 2nd-order)
        fd1 = (f(x + h) - f(x)) / h
        fd2 = (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2*h)
    elif d == 2:
        # second derivative estimates (1st and 2nd-order)
        fd1 = (f(x + 2*h) - 2*f(x + h) + f(x)) / h**2
        fd2 = (-f(x + 3*h) + 4*f(x + 2*h) - 5*f(x + h) + 2*f(x)) / h**2
    elif d == 3:
        # third derivative estimates (1st and 2nd-order)
        fd1 = (f(x + 3*h) - 3*f(x + 2*h) + 3*f(x + h) - f(x)) / h**3
        fd2 = (-3*f(x + 4*h) + 14*f(x + 3*h) - 24*f(x + 2*h) + 18*f(x + h) - 5*f(x)) / (2*h**3)
    elif d == 4:
        # fourth derivative estimates (1st and 2nd-order)
        fd1 = (f(x + 4*h) - 4*f(x + 3*h) + 6*f(x + 2*h) 
               - 4*f(x + h) + f(x)) / h**4
        fd2 = (-2*f(x + 5*h) + 11*f(x + 4*h) - 24*f(x + 3*h) 
               + 26*f(x + 2*h) - 14*f(x + h) + 3*f(x)) / h**4
    else:
        raise ValueError("Derivative order 'd' must be between 1 and 4.")

    # return both derivative estimates in a numpy array
    return np.array([fd1, fd2])


if __name__ == '__main__':
    f = lambda x: -0.1*x**4 - 0.15*x**3 - 0.5*x**2 - 0.25*x + 1.2
    
    result = frwdfnitdiff(f, 0.5, 0.25, 1)
    print("Derivative estimates:", result)

# %% Centered finite-difference formulae

def cntfnitdiff(
    f: Callable[[float], float],
    x: float,
    h: float,
    d: int
) -> NDArray[np.float64]:
    """
    Parameters:
        f : the function whose derivative is to be calculated
        x : the point at which the derivative is evaluated
        h : the step size
        d : the derivative order (1 to 4)

    Returns:
        A numpy array containing two derivative estimates,
        the second value is typically more accurate.
    """

    # check the derivative order and compute accordingly
    if d == 1:
        # first derivative estimates (1st and 2nd-order)
        cntd1 = (f(x + h) - f(x - h)) / (2 * h)
        cntd2 = (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)
    elif d == 2:
        # second derivative estimates (1st and 2nd-order)
        cntd1 = (f(x + h) - 2*f(x) + f(x - h)) / h**2
        cntd2 = (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h**2)
    elif d == 3:
        # third derivative estimates (1st and 2nd-order)
        cntd1 = (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (8 * h**3)
        cntd2 = (-f(x + 3*h) + 8*f(x + 2*h) - 13*f(x + h) + 13*f(x - h)
                 - 8*f(x - 2*h) + f(x - 3*h)) / (8 * h**3)
    elif d == 4:
        # fourth derivative estimates (1st and 2nd-order)
        cntd1 = (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / h**4
        cntd2 = (-f(x + 3*h) + 12*f(x + 2*h) - 39*f(x + h) + 56*f(x)
                 - 39*f(x - h) + 12*f(x - 2*h) - f(x - 3*h)) / (6 * h**4)
    else:
        raise ValueError("Derivative order 'd' must be between 1 and 4.")

    # return both derivative estimates in a numpy array
    return np.array([cntd1, cntd2])


if __name__ == '__main__':
    f = lambda x: -0.1*x**4 - 0.15*x**3 - 0.5*x**2 - 0.25*x + 1.2
    
    result = cntfnitdiff(f, 0.5, 0.25, 1)
    print("Derivative estimates:", result)
