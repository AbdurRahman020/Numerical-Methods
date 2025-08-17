import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from numpy.typing import NDArray

# %% Runge-Kutta 2 (Heun's) Method


def RK2System(
    func: list[Callable[[float, NDArray[np.float64]], float]],
    a: float,
    b: float,
    n: int,
    y_initial: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        func_list : each callable func(t, y) should return dy/dt for that equation
        a, b      : initial and final 't'
        n         : number of intervals
        y_initial : initial value of y

    Returns:
        vt : array of t values
        vy : array of y values (solution)
    """

    # heun's method parameters (weights and stage coefficients for RK2)
    w1, w2 = 1/2, 1/2   # weights for slope averaging
    p1, q11 = 1, 1      # coefficients for intermediate step

    # number of equations in the system (length of funcs list)
    m = len(func)

    # time array (n+1 points for n intervals)
    vt = np.zeros(n + 1)
    # solution array for all equations; shape (m equations × n+1 time points)
    vy = np.zeros((m, n + 1))

    # temporary arrays to store intermediate RK2 slopes
    k1 = np.zeros(m)
    k2 = np.zeros(m)

    # step size
    h = (b - a) / n

    # initial time
    t = a
    # initial values of y (converted to float array)
    y = np.array(y_initial, dtype=float)

    # store initial values
    vt[0] = t
    vy[:, 0] = y

    # main loop for each time step
    for i in range(n):
        # compute first slope k1 for all equations
        for j in range(m):
            k1[j] = func[j](t, y)

        # compute second slope k2 for all equations (predictor step)
        for j in range(m):
            k2[j] = func[j](t + p1 * h, y + q11 * k1 * h)

        # advance time
        t = a + (i + 1) * h

        # weighted average of slopes
        phi = w1 * k1 + w2 * k2

        # update solution vector
        y = y + phi * h

        # print intermediate results for each step
        print(f"i = {i+1:>2d}\t\t t = {t:.2f}\t\t", end="")
        for k in range(m):
            print(f"y({k+1}) = {y[k]:.5f}\t\t", end="")
        print("")

        # store results
        vt[i + 1] = t
        vy[:, i + 1] = y

    # plotting results for each equation
    for k in range(m):
        plt.plot(vt, vy[k, :], '-o', label=f'y{k+1}')

    plt.title("RK2 System")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

    return vt, vy


if __name__ == "__main__":
    def f1(t, y): return y[1]
    def f2(t, y): return -y[0]

    funcs = [f1, f2]
    a, b = 0, 10
    n = 20
    y0 = [0, 1]

    vt, vy = RK2System(funcs, a, b, n, y0)


# %% Runge-Kutta 3 Method

def RK3System(
    func: list[Callable[[float, NDArray[np.float64]], float]],
    a: float,
    b: float,
    n: int,
    y_initial: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        func_list : each callable func(t, y) should return dy/dt for that equation
        a, b      : initial and final 't'
        n         : number of intervals
        y_initial : initial value of y

    Returns:
        vt : array of t values
        vy : array of y values (solution)
    """

    # number of equations in the system (length of funcs list)
    m = len(funcs)

    # time array (n+1 points for n intervals)
    vt = np.zeros(n+1)
    # solution array for all equations; shape (m equations × n+1 time points)
    vy = np.zeros((m, n+1))

    # temporary arrays to store intermediate RK3 slopes
    k1 = np.zeros(m)
    k2 = np.zeros(m)
    k3 = np.zeros(m)

    # step size
    h = (b - a) / n

    # initial time
    t = a
    # initial values of y (converted to float array)
    y = np.array(y_initial, dtype=float)

    # store initial values in result arrays
    vt[0] = t
    vy[:, 0] = y

    # ain loop over each time step
    for i in range(1, n+1):
        # compute k1 for each equation
        for j in range(m):
            k1[j] = funcs[j](t, y)

        # compute k2 using k1 (midpoint estimate)
        for j in range(m):
            k2[j] = funcs[j](t + h/2, y + k1*h/2)

        # compute k3 using k1 and k2 (end-point estimate)
        for j in range(m):
            k3[j] = funcs[j](t + h, y - k1*h + 2*k2*h)

        # advance time
        t = a + i*h

        # weighted average slope for RK3
        phi = (k1 + 4*k2 + k3) / 6

        # update 'y' using RK3 formula
        y = y + phi * h

        # print step-by-step results to console
        print(f"i = {i:>2d}\t\t t = {t:.2f}\t", end="\t")
        for k in range(m):
            print(f"y({k+1}) = {y[k]:.5f}\t", end="\t")
        print()

        # store updated values in arrays
        vt[i] = t
        vy[:, i] = y

    # plot results for each equation
    for k in range(m):
        plt.plot(vt, vy[k, :], '-o', label=f'y{k+1}')

    plt.title("RK3 System")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

    return vt, vy


if __name__ == "__main__":
    def f1(t, y): return y[1]
    def f2(t, y): return -y[0]

    funcs = [f1, f2]
    a, b = 0, 10
    n = 20
    y_initial = [1, 0]

    vt, vy = RK3System(funcs, a, b, n, y_initial)


# %% Runge-Kutta 4 Method

def RK4System(
    func_list: list[Callable[[float, NDArray[np.float64]], float]],
    a: float,
    b: float,
    n: int,
    y_initial: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters:
        func_list : each callable func(t, y) should return dy/dt for that equation
        a, b      : initial and final 't'
        n         : number of intervals
        y_initial : initial value of y

    Returns:
        vt : array of t values
        vy : array of y values (solution)
    """

    # number of equations in the system (length of funcs list)
    m = len(func_list)

    # time array (n+1 points for n intervals)
    vt = np.zeros(n+1)
    # solution array for all equations; shape (m equations × n+1 time points)
    vy = np.zeros((m, n+1))

    # temporary arrays to store intermediate RK4 slopes
    k1 = np.zeros(m)
    k2 = np.zeros(m)
    k3 = np.zeros(m)
    k4 = np.zeros(m)

    # step size
    h = (b - a) / n

    # initialize time
    t = a
    # initial values of y (converted to float array)
    y = np.array(y_initial, dtype=float)

    # store initial values in result arrays
    vt[0] = t
    vy[:, 0] = y

    # main loop for each time step
    for i in range(1, n+1):
        # compute k1 = f(t, y)
        for j in range(m):
            k1[j] = func_list[j](t, y)

        # compute k2 = f(t + h/2, y + (k1*h)/2)
        for j in range(m):
            k2[j] = func_list[j](t + h/2, y + k1 * h/2)

        # compute k3 = f(t + h/2, y + (k2*h)/2)
        for j in range(m):
            k3[j] = func_list[j](t + h/2, y + k2 * h/2)

        # compute k4 = f(t + h, y + k3*h)
        for j in range(m):
            k4[j] = func_list[j](t + h, y + k3 * h)

        # update time
        t = a + i * h

        # weighted average of slopes (RK4 formula)
        phi = (k1 + 2*k2 + 2*k3 + k4) / 6

        # update solution
        y = y + phi * h

        # print progress for each iteration
        print(f"i = {i:>2d}\t\t t = {t:.2f}\t\t", end="")
        for k in range(m):
            print(f"y({k+1}) = {y[k]:.5f}\t\t", end="")
        print()

        # store results
        vt[i] = t
        vy[:, i] = y

    # plot results for each equation
    for k in range(m):
        plt.plot(vt, vy[k, :], '-o', label=f'y{k+1}')

    plt.title("RK4 System")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

    return vt, vy


if __name__ == "__main__":
    def f1(t, y): return y[1]
    def f2(t, y): return -y[0]

    funcs = [f1, f2]
    a, b = 0, 10
    n = 20
    y_initial = [1, 0]

    vt, vy = RK4System(funcs, a, b, n, y_initial)
