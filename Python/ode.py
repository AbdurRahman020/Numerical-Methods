import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# %% Euler's Method


def euler(
    func: Callable[[float, float], float],
    a: float,
    b: float,
    n: int,
    y_initial: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        func      : the function representing the ODE dy/dt = f(t, y)
        a         : the start value of 't' (initial time)
        b         : the end value of 't' (final time)
        n         : number of steps (intervals)
        y_initial : the initial condition y(t_0)

    Returns:
        t_values, y_values : arrays of 't' and 'y' values at each step
    """

    # create empty arrays to store the results
    t_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # step size
    h = (b - a) / n

    # initial conditions
    t = a
    y = y_initial
    t_values[0] = t
    y_values[0] = y

    # print the initial condition
    print(f"Step 00: t = {t:.4f}, y = {y:.4f}")

    # loop through and apply Euler's method
    for i in range(1, n + 1):
        # calculate the next t value
        t = a + i * h
        # calculate the derivative at the current point
        f_t_y = func(t, y)
        # update y
        y += h * f_t_y

        # store the values
        t_values[i] = t
        y_values[i] = y

        # print progress for each step
        print(f"Step {i:02d}: t = {t:.4f}, y = {y:.4f}")

    # plot the results
    plt.plot(t_values, y_values, '-o', label="Euler's Method")
    plt.title("Euler's Method for ODE")
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return t_values, y_values


if __name__ == '__main__':
    def dydt(t, y): return y - t**2 + 1

    t_vals, y_vals = euler(dydt, a=0, b=2, n=10, y_initial=0.5)


# %% Midpoint Method for solving ODEs

def midpoint(
    func: Callable[[float, float], float],
    a: float,
    b: float,
    n: int,
    y_initial: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        func      : the function representing the ODE dy/dt = f(t, y)
        a         : the start value of 't' (initial time)
        b         : the end value of 't' (final time)
        n         : number of steps (intervals)
        y_initial : the initial condition y(t_0)

    Returns:
       t_values, y_values : arrays of 't' and 'y' values at each step
    """

    # initialize arrays to store 't' and 'y' values
    t_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # step size
    h = (b - a) / n

    # initial conditions
    t = a
    y = y_initial
    t_values[0] = t
    y_values[0] = y

    # print initial condition
    print(f"Step 00: t = {t:.4f}, y = {y:.4f}")

    # midpoint method loop
    for i in range(1, n + 1):
        # calculate the next t value
        t = a + i * h
        # calculate the midpoint slope and update 'y'
        mid_slope = func(t + h / 2, y + h / 2 * func(t, y))
        y += h * mid_slope

        # store computed values
        t_values[i] = t
        y_values[i] = y

        # print progress for each step
        print(f"Step {i:02d}: t = {t:.4f}, y = {y:.4f}")

    # plot the results
    plt.plot(t_values, y_values, '-o', label="Midpoint Method")
    plt.title("Midpoint Method for ODE")
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return t_values, y_values


if __name__ == '__main__':
    def dydt(t, y): return y - t**2 + 1

    t_vals, y_vals = midpoint(dydt, a=0, b=2, n=10, y_initial=0.5)


# %% Taylor's (Order Two) Method

def taylor2(
    func: Callable[[float, float], float],
    dfunc1: Callable[[float, float], float],
    a: float,
    b: float,
    n: int,
    y_initial: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        func      : the function representing the ODE dy/dt = f(t, y)
        dfunc1    : the first derivative of f(t, y) with respect to 't' and 'y'
        a         : the start value of 't' (initial time)
        b         : the end value of 't' (final time)
        n         : number of steps (intervals)
        y_initial : the initial condition y(t_0)

    Returns:
       t_values, y_values : arrays of 't' and 'y' values at each step
    """

    # initialize arrays for 't' and 'y' values
    t_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # step size
    h = (b - a) / n

    # set initial conditions
    t = a
    y = y_initial
    t_values[0] = t
    y_values[0] = y

    # calculate the function and its derivative at the starting point
    f_t_y = func(t, y)
    df_t_y = dfunc1(t, y)

    # print the initial values
    print(f"Step {0:02d}: t = {t:.4f}, y = {y:.4f}")

    # apply Taylor's method
    for i in range(1, n + 1):
        # calculate the next 't' value
        t = a + i * h
        # update 'y' using Taylor's 2nd order method
        y += h * (f_t_y + df_t_y * h / 2)

        # update the function and its derivative for the next step
        f_t_y = func(t, y)
        df_t_y = dfunc1(t, y)

        # store the results
        t_values[i] = t
        y_values[i] = y

        # print progress for each step
        print(f"Step {i:02d}: t = {t:.4f}, y = {y:.4f}")

    # plot the results
    plt.plot(t_values, y_values, '-o', label="Taylor's 2nd Order")
    plt.title("Taylor's (Order Two) Method for ODE")
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return t_values, y_values


if __name__ == '__main__':
    def f(t, y): return y - t**2 + 1
    def df1(t, y): return -2*t + (y - t**2 + 1)

    t_vals, y_vals = taylor2(f, df1, a=0, b=2, n=10, y_initial=0.5)


# %%  Taylor's (Order Four) Method

def taylor4(
    func: Callable[[float, float], float],
    dfunc1: Callable[[float, float], float],
    dfunc2: Callable[[float, float], float],
    dfunc3: Callable[[float, float], float],
    a: float,
    b: float,
    n: int,
    y_initial: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        func      : the function representing the ODE dy/dt = f(t, y)
        dfunc1    : the first derivative of f(t, y) with respect to 't' and 'y'
        dfunc2    : the second derivative of f(t, y) with respect to 't' and 'y'
        dfunc3    : the third derivative of f(t, y) with respect to 't' and 'y'
        a         : the start value of 't' (initial time)
        b         : the end value of 't' (final time)
        n         : number of steps (intervals)
        y_initial : the initial condition y(t_0)

    Returns:
       t_values, y_values : arrays of 't' and 'y' values at each step
    """

    # initialize arrays for 't' and 'y' values
    t_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # calculate step size
    h = (b - a) / n

    # set initial conditions
    t = a
    y = y_initial
    t_values[0] = t
    y_values[0] = y

    # compute the first, second, and third derivatives at the initial point
    f_t_y = func(t, y)
    df_t_y = dfunc1(t, y)
    d2f_t_y = dfunc2(t, y)
    d3f_t_y = dfunc3(t, y)

    # print the initial condition
    print(f"Step {0:02d}: t = {t:.4f}, y = {y:.4f}")

    # apply Taylor's 4th Order method to compute the solution
    for i in range(1, n + 1):
        t = a + i * h
        # update 'y' using Taylor's 4th order method
        y += h * (f_t_y + df_t_y * h / 2 + d2f_t_y *
                  (h**2) / 6 + d3f_t_y * (h**3) / 24)

        # update the function and its derivatives for the next step
        f_t_y = func(t, y)
        df_t_y = dfunc1(t, y)
        d2f_t_y = dfunc2(t, y)
        d3f_t_y = dfunc3(t, y)

        # store the values
        t_values[i] = t
        y_values[i] = y

        # print the progress after each step
        print(f"Step {i:02d}: t = {t:.4f}, y = {y:.4f}")

    # plot the results
    plt.plot(t_values, y_values, '-o', label="Taylor's 4th Order")
    plt.title("Taylor's (Order Four) Method for ODE")
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return t_values, y_values


if __name__ == '__main__':
    def func(t, y): return y - t**2 + 1
    def dfunc1(t, y): return func(t, y)
    def dfunc2(t, y): return -2
    def dfunc3(t, y): return 0

    t_vals, y_vals = taylor4(func, dfunc1, dfunc2,
                             dfunc3, a=0, b=2, n=10, y_initial=0.5)


# %% Runge-Kutta 2 (RK2) Method

def RK2(
    func: Callable[[float, float], float],
    a: float,
    b: float,
    n: int,
    y_initial: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        func      : the function representing the ODE dy/dt = f(t, y)
        a         : the start value of 't' (initial time)
        b         : the end value of 't' (final time)
        n         : number of steps (intervals)
        y_initial : the initial condition y(t_0)

    Returns:
       t_values, y_values : arrays of 't' and 'y' values at each step

    Method used:
        Heun's Method, a second-order Runge-Kutta method where:
            - w1 = 1/2, w2 = 1/2 (weights)
            - p1 = 1, q11 = 1 (coefficients for the Runge-Kutta method)
    """

    # Heun's method constants
    w1, w2, p1, q11 = 1/2, 1/2, 1, 1

    # initialize arrays for 't' and 'y' values
    t_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # calculate the step size
    h = (b - a) / n

    # set initial conditions
    t = a
    y = y_initial
    t_values[0] = t
    y_values[0] = y

    # print the initial values
    print(f"Step {0:02d}: t = {t:.4f}, y = {y:.4f}")

    # apply Runge-Kutta method (Heun's Method) to compute the solution
    for i in range(1, n + 1):
        # compute the intermediate slopes
        k1 = func(t, y)
        k2 = func(t + p1 * h, y + q11 * k1 * h)

        # update 't' and 'y' values
        t = a + i * h
        phi = w1 * k1 + w2 * k2
        y += phi * h

        # store the results
        t_values[i] = t
        y_values[i] = y

        # print the progress for each step
        print(f"Step {i:02d}: t = {t:.4f}, y = {y:.4f}")

    # plot the results
    plt.plot(t_values, y_values, '-o', label="RK2 (Heun's Method)")
    plt.title("2nd Order Runge-Kutta (Heun's) Method")
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return t_values, y_values


if __name__ == "__main__":
    def dydt(t, y): return -2 * y

    t_vals, y_vals = RK2(dydt, a=0, b=2, n=10, y_initial=1)


# %% Runge Kutta's (RK3) Method

def RK3(
    func: Callable[[float, float], float],
    a: float,
    b: float,
    n: int,
    y_initial: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        func      : the function representing the ODE dy/dt = f(t, y)
        a         : the start value of 't' (initial time)
        b         : the end value of 't' (final time)
        n         : number of steps (intervals)
        y_initial : the initial condition y(t_0)

    Returns:
       t_values, y_values : arrays of 't' and 'y' values at each step

    Method used:
        The 3rd-order Runge-Kutta method (RK3) approximates the solution with:
            - k1 : slope at the start
            - k2 : slope at the midpoint
            - k3 : slope at the end point
            The weighted average is taken as (k1 + 4*k2 + k3) / 6.
    """

    # initialize arrays for 't' and 'y' values
    t_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # calculate the step size
    h = (b - a) / n

    # set initial conditions
    t = a
    y = y_initial
    t_values[0] = t
    y_values[0] = y

    # print initial values
    print(f"Step {0:02d}: t = {t:.4f}, y = {y:.4f}")

    # apply 3rd Order Runge-Kutta method to compute the solution
    for i in range(1, n + 1):
        # compute the slopes (k1, k2, k3)
        k1 = func(t, y)
        k2 = func(t + h / 2, y + k1 * h / 2)
        k3 = func(t + h, y - k1 * h + 2 * k2 * h)

        # update 't' and 'y' values using the weighted average of the slopes
        t = a + i * h
        phi = (k1 + 4 * k2 + k3) / 6
        y += phi * h

        # store the results
        t_values[i] = t
        y_values[i] = y

        # print the progress for each step
        print(f"Step {i:02d}: t = {t:.4f}, y = {y:.4f}")

    # plot the results
    plt.plot(t_values, y_values, '-o', label="RK3 Method")
    plt.title("3rd Order Runge-Kutta (RK3) Method")
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return t_values, y_values


if __name__ == "__main__":
    def dydt(t, y): return y - t**2 + 1

    t_vals, y_vals = RK3(dydt, a=0, b=2, n=10, y_initial=0.5)


# %% Runge-Kutta (RK4) Method

def RK4(
    func: Callable[[float, float], float],
    a: float,
    b: float,
    n: int,
    y_initial: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        func      : the function representing the ODE dy/dt = f(t, y)
        a         : the start value of 't' (initial time)
        b         : the end value of 't' (final time)
        n         : number of steps (intervals)
        y_initial : the initial condition y(t_0)

    Returns:
       t_values, y_values : arrays of 't' and 'y' values at each step

    Method used:
        The 4th-order Runge-Kutta method approximates the solution with:
            - k1 : Slope at the start
            - k2 : Slope at the midpoint
            - k3 : Another slope at the midpoint
            - k4 : Slope at the end point
       The weighted average of these slopes is used for the update formula:
                y_n+1 = y_n + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    """

    # initialize arrays for 't' and 'y' values
    t_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # calculate the step size
    h = (b - a) / n

    # set initial conditions
    t = a
    y = y_initial
    t_values[0] = t
    y_values[0] = y

    # print the initial conditions
    print(f"Step {0:02d}: t = {t:.4f}, y = {y:.4f}")

    # apply the 4th-order Runge-Kutta method to compute the solution
    for i in range(1, n + 1):
        # compute the slopes (k1, k2, k3, k4)
        k1 = func(t, y)
        k2 = func(t + h / 2, y + k1 * h / 2)
        k3 = func(t + h / 2, y + k2 * h / 2)
        k4 = func(t + h, y + k3 * h)

        # update 't' and 'y' values using the weighted average of the slopes
        t = a + i * h
        phi = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y += phi * h

        # store the results
        t_values[i] = t
        y_values[i] = y

        # print the progress for each step
        print(f"Step {i:02d}: t = {t:.4f}, y = {y:.4f}")

    # plot the results
    plt.plot(t_values, y_values, '-o', label="RK4 Method")
    plt.title("4th Order Runge-Kutta (RK4) Method")
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return t_values, y_values


if __name__ == '__main__':
    def dydt(t, y): return y - t**2 + 1

    t_vals, y_vals = RK4(dydt, a=0, b=2, n=10, y_initial=0.5)
