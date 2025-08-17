clc; clear; format short;

addpath solution_of_non_linear_eq

disp("Bisection Method")
root1 = bisection(@(x) x*sin(x) - 1, 0, 2)

disp("False Position Method")
root2 = regulafalsi(@(x) log(x) - 5 + x, 3.2, 4.0)

disp("Fixed-Point Iteration")
root3 = fixptit(@(x) exp(-x), 0.5)

disp("Newton-Raphson Iteration")
root4 = nwtraph(@(x) x^10 - 1, @(x) 10*x^9, 0.5)

disp("Secant Method")
root5 = secant(@(x) x^3 - 3*x + 2, -1.5, -1.52)
