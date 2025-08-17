clc; clear; format short;

addpath linear_system\linear_system_iteration

A = [3 -0.1 -0.2; 0.1 7 -0.3; 0.3 -0.2 10];
b = [7.85 -19.3 71.4]';
p = [0 0 0]';

disp("Gauss-Seidel Iteration Method")
X1 = gausidl(A, b, p, 10, 0.00001)

disp("Jacobi Iteration Method")
X2 = jacobi(A, b, p, 10, 0.00001)
