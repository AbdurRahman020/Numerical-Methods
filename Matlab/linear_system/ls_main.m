clc; clear; format short;

addpath linear_system

A1 = [6 15 55; 15 55 225; 55 225 979];
b1 = [76 295 1259]';

disp("Cholesky's Method")
[l1, u1, y1, x1] = cholesky(A1, b1)

A2 = [10 -1 2 0; -1 11 -1 3; 2 -1 10 -1; 0 3 -1 8];
b2 = [6 25 -11 15]';

disp("Crout's Method")
[l2, u2, y2, x2] = crout(A2, b2)

disp("Doolittle's Method")
[l3, u3, y3, x3] = doolittle(A2, b2)

A3 = randi([1, 10], [5, 5]);
b3 = randi([1, 10], [5, 1]);

disp("Cramer's Rule")
x4 = cramer(A3, b3)

disp("Gauss Elimination")
[u4, x5] = gausselimn(A3, b3)

disp("Gauss-Jordan Method")
[Ab, x6] = gaussjordan(A3, b3)

disp("QR-decomposition")
A4 = [12 -51 4; 6 167 -68; -4 24 -41];
[Q1, R] = qrDecomposition(A4)            

disp("Gram-Schmidt Process")
V = [3 1; 2 2; 4 3];
Q2 = gramSchmidt(V)
