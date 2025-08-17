clc; clear; format short;

addpath interpolation

disp("Langrange's Interpolating Polynomial")
x1 = [-40, 0, 20, 50];
y1 = [1.52, 1.29, 1.2, 1.09];
[c1,l] = lagranpoly(x1, y1)
yint1 = lagranIntpl(x1, y1, 15)

disp("Newton's Interpolating Polynomial")
x2 = [1, 4, 6];
y2 = [0, 1.386294, 1.791759];
[c2, dd] = newtpoly(x2, y2)
yint2 = newtIntpl(x2, y2, 2)