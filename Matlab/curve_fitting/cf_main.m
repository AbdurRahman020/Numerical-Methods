clc; clear; format short;

addpath curve_fitting

disp("Linear Regression Curve Fitting")
x = 10:10:80;
y = [25, 70, 380, 550, 610, 1220, 830, 1450];
[b, a, r2] = linregr(x, y)

disp("Least-Square Polynomials")
c = lstsqpoly(x, y, 2)