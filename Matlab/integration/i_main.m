clc; clear; format short;

addpath integration

disp("Composite Simpson's 1-by-3 Rule")
I1 = simpson1by3(@(x) 2 + sin(2*sqrt(x)), 0, 1, 8)

disp("Composite Simpson's 3-by-8 Rule")
I2 = simpson3by8(@(x) 2 + sin(2*sqrt(x)), 0, 1, 9)

disp("Composite Trapezoidal Rule")
I3 = trpzds(@(x) 2 + sin(2*sqrt(x)), 0, 1, 10)

disp("Composite Boole's Rule")
I4 = boole(@(x) 2 + sin(2*sqrt(x)), 0, 1, 8)

disp("Composite Weddle's Rule")
I5 = weddle(@(x) 2 + sin(2*sqrt(x)), 0, 1, 12)

disp("Gauss-Legendre Integration/Quadrature")
I6 = gausslegend(@(x) 1./x, 1, 5, 3)

disp("Unequally Spaced Trapezoidal Rule Quadrature")
x = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84];
y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123];
I7 = trapuneq(x, y)
