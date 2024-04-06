clc
clear all
format short

addpath linear_system
addpath linear_system\linear_system_iteration
addpath solution_of_non_linear_eq
addpath integration
addpath curve_fitting
addpath interpolation
addpath ode
addpath ode\ode_system
addpath differentiation

% LINEAR SYSTEM OF EQUATIONS
A = [6 15 55; 15 55 225; 55 225 979];
b = [76;295;1259];

disp("Cholesky's Method")
[l,u,y,x] = cholesky(A,b)

A = [10 -1 2 0; -1 11 -1 3; 2 -1 10 -1; 0 3 -1 8];
b = [6 25 -11 15]';

disp("Crout's Method")
[l,u,y,x] = crout(A,b)

disp("Doolittle's Method")
[l,u,y,x] = doolittle(A,b)

A = randi([1,10],[5,5]);
b = randi([1,10],[5,1]);

disp("Cramer's Rule")
x = cramer(A,b)

disp("Gauss Elimination")
[u,x] = gausselimn(A,b)

disp("Gauss-Jordan Method")
[Ab,x] = gaussjordan(A,b)

% LINEAR SYSTEM ITERATION
A = [3 -0.1 -0.2;0.1 7 -0.3;0.3 -0.2 10];
b = [7.85 -19.3 71.4]';
p = [0 0 0]';

disp("Gauss-Seidel Iteration Method")
X = gausidl(A,b,p,10,0.00001)

disp("Jacobi Iteration Method")
X = jacobi(A,b,p,10,0.00001)

% SOLUTION OF NONLINEAR SYSTEM OF EQUATIONS
disp("Bisection Method")
root = bisection(@(x) x*sin(x)-1,0,2)

disp("False Position Method")
root = regulafalsi(@(x) log(x)-5+x,3.2,4.0)

disp("Fixed-Point Iteration")
root = fixptit(@(x) exp(-x),0.5)

disp("Newton-Raphson Iteration")
root = nwtraph(@(x) x^10-1,@(x) 10*x^9,0.5)

disp("Secant Method")
root = secant(@(x) x^3-3*x+2,-1.5,-1.52)

% INTEGRATION
disp("Composite Simpson's 1-by-3 Rule")
I = simpson1by3(@(x) 2+sin(2*sqrt(x)),0,1,8)

disp("Composite Simpson's 3-by-8 Rule")
I = simpson3by8(@(x) 2+sin(2*sqrt(x)),0,1,9)

disp("Composite Trapezoidal Rule")
I = trpzds(@(x) 2+sin(2*sqrt(x)),0,1,10)

disp("Composite Boole's Rule")
I = boole(@(x) 2+sin(2*sqrt(x)),0,1,8)

disp("Composite Weddle's Rule")
I = weddle(@(x) 2+sin(2*sqrt(x)),0,1,12)

disp("Gauss-Legendre Integration/Quadrature")
I = gausslegend(@(x) 1./x,1,5,3)

disp("Unequally Spaced Trapezoidal Rule Quadrature")
x = [0,6,12,18,24,30,36,42,48,54,60,66,72,78,84];
y = [124,134,148,156,147,133,121,109,99,85,78,89,104,116,123];
I = trapuneq(x,y)

% CURVE FITTING
disp("Linear Regression Curve Fitting")
x = 10:10:80
y = [25,70,380,550,610,1220,830,1450]
[b,a,r2] = linregr(x,y)

disp("Least-Square Polynomials")
c = lstsqpoly(x,y,2)

% INTERPOLATION
disp("Langrange's Interpolating Polynomial")
x = [-40,0,20,50];
y = [1.52,1.29,1.2,1.09];
[c,l] = lagranpoly(x,y)
yint = lagranIntpl(x,y,15)

disp("Newton's Interpolating Polynomial")
x = [1,4,6];
y = [0,1.386294,1.791759];
[c,dd] = newtpoly(x,y)
yint = newtIntpl(x,y,2)

% ORDINARY DIFFERENTAIL EQUATIONS
disp("Euler's Method")
[t,y] = euler(@(t,y) (t-y)/2,0,3,12,1);

disp("Mid Point Method")
[t,y] = midpoint(@(t,y) (t-y)/2,0,3,12,1);

disp("Taylor's (Order Two) Method")
f = @(t,y) y-t^2+1;
df = @(t,y) y-t^2+1-2*t;
[t,y] = taylor2(f,df,0,2,10,0.5);

disp("Taylor's (Order Four) Method")
f = @(t,y) y-t^2+1;
df1 = @(t,y) y-t^2+1-2*t;
df2 = @(t,y) y-t^2+1-2*t-2;
df3 = @(t,y) y-t^2+1-2*t-2;
[t,y] = taylor4(f,df1,df2,df3,0,2,10,0.5);

disp("Runge Kutta's (RK2) Method")
[t,y] = RK2(@(t,y) (t-y)/2,0,3,15,1);

disp("Runge Kutta's (RK3) Method")
[t,y] = RK3(@(t,y) (t-y)/2,0,3,15,1);

disp("Runge Kutta's (RK4) Method")
[t,y] = RK4(@(t,y) (t-y)/2,0,3,15,1);

% SYSTEM OF ORDINARY DIFFERENTAIL EQUATIONS
disp("RK2 for system of ODE")
f = cell(2,1);
f{1} = @(t,y) sin(t)+cos(y(1))+sin(y(2));
f{2} = @(t,y) cos(t)+sin(y(2));
y_init = zeros(2,1);
y_init(1) = -1; y_init(2) = 1;
[t,y] = RK2System(f,0,20,40,y_init);

disp("RK3 for system of ODE")
f = cell(2,1);
f{1} = @(t,y) sin(t)+cos(y(1))+sin(y(2));
f{2} = @(t,y) cos(t)+sin(y(2));
y_init = zeros(2,1);
y_init(1) = -1; y_init(2) = 1;
[t,y] = RK3System(f,0,20,40,y_init);

disp("RK4 for system of ODE")
f = cell(2,1);
f{1} = @(t,y) sin(t)+cos(y(1))+sin(y(2));
f{2} = @(t,y) cos(t)+sin(y(2));
y_init = zeros(2,1);
y_init(1) = -1; y_init(2) = 1;
[t,y] = RK4System(f,0,20,40,y_init);

% DIFFERENTIATION
f=@(x) -0.1*x^4-0.15*x^3-0.5*x^2-0.25*x+1.2;

disp("Forward finite-difference Formula")
derv = frwdfnitdiff(f,0.5,0.25,2)

disp("Backward finite-difference Formula")
derv = bkwdfnitdiff(f,0.5,0.25,2)

disp("Centered finite-difference Formula")
derv = cntfnitdiff(f,0.5,0.25,2)