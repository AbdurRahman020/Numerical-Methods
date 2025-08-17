 function [vt, vy] = RK3System(func, a, b, n, y_initial)
%{
Runge Kutta's (RK3) Method

Parameters:
    func     : the ODE to be solve
    a, b     : initial and final points
    n        : number of intervals  
    y_intial : initial value of dependent variable
     
Returns:
    vt : values of t
    vy : values of y (solution of Initial Value Problem)
%}

    m = size(func, 1);
 
    vt = zeros(1, n+1);
    vy = zeros(m, n+1);
 
    k1 = zeros(m, 1);
    k2 = zeros(m, 1);
    k3 = zeros(m, 1);

    h = (b - a)/n;
 
    t = a; 
    y = y_initial;
 
    vt(1) = t;
    vy(:,1) = y;

    for i = 1:n
        for j = 1:m
            k1(j) = func{j}(t, y);
        end
        for j = 1:m
            k2(j) = func{j}(t + h/2, y + k1*h/2);
        end
        for j = 1:m
            k3(j) = func{j}(t+h, y - k1*h + 2*k2*h);
        end
     
        t = a + i*h;
        phi = (k1 + 4*k2 + k3)/6;
        y = y + phi*h;
     
        fprintf('i = %.3d\t\t t = %f\t\t', i, t)
        for k = 1:m
            fprintf("y(%d) = %.6f\t\t", k, y(k));
        end 
        disp(" ")
     
        vt(i+1) = t;
        vy(:, i+1) = y;
    end
end