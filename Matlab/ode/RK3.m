function [vt,vy] = RK3(func,a,b,n,y_initial)
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

    vt = zeros(1,n+1);
    vy = zeros(1,n+1);
 
    h = (b - a)/n;
 
    t = a;
    y = y_initial;
 
    vt(1) = t;
    vy(1) = y;
 
    fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\t\n', 0, t, y)

 
    for i = 1:1:n
        k1 = func(t, y);
        k2 = func(t + h/2, y + k1*h/2);
        k3 = func(t + h, y - k1*h + 2*k2*h);
        
        t = a+i*h;
        phi = (k1 + 4*k2 + k3)/6;
        y = y + phi*h;
    
        fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\n', i, t, y)
        
        vt(i+1) = t;
        vy(i+1) = y;
    end
end