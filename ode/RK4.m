function [vt,vy] = RK4(func,a,b,n,y_initial)
%{
Runge Kutta's (RK4) Method
Inputs:
    func = the ODE to be solve
    a = initial point
    b = final point
    n = number of intervals  
    y_intial = initial value of dependent variable
     
Outputs:
    vt = values of t
    vy = values of y (solution of Initial Value Problem)
%}

 vt = zeros(1,n+1); vy = zeros(1,n+1);
 h = (b-a)/n;
 t = a; y = y_initial;
 vt(1) = t; vy(1) = y;
 fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\t\n', 0, t, y)

 for i = 1:1:n
     k1 = func(t, y);
     k2 = func(t+h/2, y+k1*h/2);
     k3 = func(t+h/2, y+k2*h/2);
     k4 = func(t+h, y+k3*h);
     
     t = a+i*h;
     phi = (k1+2*k2+2*k3+k4)/6;
     y = y+phi*h;
    
     fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\n', i, t, y)
     vt(i+1) = t;
     vy(i+1) = y;
 end
 plot(vt,vy,'-o')
 title("Rk4 Method")
 ylabel("y"), xlabel("t")
 grid on;
end