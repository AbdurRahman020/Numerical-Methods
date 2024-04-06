function [vt,vy] = taylor4(func,dfunc1,dfunc2,dfunc3,a,b,n,y_initial)
%{
Taylor's (Order Four) Method
Inputs:
    func = function f(t,y)
    dfunc1 = first derivative of f(t,y)
    dfunc2 = second derivative of f(t,y)
    dfunc3 = third derivative of f(t,y)
    a = initial point
    b = final point
    n = number of intervals  
    y_intial = initial value of the dependent variable
     
Outputs:
    vt = values of t
    vy = values of y (solution of Initial Value Problem)
%}

 vt = zeros(1,n+1); vy = zeros(1,n+1);
 h = (b-a)/n;
 t = a; y = y_initial;
 vt(1) = t; vy(1) = y;
 fty = func(t,y);
 dfty1 = dfunc1(t,y); dfty2 = dfunc2(t,y); dfty3 = dfunc3(t,y);
 fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\t\n', 0, t, y)

 for i = 1:1:n
     t = a+i*h;
     y = y+h*(fty + dfty1*(h)/2 + dfty2*(h^2)/6 + dfty3*(h^3)/24);
     fty = func(t,y);
     dfty1 = dfunc1(t,y);
     dfty2 = dfunc2(t,y);
     dfty3 = dfunc3(t,y);
    
     fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\n', i, t, y)
     vt(i+1) = t;
     vy(i+1) = y;
 end
 plot(vt,vy,'-o')
 title("Taylor's (Order Four) Method")
 ylabel("y"), xlabel("t")
 grid on;
end
