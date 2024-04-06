function [vt,vy] = RK2System(func,a,b,n,y_initial)
%{
Runge Kutta's (RK2) Method
Inputs:
    func = the ODE to be solve
    a = initial point
    b = final point
    n = number of intervals  
    y_intial = initial value of dependent variable
     
Outputs:
    vt = values of t
    vy = values of y (solution of Initial Value Problem)

Methods:
    Mid point method: w1=0; w2=1; p1=1/2; q11=1/2;  
    Ralston's method: w1=1/3; w2=2/3; p1=3/4; q11=3/4;
%}
%Heun's method:
 w1 = 1/2; w2 = 1/2; p1 = 1; q11 = 1;

 m = size(func,1);
 vt = zeros(1,n+1); vy = zeros(m,n+1);
 k1 = zeros(m,1); k2 = zeros(m,1);
 
 h = (b-a)/n;
 t = a; y = y_initial;
 vt(1) = t; vy(:,1) = y;

 for i = 1:n
     for j = 1:m, k1(j) = func{j}(t, y); end
     for j = 1:m, k2(j) = func{j}(t+p1*h, y+q11*k1*h); end
    
     t = a+i*h;
     phi = w1*k1+w2*k2;
     y = y+phi*h;
     
     fprintf('i = %.3d\t\t t = %f\t\t', i, t)
     for k = 1:m
         fprintf("y(%d) = %.6f\t\t",k, y(k));
     end 
     disp(" ")
    
     vt(i+1) = t;
     vy(:,i+1) = y;
 end
 plot(vt,vy,'-o')
 title("RK2 System")
 ylabel("y"), xlabel("t")
 grid on;
end