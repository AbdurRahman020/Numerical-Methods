function [vt, vy] = taylor2(func, dfunc1, a, b, n, y_initial)
%{
Taylor's (Order Two) Method

Parameters:
    func     : function f(t,y)
    dfunc1   : first derivative of f(t,y) 
    a, b     : initial and final points
    n        : number of intervals  
    y_intial : initial value of the dependent variable
     
Outputs:
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
 
    fty = func(t, y);
    dfty1 = dfunc1(t, y);
 
    fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\t\n', 0, t, y)

 
    for i = 1:1:n
        t = a + i*h;
        y = y + h*(fty + dfty1*(h)/2);
        
        fty = func(t, y);
        dfty1 = dfunc1(t, y);
    
        fprintf('i = %.3d\t\t t = %.4f\t\t y = %.4f\n', i, t, y)
     
        vt(i+1) = t;
        vy(i+1) = y;
    end
end