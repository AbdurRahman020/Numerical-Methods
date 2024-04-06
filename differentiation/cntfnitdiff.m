function [cntdifs] = cntfnitdiff(f,x,h,d)
%{
Centered finite-difference formulae
Inputs:
    f = function whose derivative is to be calculated
    x = point at which the derivative is calculated
    h = step size
    d = derivative number (upto 4)
Output:
    cntdifs = two values of derivative, second value
              is more accurate because it incorporates
              more terms of Taylor series expanion
%}

 if d == 1
     cntd1 = (f(x+h)-f(x-h))/(2*h);
     cntd2 = (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h);
 elseif d == 2
     cntd1 = (f(x+h)-2*f(x)+f(x-h))/h^2;
     cntd2 = (-f(x+2*h)+16*f(x+h)-30*f(x)+16*f(x-h)-f(x-2*h))/(12*h^2);
 elseif d == 3
     cntd1 = (f(x+2*h)-2*f(x+h)+2*f(x-h)-f(x-2*h))/(8*h^3);
     cntd2 = (-f(x+3*h)+8*f(x+2*h)-13*f(x+h)+13*f(x-h)-8*f(x-2*h)+f(x-3*h))/(8*h^3);
 elseif d == 4
     cntd1 = (f(x+2*h)-4*f(x+h)+6*f(x)-4*f(x-h)+f(x-2*h))/h^4;
     cntd2 = (-f(x+3*h)+12*f(x+2*h)-39*f(x+h)+56*f(x)-39*f(x-h)+12*f(x-2*h)-f(x-3*h))/(6*h^4);
 else
     disp("No more derivative formulae greater than 4.")
 end
 cntdifs = [cntd1, cntd2]';
end