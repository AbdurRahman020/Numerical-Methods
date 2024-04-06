function [b,a,r2] = linregr(x,y)
%{ 
Linear Regression Curve Fitting
Input:
    x = independent variable
    y = dependent variable
Output:
    b = slope
    a = intercept
    r2 = coefficient of determination
%}

 n = length(x);
 if length(y) ~= n, error('x and y must be of same length'); end
 
 x = x(:); y = y(:);
 sx = sum(x); sy = sum(y);
 sx2 = sum(x.*x); sxy = sum(x.*y); sy2 = sum(y.*y);
 
 b = (n*sxy-sx*sy)/(n*sx2-sx^2);
 a = (sy-b*sx)/n;
 r2 = ((n*sxy-sx*sy)/sqrt(n*sx2-sx^2)/sqrt(n*sy2-sy^2))^2;

 % Plot of data and best fit line
 xp = linspace(min(x),max(x),2);
 yp = b*xp+a;
 plot(x,y,'o',xp,yp)
 grid on
end