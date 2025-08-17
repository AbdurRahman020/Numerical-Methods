function [b, a, r2] = linregr(x, y)
%{ 
Linear Regression Curve Fitting

Parameters:
    x : independent variable
    y : dependent variable

Returns:
    b  : slope
    a  : intercept
    r2 : coefficient of determination
%}

    n = length(x);
 
    if length(y) ~= n
        error('x and y must be of same length'); 
    end
 
    x = x(:); 
    y = y(:);
    
    % calculate necessary summations for the linear regression formula
    sx = sum(x);            % sum of x-values
    sy = sum(y);            % sum of y-values
    sx2 = sum(x.*x);        % sum of x^2
    sy2 = sum(y.*y);        % sum of y^2
    sxy = sum(x.*y);        % sum of x*y
    
    % compute the slope 'b' and intercept 'a' using the normal equations
    b = (n*sxy - sx*sy) / (n*sx2 - sx^2);
    a = (sy - b*sx) / n;
    
    % compute the coefficient of determination 'R²'
    r2 = ((n*sxy - sx*sy) / (sqrt(n*sx2 - sx^2) * sqrt(n*sy2 - sy^2)))^2;

    % plot of data and best fit line
    xp = linspace(min(x), max(x), 2);
    yp = b*xp + a;
    plot(x, y, 'o', xp, yp)
    grid on
end