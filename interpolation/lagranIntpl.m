function yint = lagranIntpl(x,y,xint)
%{
Langrange's Interpolating Polynomial
Input: 
    x = [x0 x1 ... xn]
    y = [y0 y1 ... yn]
    xint = value at which interpolation is calculated 
Output:
    yint = interpolated value
%}
 
 n = length(x);
 if length(y) ~= n, error("x and y must be of same length"), end
 yint = 0;
 for i = 1:n
     product = y(i);
     for j = 1:n
        if i ~= j
            product = product*(xint-x(j))/(x(i)-x(j));
        end
     end
     yint = yint+product;
 end
end