function yint = newtIntpl(x,y,xint)
%{
Newton's Interpolating Polynomial
Inputs: 
    x = [x0 x1 ... xn]
    y = [y0 y1 ... yn]
    xint = value at which interpolation is calculated 
Output:
    yint = interpolated value
%}
 
 n = length(x);
 if length(y) ~= n, error("x and y must be of same length"), end
 
 DD = zeros(n,n);
 DD(:,1) = y';
 for i = 2:n
    for j = i:n 
        DD(j,i) = (DD(j,i-1) - DD(j-1,i-1))/(x(j)- x(j-i+1));
    end
 end
 
 xt = 1; yint = DD(1,1);
 for i=1:n-1
     xt = xt*(xint-x(i));
     yint = yint+DD(i+1,i+1)*xt;
 end
end