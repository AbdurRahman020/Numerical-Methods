function [C,DD] = newtpoly(x,y)
%{
Newton's Interpolating Polynomial 
Input: 
    x = [x0 x1 ... xn]
    y = [y0 y1 ... yn]
Output:
    C = Newton polynomial coefficients of degree 'n'
    DD = divide difference table
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
  
 C = DD(n,n);
 for k = n-1:-1:1
     C = conv(C, poly(x(i)));
     m = length(C);
     C(m) = C(m)+DD(k,k);
 end
end