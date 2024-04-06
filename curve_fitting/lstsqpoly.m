function C = lstsqpoly(x,y,n)
%{
Least-Square Polynomials
Input:
    x = independent variable
    y = dependent variable
    n = degree of least-square polynomial
Output:
    C = coefficient matrix for the polynomial  
%}

 m = length(x);
 if length(y) ~= m; error('x and y should be of same length'), end
 
 B = zeros(1:n+1);
 F = zeros(m,n+1);
 
 for i = 1:n+1
     F(:,i) = x'.^(i-1);
 end
 A = F'*F;
 B = F'*y';
 C = flipud(A\B);
end 