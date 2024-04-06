function I = trapuneq(x,y)
%{
Unequally Spaced Trapezoidal Rule Quadrature
Inputs:
   x = vector of independent variables
   y = vector of dependent variables
Output:
   I = integral estimate
%}

 if nargin < 2, error('At least 2 input arguments required.'), end
 if any(diff(x) < 0), error('x is not monotonically ascending.'), end
 n = length(x);
 if n ~= length(y), error('x and y must be of same length.'), end
 
 I = 0;
 for k = 1:n-1
     I = I + 0.5*(x(k+1)-x(k))*(y(k)+y(k+1));
 end
end