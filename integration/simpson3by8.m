function I = simpson3by8(func,a,b,n)
%{
Composite Simpson's 3/8 Rule
Inputs:
    func = handle of function to be integrated
    a = lower bound
    b = upper bound
    n = number of intervals
Output:
    I = result of integration
%} 

 if nargin < 4, error('insufficient inputs'), end
 if a > b, error('b must be greater than a'), end
 if mod(n,3) == 1, error('n must be multiple of 3'), end

 x = zeros(1,n-1); y = zeros(1,n-1);
 sum_3 = 0; sum_other = 0;
 h = (b-a)/n; 

 for i = 1:1:n-1
     x(i) =  a + i*h;
     y(i) = func(x(i));
     if mod(i,3) == 0
         sum_3 = sum_3 + y(i);
     else
         sum_other = sum_other + y(i);
     end 
 end
 I = (3*h/8)*(func(a) + func(b) + 2*sum_3 + 3*sum_other);
end
