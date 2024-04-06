function I = weddle(func,a,b,n)
%{
Composite Weddle's Rule
Inputs:
    func = handle of function to be integrated
    a = lower bound
    b = upper bound
    n = number of intervals
Output:
    I = integral estimate
%} 

 if nargin < 4, error('insufficient inputs'), end
 if a > b, error('b must be greater than a'), end
 if mod(n,6) ~= 0, error('n must be multiple of 6'), end

 x = zeros(1,n-1); y = zeros(1,n-1);
 sum_1_5 = 0; sum = 0; sum_3 = 0; sum_6 = 0;
 h = (b-a)/n; 

 for i = 1:1:n-1
     x(i) =  a + i*h;
     y(i) = func(x(i));
     if mod(i,6) == 1 || mod(i,6) == 5
         sum_1_5 = sum_1_5 + y(i);
     elseif mod(i,6) == 0
         sum_6 = sum_6 + y(i);
     elseif mod(i,6) == 3 
         sum_3 = sum_3 + y(i); 
     else
         sum = sum + y(i);
     end 
 end
 I = (3*h/10)*(func(a) + sum + 2*sum_6 + 6*sum_3 + 5*sum_1_5 + func(b));
end