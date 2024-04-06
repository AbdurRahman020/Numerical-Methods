function I = trpzds(func,a,b,n)
%{
Composite Trapezidol Rule
Inputs:
    func = handle of function to be integrated
    a = lower bound
    b = upper bound
    n = number of intervals, defaults to 100, must be even
Output:
    I = result of integration
%} 

 if nargin < 4, error('insufficient inputs'), end
 if a > b, error('b must be greater than a'), end
 if mod(n,2) == 1, error('n must be even'), end

 x = zeros(1,n-1); y = zeros(1,n-1);
 sum = 0;
 h = (b-a)/n; 

 for i = 1:1:n-1
     x(i) =  a + i*h;
     y(i) = func(x(i));
     sum = sum + y(i); 
 end
 I = h/2*(func(a)+2*sum+func(b));
end
