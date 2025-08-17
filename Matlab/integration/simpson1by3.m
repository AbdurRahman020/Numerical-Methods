function I = simpson1by3(func, a, b, n)
%{
Composite Simpson's 1/3 Rule

Parameters:
    func : handle of function to be integrated
    a, b : lower and upper bounds
    n    : number of intervals, must be even

Return:
    I : result of integration
%} 

    if nargin < 4
        error('insufficient inputs')
    end
    
    if a > b
        error('b must be greater than a')
    end
    
    if mod(n, 2) ~= 0
        error('n must be even')
    end

    x = zeros(1, n-1);
    y = zeros(1, n-1);

    sum_odd = 0;
    sum_even = 0;

    h = (b - a)/n; 

    for i = 1:1:n-1
        x(i) =  a + i*h;
        y(i) = func(x(i));
        
        if mod(i, 2) == 1
            sum_odd = sum_odd + y(i);
        else
            sum_even = sum_even + y(i);
        end 
    end
    
    I = h/3 * (func(a) + func(b) + 4*sum_odd + 2*sum_even);
end
