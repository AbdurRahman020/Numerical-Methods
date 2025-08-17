function I = boole(func, a, b, n)
%{
Composite Boole's Rule

Parameters:
    func : handle of function to be integrated
    a, b : lower and upper bounds
    n    : number of intervals, must be multiple of 4

Return:
    I : integral estimate
%} 

    if nargin < 4
        error('insufficient inputs')
    end
    
    if a > b
        error('b must be greater than a')
    end
 
    if mod(n, 4) ~= 0
        error('n must be multiple of 4')
    end

    x = zeros(1, n-1);
    y = zeros(1, n-1);
    
    sum = 0;
    sum_2 = 0;
    sum_1_3 = 0; 
 
    h = (b - a) / n;
 
    for i = 1:1:n-1
        x(i) =  a + i*h;
        y(i) = func(x(i));

        if mod(i, 4) == 1 || mod(i, 4) == 3
            sum_1_3 = sum_1_3 + y(i);
        elseif mod(i, 4) == 2
            sum_2 = sum_2 + y(i); 
        else
            sum = sum + y(i);
        end 
    end
    
    I = (2*h/45) * (7*(func(a) + func(b)) + 32*sum_1_3 + 12*sum_2 + 14*sum);
end