function xr = secant(func, xl, xu, n, delta)
%{
Secant Method

Parameters:
    func   : function handle
    xl, xu : the left and right end points
    n      : no. of iterations (default is 25)
    delta  : tolerence for zero (default is 0.0001)

Return:
    xr : approximation to zero of 'func'
%}

    if nargin < 3
        error("At least 3 input argumetns are required.")
    end
 
    if nargin < 4 || isempty(n)
        n = 25;
    end
 
    if nargin < 5 || isempty(delta)
        delta = 0.0001;
    end
    
    % iterate up to 'n' times
    for i = 1:n
        % apply secant method formula to estimate the new root
        xr = xu - func(xu) * (xu - xl) / (func(xu) - func(xl));
        
        % display current iteration number, estimate, and function value
        fprintf("%d\tx2 = %.6f\t f(xr) = %.6f\n", i, xr, func(xr));
        
        % stopping iterations if 'tol' is reached  
        if abs(xr - xu) < delta
            break
        end

        % update the two most recent guesses for the next iteration
        xl = xu;
        xu = xr;
    end
end