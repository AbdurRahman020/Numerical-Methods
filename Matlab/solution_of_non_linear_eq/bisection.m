function xr = bisection(func, xl, xu, n, delta)
%{
Bisection Method

Parameters:
    func   : function handle
    xl, xu : the left and right end points
    n      : no. of iterations (default = 50)
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
    
    % check if the initial interval is valid (signs must be opposite)
    if func(xl)*func(xu) < 0
        % perform the bisection loop for a maximum of 'n' iterations
        for i = 1:n
            % find the midpoint of the current interval
            xr = (xl + xu) / 2;
            
            % stopping iterations if 'tol' is reached
            if abs(xr - xu) < delta || abs(xr - xl) < delta
                break
            end
            
            % display the iteration number, midpoint, and function value at midpoint
            fprintf("%d\t\txr = %.6f\t f(xr)=%.6f\n", i, xr, func(xr));
            
            % determine which subinterval contains the root
            if func(xl) * func(xr) < 0
                % root lies between xl and xr → update xu
                xu = xr;
            elseif func(xu) * func(xr) < 0
                % root lies between xr and xu → update xl
                xl = xr;
            end
        end
    else
        error("No root between given interval.")
    end
end