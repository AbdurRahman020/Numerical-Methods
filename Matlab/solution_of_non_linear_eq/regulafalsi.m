function xr = regulafalsi(func, xl, xu, n, delta)
%{
Regula-Falsi/False Position Method

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
    
    % calculate function values at the endpoints
    yl = func(xl); 
    yu = func(xu);
 
    % check that the function changes sign over the interval [xl, xu]
    if yl*yu < 0 && xl < xu
        % perform at most 'n' iterations
        for i = 1:n
            % compute new estimate using false position formula
            xr = (xl*yu - xu*yl) / (yu - yl);
            % evaluate function at new estimate
            yr = func(xr);
            
            % display iteration number, xr, and f(xr)
            fprintf("%d\txr = %.6f\t f(xr)=%.6f\n", i, xr, yr);
            
            % stopping iterations if 'tol' is reached 
            if abs(yr) < delta
                break, end
            
            % determine which subinterval to keep for the next iteration
            if yl*yr < 0
                % root lies between 'xl' and 'xr' → move upper bound
                xu = xr;
            elseif yu*yr < 0
                % root lies between 'xr' and 'xu' → move lower bound 
                xl = xr;
            end
        end
    end
end