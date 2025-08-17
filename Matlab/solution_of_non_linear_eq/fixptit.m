function xr = fixptit(func, x0, n, delta)
%{
Fixed-Point Iteration

Parmeters:
    func  : function handle
    x0    : initial guess to zero of 'func'
    n     : number of iterations (default is 25)    
    delta : tolerence for x0 (default is 0.0001)

Return:
    xr : approximation to zero of 'func'
%}

    if nargin < 2
        error("At least 3 input argumetns are required.")
    end
    
    if nargin < 3 || isempty(n)
        n = 25;
    end

    if nargin < 4 || isempty(delta)
        delta = 0.0001;
    end
    
    % iterate for at most 'n' iterations
    for i = 1:n
        % apply the fixed-point iteration formula: x_{r} = g(x_{0})
        xr = func(x0);

        % display the iteration number and the new approximation
        fprintf("%d\txr = %.6f\n", i, xr);
        
        % stopping iterations if 'tol' is reached
        if abs(xr - x0) < delta
            break
        end
        
        % update x0 for the next iteration
        x0 = xr;
    end
end