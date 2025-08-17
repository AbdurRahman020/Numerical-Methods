function xr = nwtraph(func,dfunc,x0,n,delta)
%{
Newton-Raphson Iteration

Parameters:
    func  : function handle
    dfunc : derivative of func
    x0    : initial approximation to zero of 'func'
    n     : number of iterations (default is 25)    
    delta : tolerence for x0 (default is 0.0001)

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
 
    % check that the derivative is not zero at the initial guess
    if dfunc(x0) ~= 0
        % perform at most 'n' iterations
        for i = 1:n
            % newton-raphson formula: xr = x0 - f(x0) / f'(x0)
            xr = x0 - func(x0) / dfunc(x0);
            
            % display current iteration number, xr, and f(xr)
            fprintf("%d\txr = %.6f\t f(xr) = %.8f\n", i, xr, func(xr));
            
            % stopping criterion: if the change is less than 'delta'
            if abs(xr - x0) < delta
                break
            end
            
            % check if derivative is zero at the new approximation (to avoid division by zero)
            if dfunc(xr) == 0
                error("Newton Rphson method has failed.")
            end 
            
            % update guess for the next iteration
            x0 = xr;
        end
    else
        % derivative was zero at the starting point → can't proceed
        error("Newton Rphson method has fails.")
    end
end