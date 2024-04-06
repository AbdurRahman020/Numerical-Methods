function xr = nwtraph(func,dfunc,x0,n,delta)
%{
Newton-Raphson Iteration
Input:
    func = function handle
    dfunc = derivative of func
    x0 = initial approximation to zero of 'func'
    n = number of iterations (default is 50)    
    delta = tolerence for x0 (default is 0.0001)
Output:
    xr = approximation to zero of 'func'
%}

 if nargin < 3, error("At least 3 input argumetns are required."), end
 if nargin < 4 || isempty(n), n = 50; end
 if nargin < 5 || isempty(delta), delta = 0.0001; end
 
 if dfunc(x0)~= 0
     for i = 1:n
         xr = x0 - func(x0)/dfunc(x0);
         fprintf("%d\txr = %.6f\t f(xr) = %.8f\n",i,xr,func(xr));
         if abs(xr-x0) < delta, break, end
         if dfunc(xr) == 0, error("Newton Rphson method has failed."), end 
         x0 = xr;
     end
 else
     error("Newton Rphson method has fails.")
 end
end