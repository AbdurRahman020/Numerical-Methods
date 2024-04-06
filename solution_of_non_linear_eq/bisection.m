function xr = bisection(func,xl,xu,n,delta)
%{
Bisection Method
Inputs:
    func = function handle
    xl, xu = the left and right end points
    n = no. of iterations (default = 50)
    delta = tolerence for zero (default is 0.0001)
Outputs:
    xr = approximation to zero of 'func'
%}

 if nargin < 3, error("At least 3 input argumetns are required."), end
 if nargin < 4 || isempty(n), n = 50; end
 if nargin < 5 || isempty(delta), delta = 0.0001; end
 
 if func(xl)*func(xu)<0
     for i = 1:n
         xr = (xl+xu)/2;
         % stopping iterations if 'tol' is reached
         if abs(xr-xu) < delta || abs(xr-xl) < delta, break, end
         fprintf("%d\t\txr = %.6f\t f(xr)=%.6f\n",i,xr,func(xr));
         if func(xl)*func(xr)<0
             xu = xr;
         elseif func(xu)*func(xr)<0
             xl = xr;
         end
     end
 else
     error("No root between given interval.")
 end
end