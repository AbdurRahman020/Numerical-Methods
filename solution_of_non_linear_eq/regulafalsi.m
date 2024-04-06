function xr = regulafalsi(func,xl,xu,n,delta)
%{
Regula-Falsi/False Position Method
Inputs:
    func = function handle
    xl, xu = the left and right end points
    n = no. of iterations (default is 50)
    delta = tolerence for zero (default is 0.0001)
Outputs:
    xr = approximation to zero of 'func'
%}

 if nargin < 3, error("At least 3 input argumetns are required."), end
 if nargin < 4 || isempty(n), n = 50; end
 if nargin < 5 || isempty(delta), delta = 0.0001; end

 yl = func(xl); yu = func(xu);
 if yl*yu < 0 && xl<xu
     for i = 1:n
         xr = (xl*yu-xu*yl)/(yu-yl);
         yr = func(xr);
         fprintf("%d\txr = %.6f\t f(xr)=%.6f\n",i,xr,yr);
         % stopping iterations if 'tol' is reached 
         if abs(yr) < delta, break, end
         if yl*yr < 0
             xu = xr;
         elseif yu*yr < 0
             xl = xr;
         end
     end
 end
end