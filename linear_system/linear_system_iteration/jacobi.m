function x = jacobi(A,b,p,max_itr,tol)
%{
Jacobi Iteration
Input:
    A = n-by-n non-singualr matrix
    b = column vector
    p = column vector of initail guess
    max_itr = the maximum number of iterations      
    tol = the tolarance of 'P'
Output:
    x = column vector, the jacobi approximation
        to the solution of Ax = b 
%}
 if nargin < 3, error("At lesat 3 input arguments required."), end
 if nargin < 4 || isempty(max_itr), max_itr = 50; end
 if nargin < 5 || isempty(tol), tol = 0.00001; end
 [m,n] = size(A);
 if m ~= n; error("Coefficient matrix 'A' should be a square matrix"), end
 
 n = length(b);
 x = zeros(n,1);
 ref = zeros(n,1);
 
 for  i = 1:max_itr
     for j = 1:n
         x(j) = b(j)/A(j,j)-(A(j,[1:j-1,j+1:n])*p([1:j-1,j+1:n]))/A(j,j);
     end
     p = x;
     fprintf("%d:\t",i);
     for k = 1:n, fprintf("%.10f ",p(k)); end
     disp(" ");
     if abs(ref-x) < tol, break, end
     ref = x;
 end
end