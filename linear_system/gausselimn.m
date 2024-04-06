function [U,x] = gausselimn(A,b)
%{
Gauss-Elimination Method
Inputs:
    A = n-by-n coefficient matrix
    b = n-by-1 source matrix
Outputs:
    U = n-by-n upper triangular matrix  
    x = n-by-1 martix, soltion of Ax = b
%}

 n = length(b);
 x = zeros(n,1);
 Ab = [A,b];
 d = det(A);
 
 if d == 0, error('A is a singular matrix'), end
 for j = 1:n-1
     for i = j+1:n
         m = Ab(i,j)/Ab(j,j);
	 Ab(i,:) = Ab(i,:) - m*Ab(j,:);
     end
 end
 
 x(n) = Ab(n,n+1)/Ab(n,n);
 for k = n-1:-1:1
     x(k) = (Ab(k,n+1)- Ab(k,k+1:n)*x(k+1:n))/Ab(k,k);
 end
 U = Ab(:,1:n);
end
