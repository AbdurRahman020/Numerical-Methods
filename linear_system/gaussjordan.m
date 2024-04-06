function [Ab,x] = gaussjordan(A,b)
%{
Gauss Jordan Method
Inputs:
    A = n-by-n coefficient matrix
    b = n-by-1 source matrix
Outputs:
    x = n-by-1 row matrix, solution of Ax=b
    Ab = n-by-n+1 augmented matrix (reduced row echelon from) 
%}
 
 n = length(b);
 Ab = [A,b];
 [r,c] = size(Ab);
 d = det(A);
 
 if d == 0, error('A is a singular matrix'), end
 % checking for pivot element if it's zero and swapping position
 for j = 1:r-1
     if Ab(j,j) == 0
         for k = j+1:r
             if Ab(k,j) ~= 0
                 tempAb = Ab(j,:);
                 Ab(j,:) = Ab(k,:);
                 Ab(k,:) = tempAb;
                 break
             end
         end
     end
 end
 
 % reducing into row echelon form to find 'x'
 for j = 1:n
     Ab(j,:) = Ab(j,:)/Ab(j,j);
     for i = 1:n
         if i ~= j
             Ab(i,:) = Ab(i,:)-Ab(i,j)*Ab(j,:);
         end
     end
 end
 x = Ab(:,c);
end