function X = backsub(A,b)
%{ 
Back-Subtituion
Input:
    A = upper triangular matrix
    b = column vector
Output: 
    X = column vecotr
%} 

 [m,n] = size(A);
 if m~=n, error("The input matrix A isn't a square matrix"); end

 n = length(b);
 X = zeros(n,1);
 X(n) = b(n)/A(n,n);

 for i = n-1:-1:1
     X(i) = (b(i) - A(i,i+1:n)*X(i+1:n))/A(i,i);
 end
end