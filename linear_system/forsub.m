function Y = forsub(A,b)
%{
Forward Substituion
Input:
    A = lower triangular matrix
    b = column vector
Output: 
    Y = column vecotr
%}

 [m,n] = size(A);
 if m~=n, error("The input matrix A isn't a square matrix"), end

 n = length(b);
 Y = zeros(n,1);
 Y(1) = b(1)/A(1,1);

 for i = 2:1:n
     Y(i) = (b(i)-A(i,1:i-1)*Y(1:i-1))/A(i,i);
 end
end