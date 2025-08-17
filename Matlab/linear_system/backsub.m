function x = backsub(A, b)
%{ 
Back-Subtituion

Parameters:
    A = upper triangular matrix
    b = column vector

Return: 
    x = column vector
%} 

    [m, n] = size(A);
    if m ~= n
         error("The input matrix A isn't a square matrix")
    end

    n = length(b);
    
    x = zeros(n, 1);

    x(n) = b(n) / A(n, n);
    for i = n-1:-1:1
         x(i) = (b(i) - A(i, i+1:n) * x(i+1:n)) / A(i, i);
    end
end