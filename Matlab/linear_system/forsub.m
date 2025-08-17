function y = forsub(A, b)
%{
Forward Substituion

Parameters:
    A : lower triangular matrix
    b : column vector

Return: 
    y : column vector
%}

    [m, n] = size(A);
    if m ~= n
         error("The input matrix A isn't a square matrix")
    end

    n = length(b);
    
    y = zeros(n, 1);

    y(1) = b(1) / A(1, 1);
    for i = 2:1:n
         y(i) = (b(i) - A(i, 1:i-1) * y(1:i-1)) / A(i, i);
    end
end