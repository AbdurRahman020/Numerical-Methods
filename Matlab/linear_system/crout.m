function [L, U, y, x] = crout(A, b)
%{
Crout's Method (LU-decomposition)
    [a11 a12 a13]   [l11  0   0 ] [ 1  u12 u13]
	[a21 a22 a23] = [l21 l22  0 ] [ 0   1  u23]
	[a31 a32 a33]   [l31 l32 l33] [ 0   0   1 ]

Parameters:
    A : n-by-n coefficient matrix
    b : n-by-1 source matrix

Returns:
    L : n-by-n lower triangular matrix    
    U : n-by-n upper triangular matrix  
    y : n-by-1 martix, soltion of Ly = b
    x : n-by-1 matrix, solution of Ux = y
%}

    n = length(b);

    L = zeros(n, n);
    U = zeros(n, n);
    y = zeros(n, 1);
    x = zeros(n, 1);
 
    for i = 1:n
         U(i, i) = 1;
    end
 
    L(:, 1) = A(:, 1);
    U(1, :) = A(1, :) / L(1, 1);
 
     for i = 2:n
         for j = i:n
             L(j, i) = A(j, i) - L(j, 1:i-1) * U(1:i-1, i);
        end
        for k = i+1:n
            U(i, k) = (A(i, k) - L(i, 1:i-1) * U(1:i-1, k)) / L(i, i);
        end
     end
 
    y(1) = b(1) / L(1, 1);
    for i = 2:n
         y(i) = (b(i) - L(i, 1:i-1) * y(1:i-1)) / L(i, i);
    end
 
    x(n) = y(n)/U(n, n);
    for i = n-1:-1:1
         x(i) = (y(i) - U(i, i+1:n) * x(i+1:n)) / U(i, i);
    end
end