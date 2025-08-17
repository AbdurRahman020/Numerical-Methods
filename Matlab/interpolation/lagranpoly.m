function [C, L] = lagranpoly(x, y)
%{
Langrange's Interpolating Polynomial

Parameters: 
    x = [x0 x1 ... xn]
    y = [y0 y1 ... yn]

Returns:
    C = row matrix that contains Lagrange polynomial
        coefficients of degree 'n'
    L = n-by-n matrix that contains Lagrange coefficient
        polynomials
%}

    n = length(x);
    if length(y) ~= n
        error("x and y must be of same length")
    end
 
    L = zeros(n, n);
 
    for i = 1:n
        V = 1;
        for j = 1:n
            if i ~= j
                V = conv(V, poly(x(j))) / (x(i) - x(j));
            end 
        end
        L(i, :) = V;
    end
    
    C = y*L;
end
