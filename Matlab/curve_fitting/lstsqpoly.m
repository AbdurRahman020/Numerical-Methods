function C = lstsqpoly(x, y, n)
%{
Least-Square Polynomials

Parameters:
    x : independent variable
    y : dependent variable
    n : degree of least-square polynomial

Return:
    C : coefficient matrix for the polynomial  
%}

    % ensure that x and y have the same length
    m = length(x);
    if length(y) ~= m
        error('x and y should be of same length')
    end
    
    % initialize the Vandermonde matrix F of size m x (n+1)
    F = zeros(m, n+1);
    
    % fill in the Vandermonde matrix F with powers of x
    for i = 1:n + 1
        % each column i is x^(i-1)
        F(:, i) = x'.^(i-1);
    end
    
    % compute the normal equation components
    A = F' * F;      % coefficient matrix
    B = F' * y';     % right-hand side vector
    
    % solve the normal equations and flip the result to match polynomial order
    C = flipud(A \ B);
end
