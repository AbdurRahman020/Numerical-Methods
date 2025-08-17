function [Q, R] = qrDecomposition(A)
    %{
    Parameters:
        A : A real-valued (m x n) matrix

    Returns:
        Q : (m x n) orthonormal matrix
        R : (n x n) upper triangular matrix
    %}

    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);

    for j = 1:n
        % take the j-th column of A
        v = A(:, j);

        % subtract projections onto previously computed Q columns
        for i = 1:j-1
            % compute projection scalar
            R(i, j) = dot(Q(:, i), A(:, j));
            % remove component in direction of Q(:, i)
            v = v - R(i, j) * Q(:, i);
        end

        % normalize v to get orthonormal vector
        R(j, j) = norm(v);

        if R(j, j) == 0
            error('matrix has linearly dependent columns');
        end

        % store normalized vector in Q
        Q(:, j) = v / R(j, j);
    end
end
