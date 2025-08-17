function Q = gramSchmidt(V)
    %{
    Parameters:
        V : A (m x n) matrix, where each column is a vector of dimension 'm'

    Return:
        Q : (m x n) orthonormal matrix
    %}

    % dimensions of input matrix
    [m, n] = size(V);
    % initialize orthonormal matrix
    Q = zeros(m, n);

    for i = 1:n
        % copy of current column vector
        vi = V(:, i);
        
        for j = 1:i-1
            qj = Q(:, j);
            % subtract projection onto previous qj
            vi = vi - (dot(qj, vi) * qj);
        end
        
        % compute euclidean norm
        norm_vi = norm(vi);  
        
        if norm_vi < 1e-10
            error('vector %d is linearly dependent or too close to zero', i);
        end
        
        % normalize
        Q(:, i) = vi / norm_vi;
    end
end
