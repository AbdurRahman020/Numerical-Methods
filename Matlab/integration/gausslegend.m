function I = gausslegend(func, a, b, g)
%{
Gauss-Legendre Integration/Quadrature

Parameters:
    func : handle of function to be integrated
    a, b : lower and upper bounds
    g    : n-point rule (2,3,4)

Return:
    I : integral estimate 
%} 

    if g == 2
        w = [1, 1];
        x = [-1/sqrt(3), 1/sqrt(3)];
    elseif g == 3
        w = [5/9, 8/9, 5/9];
        x = [-sqrt(3/5), 0, sqrt(3/5)];
    elseif g == 4
        % (18 - sqrt(30))/36 = 0.347854845137454
        % (18 + sqrt(30))/36 = 0.652145154862546
        w = [0.347854845137454, 0.347854845137454, 0.652145154862546, 0.652145154862546];
        % sqrt(3/7 + (2/7)*sqrt(6/5)) = 0.861136311594053
        % sqrt(3/7 - (2/7)*sqrt(6/5)) = 0.339981043584856 
        x = [-0.861136311594053, 0.861136311594053, -0.339981043584856, 0.339981043584856];
    else
        error("'g' is out of range, can't calculate integral.")
    end
    
    t = ((a + b)/2) - ((b - a)/2)*x;
    I = ((b - a)/2) * sum(w.*func(t));
end
