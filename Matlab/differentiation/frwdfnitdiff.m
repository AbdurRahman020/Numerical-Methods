function [frwdifs] = frwdfnitdiff(f, x, h, d)
%{
Forward finite-difference formulae

Parameters:
    f = function whose derivative is to be calculated
    x = point at which the derivative is calculated
    h = step size
    d = derivative number (upto 4)

Return:
    frwdifs = two values of derivative, second value is more accurate 
              because it incorporates more terms of Taylor series expanion
%}

    if d == 1
        fd1 = (f(x+h) - f(x)) / h;
        fd2 = (-f(x+2*h) + 4*f(x+h) - 3*f(x)) / (2*h);
    elseif d == 2
        fd1 = (f(x+2*h) - 2*f(x+h) + f(x)) / h^2;
        fd2 = (-f(x+3*h) + 4*f(x+2*h) - 5*f(x+h) + 2*f(x)) / h^2;
    elseif d == 3
        fd1 = (f(x+3*h) - 3*f(x+2*h) + 3*f(x+h) - f(x)) / h^3;
        fd2 = (-3*f(x+4*h) + 14*f(x+3*h) - 24*f(x+2*h) + 18*f(x+h) ...
            - 5*f(x)) / (2*h^3);
    elseif d == 4
        fd1 = (f(x+4*h) - 4*f(x+3*h) + 6*f(x+2*h) - 4*f(x+h) + f(x)) / h^4;
        fd2 = (-2*f(x+5*h) + 11*f(x+4*h) - 24*f(x+3*h) + 26*f(x+2*h) ...
            - 14*f(x+h) + 3*f(x)) / h^4;
    else
        disp("No more derivative formulae greater than 4.")
    end
    
    frwdifs = [fd1, fd2]';
end