clc; clear; format short;

addpath differentiation

f = @(x) -0.1*x^4-0.15*x^3-0.5*x^2-0.25*x+1.2;

disp("Forward finite-difference Formula")
derv_f = frwdfnitdiff(f,0.5,0.25,2)

disp("Backward finite-difference Formula")
derv_b = bkwdfnitdiff(f,0.5,0.25,2)

disp("Centered finite-difference Formula")
derv_i = cntfnitdiff(f,0.5,0.25,2)