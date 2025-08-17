clc; clear; format short;

addpath ode

disp("Euler's Method")
[t1, y1] = euler(@(t, y) (t - y)/2, 0, 3, 12, 1);
figure; plot(t1, y1, 'o-'); title("Euler's Method"); 
xlabel('t'); ylabel('y'); grid on;

disp("Mid Point Method")
[t2, y2] = midpoint(@(t, y) (t - y)/2, 0, 3, 12, 1);
figure; plot(t2, y2, 'o-'); title("Midpoint Method"); 
xlabel('t'); ylabel('y'); grid on;

disp("Taylor's (Order Two) Method")
f  = @(t, y) y - t^2 + 1;
df = @(t, y) -2*t + (y - t^2 + 1);
[t3, y3] = taylor2(f, df, 0, 2, 10, 0.5);
figure; plot(t3, y3, 'o-'); title("Taylor's (Order Two)"); 
xlabel('t'); ylabel('y'); grid on;

disp("Taylor's (Order Four) Method")
f   = @(t, y) y - t^2 + 1;
df1 = @(t, y) -2*t + (y - t^2 + 1);
df2 = @(t, y) -2 + df1(t, y);
df3 = @(t, y) df2(t, y);
[t4, y4] = taylor4(f, df1, df2, df3, 0, 2, 10, 0.5);
figure; plot(t4, y4, 'o-'); title("Taylor's (Order Four)"); 
xlabel('t'); ylabel('y'); grid on;

disp("Runge Kutta's (RK2) Method")
[t5, y5] = RK2(@(t, y) (t - y)/2, 0, 3, 15, 1);
figure; plot(t5, y5, 'o-'); title("Runge-Kutta 2");
xlabel('t'); ylabel('y'); grid on;

disp("Runge Kutta's (RK3) Method")
[t6, y6] = RK3(@(t, y) (t - y)/2, 0, 3, 15, 1);
figure; plot(t6, y6, 'o-'); title("Runge-Kutta 3"); 
xlabel('t'); ylabel('y'); grid on;

disp("Runge Kutta's (RK4) Method")
[t7, y7] = RK4(@(t, y) (t - y)/2, 0, 3, 15, 1);
figure; plot(t7, y7, 'o-'); title("Runge-Kutta 4"); 
xlabel('t'); ylabel('y'); grid on;
