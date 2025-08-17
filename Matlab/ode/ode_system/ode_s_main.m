clc; clear; format short

addpath ode\ode_system

f = cell(2, 1);
f{1} = @(t, y) sin(t) + cos(y(1)) + sin(y(2));
f{2} = @(t, y) cos(t) + sin(y(2));
y_init = zeros(2, 1);
y_init(1) = -1; y_init(2) = 1;

disp("RK2 for system of ODE")
[t1, y1] = RK2System(f, 0, 20, 40, y_init);
figure; plot(t1, y1,'-o'); title("RK2 System")
ylabel("y"), xlabel("t"); grid on;

disp("RK3 for system of ODE")
[t2, y2] = RK3System(f, 0, 20, 40, y_init);
figure; plot(t2, y2,'-o'); title("RK3 System")
ylabel("y"), xlabel("t"); grid on;

disp("RK4 for system of ODE")
[t3, y3] = RK4System(f, 0, 20, 40, y_init);
figure; plot(t3, y3,'-o'); title("RK4 System")
ylabel("y"), xlabel("t"); grid on;
