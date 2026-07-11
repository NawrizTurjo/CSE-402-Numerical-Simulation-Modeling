clc;
clear;
close all;

f = @(x) x.^3 - x - 1;

x_l = 1;
x_u = 2;
es = 0.001;
max_iter = 100;

fprintf('Equation: x^3 - x - 1 = 0\n');
fprintf('f(x) = x^3 - x - 1\n\n');

fprintf('f(1) = %.6f\n', f(1));
fprintf('f(2) = %.6f\n', f(2));
fprintf('Since f(1)*f(2) < 0, choose x_l = 1 and x_u = 2.\n\n');

% Graph
x = linspace(-2, 2.5, 600);
y = f(x);

figure;
plot(x, y, 'b', 'LineWidth', 1.5);
hold on;
yline(0, 'k');
xline(0, 'k');
plot([1 2], [f(1) f(2)], 'ro', 'MarkerFaceColor', 'r');
grid on;
xlabel('x');
ylabel('f(x)');
title('Graph of f(x) = x^3 - x - 1');
legend('f(x)', 'x-axis', 'y-axis', 'Initial guesses');

if f(x_l)*f(x_u) > 0
    error('Invalid interval: f(x_l) and f(x_u) must have opposite signs.');
end

fprintf('False position iteration table:\n');
fprintf('iter       x_l       x_u       x_r       f(x_r)      ea(%%)\n');

old_xr = NaN;

for iter = 1:max_iter
    f_l = f(x_l);
    f_u = f(x_u);

    x_r = (x_u*f_l - x_l*f_u)/(f_l - f_u);
    f_r = f(x_r);

    if isnan(old_xr)
        fprintf('%4d  %8.5f  %8.5f  %8.5f  %10.5f       ---\n', ...
            iter, x_l, x_u, x_r, f_r);
    else
        ea = abs((x_r - old_xr)/x_r)*100;
        fprintf('%4d  %8.5f  %8.5f  %8.5f  %10.5f  %8.5f\n', ...
            iter, x_l, x_u, x_r, f_r, ea);

        if ea <= es
            break;
        end
    end

    if f_l*f_r > 0
        x_l = x_r;
    else
        x_u = x_r;
    end

    old_xr = x_r;
end

fprintf('\nAll real roots:\n');
fprintf('%.6f\n', x_r);

