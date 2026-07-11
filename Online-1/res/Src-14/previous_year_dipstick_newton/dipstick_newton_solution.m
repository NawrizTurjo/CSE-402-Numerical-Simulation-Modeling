clc;
clear;
close all;

r = 4;
V = 5;

f = @(h) pi*h.^2.*(3*r - h)/3 - V;
df = @(h) pi*(2*r*h - h.^2);

h0 = 0.5;
es = 0.05;
max_iter = 100;

fprintf('Tank diameter = 8 ft\n');
fprintf('Radius r = 4 ft\n');
fprintf('Volume V = 5 ft^3\n');
fprintf('f(h) = pi*h^2*(12 - h)/3 - 5\n');
fprintf('f''(h) = pi*(8h - h^2)\n\n');

fprintf('f(0) = %.6f\n', f(0));
fprintf('f(1) = %.6f\n', f(1));
fprintf('Since f(0)*f(1) < 0, choose h0 = 0.5.\n\n');

fprintf('Newton-Raphson iteration table:\n');
fprintf('iter      h estimate          f(h)        ea (%%)\n');

old_h = h0;

for iter = 1:max_iter
    if df(old_h) == 0
        error('Derivative is zero. Newton-Raphson fails.');
    end

    new_h = old_h - f(old_h)/df(old_h);
    ea = abs((new_h - old_h)/new_h)*100;

    fprintf('%4d  %14.6f  %12.6f  %12.6f\n', iter, new_h, f(new_h), ea);

    if ea <= es
        root = new_h;
        break;
    end

    old_h = new_h;
end

fprintf('\nDipstick wet height h = %.6f ft\n', root);

% Graph
h = linspace(0, 8, 600);
y = f(h);

figure;
plot(h, y, 'b', 'LineWidth', 1.5);
hold on;
yline(0, 'k');
plot([0 1], [f(0) f(1)], 'ro', 'MarkerFaceColor', 'r');
grid on;
xlabel('h');
ylabel('f(h)');
title('Dipstick Height Function');
legend('f(h) = pi*h^2*(12-h)/3 - 5', 'x-axis', 'Sign check');

