clc;
clear;
close all;

f = @(x) 1000 - 298*x + 3*x.^(2/3);
df = @(x) -298 + 2*x.^(-1/3);

x0 = 4;
es = 0.05;
max_iter = 50;

fprintf('Cost: C(x) = 1000 + 2x + 3x^(2/3)\n');
fprintf('Revenue: R(x) = 300x\n');
fprintf('Break-even equation: C(x) = R(x)\n');
fprintf('f(x) = 1000 - 298x + 3x^(2/3)\n');
fprintf('f''(x) = -298 + 2x^(-1/3)\n\n');

fprintf('f(3) = %.6f\n', f(3));
fprintf('f(4) = %.6f\n', f(4));
fprintf('Since f(3) and f(4) have opposite signs, choose x0 = 4.\n\n');

fprintf('Newton-Raphson iteration table:\n');
fprintf('iter      x estimate          f(x)        ea (%%)\n');

old_x = x0;

for iter = 1:max_iter
    new_x = old_x - f(old_x)/df(old_x);
    ea = abs((new_x - old_x)/new_x)*100;

    fprintf('%4d  %14.6f  %12.6f  %12.6f\n', iter, new_x, f(new_x), ea);

    if ea <= es
        root = new_x;
        break;
    end

    old_x = new_x;
end

fprintf('\nBreak-even quantity = %.6f grams per day\n', root);

% Graph of f(x)
x = linspace(0.1, 8, 600);
y = f(x);

figure;
plot(x, y, 'b', 'LineWidth', 1.5);
hold on;
yline(0, 'k');
plot([3 4], [f(3) f(4)], 'ro', 'MarkerFaceColor', 'r');
grid on;
xlabel('x');
ylabel('f(x)');
title('Break-even Function');
legend('f(x) = 1000 - 298x + 3x^{2/3}', 'x-axis', 'Sign check points');

