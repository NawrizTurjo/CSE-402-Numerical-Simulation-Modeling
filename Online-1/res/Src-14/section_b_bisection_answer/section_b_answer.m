clc;
clear;
close all;

% Change this value if the exam gives a different sth.
sth = 1.0;

f = @(x) 0.6*log(x + 1) - sth*sin(1.7*x) - 0.08*x.^2 - 0.08;

a = 0;
b = 10;
h = 0.1;
tol = 1e-6;
max_iter = 100;

% 1. Graph plotting for 0 <= x <= 10
x = linspace(a, b, 1000);
y = f(x);

figure;
plot(x, y, 'b', 'LineWidth', 1.5);
hold on;
yline(0, 'k');
grid on;
xlabel('x');
ylabel('f(x)');
title('Graph of f(x) on [0, 10]');

% 2. Find sign-change intervals using step size 0.1
intervals = [];
x1 = a;
f1 = f(x1);

while x1 < b
    x2 = x1 + h;
    if x2 > b
        x2 = b;
    end

    f2 = f(x2);

    if f1 == 0
        intervals = [intervals; x1 x1];
    elseif f1*f2 < 0
        intervals = [intervals; x1 x2];
    end

    x1 = x2;
    f1 = f2;
end

fprintf('Sign-change intervals with step size h = 0.1:\n');
for i = 1:size(intervals, 1)
    fprintf('[%.1f, %.1f]\n', intervals(i, 1), intervals(i, 2));
end

% 3, 4, 5. Bisection for every sign-change interval and print tables
roots = [];

fprintf('\nBisection results:\n');

for k = 1:size(intervals, 1)
    left = intervals(k, 1);
    right = intervals(k, 2);

    fprintf('\nRoot %d\n', k);
    fprintf('Initial interval: [%.1f, %.1f]\n', left, right);
    fprintf('iter         a         b         mid       f(mid)\n');

    if left == right
        root = left;
        fprintf('%4d  %8.5f  %8.5f  %8.5f  %11.6f\n', 1, left, right, root, f(root));
    else
        fa = f(left);
        fb = f(right);

        if fa*fb > 0
            error('Bisection needs f(a) and f(b) to have opposite signs.');
        end

        for iter = 1:max_iter
            mid = (left + right)/2;
            fm = f(mid);

            fprintf('%4d  %8.5f  %8.5f  %8.5f  %11.6f\n', iter, left, right, mid, fm);

            if abs(fm) < tol || abs(right - left)/2 < tol
                root = mid;
                break;
            end

            if fa*fm < 0
                right = mid;
                fb = fm;
            else
                left = mid;
                fa = fm;
            end
        end
    end

    roots = [roots; root];
    fprintf('Approximate root = %.6f\n', root);
end

fprintf('\nAll roots:\n');
for i = 1:length(roots)
    fprintf('%.6f\n', roots(i));
end

fprintf('\nWhy not bisection once on [0, 10]?\n');
fprintf('Bisection needs f(0)*f(10) < 0. If the endpoint signs are same,\n');
fprintf('it cannot start. Also, one bisection run can find only one root,\n');
fprintf('so it may miss other roots inside [0, 10].\n');

