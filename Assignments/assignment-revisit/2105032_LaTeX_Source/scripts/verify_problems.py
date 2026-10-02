"""Recompute and verify M1-M5 and A1-A5 using only NumPy.

Run with Python 3. All calculations use full float64 precision; only printed
numbers are formatted. Counts are numbers of completed updates. A1 tests the
update just applied; A2 and A4 test the gradient at the current iterate.
"""

import numpy as np


def fmt(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{float(array):.6g}"
    return "(" + ", ".join(f"{float(v):.6g}" for v in array) + ")"


def close(actual, expected, atol=1e-10, rtol=1e-10):
    return bool(np.allclose(actual, expected, atol=atol, rtol=rtol))


def header(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def quartic(x):
    return x**4 - 6 * x**3 + 8 * x**2


def quartic_grad(x):
    return 4 * x**3 - 18 * x**2 + 16 * x


def quartic_hess(x):
    return 12 * x**2 - 36 * x + 16


MAXIMUM = (9 - np.sqrt(17.0)) / 4
MINIMUM = (9 + np.sqrt(17.0)) / 4
DATA_X = np.array([0.5, 2.3, 2.9])
DATA_Y = np.array([1.4, 1.9, 3.2])


def quartic_gd(start, steps=300):
    x = float(start)
    trajectory = []
    for _ in range(steps):
        x -= 0.05 * quartic_grad(x)
        trajectory.append(x)
    assert np.all(np.isfinite(trajectory))
    return np.array(trajectory)


def relative_error_steps():
    x = 0.0
    for k in range(1, 10000):
        previous_error = abs(x - 3)
        x -= 0.1 * 2 * (x - 3)
        if abs(x - 3) <= 1e-6 * 3:
            assert previous_error > 1e-6 * 3
            assert k == int(np.ceil(np.log(1e-6) / np.log(0.8))) == 62
            return k, x
    raise AssertionError("Quadratic GD failed to converge")


def m1():
    header("M1. Quartic stationary points and two GD basins")
    roots = np.array([0.0, MAXIMUM, MINIMUM])
    assert close(np.sort(np.roots([4, -18, 16, 0])), roots)
    assert close(quartic_grad(roots), np.zeros(3))
    assert np.array_equal(np.sign(quartic_hess(roots)), [1, -1, 1])
    for x in roots:
        kind = "minimum" if quartic_hess(x) > 0 else "maximum"
        print(f"  x={fmt(x)}, J''={fmt(quartic_hess(x))}, "
              f"J={fmt(quartic(x))}: {kind}")
    for start, target in [(1.0, 0.0), (1.5, MINIMUM)]:
        trajectory = quartic_gd(start)
        print(f"  GD alpha=0.05, x0={fmt(start)}: "
              f"(x1,x2,x3)={fmt(trajectory[:3])}; x300={fmt(trajectory[-1])}")
        assert close(trajectory[-1], target)
    assert close(quartic_gd(1.0)[0], 0.9)
    assert close(quartic_gd(1.5)[0], 1.65)


def m2():
    header("M2. Quadratic learning rates and relative-error count")
    for alpha, behaviour in [
        (0.1, "monotone convergence"),
        (0.5, "one-step exact"),
        (0.9, "oscillating convergence"),
        (1.1, "divergent oscillation"),
    ]:
        x = 0.0
        trajectory = []
        ratios = []
        factor = 1 - 2 * alpha
        for k in range(1, 5):
            old_error = x - 3
            x -= alpha * 2 * old_error
            error = x - 3
            trajectory.append(x)
            assert close(error, -3 * factor**k)
            if old_error != 0:
                ratios.append(error / old_error)
                assert close(ratios[-1], factor)
            else:
                ratios.append(np.nan)
                assert error == 0
        ratio_text = "(" + ", ".join(
            "undefined (0/0)" if np.isnan(r) else fmt(r) for r in ratios
        ) + ")"
        print(f"  alpha={fmt(alpha)}: (x1..x4)={fmt(trajectory)}")
        print(f"    error ratios={ratio_text}; 1-2*alpha={fmt(factor)}; {behaviour}")
        if alpha == 0.5:
            assert trajectory == [3.0] * 4
        elif alpha < 1:
            assert abs(factor) < 1 and abs(factor)**300 < 1e-25
        else:
            assert abs(factor) > 1 and abs(factor)**100 > 1e7
    k, x = relative_error_steps()
    print(f"  alpha=0.1: smallest k with |e_k| <= 1e-6*|e_0| is {k}; "
          f"x_k={fmt(x)}")
    print(f"    |e_61/e_0|={fmt(0.8**61)}, |e_62/e_0|={fmt(0.8**62)}")


def m3():
    header("M3. Anisotropic versus isotropic quadratic GD")
    start = np.array([2.0, 1.0])
    for alpha, steps in [(0.1, 3), (0.2, 4)]:
        x = start.copy()
        factors = 1 - alpha * np.array([2.0, 8.0])
        print(f"  J=x1^2+4*x2^2, alpha={fmt(alpha)}, factors={fmt(factors)}")
        for k in range(1, steps + 1):
            x -= alpha * np.array([2 * x[0], 8 * x[1]])
            assert close(x, start * factors**k)
            print(f"    step {k}: x={fmt(x)}, J={fmt(x[0]**2 + 4*x[1]**2)}")
    print("  Stability boundary alpha=2/8=0.25; convergence requires 0<alpha<0.25.")
    boundary_factors = 1 - 0.25 * np.array([2.0, 8.0])
    assert boundary_factors[1] == -1  # No decay at the boundary.
    x = start.copy()
    print("  J=x1^2+x2^2, alpha=0.1, common factor=0.8 (straight line)")
    for k in range(1, 4):
        x -= 0.1 * 2 * x
        assert close(x, start * 0.8**k)
        assert close(x[0], 2 * x[1])
        print(f"    step {k}: x={fmt(x)}, J={fmt(x @ x)}")
    for label, factors in [
        ("x1^2+4*x2^2", np.array([0.8, 0.2])),
        ("x1^2+x2^2", np.array([0.8, 0.8])),
    ]:
        x = start.copy()
        for k in range(1, 10000):
            previous_norm = np.linalg.norm(x)
            x *= factors
            if np.linalg.norm(x) < 1e-6:
                break
        else:
            raise AssertionError("M3 did not converge")
        assert k == 66 and previous_norm >= 1e-6
        assert close(x, start * factors**k)
        print(f"  {label}, alpha=0.1: ||x||<1e-6 after {k} steps; "
              f"previous norm={fmt(previous_norm)}, final norm={fmt(np.linalg.norm(x))}")


def m4():
    header("M4. SSR, half-MSE, and adding a constant")
    residual_at_zero = DATA_Y - 0.64 * DATA_X
    optimum = residual_at_zero.mean()
    initial_ssr = residual_at_zero @ residual_at_zero
    initial_gradient = -2 * residual_at_zero.sum()
    print(f"  y-0.64*x={fmt(residual_at_zero)}; b*={fmt(optimum)}")
    first_steps = []
    for label, divisor in [("SSR", 1), ("MSE_half=SSR/(2m)", 6)]:
        gradient = initial_gradient / divisor
        step = 0.1 * gradient
        first_steps.append(-step)
        b = 0.0
        for _ in range(300):
            b -= 0.1 * (-2 * np.sum(residual_at_zero - b) / divisor)
        assert close(b, optimum)
        print(f"  {label}: loss(b0)={fmt(initial_ssr/divisor)}, "
              f"gradient={fmt(gradient)}, b1={fmt(-step)}, b300={fmt(b)}")
        # Adding 2 changes the constant coefficient only; verify by differentiation.
        coefficients = np.array([3.0, -2 * residual_at_zero.sum(), initial_ssr]) / divisor
        shifted = coefficients.copy()
        shifted[-1] += 2
        assert np.array_equal(np.polyder(coefficients), np.polyder(shifted))
        assert close(np.polyval(np.polyder(shifted), 0), gradient)
        print(f"    loss+2 at b0={fmt(initial_ssr/divisor+2)}; "
              f"same gradient={fmt(gradient)}, same b1={fmt(-step)}")
    assert close(first_steps, [0.5704, 0.5704 / 6])
    assert close(first_steps[0] / first_steps[1], 6)
    print("  SSR / half-MSE first-step ratio = 2m = 6.")


def m5():
    header("M5. Newton can converge to a maximum")
    newton_quadratic = 0.0 - (2 * (0.0 - 3)) / 2
    assert newton_quadratic == 3
    print(f"  (a) Quadratic: Newton x1={fmt(newton_quadratic)} exactly; "
          f"GD alpha=0.1 needs {relative_error_steps()[0]} steps for relative error <=1e-6.")
    x = 1.4
    print(f"  (b) At x0=1.4: J'={fmt(quartic_grad(x))}, J''={fmt(quartic_hess(x))}")
    assert close(quartic_grad(x), -1.904) and close(quartic_hess(x), -10.88)
    assert quartic_hess(x) < 0
    for k in range(1, 6):
        x -= quartic_grad(x) / quartic_hess(x)
        print(f"    Newton step {k}: x={fmt(x)}")
    assert close(x, MAXIMUM) and quartic_hess(x) < 0
    print(f"    Newton limit={fmt(MAXIMUM)}: maximum, J={fmt(quartic(x))}")
    trajectory = quartic_gd(1.4)
    assert close(trajectory[-1], MINIMUM) and quartic_hess(trajectory[-1]) > 0
    print(f"    GD alpha=0.05: (x1,x2,x3)={fmt(trajectory[:3])}")
    print(f"    GD x300={fmt(trajectory[-1])}; minimum={fmt(MINIMUM)}, "
          f"J={fmt(quartic(trajectory[-1]))}")


def a1():
    header("A1. Data A: intercept-only and two-parameter regression")
    b = 0.0
    previous_step = np.inf
    print("  (i) Fixed slope=0.64, SSR, alpha=0.1; signed step=alpha*derivative, b<-b-step.")
    for k in range(1, 10000):
        gradient = -2 * np.sum(DATA_Y - b - 0.64 * DATA_X)
        step = 0.1 * gradient
        b -= step
        if k <= 3:
            print(f"    step {k}: derivative={fmt(gradient)}, "
                  f"signed step={fmt(step)}, new b={fmt(b)}")
        if k == 1:
            assert close(b, 0.5704)
        if abs(step) < 0.001:
            break
        previous_step = abs(step)
    else:
        raise AssertionError("A1 intercept GD did not converge")
    assert k == 8 and previous_step >= 0.001
    assert close(b, (DATA_Y - 0.64 * DATA_X).mean() * (1 - 0.4**k))
    print(f"    First |step|<0.001: {k} updates; final b={fmt(b)}, "
          f"last |step|={fmt(abs(step))}")
    design = np.column_stack((np.ones(3), DATA_X))
    parameters = np.array([0.0, 1.0])
    previous_size = np.inf
    print("  (ii) Both parameters, SSR, alpha=0.01, start (b,m)=(0,1).")
    for k in range(1, 100001):
        gradient = 2 * design.T @ (design @ parameters - DATA_Y)
        step = 0.01 * gradient
        parameters -= step
        if k <= 2:
            print(f"    step {k}: (dSSR/db,dSSR/dm)={fmt(gradient)}, "
                  f"new (b,m)={fmt(parameters)}")
        if k == 1:
            assert close(gradient, [-1.6, -0.8])
            assert close(parameters, [0.016, 1.008])
        if np.max(np.abs(step)) < 1e-6:
            break
        previous_size = np.max(np.abs(step))
    else:
        raise AssertionError("A1 two-parameter GD did not converge")
    assert k == 808 and previous_size >= 1e-6
    print(f"    First both |steps|<1e-6: {k} updates; (b,m)={fmt(parameters)}")
    print(f"    Last signed steps={fmt(step)}")
    n = DATA_X.size
    sum_x, sum_y = DATA_X.sum(), DATA_Y.sum()
    sum_x2, sum_xy = DATA_X @ DATA_X, DATA_X @ DATA_Y
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n
    exact = np.array([intercept, slope])
    print(f"  (iii) n={n}, sum x={fmt(sum_x)}, sum y={fmt(sum_y)}, "
          f"sum x^2={fmt(sum_x2)}, sum xy={fmt(sum_xy)}")
    print(f"    Closed form (b,m)={fmt(exact)}; GD absolute errors={fmt(abs(parameters-exact))}")
    assert close(exact, [37 / 39, 25 / 39])
    assert close(exact, np.linalg.lstsq(design, DATA_Y, rcond=None)[0])
    assert close(parameters, exact, atol=1e-4, rtol=0)


def a2():
    header("A2. Houses: scaling, stability, and least squares")
    features = np.array([[800.0, 2], [1000, 2], [1200, 3], [1500, 3]])
    prices = np.array([40.0, 50, 60, 70])
    design = np.column_stack((np.ones(4), features))
    gradient_zero = -design.T @ prices / 4
    first = -1e-7 * gradient_zero
    largest_eigenvalue = np.linalg.eigvalsh(design.T @ design / 4)[-1]
    stable_boundary = 2 / largest_eigenvalue
    assert close(gradient_zero, [-55, -64750, -142.5])
    assert close(first, [5.5e-6, 0.006475, 1.425e-5])
    assert 1e-7 < stable_boundary < 2e-6
    print(f"  (i) Unscaled grad(w=0)={fmt(gradient_zero)}")
    print(f"    alpha=1e-7: w1={fmt(first)}")
    print(f"    lambda_max(X^T X/4)={fmt(largest_eigenvalue)}; "
          f"stability boundary 2/lambda_max={fmt(stable_boundary)}")
    print("    Convergence requires alpha strictly below this boundary.")
    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=0)
    scaled = np.column_stack((np.ones(4), (features - means) / stds))
    assert close(scaled[:, 1:].mean(axis=0), [0, 0])
    assert close(scaled[:, 1:].std(axis=0), [1, 1])
    weights = np.zeros(3)
    previous_norm = np.inf
    for k in range(100001):
        gradient = scaled.T @ (scaled @ weights - prices) / 4
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm < 1e-8:
            break
        previous_norm = gradient_norm
        weights -= 0.1 * gradient
    else:
        raise AssertionError("A2 scaled GD did not converge")
    assert k == 1391 and previous_norm >= 1e-8
    original_weights = np.r_[weights[0] - np.sum(weights[1:] * means / stds),
                             weights[1:] / stds]
    least_squares = np.linalg.lstsq(design, prices, rcond=None)[0]
    assert close(least_squares, [5, 1 / 26, 35 / 13])
    assert close(original_weights, least_squares, atol=2e-7, rtol=0)
    assert close(design.T @ (design @ least_squares - prices), np.zeros(3), atol=1e-7)
    print(f"  (ii) Feature means={fmt(means)}, population stds={fmt(stds)}")
    print(f"    Scaled GD alpha=0.1: {k} updates to ||grad||_2<1e-8; "
          f"||grad||_2={fmt(gradient_norm)}")
    print(f"    Scaled weights (w0,w_size_z,w_bed_z)={fmt(weights)}")
    print(f"    Mapped original weights={fmt(original_weights)}")
    print(f"  (iii) Original-unit least-squares solution (lstsq)={fmt(least_squares)}")
    print(f"    Mapping absolute errors={fmt(abs(original_weights-least_squares))}")
    house = np.array([1.0, 1100, 2])
    prediction = house @ least_squares
    assert close(prediction, 685 / 13)
    assert close(house @ original_weights, prediction, atol=1e-6)
    print(f"    Predicted price for 1100 sqft, 2 bedrooms={fmt(prediction)} lakh")


def a3():
    header("A3. Streaming SGD, batch, and mini-batch updates")
    x = np.array([1.0, 2, 3])
    y = np.array([2.1, 3.9, 6.2])
    w = 0.0
    print("  (i) One SGD epoch, alpha=0.05, sample order 1,2,3:")
    expected = [0.105, 0.474, 1.1907]
    for index, (xi, yi) in enumerate(zip(x, y), 1):
        w -= 0.05 * (w * xi - yi) * xi
        assert close(w, expected[index - 1])
        print(f"    sample {index}: w={fmt(w)}")
    one_epoch = w
    optimum = (x @ y) / (x @ x)
    w = 0.0
    for k in range(1, 4):
        w -= 0.05 * np.mean((w * x - y) * x)
        assert close(w, optimum * (1 - (1 - 0.05 * np.mean(x*x))**k))
        if k in (1, 3):
            print(f"  (ii) Batch mean-loss GD, alpha=0.05, step {k}: w={fmt(w)}")
    assert close(w, 1.1183611111111112)
    w = 0.0
    print("  (iii) Mini-batches use their own mean gradient, alpha=0.05:")
    for batch, indices in enumerate([np.array([0, 1]), np.array([2])], 1):
        w -= 0.05 * np.mean((w * x[indices] - y[indices]) * x[indices])
        assert close(w, [0.2475, 1.066125][batch - 1])
        print(f"    update {batch}: w={fmt(w)}")
    assert close(optimum, 57 / 28)
    print(f"  (iv) w*=sum(x*y)/sum(x^2)={fmt(x@y)}/{fmt(x@x)}={fmt(optimum)}")
    finals = []
    print("  (v) Each 50-epoch run starts at w=0; fixed sample order; decay epochs k=0..49.")
    for decaying in (False, True):
        w = 0.0
        for epoch in range(50):
            alpha = 0.05 / (1 + 0.1 * epoch) if decaying else 0.05
            for xi, yi in zip(x, y):
                w -= alpha * (w * xi - yi) * xi
        finals.append(w)
        label = "decaying alpha_k=0.05/(1+0.1k)" if decaying else "constant alpha=0.05"
        print(f"    {label}: final w={fmt(w)}, w-w*={fmt(w-optimum)}")
        assert np.isfinite(w)
    epoch_factor = np.prod(1 - 0.05 * x*x)
    epoch_fixed_point = one_epoch / (1 - epoch_factor)
    assert close(finals[0], epoch_fixed_point * (1 - epoch_factor**50))
    assert not close(finals[0], optimum, atol=1e-3)
    assert abs(finals[1] - optimum) < abs(finals[0] - optimum)
    print(f"    Constant-rate epoch-end fixed point={fmt(epoch_fixed_point)} "
          "(sample noise leaves a bias).")
    print("  (vi) New sample (4,8.1), continuing each final model without restarting:")
    for label, w, alpha in [
        ("constant schedule", finals[0], 0.05),
        ("decaying schedule, next epoch k=50", finals[1], 0.05 / (1 + 0.1 * 50)),
    ]:
        gradient = (4 * w - 8.1) * 4
        updated = w - alpha * gradient
        assert abs(4 * updated - 8.1) < abs(4 * w - 8.1)
        print(f"    {label}: start w={fmt(w)}, alpha={fmt(alpha)}, "
              f"gradient={fmt(gradient)}, new w={fmt(updated)}")


def a4():
    header("A4. Fixed-volume cylindrical can")

    def area(r):
        return 2 * np.pi * r*r + 1000 / r

    def gradient(r):
        return 4 * np.pi * r - 1000 / r**2

    def hessian(r):
        return 4 * np.pi + 2000 / r**3

    exact_radius = (250 / np.pi)**(1 / 3)
    exact_height = 500 / (np.pi * exact_radius**2)
    boundary = 2 / hessian(exact_radius)
    assert close(gradient(exact_radius), 0)
    assert close(exact_height, 2 * exact_radius)
    assert close(hessian(exact_radius), 12 * np.pi)
    print(f"  Exact r*={fmt(exact_radius)} cm, h*={fmt(exact_height)} cm = 2r*")
    print(f"  A(r*)={fmt(area(exact_radius))} cm^2; "
          f"A''(r*)={fmt(hessian(exact_radius))}")
    print(f"  Local stability boundary alpha=2/A''(r*)={fmt(boundary)} "
          "(strict upper bound).")
    radius = 2.0
    previous_gradient = np.inf
    print("  GD r0=2, alpha=0.01:")
    for k in range(10001):
        current_gradient = gradient(radius)
        if abs(current_gradient) < 1e-6:
            break
        previous_gradient = abs(current_gradient)
        radius -= 0.01 * current_gradient
        assert radius > 0 and np.isfinite(radius)
        if k < 3:
            print(f"    step {k+1}: r={fmt(radius)}, A={fmt(area(radius))}")
    else:
        raise AssertionError("A4 GD did not converge")
    assert k == 32 and previous_gradient >= 1e-6
    assert close(radius, exact_radius, atol=3e-8)
    print(f"    First |A'|<1e-6: {k} updates; r={fmt(radius)}, "
          f"A'={fmt(current_gradient)}")
    multiplier = 1 - 0.06 * hessian(exact_radius)
    assert multiplier < -1 and 0.06 > boundary
    print(f"  GD r0=4, alpha=0.06: local multiplier={fmt(multiplier)}; "
          "optimum is unstable.")
    radius = 4.0
    trajectory = []
    for k in range(1, 1001):
        radius -= 0.06 * gradient(radius)
        assert radius > 0 and np.isfinite(radius)
        trajectory.append(radius)
        if k <= 5:
            print(f"    step {k}: r={fmt(radius)}, A={fmt(area(radius))}")
    assert np.all((np.array(trajectory[:5]) - exact_radius)
                  * np.array([1, -1, 1, -1, 1]) > 0)
    assert close(trajectory[-1], trajectory[-3])
    assert close(trajectory[-2], trajectory[-4])
    assert abs(trajectory[-1] - trajectory[-2]) > 1
    assert abs(gradient(trajectory[-1])) > 1
    cycle_multiplier = ((1 - 0.06 * hessian(trajectory[-1]))
                        * (1 - 0.06 * hessian(trajectory[-2])))
    assert abs(cycle_multiplier) < 1
    print(f"    (r999,r1000)={fmt(trajectory[-2:])}; "
          f"two-cycle multiplier={fmt(cycle_multiplier)}")
    print("    Oscillates toward an attracting two-cycle; does not converge to r*.")


def a5():
    header("A5. Logistic regression")
    hours = np.arange(1.0, 6.0)
    labels = np.array([0.0, 0, 1, 0, 1])
    design = np.column_stack((hours, np.ones(5)))

    def quantities(parameters):
        logits = design @ parameters
        # Stable sigmoid and log-loss, without clipping or rounding.
        probabilities = np.exp(-np.logaddexp(0, -logits))
        loss = np.mean(np.logaddexp(0, logits) - labels * logits)
        gradient = design.T @ (probabilities - labels) / 5
        return probabilities, gradient, loss

    parameters = np.zeros(2)  # Order: (w, b).
    print("  alpha=0.5; p, gradients, and loss are evaluated BEFORE each update.")
    for k in range(1, 20001):
        probabilities, gradient, loss = quantities(parameters)
        if k <= 2:
            print(f"  step {k}: start (w,b)={fmt(parameters)}")
            print(f"    p={fmt(probabilities)}")
            print(f"    (dL/dw,dL/db)={fmt(gradient)}, loss={fmt(loss)}")
        if k == 1:
            assert close(probabilities, np.full(5, 0.5))
            assert close(gradient, [-0.1, 0.1])
            assert close(loss, np.log(2))
        parameters -= 0.5 * gradient
        if k <= 2:
            print(f"    new (w,b)={fmt(parameters)}")
        if k == 1:
            assert close(parameters, [0.05, -0.05])
    probabilities, gradient, loss = quantities(parameters)
    w, b = parameters
    prediction = np.exp(-np.logaddexp(0, -(w * 3.5 + b)))
    boundary = -b / w
    hessian = design.T @ ((probabilities * (1 - probabilities))[:, None] * design) / 5
    assert np.linalg.norm(gradient) < 1e-10
    assert np.all(np.linalg.eigvalsh(hessian) > 0)
    assert close(parameters, [1.0904255603, -3.8939667463], atol=1e-9)
    assert loss < np.log(2) and 0 < prediction < 1
    assert close(np.exp(-np.logaddexp(0, -(w * boundary + b))), 0.5)
    print(f"  After 20000 total updates: (w,b)={fmt(parameters)}, loss={fmt(loss)}")
    print(f"    ||grad||_2={fmt(np.linalg.norm(gradient))}; "
          "positive-definite Hessian confirms the unique minimum.")
    print(f"    Predicted P(pass | hours=3.5)={fmt(prediction)}")
    print(f"    Decision boundary -b/w={fmt(boundary)} hours (p=0.5)")


def main():
    print("NumPy-only verification of 10 gradient-descent teaching problems")
    print("Full float64 precision internally; displayed values use 6 significant digits.")
    print("Counts mean completed updates; all checks use unrounded values.")
    for problem in (m1, m2, m3, m4, m5, a1, a2, a3, a4, a5):
        problem()
        print("  Assertions passed.")
    print("\nALL ASSERTIONS PASSED (10 problems).")


if __name__ == "__main__":
    main()
