import numpy as np
import matplotlib.pyplot as plt


def root_finder(
    f,
    method: str,
    start: float = None,
    end: float = None,
    guess: float = None,
    tol: float = 1e-4,
) -> float:
    if method == "bisection":
        print(f"{'iter':^6} {'xl':^10} {'xu':^10} {'xm':^10} {'f(xm)':^12} {'ea':^10}")
        ea = 1.0
        xl = start
        xu = end
        xm_old = None
        iter = 1
        while ea > tol:
            xm_new = (xl + xu) / 2
            if xm_old is not None:
                ea = abs((xm_new - xm_old) / xm_new)

            print(
                f"{iter:^6} {xl:^10.4f} {xu:^10.4f} {xm_new:^10.4f} {f(xm_new):^12.4f} {ea:^10.4f}"
            )

            if f(xl) * f(xm_new) < 0:
                xu = xm_new
            elif f(xl) * f(xm_new) > 0:
                xl = xm_new
            else:
                return xm_new

            xm_old = xm_new
            iter += 1

        return xm_old

    elif method == "false-position":
        print(f"{'iter':^6} {'xl':^10} {'xu':^10} {'xr':^10} {'f(xr)':^12} {'ea':^10}")
        ea = 1.0
        xl = start
        xu = end
        xr_old = None
        iter = 1
        while ea > tol:
            denom = f(xl) - f(xu)
            if denom == 0:
                print("Division by zero in false-position method.")
                return None
            xr_new = (xu * f(xl) - xl * f(xu)) / denom

            if xr_old is not None:
                ea = abs((xr_new - xr_old) / xr_new)

            print(
                f"{iter:^6} {xl:^10.4f} {xu:^10.4f} {xr_new:^10.4f} {f(xr_new):^12.4f} {ea:^10.4f}"
            )

            if f(xl) * f(xr_new) < 0:
                xu = xr_new
            elif f(xl) * f(xr_new) > 0:
                xl = xr_new
            else:
                return xr_new

            xr_old = xr_new
            iter += 1

        return xr_old

    elif method == "newton-raphson":
        print(f"{'iter':^6} {'x':^10} {'x_new':^10} {'f(x_new)':^12} {'ea':^10}")
        ea = 1.0
        x = guess
        iter = 1
        while ea > tol:
            h = 1e-8
            df = (f(x + h) - f(x)) / h
            if df == 0:
                print("Derivative is zero. Newton-Raphson failed.")
                return None
            x_new = x - (f(x) / df)

            ea = abs((x_new - x) / x_new)

            print(
                f"{iter:^6} {x:^10.4f} {x_new:^10.4f} {f(x_new):^12.4f} {ea:^10.4f}"
            )
            x = x_new
            iter += 1

        return x

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'bisection', 'false-position', or 'newton-raphson'.")


def find_interval(f, start: float, end: float, step: float):
    intervals = []
    x = start
    while x < end:
        x_next = round(x + step, 6)
        if f(x) * f(x_next) < 0 or f(x_next) == 0:
            intervals.append((round(x, 4), round(x_next, 4)))
        x = x_next
    return intervals


if __name__ == "__main__":
    f = lambda x: np.cbrt(x)
    start = -3
    end = 3
    step = 0.1
    tol = 1e-4

    x_vals = np.linspace(-4, 4, 100)
    y_vals = f(x_vals)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.show()

    intervals = find_interval(f, start, end, step)
    print(f"All sign-change intervals: {intervals}")
    first_interval = intervals[0]

    # print("\n--- Bisection Method ---")
    # root_finder(f, method="bisection", start=first_interval[0], end=first_interval[1], tol=tol)

    # print("\n--- False-Position Method ---")
    # root_finder(f, method="false-position", start=first_interval[0], end=first_interval[1], tol=tol)

    print("\n--- Newton-Raphson Method ---")
    root_finder(f, method="newton-raphson", guess=0.1, tol=1e-5)
