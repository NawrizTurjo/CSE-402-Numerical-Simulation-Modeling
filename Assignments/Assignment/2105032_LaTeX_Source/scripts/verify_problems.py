"""Recompute every number that appears in the Part B solutions (M1-M5, A1-A5).

Run:  python scripts/verify_problems.py
Every quoted value in the slides is printed here, and the key facts are
asserted, so a wrong number fails loudly.
"""
import math

import numpy as np

np.set_printoptions(precision=6, suppress=True)


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0)


def header(name):
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)


# ---------------------------------------------------------------------------
def m1():
    header("M1  f(x,y) = x^4 + y^4 - 4xy")
    f = lambda v: v[0] ** 4 + v[1] ** 4 - 4 * v[0] * v[1]
    grad = lambda v: np.array([4 * v[0] ** 3 - 4 * v[1], 4 * v[1] ** 3 - 4 * v[0]])
    hess = lambda v: np.array([[12 * v[0] ** 2, -4.0], [-4.0, 12 * v[1] ** 2]])

    # stationary points: y = x^3, x = y^3  ->  x^9 = x  ->  x in {0, 1, -1} (real)
    roots = np.roots([1, 0, 0, 0, 0, 0, 0, 0, -1, 0])
    real = sorted({round(r.real, 12) for r in roots if abs(r.imag) < 1e-9})
    print("real roots of x^9 - x:", real)
    assert real == [-1.0, 0.0, 1.0]
    for p in [(0, 0), (1, 1), (-1, -1)]:
        v = np.array(p, float)
        ev = np.linalg.eigvalsh(hess(v))
        print(f"  point {p}: grad={grad(v)}, f={f(v)}, Hessian eig={ev}")
    assert close(np.linalg.eigvalsh(hess(np.zeros(2))), [-4, 4])
    assert close(np.linalg.eigvalsh(hess(np.ones(2))), [8, 16])

    x0 = np.array([1.0, 0.0])
    g0 = grad(x0)
    x1 = x0 - 0.1 * g0
    print(f"  grad at (1,0) = {g0},  x1 = {x1},  f(x0)={f(x0)}, f(x1)={f(x1):.4f}")
    assert close(x1, [0.6, 0.4]) and close(f(x1), -0.8048)
    print(f"  local stability limit alpha < 2/16 = {2/16}")

    # descent check from the steepest-descent directional derivative
    print(f"  directional derivative along -g: {-g0 @ g0}")
    x = x0.copy()
    for _ in range(200):
        x = x - 0.1 * grad(x)
    print(f"  GD(alpha=0.1) from (1,0) after 200 its -> {x}")
    assert close(x, [1, 1], 1e-8)

    # saddle: GD from (t,-t) diverges from saddle? along eigvec (1,1) of eig -4
    # f restricted to y=x: 2x^4 - 4x^2  -> descends away from 0
    v = np.array([1e-3, 1e-3])
    for _ in range(300):
        v = v - 0.1 * grad(v)
    print(f"  start (1e-3,1e-3) escapes saddle to {v}")
    v = np.array([1e-3, -1e-3])
    for _ in range(300):
        v = v - 0.1 * grad(v)
    print(f"  start (1e-3,-1e-3) (stable manifold y=-x) -> {v}")


# ---------------------------------------------------------------------------
def m2():
    header("M2  descent lemma sanity check")
    # f(x) = log(1+exp(a.x)) + 0.5*|x|^2 has L = |a|^2/4 + 1
    rng = np.random.default_rng(0)
    a = np.array([2.0, -1.0])
    L = a @ a / 4 + 1
    f = lambda x: np.log1p(np.exp(a @ x)) + 0.5 * x @ x
    g = lambda x: a / (1 + np.exp(-(a @ x))) + x
    worst = -np.inf
    for _ in range(20000):
        x, y = rng.normal(size=2) * 3, rng.normal(size=2) * 3
        gap = f(y) - (f(x) + g(x) @ (y - x) + L / 2 * (y - x) @ (y - x))
        worst = max(worst, gap)
    print(f"  L = {L}, max over samples of f(y)-upper_bound = {worst:.3e} (must be <= 0)")
    assert worst <= 1e-12
    # decrease factor alpha(1-alpha L/2) maximised at 1/L with value 1/(2L)
    al = np.linspace(0, 2 / L, 10001)
    phi = al * (1 - al * L / 2)
    print(f"  argmax alpha = {al[phi.argmax()]:.6f} vs 1/L = {1/L:.6f}; max = {phi.max():.6f} vs 1/(2L) = {1/(2*L):.6f}")


# ---------------------------------------------------------------------------
def m3():
    header("M3  f = 1/2 x^T A x - b^T x,  A=[[3,1],[1,3]], b=(4,4)")
    A = np.array([[3.0, 1.0], [1.0, 3.0]])
    b = np.array([4.0, 4.0])
    xs = np.linalg.solve(A, b)
    lam, V = np.linalg.eigh(A)
    print("  x* =", xs, " eig =", lam, "\n  eigvecs (cols) =\n", V)
    assert close(xs, [1, 1]) and close(lam, [2, 4])
    print(f"  stability: 0 < alpha < 2/4 = {2/4}")
    a_opt = 2 / (lam[0] + lam[1])
    rho = (lam[1] - lam[0]) / (lam[1] + lam[0])
    print(f"  alpha* = {a_opt},  rho = {rho}")
    k = math.log(1e6) / math.log(1 / rho)
    print(f"  iterations for 1e-6: k >= {k:.4f} -> {math.ceil(k)}")
    assert math.ceil(k) == 13

    x = np.array([3.0, 0.0])
    for i in range(3):
        g = A @ x - b
        print(f"  k={i}: x={x}, grad={g}, e={x - xs}, |e|={np.linalg.norm(x - xs):.6f}")
        x = x - a_opt * g
    # exact fractions
    x1 = np.array([4 / 3, 1 / 3])
    x2 = np.array([11 / 9, 8 / 9])
    x = np.array([3.0, 0.0])
    x = x - a_opt * (A @ x - b)
    assert close(x, x1)
    x = x - a_opt * (A @ x - b)
    assert close(x, x2)
    print("  x1 = (4/3, 1/3), x2 = (11/9, 8/9) confirmed;  e2 = e0/9")

    for al in [0.6]:
        print(f"  alpha={al}: factors 1-alpha*lam = {1 - al * lam}")
    x = np.array([3.0, 0.0])
    for i in range(6):
        x = x - 0.6 * (A @ x - b)
    print(f"  alpha=0.6 after 6 its: e = {x - xs}")
    # decomposition of e0=(2,-1) into eigvecs (1,1),(1,-1)
    c1, c2 = (2 - 1) / 2, (2 + 1) / 2
    print(f"  e0 = {c1}*(1,1) + {c2}*(1,-1)")


# ---------------------------------------------------------------------------
def m4():
    header("M4  exact line search on 1/2(x^2 + g y^2), x0 = (g,1)")
    for gam in [10.0, 100.0]:
        A = np.diag([1.0, gam])
        x = np.array([gam, 1.0])
        r = (gam - 1) / (gam + 1)
        f0 = 0.5 * x @ A @ x
        prev_g = None
        for k in range(6):
            g = A @ x
            if prev_g is not None:
                assert abs(g @ prev_g) < 1e-9 * (1 + abs(g @ g))
            t = (g @ g) / (g @ A @ g)
            if k == 0:
                print(f"  gamma={gam}: t0 = {t} (2/(1+gamma) = {2/(1+gam)})")
            assert close(x, [gam * r ** k, (-r) ** k], 1e-9)
            prev_g = g
            x = x - t * g
        kk = math.log(1e6) / (-2 * math.log(r))
        print(f"  gamma={gam}: r={r:.6f}, f0={f0}, iterations to 1e-6 in f: {kk:.3f} -> {math.ceil(kk)}")
    print("  f0 for gamma=10:", 0.5 * (100 + 10), " x1 =", (10 * 9 / 11, -9 / 11))


# ---------------------------------------------------------------------------
def m5():
    header("M5  heavy ball vs GD, mu=1, L=100")
    mu, L = 1.0, 100.0
    k = L / mu
    alpha = 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2
    beta = ((math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))) ** 2
    print(f"  alpha = {alpha:.6f} (4/121={4/121:.6f}), beta = {beta:.6f} (81/121={81/121:.6f})")
    print(f"  (1-sqrt b)^2 = {(1-math.sqrt(beta))**2:.6f} = 4/121 ; (1+sqrt b)^2 = {(1+math.sqrt(beta))**2:.6f} = 400/121")
    print(f"  alpha*mu = {alpha*mu:.6f}, alpha*L = {alpha*L:.6f}")
    lams = np.linspace(mu, L, 20001)
    rad = []
    for lam in lams:
        r = np.roots([1, -(1 + beta - alpha * lam), beta])
        rad.append(max(abs(r)))
    rad = np.array(rad)
    print(f"  max spectral radius over [mu,L] = {rad.max():.6f}; sqrt(beta) = {math.sqrt(beta):.6f}")
    assert rad.max() <= math.sqrt(beta) + 1e-6
    disc_mu = (1 + beta - alpha * mu) ** 2 - 4 * beta
    disc_L = (1 + beta - alpha * L) ** 2 - 4 * beta
    print(f"  discriminant at mu: {disc_mu:.2e}, at L: {disc_L:.2e}  (double roots)")
    print(f"  double root at mu: {(1+beta-alpha*mu)/2:.6f}, at L: {(1+beta-alpha*L)/2:.6f}")
    rgd = (k - 1) / (k + 1)
    rhb = (math.sqrt(k) - 1) / (math.sqrt(k) + 1)
    kgd = math.log(1e6) / math.log(1 / rgd)
    khb = math.log(1e6) / math.log(1 / rhb)
    print(f"  GD rate {rgd:.6f} -> {kgd:.3f} its;  HB rate {rhb:.6f} -> {khb:.3f} its;  ratio {kgd/khb:.2f}")
    # GD with alpha=2/(mu+L)
    print(f"  GD alpha = 2/(mu+L) = {2/(mu+L):.6f}")
    # simulate on lam = 1 (slowest) to show k r^k behaviour
    x_prev, x = 1.0, 1.0
    for i in range(69):
        x, x_prev = x - alpha * mu * x + beta * (x - x_prev), x
    print(f"  HB simulated on lam=mu, |x_69| = {abs(x):.3e} (k r^k effect makes it > 1e-6)")
    x_prev, x = 1.0, 1.0
    n = 0
    while abs(x) > 1e-6:
        x, x_prev = x - alpha * mu * x + beta * (x - x_prev), x
        n += 1
    print(f"  HB on lam=mu actually needs {n} its for |x|<1e-6")
    x = 1.0
    n = 0
    while abs(x) > 1e-6:
        x = (1 - 2 / (mu + L) * mu) * x
        n += 1
    print(f"  GD on lam=mu needs {n} its")


# ---------------------------------------------------------------------------
def a1():
    header("A1  linear regression y = w x + b, data (1,2),(2,3),(3,5)")
    X = np.array([1.0, 2.0, 3.0])
    Y = np.array([2.0, 3.0, 5.0])
    n = len(X)
    H = np.array([[np.mean(X ** 2), np.mean(X)], [np.mean(X), 1.0]])
    ev = np.linalg.eigvalsh(H)
    print("  H =\n", H, "\n  eig =", ev, " kappa =", ev[1] / ev[0], " alpha_max =", 2 / ev[1])
    print(f"  exact eig: (17 +- sqrt(265))/6 = {(17-math.sqrt(265))/6:.6f}, {(17+math.sqrt(265))/6:.6f}")
    grad = lambda w, b: np.array([np.mean((w * X + b - Y) * X), np.mean(w * X + b - Y)])
    loss = lambda w, b: np.mean((w * X + b - Y) ** 2) / 2
    g0 = grad(0, 0)
    p1 = -0.1 * g0
    print(f"  L(0,0) = {loss(0,0):.6f}, grad(0,0) = {g0}, step alpha=0.1 -> {p1}, L = {loss(*p1):.6f}")
    g1 = grad(*p1)
    p2 = p1 - 0.1 * g1
    print(f"  grad at p1 = {g1}, p2 = {p2}, L = {loss(*p2):.6f}")
    w, b = np.polyfit(X, Y, 1)
    print(f"  LS optimum: w = {w:.6f}, b = {b:.6f}, L* = {loss(w,b):.6f}")
    Xc = X - X.mean()
    Hc = np.array([[np.mean(Xc ** 2), np.mean(Xc)], [np.mean(Xc), 1.0]])
    print("  centered H =\n", Hc, " kappa =", 1 / Hc[0, 0])
    # iterations to reach 1e-6 in parameter error with best fixed step
    for name, HH in [("raw", H), ("centered", Hc)]:
        e = np.linalg.eigvalsh(HH)
        kap = e[1] / e[0]
        r = (kap - 1) / (kap + 1)
        print(f"  {name}: best rate {r:.4f}, its for 1e-6: {math.ceil(math.log(1e6)/math.log(1/r))}")


# ---------------------------------------------------------------------------
def a2():
    header("A2  spring chain  K = [[k1+k2,-k2],[-k2,k2]], f = (0, F)")
    for k1 in [3.0, 1000.0]:
        k2 = 1.0
        K = np.array([[k1 + k2, -k2], [-k2, k2]])
        f = np.array([0.0, 1.0])
        u = np.linalg.solve(K, f)
        ev = np.linalg.eigvalsh(K)
        kap = ev[1] / ev[0]
        a_opt = 2 / (ev[0] + ev[1])
        rho = (kap - 1) / (kap + 1)
        its = math.ceil(math.log(1e6) / math.log(1 / rho))
        print(f"  k1={k1}: u*={u}, eig={ev}, kappa={kap:.4f}, alpha_max={2/ev[1]:.6f}, "
              f"alpha*={a_opt:.6f}, rho={rho:.6f}, its(1e-6)={its}")
    K = np.array([[4.0, -1.0], [-1.0, 1.0]])
    f = np.array([0.0, 1.0])
    print(f"  exact eig k1=3: (5 -+ sqrt13)/2 = {(5-math.sqrt(13))/2:.6f}, {(5+math.sqrt(13))/2:.6f}; rho = sqrt13/5 = {math.sqrt(13)/5:.6f}")
    u = np.zeros(2)
    for i in range(3):
        g = K @ u - f
        E = 0.5 * u @ K @ u - f @ u
        print(f"  k={i}: u={u}, grad={g}, energy={E:.6f}")
        u = u - 0.4 * g
    print(f"  energy at u*: {0.5*np.array([1/3,4/3])@K@np.array([1/3,4/3]) - f@np.array([1/3,4/3]):.6f}")


# ---------------------------------------------------------------------------
def a3():
    header("A3  profit P = 100x + 80y - 2x^2 - 2y^2 - 2xy")
    P = lambda v: 100 * v[0] + 80 * v[1] - 2 * v[0] ** 2 - 2 * v[1] ** 2 - 2 * v[0] * v[1]
    gP = lambda v: np.array([100 - 4 * v[0] - 2 * v[1], 80 - 2 * v[0] - 4 * v[1]])
    Hn = np.array([[4.0, 2.0], [2.0, 4.0]])
    xs = np.linalg.solve(Hn, [100, 80])
    print(f"  optimum {xs}, P* = {P(xs)}, eig(-H) = {np.linalg.eigvalsh(Hn)}")
    assert close(xs, [20, 10]) and close(P(xs), 1400)
    x = np.zeros(2)
    for i in range(4):
        g = gP(x)
        print(f"  k={i}: x={x}, grad={g}, P={P(x):.4f}, |e|={np.linalg.norm(x-xs):.6f}")
        x = x + 0.25 * g
    print(f"  ratio |e3|/|e0| = {np.linalg.norm(np.array([21.25,12.5])-xs)/np.linalg.norm(xs):.6f} (0.5^3 = 0.125)")
    x = np.zeros(2)
    for i in range(6):
        x = x + 0.4 * gP(x)
    print(f"  alpha=0.4 (> 1/3) after 6 its: {x}  (factor 1-0.4*6 = {1-0.4*6})")


# ---------------------------------------------------------------------------
def a4():
    header("A4  Newton cooling  J(k) = 1/2 sum (e^{-k t_i} - y_i)^2")
    t = np.array([1.0, 2.0])
    y = np.array([0.6, 0.36])
    J = lambda k: 0.5 * np.sum((np.exp(-k * t) - y) ** 2)
    dJ = lambda k: np.sum((np.exp(-k * t) - y) * (-t * np.exp(-k * t)))
    d2J = lambda k: np.sum(t ** 2 * np.exp(-2 * k * t) + (np.exp(-k * t) - y) * t ** 2 * np.exp(-k * t))
    ks = math.log(5 / 3)
    print(f"  k* = ln(5/3) = {ks:.6f}; J(k*) = {J(ks):.2e}; J''(k*) = {d2J(ks):.6f}; local alpha_max = {2/d2J(ks):.4f}")
    k = 0.0
    for i in range(4):
        print(f"  k_{i} = {k:.6f}: J = {J(k):.6f}, J' = {dJ(k):.6f}, J'' = {d2J(k):.6f}")
        k = k - 0.5 * dJ(k)
    for i in range(300):
        k = k - 0.5 * dJ(k)
    print(f"  after ~300 its from 0: k = {k:.8f}")
    # plateau
    print(f"  plateau value 0.5*sum y^2 = {0.5*np.sum(y**2):.6f};  J(6) = {J(6):.6f}, J'(6) = {dJ(6):.3e}, J'(10) = {dJ(10):.3e}")
    k = 6.0
    for i in range(1000):
        k = k - 0.5 * dJ(k)
    print(f"  1000 its from k0=6 (alpha=0.5): k = {k:.6f}")
    grid = np.linspace(0, 10, 100001)
    d2 = np.array([d2J(g) for g in grid])
    sign_change = grid[np.where(np.diff(np.sign(d2)))[0]]
    print(f"  J'' changes sign at k ~ {sign_change}  (nonconvex beyond)")
    # details for hand calc
    e1, e2 = math.exp(-0.84), math.exp(-1.68)
    print(f"  e^-0.84 = {e1:.5f}, e^-1.68 = {e2:.5f}, residuals {e1-0.6:.5f}, {e2-0.36:.5f}")


# ---------------------------------------------------------------------------
def a5():
    header("A5  streaming sensor mean by SGD")
    xi = np.array([20.4, 19.7, 20.2, 20.6, 19.9])
    # decaying: x_1 = xi_1, x_{k+1} = x_k + (1/(k+1)) (xi_{k+1} - x_k)
    x = xi[0]
    dec = [x]
    for k in range(1, 5):
        x = x + (xi[k] - x) / (k + 1)
        dec.append(x)
    print("  1/(k+1) schedule:", np.round(dec, 6), " running means:", np.round(np.cumsum(xi) / np.arange(1, 6), 6))
    assert close(dec, np.cumsum(xi) / np.arange(1, 6))
    x = xi[0]
    con = [x]
    for k in range(1, 5):
        x = x + 0.5 * (xi[k] - x)
        con.append(x)
    print("  constant alpha=0.5:", np.round(con, 6))
    s2 = 0.25
    for a in [0.5, 0.1]:
        print(f"  stationary var alpha={a}: {a*s2/(2-a):.6f} (std {math.sqrt(a*s2/(2-a)):.4f})")
    print(f"  mean of 5 readings variance sigma^2/5 = {s2/5:.4f} (std {math.sqrt(s2/5):.4f})")
    # Monte-Carlo check of stationary variance
    rng = np.random.default_rng(0)
    a = 0.1
    x = np.full(20000, 20.0)
    for _ in range(400):
        x = (1 - a) * x + a * (20 + 0.5 * rng.standard_normal(x.size))
    print(f"  MC stationary var alpha=0.1: {x.var():.5f} vs formula {a*s2/(2-a):.5f}")
    # weights of constant-step estimator: alpha (1-alpha)^j
    print("  constant-step weights on last 5 readings (alpha=0.5):", [0.5 * 0.5 ** j for j in range(4)], "+ 0.0625 on x1")


if __name__ == "__main__":
    m1(); m2(); m3(); m4(); m5(); a1(); a2(); a3(); a4(); a5()
    print("\nAll assertions passed.")
