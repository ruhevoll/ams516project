# Filename: optimal_investment_with_transaction_costs.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

# Common function to compute a0, a1, a2, a3
def get_constants(sigma, r, alpha, gamma):
    a0 = (sigma**2) / 2
    a1 = -(alpha - r - (2 - gamma) * sigma**2)
    a2 = -(alpha - r - (1 - gamma) * sigma**2)
    a3 = -(gamma * sigma**2) / 2
    return a0, a1, a2, a3

# Function to compute v_circ using interpolation
def get_v_circ(v, x, dt, a0, a1):
    beta = -a1 * dt
    chi = x * np.exp(-beta)
    f = interp1d(x, v, kind='linear', fill_value="extrapolate")
    v_circ = f(chi)
    return v_circ

# Function to build the linear matrix L = eye + dt*a2*eye + dt*D
def build_L(x, dt, a0, a2, h, n):
    D_diag = np.zeros(n)
    D_lower = np.zeros(n-1)
    D_upper = np.zeros(n-1)
    for i in range(1, n-1):
        D_lower[i-1] = a0 * x[i]**2 / h**2
        D_diag[i] = -2 * a0 * x[i]**2 / h**2
        D_upper[i] = a0 * x[i]**2 / h**2
    # For i=0
    D_diag[0] = -2 * a0 * x[0]**2 / h**2
    D_upper[0] = a0 * x[0]**2 / h**2
    # For i=n-1
    D_diag[n-1] = -2 * a0 * x[n-1]**2 / h**2
    D_lower[n-2] = a0 * x[n-1]**2 / h**2
    D = np.diag(D_diag) + np.diag(D_lower, -1) + np.diag(D_upper, 1)
    L = np.eye(n) + dt * a2 * np.eye(n) + dt * D
    return L

# Vectorized computation of B
def compute_B(v, x, h, n, a3, dt):
    x_mid = (x[1:] + x[:-1]) / 2
    B = np.zeros(n)
    for i in range(1, n-1):
        xim = x_mid[i-1]
        xip = x_mid[i]
        B[i] = (1 / h**2) * (xim**2 * (v[i]**2 - v[i-1]**2) - xip**2 * (v[i+1]**2 - v[i]**2))  # Added missing *
    return dt * a3 * B

# Vectorized computation of JB
def compute_JB(v, x, h, n, a3, dt):
    x_mid = (x[1:] + x[:-1]) / 2
    jb_diag = np.zeros(n)
    jb_lower = np.zeros(n-1)
    jb_upper = np.zeros(n-1)
    for i in range(1, n-1):
        xim = x_mid[i-1]
        xip = x_mid[i]
        jb_lower[i-1] = xim**2 * 2 * v[i] / h**2
        jb_diag[i] = -2 * v[i] * (xim**2 + xip**2) / h**2
        jb_upper[i] = xip**2 * 2 * v[i] / h**2
    JB = np.diag(jb_diag) + np.diag(jb_lower, -1) + np.diag(jb_upper, 1)
    return dt * a3 * JB

# Vectorized LCP solver
def solve_lcp(JF, rhs, l, u, omega, eps_R, h, x, lamb, max_k=20, v_init=None):
    n = len(rhs)
    V = v_init.copy() if v_init is not None else np.zeros(n)
    diag_JF = np.diag(JF)
    off_diag_lower = np.diag(JF, -1)
    off_diag_upper = np.diag(JF, 1)
    g = -1 / (x[-1] + 1 + lamb)**2
    for _ in range(max_k):
        V_old = V.copy()
        left = np.roll(V, 1)[1:] * off_diag_lower
        right = np.roll(V, -1)[:-1] * off_diag_upper
        tilde = (rhs[1:-1] - left[1:] - right[:-1]) / diag_JF[1:-1]
        V[1:-1] = V[1:-1] + omega * (tilde - V[1:-1])
        V[1:-1] = np.clip(V[1:-1], l[1:-1], u[1:-1])
        V[0] = u[0]  # Dirichlet at x = x_star
        V[-1] = V[-2] + h * g  # Neumann at x = N_x
        if np.max(np.abs(V - V_old)) < eps_R:
            break
    return V

# Main function to solve the problem
def solve_problem(sigma, r, alpha, gamma, lamb, mu, T, x_star, N_x, I, dt, omega, eps_R, eps_N, test_num):
    print(f"Starting simulation for Test {test_num}...")
    sigma, r, alpha, gamma, lamb, mu, T, x_star, N_x, dt, omega, eps_R, eps_N = map(float, [sigma, r, alpha, gamma, lamb, mu, T, x_star, N_x, dt, omega, eps_R, eps_N])
    I = int(I)

    a0, a1, a2, a3 = get_constants(sigma, r, alpha, gamma)
    h = (N_x - x_star) / (I - 1)
    x = np.linspace(x_star, N_x, I)
    n = I
    l = 1 / (x + 1 + lamb)
    u = 1 / (x + 1 - mu)
    v = u.copy()  # Initial at t=T (upper obstacle)
    M = int(T / dt)
    print(f"Initialized with M={M} time steps, h={h}, x range=[{x_star}, {N_x}]")

    L = build_L(x, dt, a0, a2, h, n)

    xs_list, xb_list, t_list, v0_list, v0_anal, f_list = [], [], [], [], [], []

    for m in range(M):
        print(f"Processing time step {m+1}/{M} (t={T - (m + 1) * dt})...")
        v_circ = get_v_circ(v, x, dt, a0, a1)

        v_old = v.copy()
        for _ in range(20):  # Limited iterations
            B = compute_B(v_old, x, h, n, a3, dt)
            F = L @ v_old + B - v_circ
            JB = compute_JB(v_old, x, h, n, a3, dt)
            JF = L + JB
            rhs = JF @ v_old - F
            V = solve_lcp(JF, rhs, l, u, omega, eps_R, h, x, lamb, v_init=v_old)
            rel_err = np.max(np.abs(V - v_old)) / np.max(np.abs(v_old + 1e-10))
            if rel_err < eps_N:
                break
            v_old = V

        v = v_old

        eps = 1e-6
        mask_sr = np.abs(v - u) < eps
        xs = x[np.where(mask_sr)[0][-1]] if np.any(mask_sr) else x_star
        xs_list.append(xs)

        mask_br = np.abs(v - l) < eps
        xb = x[np.where(mask_br)[0][0]] if np.any(mask_br) else N_x
        xb_list.append(xb)

        t = T - (m + 1) * dt
        t_list.append(t)
        v0_list.append(interp1d(x, v, kind='linear', fill_value="extrapolate")(0))

        a = alpha - r - (1 - gamma) * sigma**2
        if a > 0:
            t1 = T - (1 / a) * np.log((1 + lamb) / (1 - mu))
            v0_anal.append(1 / (1 + lamb) if t <= t1 else 1 / (1 - mu) * np.exp(-a * (T - t)))
        else:
            v0_anal.append(1 / (1 - mu))

        xs = xs_list[-1]
        numer = r * xs**2 + (alpha + r) * (1 - mu) * xs + (alpha - 0.5 * sigma**2 * (1 - gamma)) * (1 - mu)**2
        f_list.append(numer / (xs + 1 - mu)**2)

    print("Completed time stepping, computing A integral...")
    A = np.trapezoid(f_list, np.arange(1, M + 1) * dt)

    t_list, xs_list, xb_list, v0_list, v0_anal = map(np.flip, [t_list, xs_list, xb_list, v0_list, v0_anal])

    print("Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x, v, label='Numerical v')
    ax1.plot(x, l, label='Lower obstacle')
    ax1.plot(x, u, label='Upper obstacle')
    ax1.set_xlabel('x')
    ax1.set_ylabel('v')
    ax1.set_title('Computational Domain')
    ax1.legend()

    mask_nt = (v > l + 1e-6) & (v < u - 1e-6)
    ax2.plot(x[mask_nt], v[mask_nt], label='Numerical v')
    ax2.plot(x[mask_nt], l[mask_nt], label='Lower')
    ax2.plot(x[mask_nt], u[mask_nt], label='Upper')
    ax2.set_xlabel('x')
    ax2.set_ylabel('v')
    ax2.set_title('No Transaction Region')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'test{test_num}_fig1.png')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t_list, v0_anal, label='Analytical')
    ax1.plot(t_list, v0_list, label='Numerical')
    ax1.set_xlabel('t')
    ax1.set_ylabel('v(0, t)')
    ax1.legend()

    rel_err = np.abs(np.array(v0_list) - np.array(v0_anal)) / (np.abs(np.array(v0_anal)) + 1e-10)
    ax2.plot(t_list, rel_err, label='Relative error')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Relative error')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'test{test_num}_fig2.png')
    plt.close()

    plt.figure()
    plt.plot(t_list, xs_list, label='xs')
    plt.plot(t_list, xb_list, label='xb')
    plt.xlabel('t')
    plt.ylabel('Free boundaries')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'test{test_num}_fig3.png')
    plt.close()

    a = alpha - r - (1 - gamma) * sigma**2
    x_M = -a / (alpha - r) if alpha != r else 0
    upper_bound_xs = (1 - mu) * x_M
    lower_bound_xb = (1 + lamb) * x_M

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t_list, xs_list, label='xs')
    ax1.plot(t_list, [upper_bound_xs]*len(t_list), label='Upper bound')
    ax1.set_xlabel('t')
    ax1.set_ylabel('xs')
    ax1.legend()

    ax2.plot(t_list, xb_list, label='xb')
    ax2.plot(t_list, [lower_bound_xb]*len(t_list), label='Lower bound')
    if a > 0:
        t1 = T - (1 / a) * np.log((1 + lamb) / (1 - mu))
        ax2.axvline(t1, color='k', linestyle='--', label='t1')
    ax2.set_xlabel('t')
    ax2.set_ylabel('xb')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'test{test_num}_fig4.png')
    plt.close()

    xs, xb = xs_list[0], xb_list[0]
    x_M = -a / (alpha - r) if alpha != r else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    Y = np.linspace(0, 100, 100)
    X_s, X_b, X_m = xs * Y, xb * Y, x_M * Y
    ax1.plot(X_s, Y, label='Selling boundary')
    ax1.plot(X_b, Y, label='Buying boundary')
    ax1.plot(X_m, Y, label='Merton line', linestyle='dotted')
    ax1.fill_betweenx(Y, -50, X_s, color='yellow', label='SR')
    ax1.fill_betweenx(Y, X_s, X_b, color='cyan', label='NT')
    ax1.fill_betweenx(Y, X_b, 100, color='magenta', label='BR')
    ax1.set_xlim(-50, 100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Solvency Region')
    ax1.legend()

    integral = cumtrapz(v, x, initial=0)
    xs_idx = np.argmin(np.abs(x - xs))
    integral -= integral[xs_idx]
    w = A + np.log(xs + 1 - mu) + integral
    V = (1 / gamma) * np.exp(gamma * w)
    X_grid, Y_grid = np.meshgrid(np.linspace(-50, 100, 100), np.linspace(1e-6, 100, 100))
    z = X_grid / Y_grid
    f = interp1d(x, V, kind='linear', fill_value="extrapolate")
    V_grid = f(z)
    phi = Y_grid**gamma * V_grid / gamma
    cf = ax2.contourf(X_grid, Y_grid, phi, levels=20)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Value Function at t=0')
    plt.colorbar(cf, ax=ax2)
    plt.tight_layout()
    plt.savefig(f'test{test_num}_fig5.png')
    plt.close()

    print(f"Completed Test {test_num}. All figures saved.")

# Run Test 1
sigma = 0.25
r = 0.03
alpha = 0.10
gamma = 0.5
lamb = 0.08
mu = 0.02
T = 4
x_star = -0.95
N_x = 10
I = 801
dt = 0.01
omega = 1.8
eps_R = 1e-15
eps_N = 1e-4
solve_problem(sigma, r, alpha, gamma, lamb, mu, T, x_star, N_x, I, dt, omega, eps_R, eps_N, 1)

# Run Test 2
sigma = 0.25
r = 0.05
alpha = 0.08
gamma = 0.5
lamb = 0.06
mu = 0.02
T = 4
x_star = -0.95
N_x = 12
I = 201
dt = 0.01
omega = 1.8
eps_R = 1e-15
eps_N = 1e-4
solve_problem(sigma, r, alpha, gamma, lamb, mu, T, x_star, N_x, I, dt, omega, eps_R, eps_N, 2)