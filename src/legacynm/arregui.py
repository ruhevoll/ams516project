import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def solve_fbp(alpha=0.1, sigma=0.25, gamma=0.01, a=1.01, b=0.99, r=0.1, T=1.0,
              S_min=1e-3, S_max=1000, Ns=50, x_min=-10, x_max=10, Nx=50, Nt=50):
    # Log grid for S
    u_min = np.log(S_min)
    u_max = np.log(S_max)
    u = np.linspace(u_min, u_max, Ns)
    du = u[1] - u[0]
    S = np.exp(u)
    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]
    dt = T / Nt
    mu = alpha - 0.5 * sigma**2

    # Terminal condition at t = T (n = Nt)
    H = np.zeros((Nt + 1, Ns, Nx))
    for i in range(Ns):
        for j in range(Nx):
            Z = x[j] * S[i] * (a if x[j] < 0 else b)
            H[Nt, i, j] = np.exp(-gamma * Z)

    # Build the diffusion-advection matrix for S direction
    lower = (dt * (0.5 * sigma**2) / du**2 - dt * alpha / (2 * du)) * np.ones(Ns - 1)
    upper = (dt * (0.5 * sigma**2) / du**2 + dt * alpha / (2 * du)) * np.ones(Ns - 1)
    diag = (1 - dt * sigma**2 / du**2) * np.ones(Ns)
    A = diags([lower, diag, upper], [-1, 0, 1], shape=(Ns, Ns)).tocsc()

    # Backward loop in time
    residuals_buy = []
    residuals_sell = []
    s_points_buy = []
    s_points_sell = []
    x_points_buy = []
    x_points_sell = []

    for n in range(Nt - 1, -1, -1):
        t = T - n * dt  # Backward time
        H_cont = np.zeros((Ns, Nx))

        # Solve for each x slice using characteristics
        for j in range(Nx):
            u_shift = u - mu * dt
            rhs = np.interp(u_shift, u, H[n + 1, :, j], left=1.0, right=0.0)
            H_cont[:, j] = spsolve(A, rhs)

        # Apply boundary conditions and obstacles
        for i in range(Ns):
            k_a = a * gamma * S[i] * np.exp(r * t)
            k_b = b * gamma * S[i] * np.exp(r * t)
            H_current = H_cont[i, :].copy()

            for _ in range(20):  # Relaxation iterations
                # Buy region obstacle (x <= X_b)
                buy_obstacle = H_current[0] * np.exp(-k_a * (x - x[0]))
                # Sell region obstacle (x >= X_s)
                sell_obstacle = H_current[-1] * np.exp(-k_b * (x - x[-1]))

                # Project onto obstacles
                H_new = np.maximum(np.minimum(H_current, buy_obstacle), sell_obstacle)
                if np.max(np.abs(H_new - H_current)) < 1e-8:
                    break
                H_current = H_new.copy()

            H[n, i, :] = H_current

            # Error analysis at t=0 (n=0)
            if n == 0:
                # Approximate gradients for residual analysis
                grad = np.gradient(H[n, i, :], dx)
                buy_mask = H[n, i, :] <= buy_obstacle
                sell_mask = H[n, i, :] <= sell_obstacle

                # Detect boundary transitions
                if np.any(buy_mask[:-1] != buy_mask[1:]):
                    j_trans = np.where(buy_mask[:-1] != buy_mask[1:])[0][0] + 1
                    left_grad = grad[j_trans - 1] if j_trans > 0 else grad[j_trans]
                    right_grad = grad[j_trans]
                    residual = abs(left_grad + k_a * H[n, i, j_trans])
                    residuals_buy.append(residual)
                    s_points_buy.append(S[i])
                    x_points_buy.append(x[j_trans])

                if np.any(sell_mask[:-1] != sell_mask[1:]):
                    j_trans = np.where(sell_mask[:-1] != sell_mask[1:])[0][0] + 1
                    left_grad = grad[j_trans - 1] if j_trans > 0 else grad[j_trans]
                    right_grad = grad[j_trans]
                    residual = abs(right_grad + k_b * H[n, i, j_trans])
                    residuals_sell.append(residual)
                    s_points_sell.append(S[i])
                    x_points_sell.append(x[j_trans])

    # Plot residuals for error analysis
    if residuals_buy or residuals_sell:
        fig, ax = plt.subplots()
        if residuals_buy:
            ax.scatter(s_points_buy, residuals_buy, label='Buy boundary residual', color='blue')
        if residuals_sell:
            ax.scatter(s_points_sell, residuals_sell, label='Sell boundary residual', color='red')
        ax.set_xlabel('Stock Price (S)')
        ax.set_ylabel('Residual Magnitude (|∇H + k H|)')
        ax.set_title('Residuals at Free Boundaries for Error Analysis at t=0')
        ax.legend()
        ax.grid(True)
        plt.show()

    return H[0]  # Return H at t=0

# Test simulation
if __name__ == "__main__":
    params = {
        'alpha': 0.1, 'sigma': 0.25, 'gamma': 0.01, 'a': 1.01, 'b': 0.99,
        'r': 0.1, 'T': 1.0, 'S_min': 1e-3, 'S_max': 1000, 'Ns': 50,
        'x_min': -10, 'x_max': 10, 'Nx': 50, 'Nt': 50
    }
    H_t0 = solve_fbp(**params)
    print("Simulation completed. H at t=0 shape:", H_t0.shape)
    print("H at t=0 min:", H_t0.min(), "max:", H_t0.max())