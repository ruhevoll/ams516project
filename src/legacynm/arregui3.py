import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Problem Parameters (MODIFIED) ---

# Market parameters
alpha = 0.08    # Stock drift
sigma = 0.20    # Stock volatility
r = 0.04        # Risk-free rate
T = 1.0         # Time horizon (1 year)

# Transaction costs
zeta = 0.01     # Buy cost (1%)
mu = 0.01       # Sell cost (1%)
a = 1.0 + zeta
b = 1.0 - mu

# Utility parameters
# *** THIS IS THE FIX ***
# gamma=0.1 was too large, causing numerical overflow (exp(1010) = inf)
# We reduce it to a numerically stable value.
gamma = 0.001   # Risk aversion parameter

# --- 2. Grid Parameters (MODIFIED) ---
# Back to a reasonable grid, which is now stable with the new gamma
N_T = 100       # Number of time steps
N_S = 100       # Number of stock price steps
N_x = 100       # Number of share steps

S_max = 200.0   # Max stock price
S_min = 0.0
x_max = 50.0    # Max shares
x_min = -50.0   # Min shares

dt = T / N_T
ds = (S_max - S_min) / N_S
dx = (x_max - x_min) / N_x

# Stability check with new gamma
# Ca_max ~ 1.01 * 0.001 * 200 * exp(0.04) ~ 0.21
# dx = (50 - (-50)) / 100 = 1.0
print(f"Grid Stability Check: dx * Ca_max = {dx * 0.21:.4f} (must be < 1)")

t_grid = np.linspace(0, T, N_T + 1)
S_grid = np.linspace(S_min, S_max, N_S + 1)
x_grid = np.linspace(x_min, x_max, N_x + 1)

# --- 3. PSOR Parameters ---
omega = 1.5     # SOR relaxation parameter
sor_tol = 1e-6  # Convergence tolerance
max_sor_iter = 2000

# --- 4. Helper Functions ---
def Z_terminal(S, x, a, b):
    """ Terminal settlement value (Eq. 34) """
    buy_part = a * (x < 0)
    sell_part = b * (x >= 0)
    return x * S * (buy_part + sell_part)

def H_terminal(S_grid, x_grid, a, b, gamma):
    """ Terminal condition for H (Eq. 37) """
    H_term = np.zeros((N_S + 1, N_x + 1))
    for i in range(N_S + 1):
        for j in range(N_x + 1):
            Z = Z_terminal(S_grid[i], x_grid[j], a, b)
            H_term[i, j] = np.exp(-gamma * Z)
    return H_term

# --- 5. Initialization ---
H = np.zeros((N_T + 1, N_S + 1, N_x + 1))
H[N_T, :, :] = H_terminal(S_grid, x_grid, a, b, gamma)

# Storage for boundaries
X_b = np.zeros((N_T + 1, N_S + 1))
X_s = np.zeros((N_T + 1, N_S + 1))

# --- 6. Main FDM Loop (Backward in Time) ---
print("Starting FDM solver...")

for k in range(N_T - 1, -1, -1):
    t = t_grid[k]
    
    H[k, :, :] = H[k+1, :, :]
    
    A = 0.5 * dt * (sigma**2 * S_grid**2 / ds**2 - alpha * S_grid / ds)
    B = 1.0 + dt * (sigma**2 * S_grid**2 / ds**2)
    C = 0.5 * dt * (sigma**2 * S_grid**2 / ds**2 + alpha * S_grid / ds)
    
    exp_rt = np.exp(r * (T - t))
    Ca = a * gamma * S_grid * exp_rt
    Cb = b * gamma * S_grid * exp_rt

    for sor_iter in range(max_sor_iter):
        max_diff = 0.0
        
        # --- Handle boundary conditions for x ---
        for i in range(1, N_S):
            H_old = H[k, i, 0]
            denom = 1.0 - dx * Ca[i]
            if denom > 1e-8: # Stability condition
                H_star = H[k, i, 1] / denom
                H[k, i, 0] = (1 - omega) * H_old + omega * H_star

            H_old = H[k, i, N_x]
            denom = 1.0 + dx * Cb[i]
            H_star = H[k, i, N_x - 1] / denom
            H[k, i, N_x] = (1 - omega) * H_old + omega * H_star
        
        # --- Iterate over interior points (S, x) ---
        for i in range(1, N_S):
            for j in range(1, N_x):
                
                H_old = H[k, i, j]

                val_S_minus = H[k, i-1, j]
                val_S_plus = H[k, i+1, j]
                val_t_plus = H[k+1, i, j]
                
                H_pde = (val_t_plus + A[i] * val_S_minus + C[i] * val_S_plus) / B[i]
                
                if np.isnan(H_pde): # Safety check
                    continue

                val_x_minus = H[k, i, j-1]
                val_x_plus = H[k, i, j+1]
                dH_dx_term = (val_x_plus - val_x_minus) / (2.0 * dx)
                
                if Ca[i] < 1e-10 or Cb[i] < 1e-10:
                    H_buy_limit = -np.inf
                    H_sell_limit = np.inf
                else:
                    H_buy_limit = -dH_dx_term / Ca[i]
                    H_sell_limit = -dH_dx_term / Cb[i]
                
                H_star = np.minimum(H_sell_limit, np.maximum(H_pde, H_buy_limit))
                
                if np.isnan(H_star): # Safety check
                    continue
                    
                H[k, i, j] = (1.0 - omega) * H_old + omega * H_star

                if not np.isnan(H[k, i, j]) and not np.isnan(H_old):
                    max_diff = max(max_diff, abs(H[k, i, j] - H_old))
        
        # --- Handle boundary conditions for S ---
        H[k, 0, :] = 1.0
        H[k, N_S, :] = H[k, N_S - 1, :]
        
        if max_diff < sor_tol:
            break
            
    if (k * 10) % N_T == 0 or k == 0:
         print(f"Time step k = {k:3d}/{N_T} (t={t:.2f}), PSOR iters = {sor_iter+1:4d}, MaxDiff = {max_diff:.2e}")

    # --- Find Boundaries for this time step k ---
    tol_bound = 1e-5
    for i in range(1, N_S + 1): # Start from i=1 (S>0)
        if Ca[i] > 1e-10 and Cb[i] > 1e-10:
            L_buy = np.zeros(N_x + 1)
            L_sell = np.zeros(N_x + 1)
            
            H_j_plus = H[k, i, 2:]
            H_j_minus = H[k, i, :-2]
            dH_dx = (H_j_plus - H_j_minus) / (2.0 * dx)
            
            L_buy[1:-1] = dH_dx + Ca[i] * H[k, i, 1:-1]
            L_sell[1:-1] = dH_dx + Cb[i] * H[k, i, 1:-1]
        
            buy_indices = np.where(L_buy > tol_bound)[0]
            X_b[k, i] = x_grid[buy_indices[0]] if len(buy_indices) > 0 else x_max

            sell_indices = np.where(L_sell < -tol_bound)[0]
            X_s[k, i] = x_grid[sell_indices[-1]] if len(sell_indices) > 0 else x_min
        else:
            # At S=0, no transaction
            X_b[k, i] = x_min
            X_s[k, i] = x_max
            
print("Solver finished.")

# --- 7. Plotting ---
def plot_3d_surface(X, Y, Z, title, xlabel, ylabel, zlabel):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1,
                           antialiased=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=20., azim=-120)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Plot 1: Approximated Solution H at t=0
S_mesh, x_mesh = np.meshgrid(S_grid, x_grid)
plot_3d_surface(S_mesh, x_mesh, H[0, :, :].T,
                'Solution H(t, S, x) at t = 0 (gamma = 0.001)', 
                'Stock Price (S)', 'Shares (x)', 'H(0, S, x)')

# Plot 2: Buy Boundary X_b(t, S)
S_mesh, t_mesh = np.meshgrid(S_grid, t_grid)
plot_3d_surface(S_mesh, t_mesh, X_b.T,
                'Buy Boundary X_b(t, S)', 'Stock Price (S)',
                'Time (t)', 'Buy Boundary (x)')

# Plot 3: Sell Boundary X_s(t, S)
plot_3d_surface(S_mesh, t_mesh, X_s.T,
                'Sell Boundary X_s(t, S)', 'Stock Price (S)',
                'Time (t)', 'Sell Boundary (x)')