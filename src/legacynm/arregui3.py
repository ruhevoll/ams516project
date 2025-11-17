import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d
import time
import warnings
import os
import glob
try:
    import imageio.v2 as iio
except ImportError:
    print("imageio module not found. Run 'pip install imageio' to create GIFs.")


# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 0. Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Define Model Parameters & Domain ---
PARAMS = {
    'T': 1.0, 'r': 0.05, 'alpha': 0.1, 'sigma': 0.2,
    'zeta': 0.01, 'mu': 0.01, 'gamma': 0.1,
}
PARAMS['a'] = 1 + PARAMS['zeta']
PARAMS['b'] = 1 - PARAMS['mu']

DOMAIN = {
    't_min': 0.0, 't_max': PARAMS['T'],
    'S_min': 50.0, 'S_max': 150.0,
    'x_min': -50.0, 'x_max': 50.0,
}
H_SCALE = 1000.0 

# ###########################################################################
# #################### SECTION 1: PINN IMPLEMENTATION #######################
# ###########################################################################

class FCNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, activation=nn.Tanh()):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_features), activation]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_features, hidden_features), activation])
        layers.append(nn.Linear(hidden_features, out_features))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class SolutionNet(nn.Module):
    def __init__(self, in_features=3, out_features=1, hidden_layers=4, hidden_features=32):
        super().__init__()
        self.network = FCNN(in_features, out_features, hidden_layers, hidden_features)
    def forward(self, t, S, x):
        t_scaled = (t - DOMAIN['t_min']) / (DOMAIN['t_max'] - DOMAIN['t_min'])
        S_scaled = (S - DOMAIN['S_min']) / (DOMAIN['S_max'] - DOMAIN['S_min'])
        x_scaled = (x - DOMAIN['x_min']) / (DOMAIN['x_max'] - DOMAIN['x_min'])
        inputs = torch.cat([t_scaled, S_scaled, x_scaled], dim=1)
        return self.network(inputs)

class BoundaryNet(nn.Module):
    def __init__(self, in_features=2, out_features=2, hidden_layers=3, hidden_features=24):
        super().__init__()
        self.network = FCNN(in_features, out_features, hidden_layers, hidden_features)
    def forward(self, t, S):
        t_scaled = (t - DOMAIN['t_min']) / (DOMAIN['t_max'] - DOMAIN['t_min'])
        S_scaled = (S - DOMAIN['S_min']) / (DOMAIN['S_max'] - DOMAIN['S_min'])
        inputs = torch.cat([t_scaled, S_scaled], dim=1)
        outputs = self.network(inputs)
        Xb = DOMAIN['x_min'] + (DOMAIN['x_max'] - DOMAIN['x_min']) * torch.sigmoid(outputs[:, 0:1])
        log_delta_X = outputs[:, 1:2]
        Xs = Xb + torch.exp(log_delta_X) + 1e-4
        Xs = torch.clamp(Xs, max=DOMAIN['x_max'])
        return Xb, Xs

class FreeBoundaryPINN:
    def __init__(self, params, domain, n_points, device):
        self.params, self.domain, self.n_points, self.device = params, domain, n_points, device
        self.H_net = SolutionNet().to(device)
        self.Boundary_net = BoundaryNet().to(device)
        self.all_params = list(self.H_net.parameters()) + list(self.Boundary_net.parameters())
        self.optimizer = optim.Adam(self.all_params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=1000)
        self.loss_history = {'total': [], 'pde': [], 'buy': [], 'sell': [], 'terminal': []}
        self.norm_history = {'H': [], 'Xb': [], 'Xs': []}
        self.setup_convergence_grid()

    def setup_convergence_grid(self):
        n_grid = 30
        t = torch.linspace(self.domain['t_min'], self.domain['t_max'], n_grid, device=self.device)
        S = torch.linspace(self.domain['S_min'], self.domain['S_max'], n_grid, device=self.device)
        x = torch.linspace(self.domain['x_min'], self.domain['x_max'], n_grid, device=self.device)
        T_b, S_b = torch.meshgrid(t, S, indexing='ij')
        self.t_grid_b, self.S_grid_b = T_b.flatten().unsqueeze(1), S_b.flatten().unsqueeze(1)
        T_h, S_h, X_h = torch.meshgrid(t, S, x, indexing='ij')
        self.t_grid_h, self.S_grid_h, self.x_grid_h = T_h.flatten().unsqueeze(1), S_h.flatten().unsqueeze(1), X_h.flatten().unsqueeze(1)
        self.prev_H_grid = torch.zeros_like(self.t_grid_h, device=self.device)
        self.prev_Xb_grid = torch.zeros_like(self.t_grid_b, device=self.device)
        self.prev_Xs_grid = torch.zeros_like(self.t_grid_b, device=self.device)

    @staticmethod
    def compute_Z(S, x, a, b):
        return x * S * (a * (x < 0) + b * (x >= 0))

    @staticmethod
    def compute_terminal_h_tilde(S, x, a, b, gamma):
        Z = FreeBoundaryPINN.compute_Z(S, x, a, b)
        return -gamma * Z / H_SCALE

    def compute_residuals(self, t, S, x):
        t.requires_grad_(True); S.requires_grad_(True); x.requires_grad_(True)
        h_tilde = self.H_net(t, S, x)
        grads = torch.autograd.grad(h_tilde, [t, S, x], grad_outputs=torch.ones_like(h_tilde), create_graph=True)
        h_t, h_S, h_x = grads[0], grads[1], grads[2]
        h_SS = torch.autograd.grad(h_S, S, grad_outputs=torch.ones_like(h_S), create_graph=True)[0]
        res_pde = h_t + self.params['alpha'] * S * h_S + 0.5 * self.params['sigma']**2 * S**2 * (H_SCALE * h_S**2 + h_SS)
        e_rt = torch.exp(self.params['r'] * (self.params['T'] - t))
        res_buy = h_x + (1.0 / H_SCALE) * self.params['a'] * self.params['gamma'] * S * e_rt
        res_sell = h_x + (1.0 / H_SCALE) * self.params['b'] * self.params['gamma'] * S * e_rt
        t.requires_grad_(False); S.requires_grad_(False); x.requires_grad_(False)
        return res_pde, res_buy, res_sell

    def compute_loss(self):
        n_term = self.n_points['terminal']
        S_term = torch.rand(n_term, 1, device=self.device) * (self.domain['S_max'] - self.domain['S_min']) + self.domain['S_min']
        x_term = torch.rand(n_term, 1, device=self.device) * (self.domain['x_max'] - self.domain['x_min']) + self.domain['x_min']
        t_term = torch.full_like(S_term, self.domain['t_max'])
        h_pred_term = self.H_net(t_term, S_term, x_term)
        h_target_term = self.compute_terminal_h_tilde(S_term, x_term, self.params['a'], self.params['b'], self.params['gamma'])
        loss_terminal = F.mse_loss(h_pred_term, h_target_term)
        n_region = self.n_points['pde'] + self.n_points['buy'] + self.n_points['sell']
        t_region = torch.rand(n_region, 1, device=self.device) * self.domain['t_max']
        S_region = torch.rand(n_region, 1, device=self.device) * (self.domain['S_max'] - self.domain['S_min']) + self.domain['S_min']
        Xb_region, Xs_region = self.Boundary_net(t_region, S_region)
        n_pde = self.n_points['pde']
        eps_pde = torch.rand(n_pde, 1, device=self.device)
        x_pde = Xb_region[:n_pde].detach() + eps_pde * (Xs_region[:n_pde].detach() - Xb_region[:n_pde].detach() + 1e-6)
        res_pde, _, _ = self.compute_residuals(t_region[:n_pde], S_region[:n_pde], x_pde)
        loss_pde = F.mse_loss(res_pde, torch.zeros_like(res_pde))
        n_buy = self.n_points['buy']
        t_buy, S_buy = t_region[n_pde:n_pde+n_buy], S_region[n_pde:n_pde+n_buy]
        Xb_buy = Xb_region[n_pde:n_pde+n_buy]
        eps_buy = torch.rand(n_buy, 1, device=self.device)
        x_buy = self.domain['x_min'] + eps_buy * (Xb_buy.detach() - self.domain['x_min'])
        _, res_buy, _ = self.compute_residuals(t_buy, S_buy, x_buy)
        loss_buy = F.mse_loss(res_buy, torch.zeros_like(res_buy))
        n_sell = self.n_points['sell']
        t_sell, S_sell = t_region[n_pde+n_buy:], S_region[n_pde+n_buy:]
        Xs_sell = Xs_region[n_pde+n_buy:]
        eps_sell = torch.rand(n_sell, 1, device=self.device)
        x_sell = Xs_sell.detach() + eps_sell * (self.domain['x_max'] - Xs_sell.detach())
        _, _, res_sell = self.compute_residuals(t_sell, S_sell, x_sell)
        loss_sell = F.mse_loss(res_sell, torch.zeros_like(res_sell))
        loss_total = 100.0 * loss_terminal + loss_pde + loss_buy + loss_sell
        self.loss_history['total'].append(loss_total.item())
        self.loss_history['pde'].append(loss_pde.item())
        self.loss_history['buy'].append(loss_buy.item())
        self.loss_history['sell'].append(loss_sell.item())
        self.loss_history['terminal'].append(loss_terminal.item())
        return loss_total

    def update_convergence_norm(self):
        self.H_net.eval(); self.Boundary_net.eval()
        with torch.no_grad():
            h_grid = self.H_net(self.t_grid_h, self.S_grid_h, self.x_grid_h)
            Xb_grid, Xs_grid = self.Boundary_net(self.t_grid_b, self.S_grid_b)
        l2_H = torch.sqrt(torch.mean((h_grid - self.prev_H_grid)**2))
        l2_Xb = torch.sqrt(torch.mean((Xb_grid - self.prev_Xb_grid)**2))
        l2_Xs = torch.sqrt(torch.mean((Xs_grid - self.prev_Xs_grid)**2))
        self.norm_history['H'].append(l2_H.item()); self.norm_history['Xb'].append(l2_Xb.item()); self.norm_history['Xs'].append(l2_Xs.item())
        self.prev_H_grid = h_grid; self.prev_Xb_grid = Xb_grid; self.prev_Xs_grid = Xs_grid
        self.H_net.train(); self.Boundary_net.train()

    def train(self, epochs, report_every=1000, norm_every=100):
        print("--- Starting PINN Full Training ---")
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            if torch.isnan(loss): print(f"Epoch: {epoch:6d} | Loss: {loss.item():.4e} | PINN Training stopped due to NaN."); break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(loss)
            if epoch % report_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"PINN Epoch: {epoch:6d} | Loss: {loss.item():.4e} | Term: {self.loss_history['terminal'][-1]:.3e} | LR: {lr:.1e}")
            if epoch % norm_every == 0 or epoch == 1:
                self.update_convergence_norm()
        end_time = time.time()
        print(f"PINN Training finished in {end_time - start_time:.2f} seconds.")
        
    def get_solution_at_time(self, t_val, n_grid_S=50, n_grid_x=50):
        self.H_net.eval()
        S = torch.linspace(self.domain['S_min'], self.domain['S_max'], n_grid_S, device=self.device)
        x = torch.linspace(self.domain['x_min'], self.domain['x_max'], n_grid_x, device=self.device)
        S_grid, X_grid = torch.meshgrid(S, x, indexing='xy')
        T_grid = torch.full_like(S_grid, t_val)
        with torch.no_grad():
            h_tilde_pred = self.H_net(T_grid.flatten().unsqueeze(1),
                                      S_grid.flatten().unsqueeze(1),
                                      X_grid.flatten().unsqueeze(1)).reshape(n_grid_S, n_grid_x)
            h_pred = h_tilde_pred * H_SCALE
        return S_grid.cpu().numpy(), X_grid.cpu().numpy(), h_pred.cpu().numpy()

    def get_boundaries_3d(self, n_grid_t=30, n_grid_S=30):
        self.Boundary_net.eval()
        t = torch.linspace(self.domain['t_min'], self.domain['t_max'], n_grid_t, device=self.device)
        S = torch.linspace(self.domain['S_min'], self.domain['S_max'], n_grid_S, device=self.device)
        T_grid, S_grid = torch.meshgrid(t, S, indexing='ij')
        with torch.no_grad():
            Xb, Xs = self.Boundary_net(T_grid.flatten().unsqueeze(1), S_grid.flatten().unsqueeze(1))
        return (T_grid.cpu().numpy(), S_grid.cpu().numpy(),
                Xb.cpu().numpy().reshape(n_grid_t, n_grid_S),
                Xs.cpu().numpy().reshape(n_grid_t, n_grid_S))

    # --- NEW: Get 1D boundary slice at a specific time ---
    def get_boundaries_at_time(self, t_val, n_grid_S=50):
        self.Boundary_net.eval()
        S_tensor = torch.linspace(self.domain['S_min'], self.domain['S_max'], n_grid_S, device=self.device).unsqueeze(1)
        T_tensor = torch.full_like(S_tensor, t_val)
        
        with torch.no_grad():
            Xb, Xs = self.Boundary_net(T_tensor, S_tensor)
        
        return S_tensor.cpu().numpy().flatten(), Xb.cpu().numpy().flatten(), Xs.cpu().numpy().flatten()


# ###########################################################################
# #################### SECTION 2: FDM IMPLEMENTATION ########################
# ###########################################################################

class FDMValidator:
    def __init__(self, params, domain, grid_sizes):
        print("\n--- Initializing FDM Validator ---")
        self.params, self.domain = params, domain
        self.Nt, self.Ns, self.Nx = grid_sizes['Nt'], grid_sizes['Ns'], grid_sizes['Nx']
        self.dt = (domain['t_max'] - domain['t_min']) / self.Nt
        self.t = np.linspace(domain['t_min'], domain['t_max'], self.Nt + 1)
        self.S = np.linspace(domain['S_min'], domain['S_max'], self.Ns)
        self.dS = self.S[1] - self.S[0]
        self.x = np.linspace(domain['x_min'], domain['x_max'], self.Nx)
        self.dx = self.x[1] - self.x[0]
        self.S_grid, self.x_grid = np.meshgrid(self.S, self.x, indexing='ij')
        self.h_tilde = np.zeros((self.Ns, self.Nx, self.Nt + 1))
        self.Xb_fdm = np.zeros((self.Ns, self.Nt + 1))
        self.Xs_fdm = np.zeros((self.Ns, self.Nt + 1))
        self.omega = 1.4; self.tol = 1e-7; self.max_iter = 500
        print(f"FDM Grid: {self.Ns}(S) x {self.Nx}(x) x {self.Nt+1}(t)")
        self._set_terminal_condition()

    @staticmethod
    def _compute_Z(S, x, a, b):
        return x * S * (a * (x < 0) + b * (x >= 0))

    def _set_terminal_condition(self):
        Z = self._compute_Z(self.S_grid, self.x_grid, self.params['a'], self.params['b'])
        self.h_tilde[:, :, -1] = -self.params['gamma'] * Z / H_SCALE
        self.Xb_fdm[:, -1] = self.x[self.Nx // 2]
        self.Xs_fdm[:, -1] = self.x[self.Nx // 2]
        print("FDM: Terminal condition set.")

    def solve(self):
        print("--- Starting FDM Solver ---")
        start_time = time.time()
        for n in range(self.Nt - 1, -1, -1):
            if n % (max(1, self.Nt // 10)) == 0:
                print(f"FDM Time step: {n:4d}/{self.Nt} (t = {self.t[n]:.2f})")
            self._solve_timestep(n)
        end_time = time.time()
        print(f"FDM Solving finished in {end_time - start_time:.2f} seconds.")
        
    def _solve_timestep(self, n):
        h_old_timestep = self.h_tilde[:, :, n + 1]
        h_new = h_old_timestep.copy()
        
        p = self.params; dt, dS, dx = self.dt, self.dS, self.dx
        S_vec, x_vec = self.S, self.x
        
        e_rt_n = np.exp(p['r'] * (p['T'] - self.t[n]))
        
        C_buy_vec = -(1.0 / H_SCALE) * p['a'] * p['gamma'] * S_vec * e_rt_n
        C_sell_vec = -(1.0 / H_SCALE) * p['b'] * p['gamma'] * S_vec * e_rt_n

        for k in range(self.max_iter):
            h_old_iter = h_new.copy()
            err = 0.0
            for i in range(1, self.Ns - 1):
                S = S_vec[i]; C_buy = C_buy_vec[i]; C_sell = C_sell_vec[i]
                h_S_fwd = (h_old_iter[i+1, :] - h_old_iter[i, :]) / dS
                h_S_bwd = (h_old_iter[i, :] - h_old_iter[i-1, :]) / dS
                h_SS = (h_old_iter[i+1, :] - 2*h_old_iter[i, :] + h_old_iter[i-1, :]) / (dS**2)
                h_S_upwind = np.where(h_S_bwd + h_S_fwd > 0, h_S_bwd, h_S_fwd)
                pde_term = (
                    p['alpha'] * S * h_S_upwind +
                    0.5 * p['sigma']**2 * S**2 * (
                        H_SCALE * np.maximum(h_S_fwd, 0)**2 +
                        H_SCALE * np.minimum(h_S_bwd, 0)**2 +
                        h_SS
                    )
                )
                h_pde = h_old_timestep[i, :] + dt * pde_term
                h_slice_new = h_old_iter[i, :].copy()
                h_slice_new[0] = h_slice_new[1] - dx * C_buy
                h_slice_new[-1] = h_slice_new[-2] + dx * C_sell
                for j in range(1, self.Nx - 1):
                    h_sor = (1-self.omega) * h_old_iter[i,j] + self.omega * h_pde[j]
                    h_buy_limit = h_slice_new[j-1] + dx * C_buy
                    h_sell_limit = h_slice_new[j-1] + dx * C_sell
                    h_slice_new[j] = np.maximum(h_buy_limit, np.minimum(h_sell_limit, h_sor))
                h_new[i, :] = h_slice_new
                err = max(err, np.max(np.abs(h_new[i, :] - h_old_iter[i, :])))
            h_new[0, :] = h_new[1, :]; h_new[-1, :] = h_new[-2, :]
            h_old_iter = h_new.copy()
            if err < self.tol:
                break
        
        self.h_tilde[:, :, n] = h_new

        for i in range(self.Ns):
            h_slice = h_new[i, :]
            grad_x = np.gradient(h_slice, dx)
            grad_x_smooth = gaussian_filter1d(grad_x, sigma=1.0)
            C_buy = C_buy_vec[i]; C_sell = C_sell_vec[i]
            no_trade_indices = np.where((grad_x_smooth >= C_buy) & (grad_x_smooth <= C_sell))[0]
            if no_trade_indices.size == 0:
                self.Xb_fdm[i, n] = self.x[self.Nx // 2]
                self.Xs_fdm[i, n] = self.x[self.Nx // 2]
                continue
            diffs = np.diff(no_trade_indices)
            splits = np.where(diffs > 1)[0] + 1
            blocks = np.split(no_trade_indices, splits)
            if not blocks:
                self.Xb_fdm[i, n] = self.x[self.Nx // 2]
                self.Xs_fdm[i, n] = self.x[self.Nx // 2]
                continue
            largest_block = max(blocks, key=len)
            self.Xb_fdm[i, n] = self.x[largest_block[0]]
            self.Xs_fdm[i, n] = self.x[largest_block[-1]]

    def get_solution_at_time(self, t_val):
        n = np.argmin(np.abs(self.t - t_val))
        h_tilde_solution = self.h_tilde[:, :, n]
        h_solution = h_tilde_solution * H_SCALE
        return self.S, self.x, h_solution.T

    def get_boundaries_3d(self):
        T_grid, S_grid = np.meshgrid(self.t, self.S, indexing='ij')
        Xb_data = self.Xb_fdm.T
        Xs_data = self.Xs_fdm.T
        return T_grid, S_grid, Xb_data, Xs_data

    # --- NEW: Get 1D boundary slice at a specific time ---
    def get_boundaries_at_time(self, t_val):
        n = np.argmin(np.abs(self.t - t_val))
        return self.S, self.Xb_fdm[:, n], self.Xs_fdm[:, n]

# ###########################################################################
# #################### SECTION 3: PLOTTING & COMPARISON #####################
# ###########################################################################

def plot_pinn_diagnostics(pinn, norm_every):
    print("--- Plotting PINN Diagnostics ---")
    fig, ax = plt.subplots(figsize=(10, 6))
    if pinn.loss_history['total']:
        region_loss = np.array(pinn.loss_history['buy']) + np.array(pinn.loss_history['sell'])
        ax.plot(pinn.loss_history['total'], label='Total Loss', color='black', alpha=0.7)
        ax.plot(pinn.loss_history['pde'], label='PDE Loss', color='blue', linestyle='--')
        ax.plot(region_loss, label='Boundary Region Loss', color='red', linestyle='--')
        ax.plot(pinn.loss_history['terminal'], label='Terminal Loss', color='green', linestyle='--')
        ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
        ax.set_title('PINN Training Loss History'); ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5); plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if pinn.norm_history['H']:
        num_samples = len(pinn.norm_history['H'])
        epochs = [1] + [(i * norm_every) for i in range(1, num_samples)]
        if len(epochs) > num_samples: epochs = epochs[:num_samples]
        ax.plot(epochs, pinn.norm_history['H'], label=r'$||\tilde{h}_k - \tilde{h}_{k-1}||_2$', color='purple')
        ax.plot(epochs, pinn.norm_history['Xb'], label=r'$||Xb_k - Xb_{k-1}||_2$', color='blue')
        ax.plot(epochs, pinn.norm_history['Xs'], label=r'$||Xs_k - Xs_{k-1}||_2$', color='red')
        ax.set_yscale('log'); ax.set_xlabel(f'Epoch (sampled approx. every {norm_every})')
        ax.set_ylabel('L2 Norm (log scale)'); ax.set_title(r'PINN Convergence (L2 Norm of $\tilde{h}$)')
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5); plt.show()

def plot_solution_surface(S_grid, X_grid, h_solution, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    h_clean = np.nan_to_num(h_solution, nan=0.0)
    surf = ax.plot_surface(S_grid, X_grid, h_clean, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Stock Price (S)'); ax.set_ylabel('Shares Held (x)')
    ax.set_zlabel('h(t, S, x) = log(H)'); ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5); plt.show()

def plot_comparison(pinn_sol, fdm_sol, t_val):
    print(f"\n--- Generating Comparison Plots for h = log(H) at t = {t_val} ---")
    S_grid_pinn, X_grid_pinn, h_pinn = pinn_sol
    plot_solution_surface(S_grid_pinn, X_grid_pinn, h_pinn, f"PINN Solution h(t, S, x) at t = {t_val}")
    
    S_vec_fdm, x_vec_fdm, h_fdm = fdm_sol
    S_grid_fdm, X_grid_fdm = np.meshgrid(S_vec_fdm, x_vec_fdm, indexing='ij')
    plot_solution_surface(S_grid_fdm, X_grid_fdm, h_fdm.T, f"FDM Solution h(t, S, x) at t = {t_val}")
    
    print("Calculating solution error for h = log(H)...")
    fdm_interp = RectBivariateSpline(S_vec_fdm, x_vec_fdm, h_fdm)
    h_fdm_interp = fdm_interp(S_grid_pinn[:, 0], X_grid_pinn[0, :], grid=True)
    Error = h_pinn - h_fdm_interp
    
    l2_error_norm = np.sqrt(np.mean(Error**2))
    linf_error_norm = np.max(np.abs(Error))
    print(f"Error (h_PINN - h_FDM) at t = {t_val}:")
    print(f"  L2 Norm:   {l2_error_norm:.4e}")
    print(f"  L-inf Norm: {linf_error_norm:.4e}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = np.max(np.abs(Error)); vmin = -vmax
    if vmax == 0: vmin, vmax = -1.0, 1.0
    contour = ax.contourf(S_grid_pinn, X_grid_pinn, Error, 50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Stock Price (S)'); ax.set_ylabel('Shares Held (x)')
    ax.set_title(f'Error (h_PINN - h_FDM) at t = {t_val}')
    fig.colorbar(contour, label='Error'); plt.show()

# --- NEW PLOT 1: Masked solution ---
def plot_masked_solution(pinn_sol, pinn_bounds, fdm_sol, fdm_bounds, t_val):
    print(f"--- Plotting Masked 'No-Trade' Solution at t = {t_val} ---")
    
    # Unpack PINN data
    S_grid_pinn, X_grid_pinn, h_pinn = pinn_sol
    S_pinn_b, Xb_pinn, Xs_pinn = pinn_bounds
    
    # Unpack FDM data
    S_vec_fdm, x_vec_fdm, h_fdm = fdm_sol
    S_fdm_b, Xb_fdm, Xs_fdm = fdm_bounds
    S_grid_fdm, X_grid_fdm = np.meshgrid(S_vec_fdm, x_vec_fdm, indexing='ij')

    # Create 1D interpolators for the boundaries
    interp_Xb_pinn = interp1d(S_pinn_b, Xb_pinn, bounds_error=False, fill_value="extrapolate")
    interp_Xs_pinn = interp1d(S_pinn_b, Xs_pinn, bounds_error=False, fill_value="extrapolate")
    interp_Xb_fdm = interp1d(S_fdm_b, Xb_fdm, bounds_error=False, fill_value="extrapolate")
    interp_Xs_fdm = interp1d(S_fdm_b, Xs_fdm, bounds_error=False, fill_value="extrapolate")

    # Interpolate boundaries onto the 2D solution grids
    Xb_pinn_grid = interp_Xb_pinn(S_grid_pinn[:, 0]).reshape(-1, 1)
    Xs_pinn_grid = interp_Xs_pinn(S_grid_pinn[:, 0]).reshape(-1, 1)
    Xb_fdm_grid = interp_Xb_fdm(S_grid_fdm[:, 0]).reshape(-1, 1)
    Xs_fdm_grid = interp_Xs_fdm(S_grid_fdm[:, 0]).reshape(-1, 1)
    
    # Mask the solutions
    h_pinn_masked = h_pinn.copy()
    h_pinn_masked[(X_grid_pinn < Xb_pinn_grid) | (X_grid_pinn > Xs_pinn_grid)] = np.nan
    
    h_fdm_masked = h_fdm.T.copy()
    h_fdm_masked[(X_grid_fdm < Xb_fdm_grid) | (X_grid_fdm > Xs_fdm_grid)] = np.nan
    
    # Plotting
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    fig.suptitle(f"Solution in 'No-Trade' Region at t = {t_val}", fontsize=16)
    
    ax1.plot_surface(S_grid_pinn, X_grid_pinn, h_pinn_masked, cmap='viridis', edgecolor='none')
    ax1.set_title('PINN Solution')
    ax1.set_xlabel('S'); ax1.set_ylabel('x'); ax1.set_zlabel('h(t, S, x)')
    
    ax2.plot_surface(S_grid_fdm, X_grid_fdm, h_fdm_masked, cmap='viridis', edgecolor='none')
    ax2.set_title('FDM Solution')
    ax2.set_xlabel('S'); ax2.set_ylabel('x'); ax2.set_zlabel('h(t, S, x)')
    plt.show()

# --- NEW PLOT 2: Combined 3D Boundaries ---
def plot_3d_boundaries_combined(pinn, fdm):
    print("--- Plotting Combined 3D Free Boundary Comparison ---")
    
    T_pinn, S_pinn, Xb_pinn, Xs_pinn = pinn.get_boundaries_3d()
    T_fdm, S_fdm, Xb_fdm, Xs_fdm = fdm.get_boundaries_3d()

    z_min = min(np.min(Xb_pinn), np.min(Xb_fdm), np.min(Xs_pinn), np.min(Xs_fdm))
    z_max = max(np.max(Xb_pinn), np.max(Xb_fdm), np.max(Xs_pinn), np.max(Xs_fdm))
    z_min = max(z_min - 2, DOMAIN['x_min'])
    z_max = min(z_max + 2, DOMAIN['x_max'])

    fig = plt.figure(figsize=(12, 6))
    
    # PINN Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(T_pinn, S_pinn, Xb_pinn, color='blue', edgecolor='none', alpha=0.7, label="Buy Boundary")
    ax1.plot_surface(T_pinn, S_pinn, Xs_pinn, color='red', edgecolor='none', alpha=0.7, label="Sell Boundary")
    ax1.set_title('PINN Boundaries $X_b$ and $X_s$')
    ax1.set_xlabel('Time (t)'); ax1.set_ylabel('Stock Price (S)'); ax1.set_zlabel('Shares (x)')
    ax1.set_zlim(z_min, z_max)
    
    # FDM Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T_fdm, S_fdm, Xb_fdm, color='blue', edgecolor='none', alpha=0.7)
    ax2.plot_surface(T_fdm, S_fdm, Xs_fdm, color='red', edgecolor='none', alpha=0.7)
    ax2.set_title('FDM Boundaries $X_b$ and $X_s$')
    ax2.set_xlabel('Time (t)'); ax2.set_ylabel('Stock Price (S)'); ax2.set_zlabel('Shares (x)')
    ax2.set_zlim(z_min, z_max)
    
    plt.suptitle("Combined Boundary Comparison", fontsize=16)
    plt.show()

# --- NEW PLOT 3: GIF Frame Generation ---
def generate_gif_frames(pinn, fdm, n_frames=21):
    print(f"\n--- Generating {n_frames} frames for GIF ---")
    
    # Create a directory to store frames
    frame_dir = "gif_frames"
    os.makedirs(frame_dir, exist_ok=True)
    
    # Get z-limits for consistent plotting
    _, _, h_pinn_t0 = pinn.get_solution_at_time(0.0)
    _, _, h_pinn_T = pinn.get_solution_at_time(PARAMS['T'])
    _, _, h_fdm_t0 = fdm.get_solution_at_time(0.0)
    _, _, h_fdm_T = fdm.get_solution_at_time(PARAMS['T'])
    
    z_min = min(np.nanmin(h_pinn_t0), np.nanmin(h_pinn_T), np.nanmin(h_fdm_t0), np.nanmin(h_fdm_T))
    z_max = max(np.nanmax(h_pinn_t0), np.nanmax(h_pinn_T), np.nanmax(h_fdm_t0), np.nanmax(h_fdm_T))
    
    time_steps = np.linspace(0.0, PARAMS['T'], n_frames)
    
    for i, t_val in enumerate(time_steps):
        print(f"  Generating frame {i+1}/{n_frames} (t = {t_val:.2f})...")
        
        # Get solutions at time t
        S_pinn, X_pinn, h_pinn = pinn.get_solution_at_time(t_val)
        S_fdm, x_fdm, h_fdm = fdm.get_solution_at_time(t_val)
        S_fdm_grid, X_fdm_grid = np.meshgrid(S_fdm, x_fdm, indexing='ij')

        # Create plot
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f"Solution h(t, S, x) at Time t = {t_val:.2f}", fontsize=16)
        
        # PINN subplot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(S_pinn, X_pinn, h_pinn, cmap='viridis', edgecolor='none')
        ax1.set_title('PINN Solution')
        ax1.set_xlabel('S'); ax1.set_ylabel('x'); ax1.set_zlabel('h(t, S, x)')
        ax1.set_zlim(z_min, z_max)
        
        # FDM subplot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(S_fdm_grid, X_fdm_grid, h_fdm.T, cmap='viridis', edgecolor='none')
        ax2.set_title('FDM Solution')
        ax2.set_xlabel('S'); ax2.set_ylabel('x'); ax2.set_zlabel('h(t, S, x)')
        ax2.set_zlim(z_min, z_max)
        
        # Save frame
        plt.savefig(f"{frame_dir}/frame_{i:03d}.png")
        plt.close(fig) # Close figure to save memory

    print("--- Frame generation complete. ---")
    
    # --- Stitch frames into GIF ---
    try:
        print("Attempting to create GIF...")
        frames = []
        filenames = sorted(glob.glob(f"{frame_dir}/*.png"))
        for filename in filenames:
            frames.append(iio.imread(filename))
        
        # Save GIF
        gif_path = "solution_animation.gif"
        iio.mimsave(gif_path, frames, duration=200) # 200ms per frame
        print(f"\nSuccessfully created GIF: {gif_path}")
        
    except Exception as e:
        print(f"\n--- Could not create GIF ---")
        print(f"Error: {e}")
        print("Please ensure 'imageio' is installed ('pip install imageio')")
        print(f"Your frames are saved in the '{frame_dir}' directory.")


# ###########################################################################
# #################### SECTION 4: MAIN EXECUTION ############################
# ###########################################################################
if __name__ == "__main__":
    
    # --- 1. Run PINN ---
    PINN_EPOCHS = 20000          
    PINN_REPORT_EVERY = 1000
    PINN_NORM_EVERY = 500
    PINN_N_POINTS = {'terminal': 2048, 'pde': 2048, 'buy': 1024, 'sell': 1024}
    
    pinn = FreeBoundaryPINN(PARAMS, DOMAIN, PINN_N_POINTS, device)
    pinn.train(PINN_EPOCHS, PINN_REPORT_EVERY, PINN_NORM_EVERY)
    plot_pinn_diagnostics(pinn, PINN_NORM_EVERY)
    
    # --- 2. Run FDM ---
    FDM_GRID_SIZES = {
        'Nt': 100,  # 100 time steps
        'Ns': 50,   # 50 S-points
        'Nx': 50,   # 50 x-points
    }
    
    fdm = FDMValidator(PARAMS, DOMAIN, FDM_GRID_SIZES)
    fdm.solve()
    
    # --- 3. Compare 3D Boundaries ---
    # Old plot (separate)
    # plot_3d_boundaries(pinn, fdm)
    
    # New plot (combined)
    plot_3d_boundaries_combined(pinn, fdm)
    
    # --- 4. Compare Solutions at t=0.5 ---
    T_VAL = 0.5
    pinn_solution_t = pinn.get_solution_at_time(T_VAL, n_grid_S=50, n_grid_x=50)
    fdm_solution_t = fdm.get_solution_at_time(T_VAL)
    pinn_bounds_t = pinn.get_boundaries_at_time(T_VAL, n_grid_S=50)
    fdm_bounds_t = fdm.get_boundaries_at_time(T_VAL)
    
    plot_comparison(pinn_solution_t, fdm_solution_t, T_VAL)
    plot_masked_solution(pinn_solution_t, pinn_bounds_t, fdm_solution_t, fdm_bounds_t, T_VAL)

    # --- 5. Compare Solutions at t=0.0 ---
    T_VAL = 0.0
    pinn_solution_t0 = pinn.get_solution_at_time(T_VAL, n_grid_S=50, n_grid_x=50)
    fdm_solution_t0 = fdm.get_solution_at_time(T_VAL)
    pinn_bounds_t0 = pinn.get_boundaries_at_time(T_VAL, n_grid_S=50)
    fdm_bounds_t0 = fdm.get_boundaries_at_time(T_VAL)
    
    plot_comparison(pinn_solution_t0, fdm_solution_t0, T_VAL)
    plot_masked_solution(pinn_solution_t0, pinn_bounds_t0, fdm_solution_t0, fdm_bounds_t0, T_VAL)
    
    # --- 6. Generate GIF ---
    generate_gif_frames(pinn, fdm, n_frames=21) # 21 frames for 0.0 to 1.0 with dt=0.05