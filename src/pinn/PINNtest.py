import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
import time
import warnings

# Suppress runtime warnings from FDM
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

# Normalization for stability
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

    def compute_loss(self, train_pde=True, train_bc=True):
        n_term = self.n_points['terminal']
        S_term = torch.rand(n_term, 1, device=self.device) * (self.domain['S_max'] - self.domain['S_min']) + self.domain['S_min']
        x_term = torch.rand(n_term, 1, device=self.device) * (self.domain['x_max'] - self.domain['x_min']) + self.domain['x_min']
        t_term = torch.full_like(S_term, self.domain['t_max'])
        h_pred_term = self.H_net(t_term, S_term, x_term)
        h_target_term = self.compute_terminal_h_tilde(S_term, x_term, self.params['a'], self.params['b'], self.params['gamma'])
        loss_terminal = F.mse_loss(h_pred_term, h_target_term)
        loss_pde = torch.tensor(0.0, device=self.device)
        loss_buy = torch.tensor(0.0, device=self.device)
        loss_sell = torch.tensor(0.0, device=self.device)
        if train_pde or train_bc:
            n_region = self.n_points['pde'] + self.n_points['buy'] + self.n_points['sell']
            t_region = torch.rand(n_region, 1, device=self.device) * self.domain['t_max']
            S_region = torch.rand(n_region, 1, device=self.device) * (self.domain['S_max'] - self.domain['S_min']) + self.domain['S_min']
            Xb_region, Xs_region = self.Boundary_net(t_region, S_region)
            if train_pde:
                n_pde = self.n_points['pde']
                eps_pde = torch.rand(n_pde, 1, device=self.device)
                x_pde = Xb_region[:n_pde].detach() + eps_pde * (Xs_region[:n_pde].detach() - Xb_region[:n_pde].detach() + 1e-6)
                res_pde, _, _ = self.compute_residuals(t_region[:n_pde], S_region[:n_pde], x_pde)
                loss_pde = F.mse_loss(res_pde, torch.zeros_like(res_pde))
            if train_bc:
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
        if train_pde:
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

    def pretrain_terminal(self, epochs=2000):
        print("--- Starting PINN Pre-training (Terminal Condition Only) ---")
        pretrain_optim = optim.Adam(self.H_net.parameters(), lr=1e-3)
        for epoch in range(1, epochs + 1):
            pretrain_optim.zero_grad()
            loss = self.compute_loss(train_pde=False, train_bc=False)
            if torch.isnan(loss): print(f"Pre-train Epoch: {epoch:6d} | Loss: {loss.item():.4e} | NaN detected."); break
            loss.backward()
            pretrain_optim.step()
            if epoch % (epochs // 10) == 0: print(f"Pre-train Epoch: {epoch:6d} | Terminal Loss: {loss.item():.4e}")
        print("--- PINN Pre-training finished ---")

    def train(self, epochs, report_every=1000, norm_every=100, pretrain_epochs=2000):
        if pretrain_epochs > 0:
            self.pretrain_terminal(epochs=pretrain_epochs)
        print("--- Starting PINN Full Training ---")
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            loss = self.compute_loss(train_pde=True, train_bc=True)
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
        self.Xb_fdm[:, -1] = self.x[-1]
        self.Xs_fdm[:, -1] = self.x[0]
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
        h_old_iter = self.h_tilde[:, :, n + 1].copy()
        h_new = h_old_iter.copy()
        region_map = np.zeros((self.Ns, self.Nx), dtype=int)
        
        p = self.params; dt, dS, dx = self.dt, self.dS, self.dx; S = self.S
        
        e_rt = np.exp(p['r'] * (p['T'] - self.t[n]))
        C_buy = -(1.0 / H_SCALE) * p['a'] * p['gamma'] * self.S_grid * e_rt
        C_sell = -(1.0 / H_SCALE) * p['b'] * p['gamma'] * self.S_grid * e_rt

        for k in range(self.max_iter):
            err = 0.0
            for i in range(1, self.Ns - 1):
                h_new[i, 0] = h_new[i, 1] - dx * C_buy[i, 0]
                region_map[i, 0] = -1 
                for j in range(1, self.Nx - 1):
                    h_S = (h_old_iter[i+1, j] - h_old_iter[i-1, j]) / (2 * dS)
                    h_SS = (h_old_iter[i+1, j] - 2 * h_old_iter[i, j] + h_old_iter[i-1, j]) / (dS**2)
                    pde_term = p['alpha'] * S[i] * h_S + 0.5 * p['sigma']**2 * S[i]**2 * (H_SCALE * h_S**2 + h_SS)
                    h_pde = self.h_tilde[i, j, n+1] - dt * pde_term
                    h_sor = (1 - self.omega) * h_old_iter[i, j] + self.omega * h_pde
                    grad_x = (h_sor - h_new[i, j-1]) / dx 
                    if grad_x < C_buy[i, j]:
                        h_proj = h_new[i, j-1] + dx * C_buy[i, j]; region_map[i, j] = -1
                    elif grad_x > C_sell[i, j]:
                        h_proj = h_new[i, j-1] + dx * C_sell[i, j]; region_map[i, j] = 1
                    else:
                        h_proj = h_sor; region_map[i, j] = 0
                    h_new[i, j] = h_proj
                    err = max(err, abs(h_new[i, j] - h_old_iter[i, j]))
                h_new[i, -1] = h_new[i, -2] + dx * C_sell[i, -1]
                region_map[i, -1] = 1
            h_old_iter = h_new.copy()
            h_new[0, :] = h_new[1, :]; h_new[-1, :] = h_new[-2, :]
            h_old_iter = h_new.copy()
            if err < self.tol:
                break
        
        self.h_tilde[:, :, n] = h_new
        
        # --- FDM BOUNDARY OVERHAUL: Find largest contiguous No-Trade region ---
        for i in range(self.Ns):
            current_start, current_len = -1, 0
            best_start, best_len = 0, 0
            
            row_regions = region_map[i, :]
            
            for j in range(self.Nx):
                if row_regions[j] == 0: # If in No-Trade
                    if current_start == -1:
                        current_start = j # Start a new block
                    current_len += 1
                else: # If in Buy or Sell
                    if current_len > best_len: # Check if current block is the new best
                        best_len = current_len
                        best_start = current_start
                    current_start = -1 # Reset
                    current_len = 0
            
            # Check the last block
            if current_len > best_len:
                best_len = current_len
                best_start = current_start

            if best_len > 0:
                # Found a no-trade region
                self.Xb_fdm[i, n] = self.x[best_start]
                self.Xs_fdm[i, n] = self.x[best_start + best_len - 1]
            else:
                # No no-trade region found (e.g., at t=T)
                # Set boundaries to be degenerate (e.g., buy = sell)
                self.Xb_fdm[i, n] = self.x[self.Nx // 2]
                self.Xs_fdm[i, n] = self.x[self.Nx // 2]
        # --- END OF OVERHAUL ---

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
    ax.set_xlabel('Stock Price (S)')
    ax.set_ylabel('Shares Held (x)')
    ax.set_zlabel('h(t, S, x) = log(H)')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

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
    vmax = np.max(np.abs(Error))
    if vmax == 0: vmax = 1.0 # Avoid division by zero if error is zero
    contour = ax.contourf(S_grid_pinn, X_grid_pinn, Error, 50, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Stock Price (S)'); ax.set_ylabel('Shares Held (x)')
    ax.set_title(f'Error (h_PINN - h_FDM) at t = {t_val}')
    fig.colorbar(contour, label='Error'); plt.show()

def plot_3d_boundaries(pinn, fdm):
    print("--- Plotting 3D Free Boundary Comparison ---")
    
    T_pinn, S_pinn, Xb_pinn, Xs_pinn = pinn.get_boundaries_3d()
    T_fdm, S_fdm, Xb_fdm, Xs_fdm = fdm.get_boundaries_3d()

    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(121, projection='3d')
    ax1.plot_surface(T_pinn, S_pinn, Xb_pinn, cmap='Blues', edgecolor='none', alpha=0.8)
    ax1.set_title('PINN Buy Boundary $X_b(t, S)$')
    ax1.set_xlabel('Time (t)'); ax1.set_ylabel('Stock Price (S)'); ax1.set_zlabel('Shares (x)')
    
    ax2 = fig1.add_subplot(122, projection='3d')
    ax2.plot_surface(T_fdm, S_fdm, Xb_fdm, cmap='Blues', edgecolor='none', alpha=0.8)
    ax2.set_title('FDM Buy Boundary $X_b(t, S)$')
    ax2.set_xlabel('Time (t)'); ax2.set_ylabel('Stock Price (S)'); ax2.set_zlabel('Shares (x)')
    plt.suptitle("Buy Boundary Comparison", fontsize=16); plt.show()

    fig2 = plt.figure(figsize=(12, 6))
    ax3 = fig2.add_subplot(121, projection='3d')
    ax3.plot_surface(T_pinn, S_pinn, Xs_pinn, cmap='Reds', edgecolor='none', alpha=0.8)
    ax3.set_title('PINN Sell Boundary $X_s(t, S)$')
    ax3.set_xlabel('Time (t)'); ax3.set_ylabel('Stock Price (S)'); ax3.set_zlabel('Shares (x)')
    
    ax4 = fig2.add_subplot(122, projection='3d')
    ax4.plot_surface(T_fdm, S_fdm, Xs_fdm, cmap='Reds', edgecolor='none', alpha=0.8)
    ax4.set_title('FDM Sell Boundary $X_s(t, S)$')
    ax4.set_xlabel('Time (t)'); ax4.set_ylabel('Stock Price (S)'); ax4.set_zlabel('Shares (x)')
    plt.suptitle("Sell Boundary Comparison", fontsize=16); plt.show()

# ###########################################################################
# #################### SECTION 4: MAIN EXECUTION ############################
# ###########################################################################
if __name__ == "__main__":
    
    # --- 1. Run PINN ---
    PINN_PRETRAIN_EPOCHS = 5000  # Warm-up the terminal condition
    PINN_EPOCHS = 20000          # Full training
    PINN_REPORT_EVERY = 1000
    PINN_NORM_EVERY = 500
    PINN_N_POINTS = {'terminal': 2048, 'pde': 2048, 'buy': 1024, 'sell': 1024}
    
    pinn = FreeBoundaryPINN(PARAMS, DOMAIN, PINN_N_POINTS, device)
    pinn.train(PINN_EPOCHS, PINN_REPORT_EVERY, PINN_NORM_EVERY, pretrain_epochs=PINN_PRETRAIN_EPOCHS)
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
    plot_3d_boundaries(pinn, fdm)
    
    # --- 4. Compare Solutions at t=0.5 ---
    T_VAL = 0.5
    pinn_solution_t = pinn.get_solution_at_time(T_VAL, n_grid_S=50, n_grid_x=50)
    fdm_solution_t = fdm.get_solution_at_time(T_VAL)
    plot_comparison(pinn_solution_t, fdm_solution_t, T_VAL)
    
    # --- 5. Compare Solutions at t=0.0 ---
    T_VAL = 0.0
    pinn_solution_t0 = pinn.get_solution_at_time(T_VAL, n_grid_S=50, n_grid_x=50)
    fdm_solution_t0 = fdm.get_solution_at_time(T_VAL)
    plot_comparison(pinn_solution_t0, fdm_solution_t0, T_VAL)