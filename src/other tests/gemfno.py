#!/usr/bin/env python3
"""
transaction_cost_pinn_fdm_pino_final_fdmreplaced.py

- PINN: physics-informed network for u = log(H) with free boundaries Xb, Xs.
- FDM:  Replaced by FDMValidator from arregui3.py (scaled log formulation).
- FNO:  Fully independent PINO (Fourier Neural Operator) with its own
        boundary nets Xb_fno, Xs_fno.

Plots: 3×3 comparison
    Row 1: H(t=0,S,x) for PINN / FDM / FNO (same z-scale)
    Row 2: Buy boundary Xb(t,S)
    Row 3: Sell boundary Xs(t,S)
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.ndimage import gaussian_filter1d

# -----------------------------
# GLOBAL PARAMETERS
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Market parameters
T = 1.0
r = 0.05
alpha = 0.10
sigma = 0.20
gamma = 0.10
zeta = 0.01
mu = 0.01
a = 1.0 + zeta
b = 1.0 - mu

# Domain of interest
S_MIN, S_MAX = 50.0, 150.0
x_MIN, x_MAX = -1.5, 1.5

# PINN settings
N_COLLOC = 3000
N_TERM   = 1000
N_VAL    = 800
N_EPOCHS_PINN = 3000   # adjust as desired
PINN_LR = 1e-3
K_MASK = 80.0
CLIP_GRAD = 1.0

# FDM grid
FDM_nT = 120
FDM_nS = 200
FDM_nx = 320
FDM_x_MIN, FDM_x_MAX = -3.0, 3.0
S_MAX_INTERNAL = S_MAX * 4.0

# FNO / PINO settings
FNO_EPOCHS  = 200      # set to 0 to skip FNO
FNO_NS      = 96
FNO_NX      = 96
FNO_MODES1  = 20
FNO_MODES2  = 20
FNO_WIDTH   = 64
FNO_LR      = 5e-4
FNO_BATCH   = 4
FNO_K_MASK  = 60.0

# Clamping for u = log(H)
U_CLAMP_MIN, U_CLAMP_MAX = -50.0, 50.0

# Scaling used by FDMValidator (h_tilde = u / H_SCALE)
H_SCALE = 1000.0

# Seeds
torch.manual_seed(0)
np.random.seed(0)

# -----------------------------
# UTILITIES
# -----------------------------
def Z_np(S, x):
    return x * S * (a * (x < 0) + b * (x >= 0))

def Z_torch(S, x):
    return x * S * (a * (x < 0).float() + b * (x >= 0).float())

# -----------------------------
# BASIC MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=64, depth=3, activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), activation()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), activation()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ============================================================
# PINN IMPLEMENTATION
# ============================================================
u_net = MLP(3, 1, width=64, depth=4).to(DEVICE)   # (t, logS, x) -> u
Xb_net = MLP(2, 1, width=32, depth=3).to(DEVICE)  # (t, logS) -> Xb
Xs_net = MLP(2, 1, width=32, depth=3).to(DEVICE)  # (t, logS) -> Xs

def sample_collocation(n):
    t = np.random.rand(n,1) * T
    uu = np.random.rand(n,1)
    logS = np.log(S_MIN) + uu * (np.log(S_MAX) - np.log(S_MIN))
    S = np.exp(logS)
    x = np.random.rand(n,1) * (x_MAX - x_MIN) + x_MIN
    return torch.tensor(np.hstack([t,S,x]), dtype=torch.float32, device=DEVICE)

def sample_terminal(n):
    uu = np.random.rand(n,1)
    S = np.exp(np.log(S_MIN) + uu * (np.log(S_MAX) - np.log(S_MIN)))
    x = np.random.rand(n,1)*(x_MAX - x_MIN) + x_MIN
    return torch.tensor(np.hstack([S,x]), dtype=torch.float32, device=DEVICE)

def pinn_loss(collocation_pts, terminal_pts):
    t = collocation_pts[:,0:1].clone().requires_grad_(True)
    S = collocation_pts[:,1:2].clone().requires_grad_(True)
    x = collocation_pts[:,2:3].clone().requires_grad_(True)

    logS = torch.log(S)
    inp = torch.cat([t, logS, x], dim=1)
    u_pred = u_net(inp)

    grads = grad(u_pred, inp, torch.ones_like(u_pred), create_graph=True)[0]
    u_t    = grads[:,0:1]
    u_logS = grads[:,1:2]
    u_x    = grads[:,2:3]

    u_S = u_logS / S
    u_logS2 = grad(u_logS, inp, torch.ones_like(u_logS), create_graph=True)[0][:,1:2]
    u_SS = (u_logS2 - u_logS) / (S**2 + 1e-12)

    pde_res = u_t + alpha*S*u_S + 0.5*sigma**2*S**2*(u_SS + u_S**2)

    inp_b = torch.cat([t, logS], dim=1)
    Xb_pred = Xb_net(inp_b)
    Xs_pred = Xs_net(inp_b)

    mask_interior = torch.sigmoid(K_MASK*(x - Xb_pred)) * torch.sigmoid(K_MASK*(Xs_pred - x))
    mask_buy  = torch.sigmoid(K_MASK*(Xb_pred - x))
    mask_sell = torch.sigmoid(K_MASK*(x - Xs_pred))

    loss_pde = (mask_interior * pde_res**2).mean()

    rt = r*(T - t)
    exp_rt = torch.exp(torch.clamp(rt, -50, 50))

    loss_buy  = (mask_buy  * (u_x + a*gamma*S*exp_rt)**2).mean()
    loss_sell = (mask_sell * (u_x + b*gamma*S*exp_rt)**2).mean()

    St = terminal_pts[:,0:1]
    xt = terminal_pts[:,1:2]
    tt = torch.full_like(St, T)
    u_term = u_net(torch.cat([tt, torch.log(St), xt], dim=1))
    loss_term = ((u_term + gamma*Z_torch(St,xt))**2).mean()

    loss_mono = torch.relu(Xb_pred - Xs_pred).pow(2).mean()

    l2 = 1e-6 * sum(p.pow(2).sum() for p in u_net.parameters())

    loss = 50*loss_pde + 50*loss_term + loss_buy + loss_sell + 100*loss_mono + l2
    return loss, {
        'pde': loss_pde.item(),
        'term': loss_term.item()
    }

def train_pinn():
    params = list(u_net.parameters()) + list(Xb_net.parameters()) + list(Xs_net.parameters())
    opt = optim.AdamW(params, lr=PINN_LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1000, N_EPOCHS_PINN))

    coll_val = sample_collocation(N_VAL)
    term_val = sample_terminal(N_VAL//2)

    print("Training PINN...")
    for ep in range(1, N_EPOCHS_PINN+1):
        opt.zero_grad()
        coll = sample_collocation(N_COLLOC)
        term = sample_terminal(N_TERM)
        loss, comps = pinn_loss(coll, term)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP_GRAD)
        opt.step()
        sched.step()

        if ep % 200 == 0 or ep == 1:
            vloss, _ = pinn_loss(coll_val, term_val)
            print(f"Ep {ep:5d} | loss {loss.item():.2e} | val {vloss.item():.2e} | "
                  f"pde {comps['pde']:.2e} | term {comps['term']:.2e}")
    print("PINN training complete.")

def eval_pinn_on_grid(t_vals, S_vals, x_vals):
    nt, nS, nx = len(t_vals), len(S_vals), len(x_vals)
    H_surfs = {}
    Xb_surf = np.zeros((nt, nS))
    Xs_surf = np.zeros((nt, nS))

    for it, t_val in enumerate(t_vals):
        Sg, Xg = np.meshgrid(S_vals, x_vals, indexing='ij')
        t_col = np.full(Sg.size, t_val, dtype=np.float32).reshape(-1,1)
        S_col = Sg.reshape(-1,1).astype(np.float32)
        x_col = Xg.reshape(-1,1).astype(np.float32)

        t_t = torch.tensor(t_col, dtype=torch.float32, device=DEVICE)
        S_t = torch.tensor(S_col, dtype=torch.float32, device=DEVICE)
        x_t = torch.tensor(x_col, dtype=torch.float32, device=DEVICE)

        logS = torch.log(S_t)
        inp = torch.cat([t_t, logS, x_t], dim=1)
        with torch.no_grad():
            u_vals = u_net(inp).cpu().numpy().reshape(Sg.shape)
            H_surfs[t_val] = np.exp(np.clip(u_vals, U_CLAMP_MIN, U_CLAMP_MAX))

            tb = np.full((nS,1), t_val, dtype=np.float32)
            Sb = S_vals.astype(np.float32).reshape(-1,1)
            logSb = np.log(Sb)
            inp_b = torch.tensor(np.hstack([tb, logSb]), dtype=torch.float32, device=DEVICE)
            Xb_surf[it] = Xb_net(inp_b).cpu().numpy().flatten()
            Xs_surf[it] = Xs_net(inp_b).cpu().numpy().flatten()

    return H_surfs, Xb_surf, Xs_surf

# ==============================================
# FDM IMPLEMENTATION (REPLACED BY FDMValidator)
# ==============================================
class FDMValidator:
    """
    Direct port of Section 2 FDM implementation from arregui3.py,
    adapted to use our parameter/domain dictionaries and H_SCALE.
    """
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
        self.omega = 1.4
        self.tol = 1e-7
        self.max_iter = 500
        print(f"FDM Grid: {self.Ns}(S) x {self.Nx}(x) x {self.Nt+1}(t)")
        self._set_terminal_condition()

    @staticmethod
    def _compute_Z(S, x, a, b):
        return x * S * (a * (x < 0) + b * (x >= 0))

    def _set_terminal_condition(self):
        Z = self._compute_Z(self.S_grid, self.x_grid, self.params['a'], self.params['b'])
        self.h_tilde[:, :, -1] = -self.params['gamma'] * Z / H_SCALE
        # Initialize boundaries in middle of x-domain
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

        p = self.params
        dt, dS, dx = self.dt, self.dS, self.dx
        S_vec, x_vec = self.S, self.x

        e_rt_n = np.exp(p['r'] * (p['T'] - self.t[n]))

        C_buy_vec  = -(1.0 / H_SCALE) * p['a'] * p['gamma'] * S_vec * e_rt_n
        C_sell_vec = -(1.0 / H_SCALE) * p['b'] * p['gamma'] * S_vec * e_rt_n

        for k in range(self.max_iter):
            h_old_iter = h_new.copy()
            err = 0.0
            for i in range(1, self.Ns - 1):
                S = S_vec[i]
                C_buy  = C_buy_vec[i]
                C_sell = C_sell_vec[i]

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
                # Neumann-like boundaries in x
                h_slice_new[0]  = h_slice_new[1]  - dx * C_buy
                h_slice_new[-1] = h_slice_new[-2] + dx * C_sell

                for j in range(1, self.Nx - 1):
                    h_sor = (1 - self.omega) * h_old_iter[i,j] + self.omega * h_pde[j]
                    h_buy_limit  = h_slice_new[j-1] + dx * C_buy
                    h_sell_limit = h_slice_new[j-1] + dx * C_sell
                    h_slice_new[j] = np.maximum(h_buy_limit, np.minimum(h_sell_limit, h_sor))

                h_new[i, :] = h_slice_new
                err = max(err, np.max(np.abs(h_new[i, :] - h_old_iter[i, :])))

            # Neumann in S
            h_new[0, :]  = h_new[1, :]
            h_new[-1, :] = h_new[-2, :]
            if err < self.tol:
                break

        self.h_tilde[:, :, n] = h_new

        # Extract boundaries by gradient-based no-trade region
        for i in range(self.Ns):
            h_slice = h_new[i, :]
            grad_x = np.gradient(h_slice, dx)
            grad_x_smooth = gaussian_filter1d(grad_x, sigma=1.0)
            C_buy  = C_buy_vec[i]
            C_sell = C_sell_vec[i]
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

def fdm_solver(nT=FDM_nT, nS=FDM_nS, nx=FDM_nx):
    """
    Wrapper that exposes the same interface as your previous fdm_solver,
    but uses FDMValidator internally.
    Returns:
        t_grid, S_grid_full, x_grid, u_fdm (log H), Xb_fdm, Xs_fdm
    """
    params = {
        'T': T, 'r': r, 'alpha': alpha, 'sigma': sigma,
        'zeta': zeta, 'mu': mu, 'gamma': gamma,
        'a': a, 'b': b
    }
    domain = {
        't_min': 0.0, 't_max': T,
        'S_min': S_MIN, 'S_max': S_MAX_INTERNAL,
        'x_min': FDM_x_MIN, 'x_max': FDM_x_MAX
    }
    grid_sizes = {'Nt': nT, 'Ns': nS, 'Nx': nx}

    fdm = FDMValidator(params, domain, grid_sizes)
    fdm.solve()

    t_grid = fdm.t                         # (Nt+1,)
    S_grid_full = fdm.S                    # (Ns,)
    x_grid = fdm.x                         # (Nx,)

    # Convert h_tilde -> u = log H
    u_fdm = np.transpose(fdm.h_tilde * H_SCALE, (2,0,1))  # (Nt+1, Ns, Nx)
    Xb_fdm = fdm.Xb_fdm.T                  # (Nt+1, Ns)
    Xs_fdm = fdm.Xs_fdm.T                  # (Nt+1, Ns)

    return t_grid, S_grid_full, x_grid, u_fdm, Xb_fdm, Xs_fdm

# ====================================
# FNO (PINO): FULLY INDEPENDENT SOLVER
# ====================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels*out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        b, c, n1, n2 = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            b, self.out_channels, n1, n2//2+1,
            dtype=torch.cfloat, device=x.device
        )
        w = self.weights[...,0] + 1j*self.weights[...,1]
        out_ft[:,:, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:,:,:self.modes1,:self.modes2], w)
        x = torch.fft.irfft2(out_ft, s=(n1,n2), norm="ortho")
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=3, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width

        self.fc0 = nn.Linear(in_channels, width)

        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.conv4 = SpectralConv2d(width, width, modes1, modes2)

        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.w4 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, out_channels)

        self.act = nn.GELU()

    def forward(self, x):
        b, c, nS, nX = x.shape
        x = x.permute(0,2,3,1)
        x = self.fc0(x)
        x = x.permute(0,3,1,2)

        x1 = self.conv1(x); x = self.act(x1 + self.w1(x))
        x1 = self.conv2(x); x = self.act(x1 + self.w2(x))
        x1 = self.conv3(x); x = self.act(x1 + self.w3(x))
        x1 = self.conv4(x); x = self.act(x1 + self.w4(x))

        x = x.permute(0,2,3,1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

u_fno = FNO2d(FNO_MODES1, FNO_MODES2, FNO_WIDTH, in_channels=3, out_channels=1).to(DEVICE)
Xb_fno_net = MLP(2,1,width=32,depth=3).to(DEVICE)
Xs_fno_net = MLP(2,1,width=32,depth=3).to(DEVICE)

# Precompute FNO grids
S_vals_fno_np = np.exp(np.linspace(np.log(S_MIN), np.log(S_MAX), FNO_NS))
x_vals_fno_np = np.linspace(x_MIN, x_MAX, FNO_NX)
logS_grid_np   = np.log(S_vals_fno_np)[:,None].repeat(FNO_NX,axis=1)
x_grid_np      = x_vals_fno_np[None,:].repeat(FNO_NS,axis=0)

logS_grid_torch = torch.tensor(logS_grid_np, dtype=torch.float32, device=DEVICE)
x_grid_torch    = torch.tensor(x_grid_np,    dtype=torch.float32, device=DEVICE)

S_vals_fno_torch = torch.tensor(S_vals_fno_np, dtype=torch.float32, device=DEVICE)
logS_vals_fno_torch = torch.log(S_vals_fno_torch)

def train_fno_pino():
    if FNO_EPOCHS <= 0:
        print("Skipping FNO/PINO training (FNO_EPOCHS <= 0).")
        return

    params = list(u_fno.parameters()) + list(Xb_fno_net.parameters()) + list(Xs_fno_net.parameters())
    opt = optim.AdamW(params, lr=FNO_LR, weight_decay=1e-5)

    print("Training FNO as PINO (fully independent)...")
    for ep in range(1, FNO_EPOCHS+1):
        opt.zero_grad()

        B = FNO_BATCH
        t_samples = np.random.rand(B) * T

        xb = torch.zeros(B, 3, FNO_NS, FNO_NX, device=DEVICE)
        for j in range(B):
            xb[j,0,:,:] = t_samples[j]
        xb[:,1,:,:] = logS_grid_torch
        xb[:,2,:,:] = x_grid_torch
        xb = xb.requires_grad_(True)

        u_pred = u_fno(xb)

        u_sum = u_pred.sum()
        grads = grad(u_sum, xb, create_graph=True)[0]

        u_t    = grads[:,0:1,:,:]
        u_logS = grads[:,1:2,:,:]
        u_x    = grads[:,2:3,:,:]

        S_field = torch.exp(xb[:,1:2,:,:])

        u_S = u_logS / (S_field + 1e-12)
        u_logS_sum = u_logS.sum()
        grads2 = grad(u_logS_sum, xb, create_graph=True)[0]
        u_logS2 = grads2[:,1:2,:,:]
        u_SS = (u_logS2 - u_logS) / (S_field**2 + 1e-12)

        pde_res = u_t + alpha*S_field*u_S + 0.5*sigma**2*S_field**2*(u_SS + u_S**2)

        Xb_full = torch.zeros(B,1,FNO_NS,FNO_NX, device=DEVICE)
        Xs_full = torch.zeros(B,1,FNO_NS,FNO_NX, device=DEVICE)

        loss_mono = 0.0
        for j in range(B):
            t_j = xb[j,0,0,0]
            t_vec = t_j * torch.ones_like(S_vals_fno_torch).unsqueeze(1)
            inp_b = torch.cat([t_vec, logS_vals_fno_torch.unsqueeze(1)], dim=1)
            Xb_line = Xb_fno_net(inp_b)  # (NS,1)
            Xs_line = Xs_fno_net(inp_b)

            Xb_full[j,0,:,:] = Xb_line.expand(FNO_NS, FNO_NX)
            Xs_full[j,0,:,:] = Xs_line.expand(FNO_NS, FNO_NX)

            loss_mono = loss_mono + torch.relu(Xb_line - Xs_line).pow(2).mean()

        loss_mono = loss_mono / B

        x_field = xb[:,2:3,:,:]

        mask_interior = torch.sigmoid(FNO_K_MASK*(x_field - Xb_full)) * \
                        torch.sigmoid(FNO_K_MASK*(Xs_full - x_field))
        mask_buy  = torch.sigmoid(FNO_K_MASK*(Xb_full - x_field))
        mask_sell = torch.sigmoid(FNO_K_MASK*(x_field - Xs_full))

        loss_pde = (mask_interior * pde_res**2).mean()

        t_field = xb[:,0:1,:,:]
        rt_field = r*(T - t_field)
        exp_rt_field = torch.exp(torch.clamp(rt_field, -50, 50))
        buy_res  = u_x + a*gamma*S_field*exp_rt_field
        sell_res = u_x + b*gamma*S_field*exp_rt_field

        loss_buy  = (mask_buy  * buy_res**2).mean()
        loss_sell = (mask_sell * sell_res**2).mean()

        # Terminal condition
        xb_T = torch.zeros(1,3,FNO_NS,FNO_NX, device=DEVICE)
        xb_T[0,0,:,:] = T
        xb_T[0,1,:,:] = logS_grid_torch
        xb_T[0,2,:,:] = x_grid_torch
        xb_T = xb_T.requires_grad_(True)
        u_T = u_fno(xb_T)[0,0]

        S_grid_T = torch.exp(xb_T[0,1])
        x_grid_T = xb_T[0,2]
        Z_grid = Z_torch(S_grid_T, x_grid_T).view(FNO_NS,FNO_NX)
        loss_term = ((u_T + gamma*Z_grid)**2).mean()

        l2 = 1e-6 * sum(p.pow(2).sum() for p in u_fno.parameters())

        loss = 30*loss_pde + loss_buy + loss_sell + 30*loss_term + 50*loss_mono + l2
        loss.backward()
        opt.step()

        if ep % 20 == 0 or ep == 1:
            print(f"FNO Ep {ep:4d} | loss {loss.item():.3e} | pde {loss_pde.item():.3e} "
                  f"| term {loss_term.item():.3e}")

    print("FNO/PINO training complete.")

def eval_fno_on_grid(t_vals, S_vals, x_vals):
    if FNO_EPOCHS <= 0:
        return {}, {}
    u_fno.eval()
    H_surfs = {}
    u_surfs = {}

    nS, nX = len(S_vals), len(x_vals)
    Sg, Xg = np.meshgrid(S_vals, x_vals, indexing='ij')
    logS_grid = np.log(Sg).astype(np.float32)
    x_grid = Xg.astype(np.float32)

    for t_val in t_vals:
        feat = np.stack([
            np.full((nS,nX), t_val, dtype=np.float32),
            logS_grid,
            x_grid
        ], axis=0)
        xb = torch.tensor(feat[None,...], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            u_pred = u_fno(xb)[0,0].cpu().numpy()
        u_clamped = np.clip(u_pred, U_CLAMP_MIN, U_CLAMP_MAX)
        u_surfs[t_val] = u_clamped
        H_surfs[t_val] = np.exp(u_clamped)
    return H_surfs, u_surfs

def eval_fno_boundaries(t_vals, S_vals):
    nt, nS = len(t_vals), len(S_vals)
    Xb_arr = np.zeros((nt, nS))
    Xs_arr = np.zeros((nt, nS))

    S_tensor = torch.tensor(S_vals, dtype=torch.float32, device=DEVICE).view(-1,1)
    logS_tensor = torch.log(S_tensor)

    for it, t_val in enumerate(t_vals):
        t_tensor = torch.full_like(S_tensor, t_val)
        inp_b = torch.cat([t_tensor, logS_tensor], dim=1)
        with torch.no_grad():
            Xb_line = Xb_fno_net(inp_b).cpu().numpy().flatten()
            Xs_line = Xs_fno_net(inp_b).cpu().numpy().flatten()
        Xb_arr[it] = Xb_line
        Xs_arr[it] = Xs_line

    return Xb_arr, Xs_arr

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    start = time.time()

    # 1) Train PINN
    if N_EPOCHS_PINN > 0:
        train_pinn()
    else:
        print("Skipping PINN training")

    # 2) Evaluate PINN on plot grid
    t_plot = [0.0, 0.25*T, 0.5*T, 0.75*T, T]
    S_plot = np.linspace(S_MIN, S_MAX, 90)
    x_plot = np.linspace(x_MIN, x_MAX, 130)
    H_pinn, Xb_pinn, Xs_pinn = eval_pinn_on_grid(t_plot, S_plot, x_plot)

    # 3) FDM using FDMValidator
    t_grid, S_grid_full, x_grid, u_fdm, Xb_fdm, Xs_fdm = fdm_solver()
    mask_S = (S_grid_full >= S_MIN) & (S_grid_full <= S_MAX)
    S_grid = S_grid_full[mask_S]
    u_fdm_cropped = u_fdm[:, mask_S, :]
    Xb_fdm = Xb_fdm[:, mask_S]
    Xs_fdm = Xs_fdm[:, mask_S]

    # 4) Train FNO as PINO
    if FNO_EPOCHS > 0:
        train_fno_pino()
        H_fno, u_fno_surfs = eval_fno_on_grid(t_plot, S_plot, x_plot)
        Xb_fno, Xs_fno = eval_fno_boundaries(t_plot, S_plot)
    else:
        H_fno = {}
        Xb_fno = np.zeros_like(Xb_pinn)
        Xs_fno = np.zeros_like(Xs_pinn)

    # 5) Plot 3×3 comparison: PINN / FDM / FNO
    fig = plt.figure(figsize=(24, 12))

    # Row 1: H(t=0,S,x) with SHARED z-scale for PINN/FDM/FNO
    mask_x = (x_grid >= x_MIN) & (x_grid <= x_MAX)
    x_grid_plot = x_grid[mask_x]
    u_fdm0_plot = u_fdm_cropped[0][:, mask_x]
    H_fdm0 = np.exp(np.clip(u_fdm0_plot, U_CLAMP_MIN, U_CLAMP_MAX))

    Sg_pin, Xg_pin = np.meshgrid(S_plot, x_plot, indexing='ij')

    # Determine global z-limits from PINN + FNO (and FDM for safety if available)
    if 0.0 in H_fno:
        hmin0 = min(np.min(H_pinn[0.0]), np.min(H_fno[0.0]), np.min(H_fdm0))
        hmax0 = max(np.max(H_pinn[0.0]), np.max(H_fno[0.0]), np.max(H_fdm0))
    else:
        hmin0 = min(np.min(H_pinn[0.0]), np.min(H_fdm0))
        hmax0 = max(np.max(H_pinn[0.0]), np.max(H_fdm0))

    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    ax1.plot_surface(Sg_pin, Xg_pin, H_pinn[0.0], cmap=cm.viridis, alpha=0.9)
    ax1.set_title("PINN: H(t=0)")
    ax1.set_xlabel("S"); ax1.set_ylabel("x")
    ax1.set_zlim(hmin0, hmax0)

    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    Sg_fdm, Xg_fdm = np.meshgrid(S_grid, x_grid_plot, indexing='ij')
    ax2.plot_surface(Sg_fdm, Xg_fdm, H_fdm0, cmap=cm.plasma, alpha=0.9)
    ax2.set_title("FDM: H(t=0)")
    ax2.set_xlabel("S"); ax2.set_ylabel("x")
    ax2.set_zlim(hmin0, hmax0)

    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    if 0.0 in H_fno:
        ax3.plot_surface(Sg_pin, Xg_pin, H_fno[0.0], cmap=cm.cividis, alpha=0.9)
    ax3.set_title("FNO/PINO: H(t=0)")
    ax3.set_xlabel("S"); ax3.set_ylabel("x")
    ax3.set_zlim(hmin0, hmax0)

    # Row 2: Buy boundaries
    Tpin, Spin = np.meshgrid(t_plot, S_plot, indexing='ij')
    idxs = [np.argmin(abs(t_grid - tt)) for tt in t_plot]
    Tf, Sf = np.meshgrid(t_grid[idxs], S_grid, indexing='ij')

    ax4 = fig.add_subplot(3, 3, 4, projection='3d')
    ax4.plot_surface(Spin, Tpin, Xb_pinn, color='tab:blue', alpha=0.8)
    ax4.set_title("PINN: Buy Xb")
    ax4.set_xlabel("S"); ax4.set_ylabel("t")

    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    ax5.plot_surface(Sf, Tf, Xb_fdm[idxs,:], color='tab:cyan', alpha=0.8)
    ax5.set_title("FDM: Buy Xb")
    ax5.set_xlabel("S"); ax5.set_ylabel("t")

    ax6 = fig.add_subplot(3, 3, 6, projection='3d')
    ax6.plot_surface(Spin, Tpin, Xb_fno, color='lightblue', alpha=0.8)
    ax6.set_title("FNO/PINO: Buy Xb")
    ax6.set_xlabel("S"); ax6.set_ylabel("t")

    # Row 3: Sell boundaries
    ax7 = fig.add_subplot(3, 3, 7, projection='3d')
    ax7.plot_surface(Spin, Tpin, Xs_pinn, color='tab:red', alpha=0.8)
    ax7.set_title("PINN: Sell Xs")
    ax7.set_xlabel("S"); ax7.set_ylabel("t")

    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    ax8.plot_surface(Sf, Tf, Xs_fdm[idxs,:], color='tab:orange', alpha=0.8)
    ax8.set_title("FDM: Sell Xs")
    ax8.set_xlabel("S"); ax8.set_ylabel("t")

    ax9 = fig.add_subplot(3, 3, 9, projection='3d')
    ax9.plot_surface(Spin, Tpin, Xs_fno, color='salmon', alpha=0.8)
    ax9.set_title("FNO/PINO: Sell Xs")
    ax9.set_xlabel("S"); ax9.set_ylabel("t")

    plt.tight_layout()
    plt.show()

    print(f"Total runtime: {time.time()-start:.1f} seconds")
