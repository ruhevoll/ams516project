#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDM + PINN + PFNO (physics-informed FNO) solver for utility-indifference H(t,S,x)
with proportional transaction costs.

Features:
- FDM reference solution (t,S,x) -> H, plus X_b(t,S), X_s(t,S) via gradients
- PINN (PyTorch) solving PDE with free-boundary gradient constraints, plus
  X_b(t,S), X_s(t,S)
- PFNO (Fourier Neural Operator) trained ONLY on PDE residual + terminal condition
  (no supervision from FDM or PINN)
- 3×3 grid of:
    rows: FDM, PINN, PFNO
    cols: H(t=0,S,x), X_b(t,S), X_s(t,S)
- GIF animation of H(t,S,x) surfaces from t=0..1 (step 0.1) showing side-by-side
  FDM / PINN / PFNO.

NOTE:
- This is research-grade code. Hyperparameters (epochs, widths, etc.) will likely
  need tuning for your use case and hardware.
- Uses double precision and clipping to mitigate exponential overflow.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import imageio

# ----------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Financial / model parameters
T = 1.0            # maturity
r = 0.05           # risk-free rate
alpha = 0.10       # stock drift
sigma = 0.2        # volatility
gamma = 0.5        # risk aversion

zeta = 0.01        # proportional cost (buy)
mu = 0.01          # proportional cost (sell)
a = 1.0 + zeta
b = 1.0 - mu

# Domain bounds – choose moderate ranges to avoid exponential overflow
S_min, S_max = 1e-2, 4.0
x_min, x_max = -2.0, 2.0

# Discretization for FDM
N_t = 40
N_S = 60
N_x = 41

# PFNO time grid (for physics-informed FNO)
N_t_pfno = 11  # 0, 0.1, ..., 1.0

# Training configs (tune as needed)
PINN_EPOCHS = 5000
PINN_LR = 1e-3
PFNO_EPOCHS = 4000
PFNO_LR = 1e-3

# GIF settings
GIF_FILENAME = "H_evolution_FDM_PINN_PFNO.gif"
GIF_FPS = 3

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def Z_payoff(S, x):
    """
    Terminal settlement value Z(S,x) for stock without option:
    Z(S,x) = x S (a I_{x<0} + b I_{x>=0})
    """
    S = np.asarray(S)
    x = np.asarray(x)
    res = np.where(x < 0.0, a * x * S, b * x * S)
    return res


def Z_payoff_torch(S, x):
    S = torch.as_tensor(S, dtype=torch.get_default_dtype(), device=device)
    x = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)
    return torch.where(x < 0.0, a * x * S, b * x * S)


def H_terminal_np(S_grid, x_grid):
    """
    H(T,S,x) = exp(-gamma * Z(S,x)), numpy version.
    """
    Z = Z_payoff(S_grid, x_grid)
    arg = -gamma * Z
    arg = np.clip(arg, -50.0, 50.0)
    return np.exp(arg)


def H_terminal_torch(S, x):
    Z = Z_payoff_torch(S, x)
    arg = -gamma * Z
    arg = torch.clamp(arg, -50.0, 50.0)
    return torch.exp(arg)


# ----------------------------------------------------------------------
# Black–Scholes analytical solution (no transaction costs) and binomial tree
# ----------------------------------------------------------------------

def black_scholes_call(S0, K, T, r, sigma):
    """
    Standard Black–Scholes European call (no transaction costs).
    """
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from math import erf

    def N(z):
        return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))

    return S0 * N(d1) - K * math.exp(-r * T) * N(d2)


def binomial_tree_call(S0, K, T, r, sigma, N=200):
    """
    Cox–Ross–Rubinstein binomial tree for European call (no transaction costs).
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    q = (math.exp(r * dt) - d) / (u - d)

    ST = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    C = np.maximum(ST - K, 0.0)

    disc = math.exp(-r * dt)
    for n in range(N - 1, -1, -1):
        C = disc * (q * C[1:n + 2] + (1 - q) * C[0:n + 1])

    return C[0]


# ----------------------------------------------------------------------
# Finite-Difference Method (FDM) for PDE in (t,S) for each x
# ----------------------------------------------------------------------

def build_S_grid():
    S_vals = np.linspace(S_min, S_max, N_S)
    dS = S_vals[1] - S_vals[0]
    return S_vals, dS


def build_x_grid():
    x_vals = np.linspace(x_min, x_max, N_x)
    dx = x_vals[1] - x_vals[0]
    return x_vals, dx


def build_t_grid():
    t_vals = np.linspace(0.0, T, N_t + 1)  # t_0=0, t_N=T
    dt = t_vals[1] - t_vals[0]
    return t_vals, dt


def fdm_solve():
    """
    FDM solver for the PDE in t,S for each x independently.
    Crank–Nicolson scheme in S, backward in time.
    """
    S_vals, dS = build_S_grid()
    x_vals, dx = build_x_grid()
    t_vals, dt = build_t_grid()

    H = np.zeros((N_t + 1, N_S, N_x), dtype=np.float64)

    S_grid, x_grid = np.meshgrid(S_vals, x_vals, indexing='ij')
    H[-1, :, :] = H_terminal_np(S_grid, x_grid)

    A_diag = np.zeros(N_S)
    A_upper = np.zeros(N_S - 1)
    A_lower = np.zeros(N_S - 1)

    for j in range(1, N_S - 1):
        S = S_vals[j]
        sigma2S2 = 0.5 * sigma ** 2 * S ** 2
        alphaS = alpha * S
        a_j = sigma2S2 / dS ** 2 - alphaS / (2 * dS)
        b_j = -2.0 * sigma2S2 / dS ** 2
        c_j = sigma2S2 / dS ** 2 + alphaS / (2 * dS)
        A_lower[j - 1] = a_j
        A_diag[j] = b_j
        A_upper[j] = c_j

    A_diag[0] = 1.0
    A_upper[0] = 0.0

    A_diag[-1] = 1.0
    A_lower[-1] = -1.0

    I = np.eye(N_S)
    A = np.zeros((N_S, N_S))
    for j in range(N_S):
        A[j, j] = A_diag[j]
        if j > 0:
            A[j, j - 1] = A_lower[j - 1]
        if j < N_S - 1:
            A[j, j + 1] = A_upper[j]

    M_left = I - 0.5 * dt * A
    M_right = I + 0.5 * dt * A

    from numpy.linalg import solve

    for ix, x_val in enumerate(x_vals):
        for n in range(N_t - 1, -1, -1):
            rhs = M_right @ H[n + 1, :, ix]
            rhs[0] = H_terminal_np(S_vals[0], x_val)
            H[n, :, ix] = solve(M_left, rhs)

    return H, t_vals, S_vals, x_vals


def extract_boundaries_from_H(H, t_vals, S_vals, x_vals):
    """
    Given H(t,S,x) on a grid, approximate X_b(t,S) and X_s(t,S) by locating
    zeros of the gradient constraint in x:
        buy:  dH/dx + a gamma S e^{r(T-t)} H = 0
        sell: dH/dx + b gamma S e^{r(T-t)} H = 0
    """
    dx = x_vals[1] - x_vals[0]
    Nt = len(t_vals)
    Ns = len(S_vals)

    Xb = np.full((Nt, Ns), np.nan)
    Xs = np.full((Nt, Ns), np.nan)

    for n, t in enumerate(t_vals):
        dfac = math.exp(r * (T - t))
        for j, S in enumerate(S_vals):
            H_line = H[n, j, :]
            dHdx = np.gradient(H_line, dx)

            buy_res = dHdx + a * gamma * S * dfac * H_line
            sell_res = dHdx + b * gamma * S * dfac * H_line

            # buy boundary
            for k in range(len(x_vals) - 1):
                if buy_res[k] * buy_res[k + 1] <= 0:
                    Xb[n, j] = x_vals[k]
                    break

            # sell boundary
            for k in range(len(x_vals) - 1):
                if sell_res[k] * sell_res[k + 1] <= 0:
                    Xs[n, j] = x_vals[k]
                    break

    return Xb, Xs


# ----------------------------------------------------------------------
# PINN for H(t,S,x) + boundary networks X_b(t,S), X_s(t,S)
# ----------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=64, depth=4, act=nn.Tanh):
        super().__init__()
        layers = []
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BoundaryNet(nn.Module):
    """
    Parameterization for buy and sell boundary:
        mid(t,S), hw(t,S) -> Xb = mid - softplus(hw), Xs = mid + softplus(hw)
    """
    def __init__(self, in_dim=2, width=64, depth=3, act=nn.Tanh):
        super().__init__()
        self.mid_net = MLP(in_dim, 1, width=width, depth=depth, act=act)
        self.hw_net = MLP(in_dim, 1, width=width, depth=depth, act=act)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, ts):
        mid = self.mid_net(ts)
        hw_raw = self.hw_net(ts)
        hw = self.softplus(hw_raw)
        Xb = mid - hw
        Xs = mid + hw
        return Xb, Xs


class PINN_H(nn.Module):
    def __init__(self, width=64, depth=5, act=nn.Tanh):
        super().__init__()
        self.mlp = MLP(3, 1, width=width, depth=depth, act=act)

    def forward(self, t, S, x):
        t_norm = 2.0 * t / T - 1.0
        S_norm = 2.0 * (S - S_min) / (S_max - S_min) - 1.0
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        inp = torch.stack([t_norm, S_norm, x_norm], dim=-1)
        return self.mlp(inp).squeeze(-1)


def pinn_loss(pinn, bnet, n_int=4000, n_bc=2000, n_term=2000):
    pinn.train()
    bnet.train()

    # Interior points
    t_int = torch.rand(n_int, device=device) * T
    S_int = torch.rand(n_int, device=device) * (S_max - S_min) + S_min
    x_int = torch.rand(n_int, device=device) * (x_max - x_min) + x_min

    t_int.requires_grad_(True)
    S_int.requires_grad_(True)
    x_int.requires_grad_(True)

    H_int = pinn(t_int, S_int, x_int)

    dH_dt = torch.autograd.grad(H_int, t_int,
                                grad_outputs=torch.ones_like(H_int),
                                retain_graph=True,
                                create_graph=True)[0]
    dH_dS = torch.autograd.grad(H_int, S_int,
                                grad_outputs=torch.ones_like(H_int),
                                retain_graph=True,
                                create_graph=True)[0]
    d2H_dS2 = torch.autograd.grad(dH_dS, S_int,
                                  grad_outputs=torch.ones_like(dH_dS),
                                  retain_graph=True,
                                  create_graph=True)[0]

    pde_res = dH_dt + alpha * S_int * dH_dS + 0.5 * sigma ** 2 * S_int ** 2 * d2H_dS2
    loss_pde = torch.mean(pde_res ** 2)

    # Boundary gradient conditions at x = Xb, Xs
    t_bc = torch.rand(n_bc, device=device) * T
    S_bc = torch.rand(n_bc, device=device) * (S_max - S_min) + S_min
    ts = torch.stack([t_bc, S_bc], dim=-1)
    Xb, Xs = bnet(ts)

    Xb = torch.clamp(Xb.squeeze(-1), x_min, x_max)
    Xs = torch.clamp(Xs.squeeze(-1), x_min, x_max)

    # buy boundary
    t_b = t_bc.clone().detach().requires_grad_(True)
    S_b = S_bc.clone().detach().requires_grad_(True)
    x_b = Xb.clone().detach().requires_grad_(True)
    H_b = pinn(t_b, S_b, x_b)
    dH_dx_b = torch.autograd.grad(H_b, x_b,
                                  grad_outputs=torch.ones_like(H_b),
                                  retain_graph=True,
                                  create_graph=True)[0]
    dfac_b = torch.exp(r * (T - t_b))
    buy_res = dH_dx_b + a * gamma * S_b * dfac_b * H_b
    loss_buy = torch.mean(buy_res ** 2)

    # sell boundary
    t_s = t_bc.clone().detach().requires_grad_(True)
    S_s = S_bc.clone().detach().requires_grad_(True)
    x_s = Xs.clone().detach().requires_grad_(True)
    H_s = pinn(t_s, S_s, x_s)
    dH_dx_s = torch.autograd.grad(H_s, x_s,
                                  grad_outputs=torch.ones_like(H_s),
                                  retain_graph=True,
                                  create_graph=True)[0]
    dfac_s = torch.exp(r * (T - t_s))
    sell_res = dH_dx_s + b * gamma * S_s * dfac_s * H_s
    loss_sell = torch.mean(sell_res ** 2)

    # Terminal condition
    S_T = torch.rand(n_term, device=device) * (S_max - S_min) + S_min
    x_T = torch.rand(n_term, device=device) * (x_max - x_min) + x_min
    t_T = torch.full_like(S_T, T, requires_grad=False)
    H_T_pred = pinn(t_T, S_T, x_T)
    H_T_true = H_terminal_torch(S_T, x_T)
    loss_term = torch.mean((H_T_pred - H_T_true) ** 2)

    # S-boundary regularization (optional)
    n_sb = n_bc
    t_sb = torch.rand(n_sb, device=device) * T
    x_sb = torch.rand(n_sb, device=device) * (x_max - x_min) + x_min
    S_sb_min = torch.full_like(x_sb, S_min)
    S_sb_max = torch.full_like(x_sb, S_max)
    H_sb_min = pinn(t_sb, S_sb_min, x_sb)
    H_sb_max = pinn(t_sb, S_sb_max, x_sb)
    H_sb_min_target = H_terminal_torch(S_sb_min, x_sb)
    H_sb_max_target = H_terminal_torch(S_sb_max, x_sb)
    loss_sbd = torch.mean((H_sb_min - H_sb_min_target) ** 2) + \
               torch.mean((H_sb_max - H_sb_max_target) ** 2)

    loss = loss_pde + loss_term + 0.1 * (loss_buy + loss_sell) + 0.01 * loss_sbd
    return loss


def train_pinn():
    pinn = PINN_H().to(device)
    bnet = BoundaryNet().to(device)

    params = list(pinn.parameters()) + list(bnet.parameters())
    optimizer = torch.optim.Adam(params, lr=PINN_LR)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=500)

    for epoch in range(1, PINN_EPOCHS + 1):
        optimizer.zero_grad()
        loss = pinn_loss(pinn, bnet)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if epoch % 500 == 0:
            print(f"[PINN] Epoch {epoch:5d} | loss={loss.item():.3e}")

    return pinn, bnet


def evaluate_pinn_on_grid(pinn, bnet, t_eval=0.0, N_S_plot=60, N_x_plot=41):
    pinn.eval()
    bnet.eval()

    S_vals = np.linspace(S_min, S_max, N_S_plot)
    x_vals = np.linspace(x_min, x_max, N_x_plot)
    S_grid, x_grid = np.meshgrid(S_vals, x_vals, indexing='ij')

    S_t = torch.as_tensor(S_grid, device=device)
    x_t = torch.as_tensor(x_grid, device=device)
    t_t = torch.full_like(S_t, t_eval, device=device)

    with torch.no_grad():
        H_grid = pinn(t_t.flatten(), S_t.flatten(), x_t.flatten())
    H_grid = H_grid.view_as(S_t).cpu().numpy()

    # boundaries vs S at this fixed t
    ts = torch.tensor([[t_eval, s] for s in S_vals],
                      dtype=torch.get_default_dtype(), device=device)
    with torch.no_grad():
        Xb_line, Xs_line = bnet(ts)
    Xb_line = Xb_line.squeeze(-1).cpu().numpy()
    Xs_line = Xs_line.squeeze(-1).cpu().numpy()

    return S_vals, x_vals, H_grid, Xb_line, Xs_line


def evaluate_pinn_boundaries_on_grid(bnet, N_t_plot=20, N_S_plot=50):
    """
    Evaluate PINN-predicted X_b(t,S), X_s(t,S) over a (t,S) grid.
    """
    bnet.eval()
    t_vals = np.linspace(0.0, T, N_t_plot)
    S_vals = np.linspace(S_min, S_max, N_S_plot)
    Xb_grid = np.zeros((N_t_plot, N_S_plot))
    Xs_grid = np.zeros((N_t_plot, N_S_plot))

    with torch.no_grad():
        for i, t in enumerate(t_vals):
            ts = torch.tensor([[t, s] for s in S_vals],
                              dtype=torch.get_default_dtype(), device=device)
            Xb, Xs = bnet(ts)
            Xb_grid[i, :] = torch.clamp(Xb.squeeze(-1), x_min, x_max).cpu().numpy()
            Xs_grid[i, :] = torch.clamp(Xs.squeeze(-1), x_min, x_max).cpu().numpy()

    return t_vals, S_vals, Xb_grid, Xs_grid


# ----------------------------------------------------------------------
# PFNO (Physics-Informed Fourier Neural Operator)
# ----------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """
    Basic 2D Fourier layer for FNO: in R^{batch x C x H x W}.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                     dtype=torch.get_default_dtype())
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_c, H, W), (in_c, out_c, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        x: (batch, in_c, H, W), real-valued
        """
        batchsize = x.shape[0]

        # FFT
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.complex128, device=x.device
        )

        # Cast weights to complex before multiplication
        weights_complex = self.weights1.to(x_ft.dtype)

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], weights_complex
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


class PFNO2d(nn.Module):
    """
    2D FNO over (S,x); t enters as an additional input channel.
    Input: coords (batch, H, W, 3) where channels = (t_norm, S_norm, x_norm).
    Output: H field (batch, H, W).
    """
    def __init__(self, modes1=20, modes2=20, width=32):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(3, self.width)  # input: (t,S,x) coords

        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.GELU()

    def forward(self, coords):
        """
        coords: (batch, H, W, 3) with normalized (t,S,x).
        For PFNO we typically use batch = N_t_pfno, H=N_S, W=N_x.
        """
        batchsize, H, W, _ = coords.shape
        x = self.fc0(coords)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.act(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.act(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.act(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.act(x1 + x2)

        x = x.permute(0, 2, 3, 1)  # (batch, H, W, width)
        x = self.act(self.fc1(x))
        x = self.fc2(x).squeeze(-1)  # (batch, H, W)
        return x


def build_pfno_grid():
    """
    Builds fixed (t,S,x) grid for PFNO training:
    - time: N_t_pfno points in [0,T]
    - S: use same S grid as FDM
    - x: use same x grid as FDM
    Returns:
        coords (torch): (N_t_pfno, N_S, N_x, 3) normalized
        t_vals_pfno (np array)
        S_vals (np array)
        x_vals (np array)
        times_t (torch), S_t (torch), x_t (torch)
    """
    t_vals_pfno = np.linspace(0.0, T, N_t_pfno)
    S_vals, _ = build_S_grid()
    x_vals, _ = build_x_grid()

    times_t = torch.tensor(t_vals_pfno, device=device)
    S_t = torch.tensor(S_vals, device=device)
    x_t = torch.tensor(x_vals, device=device)

    t_grid, S_grid, x_grid = torch.meshgrid(times_t, S_t, x_t, indexing='ij')

    t_norm = 2.0 * t_grid / T - 1.0
    S_norm = 2.0 * (S_grid - S_min) / (S_max - S_min) - 1.0
    x_norm = 2.0 * (x_grid - x_min) / (x_max - x_min) - 1.0

    coords = torch.stack([t_norm, S_norm, x_norm], dim=-1)  # (Nt, Ns, Nx, 3)
    return coords, t_vals_pfno, S_vals, x_vals, times_t, S_t, x_t


def pfno_loss(fno, coords, times_t, S_t, x_t):
    """
    Physics-informed loss for FNO:
      - PDE residual via finite differences in (t,S)
      - Terminal condition at t=T
    """
    fno.train()
    H_pred = fno(coords)  # (Nt, Ns, Nx)

    Nt, Ns, Nx = H_pred.shape
    dt = times_t[1] - times_t[0]
    dS = S_t[1] - S_t[0]

    # Finite differences for PDE residual:
    # interior indices: time 1..Nt-2, S 1..Ns-2
    H = H_pred

    # time derivative (central)
    H_t = (H[2:, :, :] - H[:-2, :, :]) / (2.0 * dt)  # (Nt-2, Ns, Nx)

    # S derivatives (central)
    H_S = (H[1:-1, 2:, :] - H[1:-1, :-2, :]) / (2.0 * dS)           # (Nt-2, Ns-2, Nx)
    H_SS = (H[1:-1, 2:, :] - 2.0 * H[1:-1, 1:-1, :] + H[1:-1, :-2, :]) / (dS ** 2)

    # S grid mid
    S_mid = S_t[1:-1].view(1, -1, 1)  # (1, Ns-2, 1)
    # Align time dimension: 1..Nt-2
    H_t_mid = H_t[:, 1:-1, :]  # (Nt-2, Ns-2, Nx)

    # PDE residual
    pde_res = H_t_mid + alpha * S_mid * H_S + 0.5 * sigma ** 2 * S_mid ** 2 * H_SS
    loss_pde = torch.mean(pde_res ** 2)

    # Terminal condition at last time index (t = T)
    H_T = H[-1, :, :]  # (Ns, Nx)
    S_grid_T, x_grid_T = torch.meshgrid(S_t, x_t, indexing='ij')
    H_T_true = H_terminal_torch(S_grid_T, x_grid_T)
    loss_term = torch.mean((H_T - H_T_true) ** 2)

    loss = loss_pde + loss_term
    return loss, H_pred


def train_pfno():
    fno = PFNO2d().to(device)
    coords, t_vals_pfno, S_vals, x_vals, times_t, S_t, x_t = build_pfno_grid()

    optimizer = torch.optim.Adam(fno.parameters(), lr=PFNO_LR)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=300)

    coords = coords  # (Nt, Ns, Nx, 3), on device

    for epoch in range(1, PFNO_EPOCHS + 1):
        optimizer.zero_grad()
        loss, H_pred = pfno_loss(fno, coords, times_t, S_t, x_t)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        if epoch % 500 == 0:
            print(f"[PFNO] Epoch {epoch:5d} | loss={loss.item():.3e}")

    # Final prediction after training
    with torch.no_grad():
        H_final = fno(coords).cpu().numpy()
    return fno, H_final, t_vals_pfno, S_vals, x_vals


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

def plot_surface_on_ax(ax, X_vals, Y_vals, Z, title, xlabel, ylabel, zlabel):
    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals, indexing='ij')
    ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)


def plot_boundary_surface(t_vals, S_vals, Xb, title):
    T_grid, S_grid = np.meshgrid(t_vals, S_vals, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_grid, S_grid, Xb, cmap='plasma')
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("S")
    ax.set_zlabel("x")
    plt.tight_layout()
    plt.show()


def make_3x3_grid(H0_fdm, Xb_fdm, Xs_fdm,
                  H0_pinn, Xb_pinn_3d, Xs_pinn_3d,
                  H0_pfno, Xb_pfno, Xs_pfno,
                  t_vals_fdm, S_vals_fdm, x_vals_fdm,
                  t_vals_pinn, S_vals_pinn,
                  t_vals_pfno, S_vals_pfno):
    """
    3x3 grid:
      Row 1: FDM   H0, Xb, Xs
      Row 2: PINN  H0, Xb, Xs
      Row 3: PFNO  H0, Xb, Xs
    """
    fig = plt.figure(figsize=(18, 12))

    # Row 1: FDM
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    plot_surface_on_ax(ax1, S_vals_fdm, x_vals_fdm, H0_fdm,
                       "FDM H(t=0,S,x)", "S", "x", "H")

    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    plot_surface_on_ax(ax2, t_vals_fdm, S_vals_fdm, Xb_fdm,
                       "FDM X_b(t,S)", "t", "S", "x")

    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    plot_surface_on_ax(ax3, t_vals_fdm, S_vals_fdm, Xs_fdm,
                       "FDM X_s(t,S)", "t", "S", "x")

    # Row 2: PINN
    ax4 = fig.add_subplot(3, 3, 4, projection='3d')
    plot_surface_on_ax(ax4, S_vals_fdm, x_vals_fdm, H0_pinn,
                       "PINN H(t=0,S,x)", "S", "x", "H")

    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    plot_surface_on_ax(ax5, t_vals_pinn, S_vals_pinn, Xb_pinn_3d,
                       "PINN X_b(t,S)", "t", "S", "x")

    ax6 = fig.add_subplot(3, 3, 6, projection='3d')
    plot_surface_on_ax(ax6, t_vals_pinn, S_vals_pinn, Xs_pinn_3d,
                       "PINN X_s(t,S)", "t", "S", "x")

    # Row 3: PFNO
    ax7 = fig.add_subplot(3, 3, 7, projection='3d')
    plot_surface_on_ax(ax7, S_vals_fdm, x_vals_fdm, H0_pfno,
                       "PFNO H(t=0,S,x)", "S", "x", "H")

    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    plot_surface_on_ax(ax8, t_vals_pfno, S_vals_pfno, Xb_pfno,
                       "PFNO X_b(t,S)", "t", "S", "x")

    ax9 = fig.add_subplot(3, 3, 9, projection='3d')
    plot_surface_on_ax(ax9, t_vals_pfno, S_vals_pfno, Xs_pfno,
                       "PFNO X_s(t,S)", "t", "S", "x")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# GIF animation
# ----------------------------------------------------------------------

def make_animation(H_fdm, t_vals_fdm, S_vals_fdm, x_vals_fdm,
                   pinn, bnet,
                   H_pfno, t_vals_pfno,
                   filename=GIF_FILENAME, fps=GIF_FPS):
    """
    Create a GIF where each frame shows THREE side-by-side 3D surfaces
    of H(t,S,x):

        [ FDM ]   [ PINN ]   [ PFNO ]

    for t going from 0 to 1 in steps of 0.1.
    All three use the same (S,x) grid defined by S_vals_fdm, x_vals_fdm.
    """
    frames = []

    # We’ll use the PFNO time grid (t_vals_pfno), which in this setup is 0, 0.1, ..., 1
    times_anim = t_vals_pfno

    dt_fdm = t_vals_fdm[1] - t_vals_fdm[0]

    for idx_t, t in enumerate(times_anim):
        # --- FDM: pick nearest time index ---
        n_fdm = int(round(t / dt_fdm))
        n_fdm = max(0, min(n_fdm, len(t_vals_fdm) - 1))
        H_fdm_t = H_fdm[n_fdm, :, :]  # (N_S, N_x)

        # --- PINN: evaluate on same (S,x) grid ---
        S_grid, x_grid = np.meshgrid(S_vals_fdm, x_vals_fdm, indexing='ij')
        S_t = torch.as_tensor(S_grid, dtype=torch.get_default_dtype(), device=device)
        x_t = torch.as_tensor(x_grid, dtype=torch.get_default_dtype(), device=device)
        t_t = torch.full_like(S_t, t, device=device)
        with torch.no_grad():
            H_pinn_flat = pinn(t_t.flatten(), S_t.flatten(), x_t.flatten())
        H_pinn_t = H_pinn_flat.view_as(S_t).cpu().numpy()

        # --- PFNO: H_pfno already on (t_idx, S, x) grid ---
        # H_pfno has shape (N_t_pfno, N_S, N_x)
        H_pfno_t = H_pfno[idx_t, :, :]

        # --- Build figure with 3 side-by-side subplots ---
        fig = plt.figure(figsize=(18, 5))

        # Subplot 1: FDM
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        plot_surface_on_ax(ax1, S_vals_fdm, x_vals_fdm, H_fdm_t,
                           f"FDM H(t={t:.2f})", "S", "x", "H")

        # Subplot 2: PINN
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        plot_surface_on_ax(ax2, S_vals_fdm, x_vals_fdm, H_pinn_t,
                           f"PINN H(t={t:.2f})", "S", "x", "H")

        # Subplot 3: PFNO
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        plot_surface_on_ax(ax3, S_vals_fdm, x_vals_fdm, H_pfno_t,
                           f"PFNO H(t={t:.2f})", "S", "x", "H")

        plt.tight_layout()

        # --- Render to an image for the GIF ---
        fig.canvas.draw()

        try:
            # Works for Agg and some other backends
            buf = fig.canvas.tostring_rgb()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except Exception:
            # Fallback for Qt backends that only have tostring_argb
            buf = fig.canvas.tostring_argb()
            arr = np.frombuffer(buf, dtype=np.uint8)
            arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = arr[:, :, 1:4]  # drop alpha (ARGB -> RGB)

        frames.append(image)
        plt.close(fig)

    if not frames:
        raise RuntimeError("No frames were generated for the animation.")

    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved as {filename}")




# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------

def main():
    # ---------------- FDM ----------------
    print("=== FDM solver (reference) ===")
    H_fdm, t_vals, S_vals_fdm, x_vals_fdm = fdm_solve()
    Xb_fdm, Xs_fdm = extract_boundaries_from_H(H_fdm, t_vals, S_vals_fdm, x_vals_fdm)
    H0_fdm = H_fdm[0, :, :]

    # ---------------- PINN ----------------
    print("=== PINN training ===")
    pinn, bnet = train_pinn()
    S_vals_pinn0, x_vals_pinn0, H0_pinn, _, _ = evaluate_pinn_on_grid(
        pinn, bnet, t_eval=0.0,
        N_S_plot=len(S_vals_fdm),
        N_x_plot=len(x_vals_fdm)
    )
    t_vals_pinn, S_vals_pinn, Xb_pinn_3d, Xs_pinn_3d = evaluate_pinn_boundaries_on_grid(
        bnet, N_t_plot=20, N_S_plot=50
    )

    # ---------------- PFNO (physics-informed FNO) ----------------
    print("=== PFNO training (physics-informed FNO) ===")
    pfno, H_pfno, t_vals_pfno, S_vals_pfno, x_vals_pfno = train_pfno()
    # H_pfno shape: (N_t_pfno, N_S, N_x)
    H0_pfno = H_pfno[0, :, :]
    Xb_pfno, Xs_pfno = extract_boundaries_from_H(H_pfno, t_vals_pfno, S_vals_pfno, x_vals_pfno)

    # ---------------- 3x3 grid ----------------
    print("=== Producing 3×3 comparison grid ===")
    make_3x3_grid(H0_fdm, Xb_fdm, Xs_fdm,
                  H0_pinn, Xb_pinn_3d, Xs_pinn_3d,
                  H0_pfno, Xb_pfno, Xs_pfno,
                  t_vals, S_vals_fdm, x_vals_fdm,
                  t_vals_pinn, S_vals_pinn,
                  t_vals_pfno, S_vals_pfno)

    # ---------------- GIF animation ----------------
    print("=== Producing GIF animation of H(t,S,x) for FDM, PINN, PFNO ===")
    make_animation(H_fdm, t_vals, S_vals_fdm, x_vals_fdm,
                   pinn, bnet,
                   H_pfno, t_vals_pfno,
                   filename=GIF_FILENAME, fps=GIF_FPS)

    # ---------------- Black–Scholes & Binomial Check ----------------
    print("=== Black–Scholes / Binomial check (frictionless reference) ===")
    S0 = 1.0
    K = 1.0
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    bt_price = binomial_tree_call(S0, K, T, r, sigma, N=300)
    print(f"Black–Scholes call (S0=K=1): {bs_price:.6f}")
    print(f"Binomial tree call       : {bt_price:.6f}")

    print("Done. Adjust epochs / widths / grids as needed for accuracy and speed.")


if __name__ == "__main__":
    main()
