"""Temporal consistency for annual predictions: a per-pixel robust Kalman smoother.

Each pixel's annual predictions z_y (with calibrated SD s_y) are treated as noisy
observations of a latent height h_y that follows a local linear trend:

    h_{y+1} = h_y + g_y + w_h,   g_{y+1} = g_y + w_g,   z_y = h_y + v_y

with w_h ~ N(0, q_h), w_g ~ N(0, q_g), v_y ~ N(0, r * s_y^2). Disturbances (clearing,
fire) are abrupt drops, which a Gaussian random walk would smear over several years.
The filter is therefore robust: when the standardized innovation of a *decrease*
exceeds `jump_z`, the level's process variance for that step is inflated by
`jump_var`, so the state can jump. The filter is followed by a Rauch-Tung-Striebel
backward pass.

All operations are vectorized over pixels (arrays shaped (n_pixels, n_years)), so the
smoother runs over whole map tiles on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmootherParams:
    q_h: float = 0.5 ** 2      # level process variance per year (m^2)
    q_g: float = 0.15 ** 2     # growth-rate process variance per year ((m/yr)^2)
    r_scale: float = 1.0       # multiplier on observation variances
    jump_z: float = 2.5        # standardized negative innovation that triggers a jump
    jump_var: float = 15.0 ** 2  # extra level variance for a jump step
    g0_sd: float = 1.0         # prior SD of the initial growth rate (m/yr)
    h_max: float = 60.0


def smooth(z: np.ndarray, s: np.ndarray, p: SmootherParams = SmootherParams(),
           return_cov: bool = False):
    """Smooth annual series.

    z, s: (n, T) predictions and their SDs (NaN = missing year).
    Returns smoothed heights (n, T), their SDs, the smoothed growth rate (n, T) and,
    optionally, the lag-one covariance needed for change SDs.
    """
    z = np.asarray(z, float)
    s = np.asarray(s, float)
    n, T = z.shape
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[p.q_h, 0.0], [0.0, p.q_g]])
    # Filter storage.
    xf = np.zeros((n, T, 2))
    Pf = np.zeros((n, T, 2, 2))
    xp = np.zeros((n, T, 2))
    Pp = np.zeros((n, T, 2, 2))
    # Initial state from the first available observation.
    first = np.where(np.isfinite(z), z, np.nan)
    h0 = np.nan_to_num(first[np.arange(n), np.argmax(np.isfinite(first), axis=1)], nan=10.0)
    x = np.stack([h0, np.zeros(n)], axis=1)
    P = np.zeros((n, 2, 2))
    P[:, 0, 0] = 20.0 ** 2
    P[:, 1, 1] = p.g0_sd ** 2
    for t in range(T):
        if t > 0:
            x = x @ F.T
            P = F @ P @ F.T + Q
        xp[:, t], Pp[:, t] = x, P
        obs = np.isfinite(z[:, t])
        if obs.any():
            R = p.r_scale * np.nan_to_num(s[:, t], nan=1.0) ** 2
            innov = z[:, t] - x[:, 0]
            S = P[:, 0, 0] + R
            zstd = innov / np.sqrt(S)
            jump = obs & (zstd < -p.jump_z) & (t > 0)
            if jump.any():
                P[jump, 0, 0] += p.jump_var
                P[jump, 1, 1] += p.g0_sd ** 2  # regrowth rate after a disturbance is new
                xp[jump, t] = x[jump]
                Pp[jump, t] = P[jump]
                S = P[:, 0, 0] + R
            K = P[:, :, 0] / S[:, None]
            upd = obs[:, None]
            x = np.where(upd, x + K * innov[:, None], x)
            KH = np.zeros_like(P)
            KH[:, :, 0] = K
            P = np.where(upd[:, :, None], P - KH @ P, P)
        x[:, 0] = np.clip(x[:, 0], 0, p.h_max)
        xf[:, t], Pf[:, t] = x, P
    # RTS smoother.
    xs = xf.copy()
    Ps = Pf.copy()
    lag = np.zeros((n, T, 2, 2))
    for t in range(T - 2, -1, -1):
        Ppn_inv = np.linalg.inv(Pp[:, t + 1] + 1e-9 * np.eye(2))
        C = Pf[:, t] @ F.T @ Ppn_inv
        xs[:, t] = xf[:, t] + (C @ (xs[:, t + 1] - xp[:, t + 1])[..., None])[..., 0]
        Ps[:, t] = Pf[:, t] + C @ (Ps[:, t + 1] - Pp[:, t + 1]) @ np.swapaxes(C, 1, 2)
        lag[:, t + 1] = Ps[:, t + 1] @ np.swapaxes(C, 1, 2)  # Cov(x_{t+1}, x_t)
    h = np.clip(xs[:, :, 0], 0, p.h_max)
    sd = np.sqrt(np.maximum(Ps[:, :, 0, 0], 0))
    g = xs[:, :, 1]
    if return_cov:
        return h, sd, g, Ps, lag
    return h, sd, g


def change_sd(Ps: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """SD of h_t - h_{t-1} from smoothed marginal and lag-one covariances (n, T)."""
    var = np.full(Ps.shape[:2], np.nan)
    var[:, 1:] = (Ps[:, 1:, 0, 0] + Ps[:, :-1, 0, 0] - 2 * lag[:, 1:, 0, 0])
    return np.sqrt(np.maximum(var, 0))
