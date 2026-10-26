"""Vectorized per-pixel Kalman filter / RTS smoother for monthly canopy height.

State (per pixel), with dt the step length in years:
    x = [h, g, b_opt, b_sar]
    h      canopy height (m)
    g      growth rate (m/yr)
    b_opt  bias of the optical (HLS) observation operator at this pixel (m)
    b_sar  bias of the radar (Sentinel-1) observation operator at this pixel (m)

Dynamics: h' = h + dt*g + w_h,  g' = g + w_g,  b' = b + w_b, with variances scaled by dt.
Observation types, each a scalar update y = H x + v:
    lidar  (GEDI / ICESat-2, calibrated):  H = [1, 0, 0, 0], var = sigma_label^2
    annual (v1 AlphaEarth model):          H = [1, 0, 0, 0], var = model error variance
    optical (HLS monthly regressor):       H = [1, 0, 1, 0], var = regressor residual var
    radar   (S1 monthly regressor):        H = [1, 0, 0, 1], var = regressor residual var
The bias states absorb the strong month-to-month error correlation of the data-driven
operators, which the filter would otherwise treat as independent and over-weight.
Disturbances: if a lidar/annual/optical innovation is a large *decrease* (z < -jump_z),
the height variance is inflated before the update so the state can jump.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NX = 4
H_ROWS = {
    "lidar": np.array([1.0, 0, 0, 0]),
    "annual": np.array([1.0, 0, 0, 0]),
    "optical": np.array([1.0, 0, 1.0, 0]),
    "radar": np.array([1.0, 0, 0, 1.0]),
}
JUMP_TYPES = ("lidar", "annual", "optical")


@dataclass
class SSMParams:
    q_h: float = 0.4 ** 2          # m^2 / yr
    q_g: float = 0.3 ** 2          # (m/yr)^2 / yr
    q_b: float = 0.3 ** 2          # m^2 / yr (operator-bias drift)
    b0_sd: dict = field(default_factory=lambda: {"optical": 5.0, "radar": 6.0})
    h0_sd: float = 15.0
    g0_sd: float = 1.0
    jump_z: float = 2.5
    jump_var: float = 15.0 ** 2
    r_scale: dict = field(default_factory=lambda: {"lidar": 1.0, "annual": 1.0,
                                                   "optical": 1.0, "radar": 1.0})
    h_max: float = 60.0
    jump_types: tuple = JUMP_TYPES

    @classmethod
    def from_config(cls) -> "SSMParams":
        from ..config import load_config

        c = load_config().get("ssm", {})
        ro = c.get("r_opt", 1.0)
        bsd = c.get("bias_prior_sd", 1.0)
        return cls(q_h=c.get("q_h", 0.16), jump_z=c.get("jump_z", 2.5),
                   jump_types=tuple(c.get("jump_types", JUMP_TYPES)),
                   b0_sd={"optical": bsd, "radar": bsd},
                   r_scale={"lidar": 1.0, "annual": 1.0, "optical": ro, "radar": ro})


def run(obs: dict, T: int, dt: float, p: SSMParams, h0=None, return_cov: bool = False):
    """Filter + smooth.

    obs: {type: (values (n, T), variances (n, T))} with NaN where missing.
    Returns smoothed state means (n, T, 4) and covariances (n, T, 4, 4) [+ lag-one cov].
    """
    n = next(iter(obs.values()))[0].shape[0]
    F = np.eye(NX)
    F[0, 1] = dt
    Q = np.diag([p.q_h * dt, p.q_g * dt, p.q_b * dt, p.q_b * dt])
    x = np.zeros((n, NX))
    if h0 is None:
        a = obs.get("annual", obs.get("lidar"))[0]
        h0 = np.nanmean(a, axis=1) if a is not None else np.full(n, 10.0)
        h0 = np.nan_to_num(h0, nan=10.0)
    x[:, 0] = h0
    P = np.zeros((n, NX, NX))
    P[:, 0, 0] = p.h0_sd ** 2
    P[:, 1, 1] = p.g0_sd ** 2
    P[:, 2, 2] = p.b0_sd.get("optical", 5.0) ** 2
    P[:, 3, 3] = p.b0_sd.get("radar", 5.0) ** 2
    xf = np.zeros((n, T, NX))
    Pf = np.zeros((n, T, NX, NX))
    xp = np.zeros((n, T, NX))
    Pp = np.zeros((n, T, NX, NX))
    order = [k for k in ("annual", "optical", "radar", "lidar") if k in obs]
    for t in range(T):
        if t > 0:
            x = x @ F.T
            P = F @ P @ F.T + Q
        jumped = np.zeros(n, bool)
        xp_t, Pp_t = x.copy(), P.copy()
        for k in order:
            yv, rv = obs[k][0][:, t], obs[k][1][:, t]
            m = np.isfinite(yv) & np.isfinite(rv)
            if not m.any():
                continue
            Hr = H_ROWS[k]
            R = rv * p.r_scale.get(k, 1.0)
            pred = x @ Hr
            PH = P @ Hr                       # (n, 4)
            S = PH @ Hr + R
            innov = yv - pred
            if k in p.jump_types and t > 0:
                jz = m & ~jumped & (innov / np.sqrt(S) < -p.jump_z)
                if jz.any():
                    P[jz, 0, 0] += p.jump_var
                    P[jz, 1, 1] += p.g0_sd ** 2
                    Pp_t[jz] = P[jz]
                    jumped |= jz
                    PH = P @ Hr
                    S = PH @ Hr + R
            K = PH / S[:, None]
            K[~m] = 0
            innov = np.where(m, innov, 0)
            x = x + K * innov[:, None]
            P = P - K[:, :, None] * PH[:, None, :]
        x[:, 0] = np.clip(x[:, 0], 0, p.h_max)
        xp[:, t], Pp[:, t] = xp_t, Pp_t
        xf[:, t], Pf[:, t] = x, P
    xs = xf.copy()
    Ps = Pf.copy()
    lag = np.zeros((n, T, NX, NX)) if return_cov else None
    eye = 1e-9 * np.eye(NX)
    for t in range(T - 2, -1, -1):
        C = Pf[:, t] @ F.T @ np.linalg.inv(Pp[:, t + 1] + eye)
        xs[:, t] = xf[:, t] + (C @ (xs[:, t + 1] - xp[:, t + 1])[..., None])[..., 0]
        Ps[:, t] = Pf[:, t] + C @ (Ps[:, t + 1] - Pp[:, t + 1]) @ np.swapaxes(C, 1, 2)
        if return_cov:
            lag[:, t + 1] = Ps[:, t + 1] @ np.swapaxes(C, 1, 2)
    xs[:, :, 0] = np.clip(xs[:, :, 0], 0, p.h_max)
    if return_cov:
        return xs, Ps, lag
    return xs, Ps
