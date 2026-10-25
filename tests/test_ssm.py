import numpy as np

from azuero.ssm.kalman import SSMParams, run


def test_ssm_bias_states_and_jump():
    rng = np.random.default_rng(1)
    n, T, dt = 400, 48, 1 / 12
    t = np.arange(T) * dt
    true = 8 + 1.2 * t[None, :] * np.ones((n, 1))
    true[: n // 2, 30:] = 1.0  # clearing at month 30 for half
    bias = rng.normal(0, 4, n)[:, None]  # per-pixel optical operator bias
    opt = true + bias + rng.normal(0, 2, (n, T))
    opt[:, rng.random(T) < 0.4] = np.nan  # cloudy months
    ann = np.full((n, T), np.nan)
    ann[:, 6::12] = true[:, 6::12] + rng.normal(0, 3, (n, 4))
    lid = np.full((n, T), np.nan)
    pick = rng.random((n, T)) < 0.03
    lid[pick] = (true + rng.normal(0, 2, (n, T)))[pick]
    obs = {"optical": (opt, np.full((n, T), 4.0)), "annual": (ann, np.full((n, T), 9.0)),
           "lidar": (lid, np.full((n, T), 4.0))}
    xs, Ps = run(obs, T, dt, SSMParams())
    h = xs[:, :, 0]
    rmse = np.sqrt(np.mean((h - true) ** 2))
    naive = np.sqrt(np.nanmean((opt - true) ** 2))
    assert rmse < 0.6 * naive
    # the optical bias state learns the per-pixel bias
    assert np.corrcoef(xs[:, -1, 2], bias[:, 0])[0, 1] > 0.5
    # clearing recovered as a drop
    assert np.median(h[: n // 2, 34] - h[: n // 2, 26]) < -4
