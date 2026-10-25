import numpy as np

from azuero.temporal import SmootherParams, change_sd, smooth


def test_smoother_reduces_noise_and_keeps_jump():
    rng = np.random.default_rng(0)
    n, T = 2000, 9
    true = np.tile(5 + 1.0 * np.arange(T), (n, 1)).astype(float)
    true[: n // 2, 5:] = 0.5  # clearing in year 5 for half the pixels
    z = true + rng.normal(0, 1.5, true.shape)
    s = np.full_like(z, 1.5)
    h, sd, g, Ps, lag = smooth(z, s, SmootherParams(), return_cov=True)
    assert np.sqrt(np.mean((h - true) ** 2)) < np.sqrt(np.mean((z - true) ** 2))
    # the clearing is recovered as a large drop
    assert np.median(h[: n // 2, 5] - h[: n // 2, 4]) < -6
    # smoothed annual change of undisturbed pixels is close to +1 m/yr
    d = np.diff(h[n // 2:], axis=1)
    assert abs(np.median(d) - 1.0) < 0.3
    assert np.all(change_sd(Ps, lag)[:, 1:] > 0)


def test_missing_years():
    z = np.array([[5.0, np.nan, 7.0, np.nan, 9.0]])
    h, sd, g = smooth(z, np.ones_like(z))
    assert np.all(np.isfinite(h)) and sd[0, 1] > sd[0, 2]
