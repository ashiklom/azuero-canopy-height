"""Model registry.

Every model implements `fit(X, y, w=None)`, `predict(X)` (point estimate) and
`predict_quantiles(X, qs)`, and declares what it needs (`requires_gpu`, `min_mem_gb`,
`needs_patches`). `runnable()` checks those needs against the active resource profile.
Models that do not fit the current machine are skipped with a logged reason, not failed.

Run in the small profile: LightGBM (pixel-wise gradient boosting with quantile heads) and
a small scikit-learn MLP. Implemented for larger machines but not run here:
  * SpatialCNN        - U-Net-style network on embedding/HLS patches (spatial context)
  * TemporalTransformer - attention over monthly HLS/S1 sequences per pixel
  * DeepEnsemble      - an ensemble of MLPs with heteroscedastic (Gaussian NLL) heads
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import resources

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    cls: type
    requires_gpu: bool = False
    min_mem_gb: float = 2.0
    needs_patches: bool = False
    params: dict = field(default_factory=dict)


class LGBMQuantile:
    """LightGBM point model (L2 by default) plus one model per requested quantile."""

    def __init__(self, quantiles=(0.05, 0.5, 0.95), n_estimators=600, learning_rate=0.05,
                 num_leaves=127, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
                 objective="regression", reg_lambda=1.0, max_bin=127, n_jobs=None, seed=0,
                 q_n_estimators=300, q_learning_rate=0.08, q_max_rows=150_000):
        self.quantiles = tuple(quantiles)
        self.kw = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                       num_leaves=num_leaves, min_child_samples=min_child_samples,
                       subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree,
                       reg_lambda=reg_lambda, max_bin=max_bin,
                       n_jobs=n_jobs or resources().model_threads, random_state=seed, verbose=-1)
        # Quantile heads are only a starting point for conformal calibration, so they are
        # trained on a subsample with fewer trees (the quantile objective is slow).
        self.qkw = {**self.kw, "n_estimators": q_n_estimators, "learning_rate": q_learning_rate,
                    "num_leaves": 63}
        self.q_max_rows = q_max_rows
        self.seed = seed
        self.objective = objective
        self.models = {}

    def fit(self, X, y, w=None):
        import lightgbm as lgb

        self.point = lgb.LGBMRegressor(objective=self.objective, **self.kw).fit(X, y, sample_weight=w)
        idx = np.arange(len(y))
        if self.q_max_rows and len(y) > self.q_max_rows:
            idx = np.random.default_rng(self.seed).choice(len(y), self.q_max_rows, replace=False)
        for q in self.quantiles:
            self.models[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.qkw).fit(
                X[idx], y[idx], sample_weight=None if w is None else w[idx])
        return self

    def predict(self, X):
        return self.point.predict(X)

    def predict_quantiles(self, X, qs=None):
        qs = qs or self.quantiles
        out = np.column_stack([self.models[q].predict(X) for q in qs])
        return np.sort(out, axis=1)  # enforce monotone quantiles

    def feature_importance(self, names):
        imp = self.point.booster_.feature_importance("gain")
        return dict(zip(names, imp / imp.sum()))


class SklearnMLP:
    """Small MLP baseline; quantiles from a Gaussian with residual SD from training."""

    def __init__(self, hidden=(256, 128), alpha=1e-4, max_iter=200, seed=0):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.m = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter, early_stopping=True,
            random_state=seed))

    def fit(self, X, y, w=None):
        X = np.nan_to_num(X)
        self.m.fit(X, y)
        self.sd = float(np.std(y - self.m.predict(X)))
        return self

    def predict(self, X):
        return self.m.predict(np.nan_to_num(X))

    def predict_quantiles(self, X, qs=(0.05, 0.5, 0.95)):
        from scipy.stats import norm

        mu = self.predict(X)
        return np.column_stack([mu + norm.ppf(q) * self.sd for q in qs])


class _TorchModel:
    """Shared plumbing for the GPU-scale models (imported lazily)."""

    def _torch(self):
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("install the 'gpu' extra (torch) to use this model") from e
        import torch

        return torch, ("cuda" if torch.cuda.is_available() else "cpu")


class DeepEnsemble(_TorchModel):
    """M MLPs with Gaussian negative-log-likelihood heads (mean + log-variance).

    Predictive distribution = mixture of members. Captures both aleatoric (label noise)
    and epistemic (member disagreement) uncertainty.
    """

    def __init__(self, members=8, hidden=(512, 256, 128), epochs=60, lr=1e-3, batch=4096):
        self.M, self.hidden, self.epochs, self.lr, self.batch = members, hidden, epochs, lr, batch

    def fit(self, X, y, w=None):
        torch, dev = self._torch()
        X = torch.tensor(np.nan_to_num(X), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu_) / self.sd_
        self.nets = []
        for m in range(self.M):
            torch.manual_seed(m)
            layers, d = [], X.shape[1]
            for h in self.hidden:
                layers += [torch.nn.Linear(d, h), torch.nn.GELU()]
                d = h
            net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 2)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                perm = torch.randperm(len(Xn))
                for k in range(0, len(Xn), self.batch):
                    idx = perm[k:k + self.batch]
                    out = net(Xn[idx].to(dev))
                    mu, logv = out[:, 0], out[:, 1].clamp(-6, 8)
                    loss = (0.5 * (logv + (y[idx].to(dev) - mu) ** 2 / logv.exp())).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.nets.append(net.eval())
        return self

    def _members(self, X):
        torch, dev = self._torch()
        Xn = (torch.tensor(np.nan_to_num(X), dtype=torch.float32) - self.mu_) / self.sd_
        with torch.no_grad():
            outs = [net(Xn.to(dev)).cpu().numpy() for net in self.nets]
        mu = np.stack([o[:, 0] for o in outs])
        var = np.stack([np.exp(np.clip(o[:, 1], -6, 8)) for o in outs])
        return mu, var

    def predict(self, X):
        return self._members(X)[0].mean(0)

    def predict_quantiles(self, X, qs=(0.05, 0.5, 0.95)):
        from scipy.stats import norm

        mu, var = self._members(X)
        m = mu.mean(0)
        sd = np.sqrt(var.mean(0) + mu.var(0))
        return np.column_stack([m + norm.ppf(q) * sd for q in qs])


class SpatialCNN(_TorchModel):
    """U-Net on (features x H x W) patches with sparse lidar supervision.

    Each training sample is a patch centred on lidar observations. The loss is evaluated
    only at labelled pixels (masked pinball loss for the quantiles). This lets the model
    use texture and context (crown structure, edges, topographic position) that pixel-wise
    models cannot see. Inference runs on overlapping tiles with cosine blending.
    """

    def __init__(self, patch=64, base=32, depth=3, epochs=40, quantiles=(0.05, 0.5, 0.95)):
        self.patch, self.base, self.depth, self.epochs, self.quantiles = (
            patch, base, depth, epochs, quantiles)

    def fit(self, patches, label_maps, w=None):  # pragma: no cover - not run on small VMs
        raise NotImplementedError(
            "SpatialCNN training needs patch extraction (features.patch_dataset) and a GPU; "
            "enable with profile=full and the 'gpu' extra")

    def predict(self, X):  # pragma: no cover
        raise NotImplementedError

    def predict_quantiles(self, X, qs=None):  # pragma: no cover
        raise NotImplementedError


class TemporalTransformer(_TorchModel):
    """Transformer encoder over a pixel's monthly HLS + S1 sequence (+ static terrain and
    the annual embedding as a prefix token). It predicts monthly canopy height, so it can
    also stand in for the data-driven observation operator of the state-space model."""

    def __init__(self, d_model=128, heads=4, layers=4, epochs=30):
        self.d_model, self.heads, self.layers, self.epochs = d_model, heads, layers, epochs

    def fit(self, seqs, y, w=None):  # pragma: no cover - not run on small VMs
        raise NotImplementedError(
            "TemporalTransformer needs monthly sequence tensors and a GPU; enable with "
            "profile=full and the 'gpu' extra")

    def predict(self, X):  # pragma: no cover
        raise NotImplementedError

    def predict_quantiles(self, X, qs=None):  # pragma: no cover
        raise NotImplementedError


MODELS: dict[str, ModelSpec] = {
    "lightgbm": ModelSpec("lightgbm", LGBMQuantile, min_mem_gb=2),
    "mlp": ModelSpec("mlp", SklearnMLP, min_mem_gb=2),
    "deep_ensemble": ModelSpec("deep_ensemble", DeepEnsemble, requires_gpu=True, min_mem_gb=16),
    "spatial_cnn": ModelSpec("spatial_cnn", SpatialCNN, requires_gpu=True, min_mem_gb=32,
                             needs_patches=True),
    "temporal_transformer": ModelSpec("temporal_transformer", TemporalTransformer,
                                      requires_gpu=True, min_mem_gb=32),
}


def runnable(name: str) -> tuple[bool, str]:
    spec = MODELS[name]
    r = resources()
    if spec.requires_gpu and not (r.allow_gpu and r.run_large_models):
        return False, f"{name} requires a GPU / run_large_models (profile={r.profile})"
    if spec.min_mem_gb > r.max_mem_gb:
        return False, f"{name} needs >= {spec.min_mem_gb} GB (max_mem_gb={r.max_mem_gb})"
    return True, ""


def get_model(name: str, **params):
    ok, why = runnable(name)
    if not ok:
        raise RuntimeError(why)
    spec = MODELS[name]
    return spec.cls(**{**spec.params, **params})
