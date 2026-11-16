"""External hidden-state baselines. No source text or source forward calls."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from scipy.special import expit
from clr.baselines.probes import Probe


class HSRM(nn.Module):
    """HSRM Section 3.2: projection, encoder, masked mean, LN, scalar."""
    def __init__(self, dim):
        super().__init__()
        self.project = nn.Linear(dim, 256)
        layer = nn.TransformerEncoderLayer(256, 4, 1024, .1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 1)

    def forward(self, z, mask):
        h = self.encoder(self.project(z), src_key_padding_mask=~mask)
        h = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        s = self.head(self.norm(h)).squeeze(-1)
        return torch.stack([torch.zeros_like(s), s], -1)


def linear_probe(dim, qtype):
    model = Probe('B2', qtype)
    model.head = nn.Linear(dim, 1)
    if qtype == 'Q3':
        # Preserve the original forward's sqrt(768) scaling exactly while
        # extending only the state/option projection input dimension.
        model.option = nn.Linear(8, dim, bias=False)
    return model


def delta(bank, state):
    ix = torch.as_tensor(state)
    return bank.z[ix] - bank.z[bank.pi[ix, 0]]


def centroid_score(x, centroids):
    # Positive means closer to success; Euclidean, not squared distance.
    return torch.linalg.vector_norm(x-centroids[0], dim=-1) - torch.linalg.vector_norm(x-centroids[1], dim=-1)


def fit_logistic(score, targets):
    """Deterministic unregularized cal_A logistic, separate by question type."""
    score = np.asarray(score, dtype=np.float64)
    center = float(score.mean()); scale = max(float(score.std()), 1e-8)
    x = (score-center)/scale; y = np.asarray(targets, dtype=np.float64)
    def loss(ab):
        s = ab[0]*x+ab[1]
        return np.mean(np.logaddexp(0, s)-y*s), np.array([np.mean((expit(s)-y)*x), np.mean(expit(s)-y)])
    fit = minimize(loss, [0., 0.], jac=True, method='BFGS', options={'gtol': 1e-8, 'maxiter': 1000})
    if not fit.success and np.linalg.norm(fit.jac) > 1e-6:
        raise RuntimeError(f'Logistic calibration failed: {fit.message}')
    return dict(slope=float(fit.x[0]), intercept=float(fit.x[1]), center=center, scale=scale)


def calibrated_logits(score, fit):
    s = fit['slope']*(np.asarray(score)-fit['center'])/fit['scale']+fit['intercept']
    return np.stack([np.zeros_like(s), s], -1)
