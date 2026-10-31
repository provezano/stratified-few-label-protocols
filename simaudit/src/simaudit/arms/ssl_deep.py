"""Deep tabular SSL baselines reimplemented from the primary papers (CPU, float64).

- VIME (Yoon et al., NeurIPS 2020): mask/feature estimation pretraining, then a
  predictor on the frozen encoder with the Eq. 10 consistency loss.
- SCARF (Bahri et al., ICLR 2022): InfoNCE with random feature corruption
  (Alg. 1), then joint fine-tuning; optional self-training (threshold 0.75,
  10 rounds) as in the paper's Table 1.
- 1-NN and nearest class mean on z-scored features (zero hyperparameters).

No stage uses labeled validation data: stages that early-stopped on labels in
the papers run for fixed step budgets (Study 2, Sec. 3; spec in research_notes/
Modern tabular SSL baselines/vime_scarf_tabular_ssl.md, Sec. 6).
"""

import copy
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

DTYPE = torch.float64


@dataclass(frozen=True)
class VimeConfig:
    """VIME defaults from the official ``main_vime.py``."""

    p_m: float = 0.3
    alpha: float = 2.0
    beta: float = 1.0
    K: int = 3
    pretrain_steps: int = 4000
    semi_steps: int = 1000
    batch: int = 128
    min_encoder_width: int = 32


@dataclass(frozen=True)
class ScarfConfig:
    """SCARF defaults from the paper (Sec. 4)."""

    corruption: float = 0.6
    tau: float = 1.0
    width: int = 256
    batch: int = 128
    patience: int = 3
    min_pseudo_epoch_steps: int = 20
    max_pretrain_steps: int = 5000
    finetune_steps: int = 100
    self_training: bool = False
    st_threshold: float = 0.75
    st_rounds: int = 10


def grid(name: str) -> list[Any]:
    """The 16 pre-registered configurations of a method (default included)."""
    if name == "vime":
        return [
            VimeConfig(p_m=p, beta=b)
            for p, b in itertools.product((0.1, 0.3, 0.5, 0.7), (0.1, 1.0, 3.0, 10.0))
        ]
    if name in ("scarf", "scarf_st"):
        return [
            ScarfConfig(corruption=c, tau=t, self_training=name == "scarf_st")
            for c, t in itertools.product((0.2, 0.4, 0.6, 0.8), (0.1, 0.5, 1.0, 2.0))
        ]
    raise ValueError(f"no grid for {name}")


# --- corruption and losses ----------------------------------------------------


def vime_corrupt(
    U: torch.Tensor, p_m: float, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bernoulli(p_m) mask; masked entries take column-shuffled values (Supp. Fig 3)."""
    mask = (torch.rand(U.shape, generator=gen, dtype=DTYPE) < p_m).to(DTYPE)
    shuffled = torch.stack(
        [U[torch.randperm(len(U), generator=gen), j] for j in range(U.shape[1])], dim=1
    )
    return mask, mask * shuffled + (1 - mask) * U


def scarf_q(corruption: float, d: int) -> int:
    """Number of corrupted features; at least one (adaptation for small d)."""
    return max(1, int(np.floor(corruption * d)))


def scarf_corrupt(
    x: torch.Tensor, U: torch.Tensor, corruption: float, gen: torch.Generator
) -> torch.Tensor:
    """Replace q random features per row by draws from U's column marginals."""
    n, d = x.shape
    q = scarf_q(corruption, d)
    chosen = torch.rand((n, d), generator=gen).argsort(dim=1)[:, :q]
    mask = torch.zeros((n, d), dtype=torch.bool)
    mask.scatter_(1, chosen, True)
    rows = torch.randint(len(U), (n, d), generator=gen)
    draws = U[rows, torch.arange(d).expand(n, d)]
    return torch.where(mask, draws, x)


def info_nce(z: torch.Tensor, z_tilde: torch.Tensor, tau: float) -> torch.Tensor:
    """Compute SCARF's InfoNCE (Alg. 1, line 8): clean anchors vs corrupted views."""
    s = nn.functional.normalize(z, dim=1) @ nn.functional.normalize(z_tilde, dim=1).T
    log_mean_exp = torch.logsumexp(s / tau, dim=1) - np.log(len(z))
    loss: torch.Tensor = (log_mean_exp - s.diag() / tau).mean()
    return loss


def vime_consistency(clean: torch.Tensor, corrupted: torch.Tensor) -> torch.Tensor:
    """Eq. 10: mean squared gap between corrupted and clean logits."""
    gap: torch.Tensor = ((corrupted - clean[:, None, :]) ** 2).mean()
    return gap


# --- helpers ------------------------------------------------------------------


def _scale(X: NDArray[np.float64], kind: str) -> torch.Tensor:
    if kind == "minmax":
        low, span = X.min(axis=0), np.ptp(X, axis=0)
        Z = (X - low) / np.where(span > 0, span, 1.0)
    else:
        std = X.std(axis=0)
        Z = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    return torch.as_tensor(Z, dtype=DTYPE)


def _mlp(sizes: list[int], last_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(itertools.pairwise(sizes)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or last_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers).to(DTYPE)


def _step(opt: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    opt.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    opt.step()


def _seeded(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _batch(n: int, size: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(n, (min(size, n),), generator=gen)


def _softmax(logits: torch.Tensor) -> NDArray[np.float64]:
    return np.asarray(torch.softmax(logits, dim=1).detach().numpy(), dtype=np.float64)


# --- VIME ---------------------------------------------------------------------


def _vime_pretrain(X: NDArray[np.float64], seed: int, cfg: VimeConfig) -> nn.Module:
    U = _scale(X, "minmax")
    d = U.shape[1]
    width = max(d, cfg.min_encoder_width)
    gen = _seeded(seed)
    encoder = _mlp([d, width], last_activation=True)
    mask_head = nn.Sequential(nn.Linear(width, d), nn.Sigmoid()).to(DTYPE)
    feature_head = nn.Sequential(nn.Linear(width, d), nn.Sigmoid()).to(DTYPE)
    params = [
        *encoder.parameters(),
        *mask_head.parameters(),
        *feature_head.parameters(),
    ]
    opt = torch.optim.RMSprop(params, lr=1e-3)
    for _ in range(cfg.pretrain_steps):
        x = U[_batch(len(U), cfg.batch, gen)]
        mask, x_tilde = vime_corrupt(x, cfg.p_m, gen)
        z = encoder(x_tilde)
        loss = nn.functional.binary_cross_entropy(mask_head(z), mask)
        loss = loss + cfg.alpha * nn.functional.mse_loss(feature_head(z), x)
        _step(opt, loss)
    return encoder


def _vime_finetune(
    encoder: nn.Module,
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    cfg: VimeConfig,
) -> NDArray[np.float64]:
    U = _scale(X, "minmax")
    n_classes = int(y_lab.max()) + 1
    gen = _seeded(seed + 1)
    encoder = copy.deepcopy(encoder).requires_grad_(False)
    width = int(next(encoder.parameters()).shape[0])  # first Linear's out_features
    predictor = _mlp([width, 100, 100, n_classes], last_activation=False)
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    x_lab, y_t = U[torch.as_tensor(labeled)], torch.as_tensor(y_lab)
    for _ in range(cfg.semi_steps):
        x_u = U[_batch(len(U), cfg.batch, gen)]
        copies = torch.stack(
            [vime_corrupt(x_u, cfg.p_m, gen)[1] for _ in range(cfg.K)], 1
        )
        clean = predictor(encoder(x_u))
        corrupted = predictor(encoder(copies.reshape(-1, U.shape[1])))
        corrupted = corrupted.reshape(len(x_u), cfg.K, n_classes)
        loss = nn.functional.cross_entropy(predictor(encoder(x_lab)), y_t)
        loss = loss + cfg.beta * vime_consistency(clean, corrupted)
        _step(opt, loss)
    with torch.no_grad():
        return _softmax(predictor(encoder(U)))


# --- SCARF --------------------------------------------------------------------


def _scarf_pretrain(X: NDArray[np.float64], seed: int, cfg: ScarfConfig) -> nn.Module:
    U = _scale(X, "zscore")
    gen = _seeded(seed)
    order = torch.randperm(len(U), generator=gen)
    n_val = max(1, len(U) // 10)
    U_val, U_train = U[order[:n_val]], U[order[n_val:]]
    val_clean = U_val.repeat(10, 1)
    val_tilde = scarf_corrupt(val_clean, U, cfg.corruption, gen)
    encoder = _mlp([U.shape[1], *[cfg.width] * 4], last_activation=True)
    head = _mlp([cfg.width, cfg.width, cfg.width], last_activation=False)
    opt = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=1e-3)
    pseudo_epoch = max(-(-len(U_train) // cfg.batch), cfg.min_pseudo_epoch_steps)
    best, best_state, waited, step = np.inf, copy.deepcopy(encoder.state_dict()), 0, 0
    while step < cfg.max_pretrain_steps and waited < cfg.patience:
        for _ in range(pseudo_epoch):
            x = U_train[_batch(len(U_train), cfg.batch, gen)]
            loss = info_nce(
                head(encoder(x)),
                head(encoder(scarf_corrupt(x, U, cfg.corruption, gen))),
                cfg.tau,
            )
            _step(opt, loss)
            step += 1
        with torch.no_grad():
            val = _val_info_nce(encoder, head, val_clean, val_tilde, cfg)
        if val < best:
            best, best_state, waited = val, copy.deepcopy(encoder.state_dict()), 0
        else:
            waited += 1
    encoder.load_state_dict(best_state)
    return encoder


def _val_info_nce(
    encoder: nn.Module,
    head: nn.Module,
    clean: torch.Tensor,
    tilde: torch.Tensor,
    cfg: ScarfConfig,
) -> float:
    losses = [
        info_nce(
            head(encoder(clean[i : i + cfg.batch])),
            head(encoder(tilde[i : i + cfg.batch])),
            cfg.tau,
        ).item()
        for i in range(0, len(clean), cfg.batch)
    ]
    return float(np.mean(losses))


def _scarf_fit(
    encoder: nn.Module,
    U: torch.Tensor,
    rows: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    cfg: ScarfConfig,
) -> NDArray[np.float64]:
    encoder = copy.deepcopy(encoder)
    head = _mlp([cfg.width, cfg.width, n_classes], last_activation=False)
    opt = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=1e-3)
    for _ in range(cfg.finetune_steps):
        loss = nn.functional.cross_entropy(head(encoder(U[rows])), targets)
        _step(opt, loss)
    with torch.no_grad():
        return _softmax(head(encoder(U)))


def _scarf_finetune(
    encoder: nn.Module,
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    cfg: ScarfConfig,
) -> NDArray[np.float64]:
    U = _scale(X, "zscore")
    n_classes = int(y_lab.max()) + 1
    _seeded(seed + 1)
    rows, targets = torch.as_tensor(labeled), torch.as_tensor(y_lab)
    proba = _scarf_fit(encoder, U, rows, targets, n_classes, cfg)
    if not cfg.self_training:
        return proba
    is_labeled = np.zeros(len(U), dtype=bool)
    is_labeled[labeled] = True
    for _ in range(cfg.st_rounds):
        confident = (proba.max(axis=1) >= cfg.st_threshold) & ~is_labeled
        rows = torch.as_tensor(np.concatenate([labeled, np.flatnonzero(confident)]))
        pseudo = proba.argmax(axis=1)[np.flatnonzero(confident)]
        targets = torch.as_tensor(np.concatenate([y_lab, pseudo]))
        proba = _scarf_fit(encoder, U, rows, targets, n_classes, cfg)
    return proba


# --- zero-hyperparameter companions ------------------------------------------


def _nearest(
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    class_means: bool,
) -> NDArray[np.float64]:
    Z = _scale(X, "zscore").numpy()
    n_classes = int(y_lab.max()) + 1
    if class_means:
        refs = np.stack([Z[labeled[y_lab == c]].mean(axis=0) for c in range(n_classes)])
        ref_labels = np.arange(n_classes)
    else:
        refs, ref_labels = Z[labeled], y_lab
    d = ((Z[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(np.eye(n_classes)[ref_labels[d.argmin(axis=1)]], dtype=np.float64)


# --- public API ---------------------------------------------------------------


def pretrain(name: str, X: NDArray[np.float64], seed: int, config: Any) -> nn.Module:
    """Label-free stage; reusable across label draws and folds of the same data."""
    if name == "vime":
        return _vime_pretrain(X, seed, config)
    return _scarf_pretrain(X, seed, config)


def finetune(
    name: str,
    encoder: nn.Module,
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    config: Any,
) -> NDArray[np.float64]:
    """Label-dependent stage; returns class probabilities for every row."""
    if name == "vime":
        return _vime_finetune(encoder, X, labeled, y_lab, seed, config)
    return _scarf_finetune(encoder, X, labeled, y_lab, seed, config)


def fit_predict(
    name: str,
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    config: Any,
) -> NDArray[np.float64]:
    """Run one method end to end; only ``y_lab`` (labels of ``labeled``) is read."""
    if name in ("knn1", "ncm"):
        return _nearest(X, labeled, y_lab, class_means=name == "ncm")
    method = "vime" if name == "vime" else "scarf"
    encoder = pretrain(method, X, seed, config)
    return finetune(method, encoder, X, labeled, y_lab, seed, config)
