"""STUNT (Nam et al., ICLR 2023), reimplemented from the paper (no official licence).

Self-generated tasks: k-means pseudo-labels on a random column subset, whose
columns are then replaced by marginal samples; a ProtoNet is meta-trained on
them, with label-free pseudo-validation (k-means, k = C, on a 20% split) for
checkpoint selection. Inference: nearest class prototype of the labeled rows.
Spec and every flagged adaptation: research_notes/Baselines SSL tabular
modernos/vime_scarf_tabular_ssl.md, Sec. 7.6.
"""

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from torch import nn

DTYPE = torch.float32  # the paper's precision; distances only feed an argmin
MAX_RETRIES = 20


@dataclass(frozen=True)
class StuntConfig:
    """Table 6/7 knobs; the rest are fixed by the paper (App. B)."""

    shot: int = 1
    query: int = 15
    way: int = 10
    steps: int = 3000
    task_batch: int = 4
    eval_every: int = 50
    pv_episodes: int = 1000
    width: int = 1024
    r1: float = 0.2
    r2: float = 0.5


def grid() -> list[StuntConfig]:
    """16 configurations, default (1, 15, 10) first."""
    rest = [
        StuntConfig(shot=s, query=q, way=w)
        for (s, q), w in itertools.product(
            ((1, 5), (1, 15), (5, 10), (5, 20)), (3, 5, 10, 20)
        )
        if (s, q, w) != (1, 15, 10)
    ]
    return [StuntConfig(), *rest]


@dataclass
class StuntModel:
    """Selected checkpoint, its pseudo-validation accuracy and effective sizes."""

    net: nn.Module
    pv_accuracy: float
    way_eff: int
    query_eff: int


def _kmeans(
    X: torch.Tensor, k: int, gen: torch.Generator, iters: int = 20
) -> torch.Tensor:
    """Lloyd's algorithm, k distinct random rows as init, one restart."""
    centers = X[torch.randperm(len(X), generator=gen)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(X, centers).argmin(dim=1)
        for c in range(k):
            members = X[assign == c]
            if len(members):
                centers[c] = members.mean(dim=0)
    return torch.cdist(X, centers).argmin(dim=1)


def _effective_sizes(n_meta: int, cfg: StuntConfig) -> tuple[int, int]:
    way = min(cfg.way, max(2, n_meta // (cfg.shot + 1)))
    query = min(cfg.query, max(1, (n_meta // 2) // way - cfg.shot))
    return way, query


def make_task(
    U: torch.Tensor,
    cfg: StuntConfig,
    way: int,
    query: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One self-generated task: (support x, support y, query x, query y, query used)."""
    n, d = U.shape
    while True:
        for _ in range(MAX_RETRIES):
            p = cfg.r1 + (cfg.r2 - cfg.r1) * torch.rand(1, generator=gen).item()
            n_col = max(1, int(np.floor(d * p)))
            cols = torch.randperm(d, generator=gen)[:n_col]
            labels = _kmeans(U[:, cols], way, gen)
            counts = torch.bincount(labels, minlength=way)
            if counts.min() >= cfg.shot + query:
                return (*_split_task(U, cols, labels, way, cfg.shot, query, gen), query)
        if query == 1:
            raise ValueError("no feasible task: clusters smaller than shot + 1")
        query -= 1


def _task_or_shrink(
    U: torch.Tensor,
    cfg: StuntConfig,
    way: int,
    query: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """make_task, dropping one cluster at a time (down to 2) when infeasible.

    Flagged adaptation (not in the paper): tiny tables can have k-means clusters
    smaller than shot + 1 for every column subset; the paper never hits this.
    """
    while True:
        try:
            *task, used = make_task(U, cfg, way, query, gen)
            return (*task, used, way)  # type: ignore[return-value]
        except ValueError:
            if way == 2:
                raise
            way -= 1


def _split_task(
    U: torch.Tensor,
    cols: torch.Tensor,
    labels: torch.Tensor,
    way: int,
    shot: int,
    query: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = U.clone()
    for c in cols.tolist():  # marginal replacement: independent column permutations
        X[:, c] = X[torch.randperm(len(X), generator=gen), c]
    order = torch.randperm(way, generator=gen)  # random relabeling of the clusters
    support, query_rows, ys, yq = [], [], [], []
    for new, old in enumerate(order.tolist()):
        idx = torch.nonzero(labels == old).flatten()
        idx = idx[torch.randperm(len(idx), generator=gen)]
        support.append(idx[:shot])
        query_rows.append(idx[shot : shot + query])
        ys += [new] * shot
        yq += [new] * query
    return (
        X[torch.cat(support)],
        torch.tensor(ys),
        X[torch.cat(query_rows)],
        torch.tensor(yq),
    )


def proto_logits(
    z_support: torch.Tensor, y_support: torch.Tensor, z_query: torch.Tensor, way: int
) -> torch.Tensor:
    """Negative squared Euclidean distance to each class prototype (Eq. 2)."""
    protos = torch.stack([z_support[y_support == c].mean(dim=0) for c in range(way)])
    logits: torch.Tensor = -torch.cdist(z_query, protos) ** 2
    return logits


def _pv_episodes(
    U_val: NDArray[np.float32], n_classes: int, cfg: StuntConfig, seed: int
) -> list[
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]
]:
    """Fixed 1-shot pseudo-validation episodes; empty if under 2 usable clusters."""
    labels = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit_predict(
        U_val
    )
    usable = [c for c in range(n_classes) if (labels == c).sum() >= 2]
    if len(usable) < 2:
        return []
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(cfg.pv_episodes):
        s_idx, s_y, q_idx, q_y = [], [], [], []
        for new, c in enumerate(usable):
            idx = rng.permutation(np.flatnonzero(labels == c))
            s_idx.append(idx[:1])
            q_idx.append(idx[1:31])
            s_y.append([new])
            q_y.append([new] * len(idx[1:31]))
        episodes.append(
            (
                np.concatenate(s_idx),
                np.concatenate(s_y),
                np.concatenate(q_idx),
                np.concatenate(q_y),
            )
        )
    return episodes


Episode = tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]
]


def _pv_accuracy(net: nn.Module, U_val: torch.Tensor, episodes: list[Episode]) -> float:
    with torch.no_grad():
        Z = net(U_val)
        correct = total = 0
        for s_idx, s_y, q_idx, q_y in episodes:
            way = int(s_y.max()) + 1
            logits = proto_logits(Z[s_idx], torch.as_tensor(s_y), Z[q_idx], way)
            correct += int((logits.argmax(dim=1) == torch.as_tensor(q_y)).sum())
            total += len(q_y)
    return correct / max(total, 1)


def _minmax(X: NDArray[np.float64]) -> NDArray[np.float32]:
    low, span = X.min(axis=0), np.ptp(X, axis=0)
    return ((X - low) / np.where(span > 0, span, 1.0)).astype(np.float32)


def pretrain(
    X: NDArray[np.float64], n_classes: int, seed: int, cfg: StuntConfig
) -> StuntModel:
    """Label-free meta-training on all rows (only the class count C is used)."""
    U_all = _minmax(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(U_all))
    n_val = max(2 * n_classes, len(U_all) // 5)
    U_val, U_meta = U_all[order[:n_val]], torch.as_tensor(U_all[order[n_val:]])
    episodes = _pv_episodes(U_val, n_classes, cfg, seed)
    U_val_t = torch.as_tensor(U_val)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(X.shape[1], cfg.width), nn.ReLU(), nn.Linear(cfg.width, cfg.width)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    way, query = _effective_sizes(len(U_meta), cfg)
    best_acc, best_state = -1.0, copy.deepcopy(net.state_dict())
    for step in range(1, cfg.steps + 1):
        loss = torch.zeros(())
        for _ in range(cfg.task_batch):
            xs, ys, xq, yq, query, way = _task_or_shrink(U_meta, cfg, way, query, gen)
            logits = proto_logits(net(xs), ys, net(xq), way)
            loss = loss + nn.functional.cross_entropy(logits, yq) / cfg.task_batch
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        if episodes and step % cfg.eval_every == 0:
            acc = _pv_accuracy(net, U_val_t, episodes)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict())
    if episodes:
        net.load_state_dict(best_state)
    else:  # early stopping disabled (flagged adaptation): keep the last checkpoint
        best_acc = float("nan")
    return StuntModel(net, best_acc, way, query)


def predict(
    model: StuntModel,
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Nearest labeled-class prototype for every row, as one-hot probabilities."""
    n_classes = int(y_lab.max()) + 1
    with torch.no_grad():
        Z = model.net(torch.as_tensor(_minmax(X)))
        logits = proto_logits(Z[labeled], torch.as_tensor(y_lab), Z, n_classes)
    return np.asarray(np.eye(n_classes)[logits.argmax(dim=1).numpy()], dtype=np.float64)
