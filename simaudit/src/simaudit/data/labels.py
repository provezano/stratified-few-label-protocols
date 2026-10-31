"""Label-spelling normalisation (Study 13 Amendment 1 (A); Study 9 Amendment 2).

The stored npz tables hold integer codes only: ``openml_loader.encode`` codes
the target as the rank of its string among the sorted distinct strings. So
code c of a table is the c-th sorted distinct raw label string. The raw
strings come from the table's raw OpenML cache (``description.xml`` for the
default target and ``dataset_{did}.pq`` for its column), read without
``openml`` (no network) and without any pickle. They are trusted only if they
re-encode, row by row, to exactly the stored codes.

Rule (declared before any HELD-OUT access):

- a numeric label string (after stripping) is left unchanged;
- any other label is lowercased, every run of whitespace, ``_``, ``-`` and
  ``.`` becomes one space, and the result is stripped;
- labels whose normalised strings are equal are merged into one class;
- new codes keep the old order: each merged class takes the smallest old code
  of its members, and the codes are then made contiguous (0..K'-1). With no
  merge, the mapping is the identity, so nothing about the table changes.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
SEPARATORS = re.compile(r"[\s_.\-]+")
STATUS_VERIFIED = "verified"
STATUS_NOT_FOUND = "raw labels not found"
STATUS_MISMATCH = "raw labels do not reproduce the stored codes"

IntArray = NDArray[np.int64]
Log = Callable[[Path, str], None]


def is_numeric(label: str) -> bool:
    """Whether a raw label string is a plain number (left unchanged)."""
    return bool(NUMERIC.match(label.strip()))


def normalise_label(label: str) -> str:
    """The declared spelling rule for one raw label string."""
    if is_numeric(label):
        return label
    return SEPARATORS.sub(" ", label.lower()).strip()


def merge_codes(names: Sequence[str]) -> IntArray:
    """New code per old code; ``names[c]`` is the raw string of old code c."""
    first: dict[str, int] = {}
    for code, name in enumerate(names):
        first.setdefault(normalise_label(name), code)
    keep = sorted(set(first.values()))
    rank = {old: new for new, old in enumerate(keep)}
    return np.array([rank[first[normalise_label(n)]] for n in names], dtype=np.int64)


@dataclass(frozen=True)
class LabelRecord:
    """What the step did to one table (written to the integrity log)."""

    did: int
    status: str
    raw: list[str] = field(default_factory=list)
    normalised: list[str] = field(default_factory=list)
    mapping: list[int] = field(default_factory=list)
    merges: list[list[str]] = field(default_factory=list)

    @property
    def merged(self) -> bool:
        """Whether any two raw labels were merged."""
        return bool(self.merges)

    def as_row(self) -> dict[str, Any]:
        """One CSV row (lists joined with ' | ' and ';')."""
        return dict(
            did=self.did,
            status=self.status,
            n_raw=len(self.raw),
            n_after=len(set(self.mapping)) if self.mapping else len(self.raw),
            merged=self.merged,
            raw=" | ".join(self.raw),
            normalised=" | ".join(self.normalised),
            mapping=";".join(map(str, self.mapping)),
            merges=" || ".join(" | ".join(g) for g in self.merges),
        )


def record_for(did: int, names: Sequence[str]) -> LabelRecord:
    """Mapping and merge groups of a table whose code c has raw ``names[c]``."""
    mapping = merge_codes(names)
    groups: dict[int, list[str]] = {}
    for name, new in zip(names, mapping.tolist()):
        groups.setdefault(new, []).append(name)
    return LabelRecord(
        did=did,
        status=STATUS_VERIFIED,
        raw=list(names),
        normalised=[normalise_label(n) for n in names],
        mapping=mapping.tolist(),
        merges=[g for _, g in sorted(groups.items()) if len(g) > 1],
    )


def apply(y: IntArray, record: LabelRecord) -> IntArray:
    """Recode ``y`` with a verified record (identity otherwise)."""
    if record.status != STATUS_VERIFIED or not record.merged:
        return y
    return np.asarray(record.mapping, dtype=np.int64)[y]


# ---------------------------------------------------------------- raw labels
def raw_dir(root: Path, did: int) -> Path | None:
    """The OpenML cache folder ``.../datasets/{did}`` under ``root``.

    It must hold ``description.xml`` and ``dataset_{did}.pq``. Several such
    folders are accepted only if their two files are byte-identical;
    otherwise the table counts as not found.
    """
    hits = [
        p.parent
        for p in sorted(root.rglob("description.xml"))
        if p.parent.name == str(did)
        and p.parent.parent.name == "datasets"
        and (p.parent / f"dataset_{did}.pq").exists()
    ]
    if not hits:
        return None
    names = ("description.xml", f"dataset_{did}.pq")
    first = [(hits[0] / n).read_bytes() for n in names]
    same = all([(h / n).read_bytes() for n in names] == first for h in hits[1:])
    return hits[0] if same else None


def default_target(description: Path) -> str:
    """``default_target_attribute`` of an OpenML ``description.xml``."""
    for element in ET.parse(description).getroot().iter():
        if element.tag.split("}")[-1] == "default_target_attribute":
            return str(element.text or "").strip()
    raise ValueError(f"{description}: no default_target_attribute")


def raw_labels(root: Path, did: int, log: Log | None = None) -> NDArray[np.str_] | None:
    """Raw target strings of table ``did`` in stored row order, or None."""
    folder = raw_dir(root, did)
    if folder is None:
        return None
    description = folder / "description.xml"
    parquet = folder / f"dataset_{did}.pq"
    if log is not None:
        log(description, "read default target name (raw OpenML cache)")
    target = default_target(description)
    if not target or "," in target:
        return None
    if log is not None:
        log(parquet, f"read target column {target!r} only (raw OpenML cache)")
    column = pd.read_parquet(parquet, columns=[target])[target]
    return np.asarray(column.astype(str).to_numpy(), dtype=np.str_)


def label_record(did: int, y: IntArray, labels: NDArray[np.str_] | None) -> LabelRecord:
    """Verify raw strings against the stored codes, then build the record."""
    if labels is None:
        return LabelRecord(did=did, status=STATUS_NOT_FOUND)
    names, codes = np.unique(labels, return_inverse=True)
    if len(codes) != len(y) or not np.array_equal(codes.astype(np.int64), y):
        return LabelRecord(did=did, status=STATUS_MISMATCH)
    return record_for(did, [str(n) for n in names])
