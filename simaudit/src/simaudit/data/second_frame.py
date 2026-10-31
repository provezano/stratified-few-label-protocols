"""Study 15 second frame: 20 post-FM tables (Zenodo, figshare, Kaggle).

Prereg: prereg/STUDY15_SECOND_FRAME.md (Sec. 1). The 20 rows with
``status_pool == SAFE`` of research_notes/Autoresearch idea bank/
POOL_POS_FM_2026-10-30.csv. Each table has one declared target and declared
drops (identifier, leakage, post-outcome and date columns), fixed from the
source documentation and column names only.

Raw files are untrusted: they are read as text (csv module) or through the
standard-library xlsx reader (:mod:`simaudit.data.xlsx_lite`); nothing in
them is executed. Run the build with ``python -I``.

Encoding rule (prereg Sec. 1.3), the same for every table:
1. cells are stripped; ``NA_TOKENS`` are missing; rows with a missing target
   are dropped;
2. a feature column is numeric when at least ``NUMERIC_SHARE`` (95%) of its
   non-missing cells parse as numbers (a decimal comma is accepted); the
   cells that do not parse are then missing. Otherwise it is nominal;
3. nominal columns with more than ``MAX_LEVELS`` levels (free text, IDs) are
   dropped; nominal levels are coded by their sorted string order;
4. columns that are entirely missing or constant are dropped;
5. missing values are imputed without labels, from all rows: numeric by the
   column median, nominal as one extra level;
6. target strings get the Study 13 Amendment 1 (A) spelling rule
   (``data.labels``) and are coded in sorted order of the raw strings.

Eligibility (as Study 13 / pilot I2): 2 <= C <= 10, smallest class >= 11,
n >= 100, d >= 1 after encoding.
"""

import csv
import hashlib
import io
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simaudit.data import labels
from simaudit.data.openml_loader import Dataset, content_hash
from simaudit.data.xlsx_lite import read_sheet

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

NA_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "#null!", "#n/a", "-", ".", "?"})
MAX_LEVELS = 100
NUMERIC_SHARE = 0.95  # share of non-missing cells that must parse as numbers
MIN_CLASS = 11
MIN_ROWS = 100
C_RANGE = (2, 10)
ID_NAME = re.compile(
    r"(?i)^(id|.*_id|id_.*|sn|no\.?|ind|subjid|study_id|respondent_id)$"
)
SOURCE_OFFSET = {"Zenodo": 10**10, "figshare": 2 * 10**10, "Kaggle": 3 * 10**10}
NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

Z = "https://zenodo.org/api/records/{}/files/{}/content"
FS = "https://ndownloader.figshare.com/files/{}"
KG = "https://www.kaggle.com/api/v1/datasets/download/{}"


@dataclass(frozen=True)
class Part:
    """One raw file (and sheet / header row) of a table."""

    name: str
    sheet: int | str | None = None
    header: int = 0


@dataclass(frozen=True)
class Spec:
    """Declared source, target and drops of one table (prereg Sec. 1.2)."""

    key: str
    source: str
    record: str
    first_public: str
    license: str
    parts: tuple[Part, ...]
    target: str
    drop: tuple[str, ...] = ()
    drop_re: str | None = None
    tabpfn35_safe: bool = False
    urls: tuple[str, ...] = field(default=())

    @property
    def did(self) -> int:
        """Numeric ID: source offset + repository ID (disjoint from OpenML/PMLB)."""
        return SOURCE_OFFSET[self.source] + int(self.record)


def _p(name: str, sheet: int | str | None = None, header: int = 0) -> Part:
    return Part(name, sheet, header)


SPECS: tuple[Spec, ...] = (
    Spec("k11395735_malaria", "Kaggle", "11395735", "2026-07-28", "CC0-1.0",
         (_p("Malaria Diseases dataset - .csv"),), "Result",
         urls=(KG.format("mdtuhinalambijoy/malaria-associated-clinical-data-from-"
                         "bangladesh"),)),
    Spec("f33111881_frailty", "figshare", "33111881", "2026-07-29", "CC BY 4.0",
         (_p("Heart failure frailty prediction training dataset.xlsx", 0, 1),
          _p("Heart failure frailty prediction testing dataset.xlsx", 0, 1)),
         "Frailty", urls=(FS.format(67087244), FS.format(67087241))),
    Spec("f33126371_nonadh", "figshare", "33126371", "2026-07-30", "CC BY 4.0",
         (_p("pone.0354682.s004.xlsx", 0, 0),), "Adherence",
         urls=(FS.format(67133465),)),
    Spec("z21713220_optiaged", "Zenodo", "21713220", "2026-07-31", "CC-BY-4.0",
         (_p("OPTIAGED DATA SHARE.xlsx", 0, 0),), "CJP_COMPOSITE",
         drop=("CENTER_NOM", "n_ctrl", "n_exp"),
         drop_re=r"(?i)date|^dt_|^dvisit$",
         urls=(Z.format(21713220, "OPTIAGED%20DATA%20SHARE.xlsx"),)),
    Spec("z21872549_vacc", "Zenodo", "21872549", "2026-08-10", "CC-BY-4.0",
         (_p("views about vaccination in the medical students baza.xlsx", 0, 0),),
         "kierunek", drop=("Aktualnie jestem studentką/studentem",),
         urls=(Z.format(21872549, "views%20about%20vaccination%20in%20the%20medical"
                        "%20students%20baza.xlsx"),)),
    Spec("z22705412_hpv", "Zenodo", "22705412", "2026-08-13", "CC-BY-4.0",
         (_p("HPV_CaMau_2026_dataset.xlsx", "Data", 0),), "hpv_result",
         drop=("n_genotypes_detected", "genotypes_detected", "assay_risk_class",
               "any_hr_assay", "any_iarc_group1", "hpv16_or_hpv18",
               "any_nonavalent_genotype", "multiple_infection"),
         drop_re=r"^hpv\d+$",
         urls=(Z.format(22705412, "HPV_CaMau_2026_dataset.xlsx"),)),
    Spec("z22065799_school", "Zenodo", "22065799", "2026-08-23", "CC-BY-4.0",
         (_p("infra_academic_performance_survey.csv"),), "Q4. Current CGPA",
         urls=(Z.format(22065799, "infra_academic_performance_survey.csv"),)),
    Spec("z22133098_trees", "Zenodo", "22133098", "2026-08-27", "CC-BY-4.0",
         (_p("urban_forest_tree_dataset_2023_2026.xlsx", 0, 0),), "Status_2026",
         drop=("Tree_ID", "CBH_2026_cm", "DBH_2026_cm"),
         urls=(Z.format(22133098, "urban_forest_tree_dataset_2023_2026.xlsx"),)),
    Spec("z22172178_perio", "Zenodo", "22172178", "2026-08-30", "CC-BY-4.0",
         (_p("S1_Dataset.xlsx", 0, 0),), "Periodontal_status",
         drop=("Mean_PPD_mm", "Maximum_PPD_mm", "Mean_CAL_mm", "Maximum_CAL_mm",
               "CPI"),
         urls=(Z.format(22172178, "S1_Dataset.xlsx"),)),
    Spec("z22216010_pvt", "Zenodo", "22216010", "2026-08-31", "CC-BY-4.0",
         (_p("database PVT.csv"),), "PVT", drop=("Predicted - PVT", "Prediction"),
         urls=(Z.format(22216010, "database%20PVT.csv"),)),
    Spec("z22751055_fairtrade", "Zenodo", "22751055", "2026-09-14", "CC-BY-4.0",
         (_p("fair_trade_credit_burundi_anonymized_data.csv"),), "credit_access",
         urls=(Z.format(22751055, "fair_trade_credit_burundi_anonymized_data.csv"),)),
    Spec("z22767593_edmort", "Zenodo", "22767593", "2026-09-15", "CC-BY-4.0",
         (_p("clean data ML.csv"),), "Mortality",
         drop_re=r"(?i)date|time of|resuscitation|discharge|death",
         urls=(Z.format(22767593, "clean%20data%20ML.csv"),)),
    Spec("f33868873_adherence", "figshare", "33868873", "2026-09-17", "CC BY 4.0",
         (_p("S1_Data.xlsx", 0, 0),), "adherence", urls=(FS.format(68754607),)),
    Spec("z22900584_foodwaste", "Zenodo", "22900584", "2026-09-22", "CC-BY-4.0",
         (_p("01_Tulcan_Household_Food_Waste_2023.csv"),), "waste_occurrence",
         drop=("household_id", "cluster_id", "discard_frequency"),
         drop_re=r"(?i)waste|^cause_|^solution_",
         urls=(Z.format(22900584, "01_Tulcan_Household_Food_Waste_2023.csv"),)),
    Spec("z23293577_pain", "Zenodo", "23293577", "2026-09-24", "CC-BY-4.0",
         (_p("data set.xlsx", "women", 1),), "pain",
         drop=("nrs", "頭", "顔（口・顎）", "首", "肩", "上腕", "肘（外側）", "肘（内側）",
               "前腕", "手首・手・指", "胸", "お腹", "股関節", "太もも", "膝", "すね",
               "足首", "足", "背中", "腰・臀部", "全身"),
         tabpfn35_safe=True, urls=(Z.format(23293577, "data%20set.xlsx"),)),
    Spec("f33328002_tea", "figshare", "33328002", "2026-10-10", "CC BY 4.0",
         (_p("tea_farmer_data.csv"),), "adoption", drop=("swb",),
         tabpfn35_safe=True, urls=(FS.format(67817394),)),
    Spec("z23318225_hlit", "Zenodo", "23318225", "2026-10-12", "CC-BY-4.0",
         (_p("iHF_health_literacy_survey_anonymised_v2.csv"),),
         "long_standing_condition", tabpfn35_safe=True,
         urls=(Z.format(23318225, "iHF_health_literacy_survey_anonymised_v2.csv"),)),
    Spec("f34554682_pcos", "figshare", "34554682", "2026-10-15", "CC BY 4.0",
         (_p("pone.0358554.s001.xlsx", 0, 0),), "group", drop=("gr",),
         tabpfn35_safe=True, urls=(FS.format(70201219),)),
    Spec("z23168913_volunteer", "Zenodo", "23168913", "2026-10-19", "CC-BY-4.0",
         (_p("S1_Dataset_Predictors_of_Volunteer_Satisfaction.xlsx", 0, 0),),
         "Nursing_Group", drop=("Academic_Discipline_Group", "College_CAT"),
         tabpfn35_safe=True,
         urls=(Z.format(23168913, "S1_Dataset_Predictors_of_Volunteer_"
                        "Satisfaction.xlsx"),)),
    Spec("f34737363_pneumonia", "figshare", "34737363", "2026-10-21", "CC BY 4.0",
         (_p("occult_pneumonia.csv"),), "分组", drop=("随机编号排序",),
         tabpfn35_safe=True, urls=(FS.format(70458786),)),
)  # fmt: skip


def spec_by_did(did: int) -> Spec:
    """The spec of one table ID."""
    return next(s for s in SPECS if s.did == did)


def dataset_key(did: int) -> str:
    """File-name-safe key of a second-frame table (its spec key)."""
    return spec_by_did(did).key


# ---------------------------------------------------------------- reading
def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def read_part(path: Path, part: Part) -> tuple[list[str], list[list[str]]]:
    """(header, rows) of one raw file, all cells as stripped strings."""
    if path.suffix.lower() == ".xlsx":
        grid = read_sheet(path, 0 if part.sheet is None else part.sheet)
    else:
        text = _decode(path.read_bytes())
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t")
        grid = list(csv.reader(io.StringIO(text), dialect))
    grid = [[c.strip() for c in row] for row in grid]
    header, body = grid[part.header], grid[part.header + 1 :]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body if any(r)]
    return header, body


def read_table(spec: Spec, raw_dir: Path) -> tuple[list[str], list[list[str]]]:
    """Concatenate the parts of a table (aligned by position; equal widths)."""
    parts = [read_part(raw_dir / p.name, p) for p in spec.parts]
    widths = {len(h) for h, _ in parts}
    if len(widths) != 1:
        raise ValueError(f"{spec.key}: parts differ in width {sorted(widths)}")
    header = _unique_names(parts[0][0])
    return header, [row for _, body in parts for row in body]


def _unique_names(names: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        k = seen.get(name, 0)
        seen[name] = k + 1
        out.append(name if k == 0 else f"{name}.{k}")
    return out


# ---------------------------------------------------------------- encoding
def is_missing(cell: str) -> bool:
    """Whether a stripped cell is a declared missing token."""
    return cell.strip().lower() in NA_TOKENS


def _number(cell: str) -> float | None:
    text = cell.replace(",", ".") if cell.count(",") == 1 else cell
    return float(text) if NUMBER.match(text) else None


def dropped_columns(header: Sequence[str], spec: Spec) -> dict[str, str]:
    """Declared drops before encoding: name -> reason (target excluded)."""
    rx = re.compile(spec.drop_re) if spec.drop_re else None
    out = {}
    for name in header:
        if name == spec.target:
            continue
        if name in spec.drop:
            out[name] = "declared"
        elif rx is not None and rx.search(name):
            out[name] = "declared pattern"
        elif ID_NAME.match(name.strip()):
            out[name] = "identifier name"
    return out


def encode_column(cells: Sequence[str]) -> tuple[FloatArray | None, str]:
    """Encoded, imputed column, or None with the reason it is dropped."""
    miss = np.array([is_missing(c) for c in cells])
    present = [c for c, m in zip(cells, miss) if not m]
    if not present:
        return None, "all missing"
    numbers = [_number(c) for c in cells]
    parsed = np.array([v is not None for v in numbers]) & ~miss
    if parsed.sum() >= NUMERIC_SHARE * len(present):
        col = np.array([np.nan if v is None else v for v in numbers])
        col[~parsed] = np.nan  # missing tokens and unparsable cells
        col[~parsed] = np.median(col[parsed])
        kind = "numeric"
    else:
        levels = sorted(set(present))
        if len(levels) > MAX_LEVELS:
            return None, f"nominal with > {MAX_LEVELS} levels"
        code = {lv: i for i, lv in enumerate(levels)}
        col = np.array(
            [len(levels) if m else code[c] for c, m in zip(cells, miss)],
            dtype=np.float64,
        )
        kind = "nominal"
    if np.unique(col).size < 2:
        return None, "constant"
    return col, kind


def encode_labels(did: int, raw: Sequence[str]) -> tuple[IntArray, dict[str, Any]]:
    """Codes in sorted raw-string order, then the Study 13 spelling merge."""
    levels, codes = np.unique(np.asarray(raw, dtype=str), return_inverse=True)
    record = labels.record_for(did, [str(x) for x in levels])
    y = labels.apply(codes.astype(np.int64), record)
    return y, record.as_row()


def encode(
    spec: Spec, header: Sequence[str], rows: Sequence[Sequence[str]]
) -> tuple[FloatArray, IntArray, dict[str, Any]]:
    """(X, y, info) by the rule of the module docstring."""
    if spec.target not in header:
        raise KeyError(f"{spec.key}: target {spec.target!r} not in columns")
    t = list(header).index(spec.target)
    rows = [r for r in rows if not is_missing(r[t])]
    drops = dropped_columns(header, spec)
    cols, kinds = [], {}
    for j, name in enumerate(header):
        if j == t or name in drops:
            continue
        col, kind = encode_column([r[j] for r in rows])
        if col is None:
            drops[name] = kind
            continue
        cols.append(col)
        kinds[name] = kind
    if not cols:
        raise ValueError(f"{spec.key}: no feature left")
    y, label_row = encode_labels(spec.did, [r[t] for r in rows])
    X = np.column_stack(cols)
    counts = np.bincount(y)
    info = dict(
        key=spec.key,
        did=spec.did,
        n=int(len(y)),
        d=int(X.shape[1]),
        C=int(len(counts)),
        class_counts=[int(c) for c in counts],
        features=kinds,
        dropped=drops,
        labels=label_row,
    )
    return X, y, info


def ineligible(info: Mapping[str, Any]) -> str | None:
    """Reason a table is excluded by the Study 13 / pilot rule, or None."""
    if not C_RANGE[0] <= info["C"] <= C_RANGE[1]:
        return f"C = {info['C']} outside {C_RANGE}"
    if min(info["class_counts"]) < MIN_CLASS:
        return f"smallest class {min(info['class_counts'])} < {MIN_CLASS}"
    if info["n"] < MIN_ROWS:
        return f"n = {info['n']} < {MIN_ROWS}"
    if info["d"] < 1:
        return "no feature"
    return None


# ---------------------------------------------------------------- dedup
def overlap_checks(
    X: FloatArray,
    y: IntArray,
    hashes: Mapping[str, str],
    arrays: Mapping[str, FloatArray],
) -> dict[str, Any]:
    """Content-hash match against reference sets and row overlap (A2.3b).

    ``hashes``: reference name -> content hash (DEV, T1', HELD-OUT lists).
    ``arrays``: reference name -> X of non-sealed tables (DEV, T1').
    """
    from simaudit.data.integrity import overlap_share

    sha = content_hash(X, y)
    match = sorted(name for name, h in hashes.items() if h == sha)
    shares = {name: overlap_share(X, A) for name, A in arrays.items()}
    worst = max(shares.items(), key=lambda kv: kv[1], default=("", 0.0))
    return dict(
        sha256=sha,
        hash_match=match,
        max_overlap=float(worst[1]),
        max_overlap_with=worst[0],
    )


# ---------------------------------------------------------------- build
def file_digest(path: Path, algo: str) -> str:
    """Hex digest of a file's bytes."""
    return hashlib.new(algo, path.read_bytes()).hexdigest()


def save_npz(path: Path, X: FloatArray, y: IntArray, name: str) -> str:
    """Write the table in the openml_loader npz layout; return its content hash."""
    sha = content_hash(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, X=X, y=y, name=name, version=1, sha=sha)
    tmp.replace(path)
    return sha


def build(
    root: Path,
    hashes: Mapping[str, str],
    arrays: Mapping[str, FloatArray],
    specs: Sequence[Spec] = SPECS,
) -> list[dict[str, Any]]:
    """Encode every table under ``root/raw/<key>/``; npz to ``root/npz/``.

    Returns one record per table (eligible or not, with the reason). Tables
    that fail eligibility or the dedup rule get no npz.
    """
    out = []
    for spec in specs:
        rec: dict[str, Any] = dict(key=spec.key, did=spec.did, source=spec.source)
        try:
            header, rows = read_table(spec, root / "raw" / spec.key)
            X, y, info = encode(spec, header, rows)
        except (KeyError, ValueError, OSError) as exc:
            out.append(dict(rec, excluded=f"unreadable: {exc}"))
            continue
        checks = overlap_checks(X, y, hashes, arrays)
        reason = ineligible(info)
        if reason is None and checks["hash_match"]:
            reason = f"duplicate of {checks['hash_match']}"
        if reason is None and checks["max_overlap"] >= OVERLAP_LIMIT:
            reason = f"row overlap {checks['max_overlap']:.2f} with "
            reason += checks["max_overlap_with"]
        if reason is None:
            save_npz(root / "npz" / f"{spec.did}.npz", X, y, spec.key)
        out.append(dict(rec, **info, **checks, excluded=reason))
    return out


OVERLAP_LIMIT = 0.5  # integrity.OVERLAP (Study 8 A2.3a)


# ---------------------------------------------------------------- loading
class SecondFrameLoader:
    """Picklable loader of the built npz files; checks the frozen content hash."""

    def __init__(self, npz_dir: Path, pool_hashes: Mapping[int, str]) -> None:
        """Bind the npz folder and the frozen hashes (data/pool_second_frame)."""
        self.npz_dir = npz_dir
        self.pool_hashes = dict(pool_hashes)

    def __call__(self, did: int, cache_dir: Path) -> Dataset:  # noqa: ARG002
        """Load table ``did`` (``cache_dir`` unused: Loader signature)."""
        with np.load(self.npz_dir / f"{did}.npz", allow_pickle=False) as z:
            data = Dataset(
                did,
                str(z["name"]),
                int(z["version"]),
                np.asarray(z["X"], dtype=np.float64),
                np.asarray(z["y"], dtype=np.int64),
                str(z["sha"]),
            )
        want = self.pool_hashes.get(did)
        if want is None or data.sha256 != want or content_hash(data.X, data.y) != want:
            raise ValueError(f"second frame {did}: content hash != frozen pool entry")
        return data


def reference_hashes(project_root: Path) -> dict[str, str]:
    """Content hashes of DEV and HELD-OUT (pool_i2.yaml) and T1' (npz cache).

    HELD-OUT enters only through its hash list in ``pool_i2.yaml``;
    ``sealed_heldout/`` is never read.
    """
    import yaml

    pool = yaml.safe_load((project_root / "data" / "pool_i2.yaml").read_text())
    out = {
        f"{part}:{d['did']}": str(d["sha256"])
        for part in ("dev", "heldout")
        for d in pool[part]
    }
    for did, X, y in _t1prime(project_root):
        out[f"t1prime:{did}"] = content_hash(X, y)
    return out


def reference_arrays(project_root: Path) -> dict[str, FloatArray]:
    """X of the non-sealed reference tables (DEV and T1') for row overlap."""
    import yaml

    pool = yaml.safe_load((project_root / "data" / "pool_i2.yaml").read_text())
    npz = project_root / "data" / "cache" / "npz"
    out = {}
    for d in pool["dev"]:
        with np.load(npz / f"{d['did']}.npz", allow_pickle=False) as z:
            out[f"dev:{d['did']}"] = np.asarray(z["X"], dtype=np.float64)
    for did, X, _ in _t1prime(project_root):
        out[f"t1prime:{did}"] = X
    return out


def _t1prime(project_root: Path) -> list[tuple[int, FloatArray, IntArray]]:
    import yaml

    pool = yaml.safe_load((project_root / "data" / "pool_t1prime.yaml").read_text())
    npz = project_root / "data" / "cache" / "npz"
    out = []
    for d in pool["datasets"]:
        with np.load(npz / f"{d['did']}.npz", allow_pickle=False) as z:
            out.append(
                (
                    int(d["did"]),
                    np.asarray(z["X"], dtype=np.float64),
                    np.asarray(z["y"], dtype=np.int64),
                )
            )
    return out


def pool_entries(
    records: Sequence[Mapping[str, Any]], files: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The frozen pool (``data/pool_second_frame.yaml``) from a build report.

    ``files`` is the fetch log (one entry per raw file: key, name, url, md5,
    sha256, ...). Excluded tables are listed with their reason.
    """
    specs = {s.key: s for s in SPECS}
    kept, excluded = [], []
    for r in records:
        spec = specs[r["key"]]
        if r.get("excluded"):
            excluded.append(dict(key=r["key"], did=r["did"], reason=r["excluded"]))
            continue
        kept.append(
            dict(
                key=spec.key,
                did=spec.did,
                source=spec.source,
                record=spec.record,
                first_public=spec.first_public,
                license=spec.license,
                target=spec.target,
                tabpfn35_safe=spec.tabpfn35_safe,
                n=r["n"],
                d=r["d"],
                C=r["C"],
                class_counts=r["class_counts"],
                sha256=r["sha256"],
                raw_files=[
                    {k: f[k] for k in ("name", "url", "md5", "md5_repo", "sha256")}
                    for f in files
                    if f["key"] == spec.key
                ],
            )
        )
    return dict(datasets=kept, excluded=excluded)


def main(argv: Sequence[str]) -> None:  # pragma: no cover - one-off build
    """``python -I -m simaudit.data.second_frame <root> <report.json> <pool.yaml>``."""
    import yaml

    from simaudit.settings import PROJECT_ROOT

    root, report, pool = Path(argv[0]), Path(argv[1]), Path(argv[2])
    records = build(
        root, reference_hashes(PROJECT_ROOT), reference_arrays(PROJECT_ROOT)
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=1, ensure_ascii=False))
    files = json.loads((root / "fetch_log.json").read_text())["files"]
    pool.write_text(
        yaml.safe_dump(
            pool_entries(records, files), allow_unicode=True, sort_keys=False
        )
    )
    for r in records:
        print(r["key"], r.get("n"), r.get("d"), r.get("C"), r.get("excluded"))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
