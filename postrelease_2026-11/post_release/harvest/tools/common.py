"""P1 CONFIRM 2026-11-08 copy of the CONF3 helpers: CUT 2026-09-24 (after TabPFN v3.5 2026-09-09 and Kumo-Tabular 2026-09-23);
reserved ids extended with the S2 frame/manifest/decisions, the G4 frames, conf2/conf3 decisions/specs/reports.
CONF3 copy (2026-11-07) of the S2 helpers: CUT 2026-09-10, outcome/ecology keyword score, imbalance-aware mech screen.
Shared helpers for the S2 K>=5 fresh harvest (metadata only). Reserved record/concept ids are read as text from
reserved-pool files (same rule as E7 s3_rank.py) + every record in the E7 manifest/decisions. sealed_heldout never read."""
import os, re, json, time, urllib.request, urllib.error
from pathlib import Path
# HARVEST_WORKSPACE: folder holding simaudit/ and the earlier harvest copies; HARVEST_PROBES: the earlier
# probe folders (E7, S2, CONF3, semprior). Missing files are skipped, as before.
G = Path(os.environ.get("HARVEST_WORKSPACE", "<workspace>")); H = Path(__file__).resolve().parents[1]
PROBES = Path(os.environ.get("HARVEST_PROBES", str(G / "probes")))
E7 = PROBES / "fenomeno_bet1_e7_2026-11-03/harvest"; SP = PROBES / "semprior_v6_conf_2026-11-02"
CUT = "2026-09-24"   # P1 confirm window (strictly after every FM release)
UA = {"User-Agent": "Mozilla/5.0 (compatible; research-metadata-harvest)"}

def reserved_ids():
    src = [G / "simaudit/data" / f for f in ("pool_second_frame.yaml", "pool_study17.yaml", "pool_study18.yaml", "pool_study16_candidates.csv",
                                            "pool_t2.yaml", "pool_t1prime.yaml", "t2_metadata_candidates_DRAFT.csv", "pool_study18_candidates.csv")]
    src += [G / "simaudit/tools/study17/postrelease_specs.jsonl", G / "simaudit/tools/study17/postrelease_specs_v2.jsonl", SP / "curation/pass_v6.jsonl"]
    src += list((PROBES / "pool_posfm_confirm_2026-11-01").glob("*.csv")) + list((PROBES / "pool_posfm_confirm_v2_2026-11-01").glob("*.csv"))
    src += [E7 / "manifest.csv", E7 / "curation/specs_e7.jsonl"] + list((E7 / "curation").glob("decisions_*.py"))
    S2 = PROBES / "S2_k5_fresh_2026-11-07"; G4 = PROBES / "class_balancing_g4_gates_2026-11-05"
    src += [S2 / "frame.csv", S2 / "harvest/manifest.csv", S2 / "harvest/curation/decisions.py", S2 / "harvest/curation/build_report.csv", S2 / "harvest/specs.jsonl",
            G4 / "out/frame/frame.csv", G4 / "conf2/frame_conf2.csv", G4 / "conf3/harvest/curation/decisions.py", G4 / "conf3/harvest/curation/build_report.csv",
            G4 / "conf3/harvest/specs.jsonl", E7 / "curation/build_report.csv"]
    txt = "\n".join(p.read_text(errors="ignore") for p in src if p.exists())
    ids = set(re.findall(r"(?<!\d)(\d{6,9})(?!\d)", txt))
    ids |= {m.lower() for m in re.findall(r"(?:mnd|m_)([a-z0-9]{10})(?![a-z0-9])", txt)}          # Mendeley dataset ids
    ids |= {m.upper() for m in re.findall(r"(?:dvn|d_)([A-Z0-9]{6})(?![A-Za-z0-9])", txt)}        # Dataverse DOI suffixes
    ids |= {m.upper() for m in re.findall(r"10\.\d{4,5}/[A-Za-z]+/([A-Z0-9]{6})", txt)}
    return ids

def get(url, cap=None, timeout=60):
    for t in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                if cap and int(r.headers.get("Content-Length") or 0) > cap:
                    raise ValueError("too big")
                b = r.read((cap or 80 * 2**20) + 1)
            if cap and len(b) > cap:
                raise ValueError("too big")
            return b
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After") or 60)); continue
            if e.code in (400, 401, 403, 404, 410, 422):
                raise
            time.sleep(3 + 5 * t)
        except ValueError:
            raise
        except Exception:
            time.sleep(3 + 5 * t)
    raise RuntimeError("failed")

import re as _re
MC = _re.compile(r"species|genus|cultivar|variet|land.?cover|land.?use|facies|litholog|soil|fault|activity|gesture|grade|stage|severity|subtype|phenotyp|genotyp|"
                 r"emotion|quality|blood|tumou?r|cell type|weather|crop|material|defect|behaviou?r|diet|origin|provenance|classif|multi-?class|categor|type|taxonom|"
                 r"morpholog|identif|discriminat|authentic|accession|landrace|breed|strain|isolate|mineral|rock|sediment|pollen|insect|fish|bird|plant|tree|seed|"
                 r"leaf|wood|honey|wine|tea\b|coffee|milk|rice|wheat|maize|diagnos|patient|disease|stress|recognition|machine learning|habitat|community|forest|"
                 r"geochem|archaeolog|ceramic|lithic|bone|skull|fossil|otolith|morphometr|survey|clinical|cohort|sensor|vibration|bearing", _re.I)
NEG = _re.compile(r"synthetic|simulat|bibliometric|meta-analys|systematic review|scoping review|benchmark of llm|LLM|price|stock market|cryptocurr", _re.I)

def score(text):
    return len(set(w.lower()[:6] for w in MC.findall(text))) - 3 * bool(NEG.search(text))

def probe_df(df):
    df = df.dropna(how="all"); n, d = df.shape
    cands = []
    for c in df.columns:
        vc = df[c].dropna().astype(str).str.strip(); vc = vc[~vc.isin(["", "NA", "nan"])].value_counts()
        if 2 <= len(vc) <= 10 and vc.min() >= 10 and vc.sum() >= 0.8 * n:
            cands.append(f"{c}({len(vc)}:{vc.min()})")
    return dict(n=n, d=d, cands="|".join(cands)[:800], cols="|".join(map(str, df.columns[:120]))[:2500])

def probe_csv(p):
    import pandas as pd
    head = p.open(errors="ignore").read(20000); first = head.split("\n", 1)[0]
    sep = max([",", ";", "\t", "|"], key=first.count)
    df = pd.read_csv(p, sep=sep, nrows=20000, low_memory=False, encoding_errors="ignore", on_bad_lines="skip", dtype=str)
    return probe_df(df)

def probe_grid(g):
    import pandas as pd
    g = [r for r in g if any(str(c).strip() for c in r)]
    if len(g) < 3:
        return dict(n=0, d=0, cands="", cols="")
    hdr = [str(h) for h in g[0]]; w = len(hdr)
    body = [list(r[:w]) + [""] * (w - len(r[:w])) for r in g[1:]]
    df = pd.DataFrame(body, columns=[f"{h}" for h in hdr]).replace("", None)
    df = df.loc[:, ~df.columns.duplicated()]
    return probe_df(df)


# ---- CONF3 additions: outcome/ecology keyword score and the mechanical screen (imbalanced binary or 3..10-level column)
DOM3 = _re.compile(r"patient|clinic|hospital|cohort|mortality|surviv|disease|infect|cancer|diagnos|prognos|outcome|predict|species|presence|occurrence|"
                   r"occupancy|ecolog|wildlife|fish|bird|plant|pest|pathogen|farm|cattle|livestock|dairy|veterin|epidem|surveillance|complication|adverse|"
                   r"icu|pregnan|neonat|maternal|stroke|sepsis|tubercul|malaria|hiv|covid|retrospective|case-control|nest|predat|mosquito|parasit|tumou?r|"
                   r"surgery|surgical|postoperative|risk factor|incidence|prevalence|seropreval|antimicrobial|resistance|forest|tree|habitat", _re.I)
NEG3 = _re.compile(r"synthetic|simulat|bibliometric|meta-analys|systematic review|scoping review|questionnaire|perception|attitude|students?|teacher|"
                   r"likert|LLM|benchmark|price|stock market|crypto|software|literature", _re.I)

def score(text):
    return len(set(w.lower()[:6] for w in DOM3.findall(text))) - 3 * bool(NEG3.search(text))

def mech3(pr):
    n = pr["n"] or 0; ok = False
    for k, m in _re.findall(r"\((\d+):(\d+)\)", pr["cands"]):
        k, m = int(k), int(m)
        ok |= k >= 3 or (k == 2 and m <= n / 4.5)
    return int(100 <= n <= 5000 and 2 <= (pr["d"] or 0) <= 160 and ok)
