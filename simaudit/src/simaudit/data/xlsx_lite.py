"""Minimal standard-library reader for .xlsx sheets (values as strings).

Used for untrusted downloaded workbooks (Study 15): no formula is evaluated,
no macro or external link is followed, nothing is executed. Only
``xl/workbook.xml``, ``xl/_rels/workbook.xml.rels``, ``xl/sharedStrings.xml``
and the chosen sheet XML are parsed. Cached cell values are returned as
written in the file (numbers in their stored text form, so dates stay serial
numbers); inline and shared strings are decoded.
"""

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
MAX_MEMBER = 200 * 2**20  # refuse decompressed members above 200 MB
_REF = re.compile(r"^([A-Z]+)(\d+)$")


def _read(z: zipfile.ZipFile, name: str) -> bytes:
    info = z.getinfo(name)
    if info.file_size > MAX_MEMBER:
        raise ValueError(f"{name}: member too large ({info.file_size} bytes)")
    return z.read(info)


def _col_index(letters: str) -> int:
    out = 0
    for ch in letters:
        out = out * 26 + (ord(ch) - 64)
    return out - 1


def _text(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.iter(f"{{{NS['m']}}}t"))


def sheet_names(path: Path) -> list[str]:
    """Sheet names in workbook order."""
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(_read(z, "xl/workbook.xml"))
    return [s.get("name", "") for s in wb.iter(f"{{{NS['m']}}}sheet")]


def _sheet_member(z: zipfile.ZipFile, sheet: int | str) -> str:
    wb = ET.fromstring(_read(z, "xl/workbook.xml"))
    sheets = list(wb.iter(f"{{{NS['m']}}}sheet"))
    chosen = (
        sheets[sheet]
        if isinstance(sheet, int)
        else next(s for s in sheets if s.get("name") == sheet)
    )
    rid = chosen.get(f"{{{NS['r']}}}id")
    rels = ET.fromstring(_read(z, "xl/_rels/workbook.xml.rels"))
    target = next(
        r.get("Target", "")
        for r in rels.iter(f"{{{NS['pr']}}}Relationship")
        if r.get("Id") == rid
    )
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.get("t")
    value = cell.find("m:v", NS)
    if kind == "inlineStr":
        return _text(cell)
    if value is None or value.text is None:
        return ""
    return shared[int(value.text)] if kind == "s" else value.text


def read_sheet(path: Path, sheet: int | str = 0) -> list[list[str]]:
    """Rows of one sheet as lists of strings ('' for empty cells), rectangular."""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = ET.fromstring(_read(z, "xl/sharedStrings.xml"))
            shared = [_text(si) for si in sst.findall("m:si", NS)]
        root = ET.fromstring(_read(z, _sheet_member(z, sheet)))
    rows: dict[int, dict[int, str]] = {}
    for row in root.iter(f"{{{NS['m']}}}row"):
        for cell in row.findall("m:c", NS):
            match = _REF.match(cell.get("r", ""))
            text = _cell_text(cell, shared)
            if match is not None and text:  # empty cells add no row or column
                col, line = _col_index(match.group(1)), int(match.group(2)) - 1
                rows.setdefault(line, {})[col] = text
    if not rows:
        return []
    width = 1 + max(c for r in rows.values() for c in r)
    return [
        [rows.get(i, {}).get(j, "") for j in range(width)] for i in range(1 + max(rows))
    ]
