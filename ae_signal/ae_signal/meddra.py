"""MedDRA dictionary loader and LLT candidate search.

Reads the standard MedDRA ASCII distribution (``$``-delimited ``.asc`` files):

  llt.asc     llt_code$llt_name$pt_code$...$llt_currency$...
  mdhier.asc  pt_code$hlt_code$hlgt_code$soc_code$pt_name$hlt_name$hlgt_name$
              soc_name$soc_abbrev$null$pt_soc_code$primary_soc_fg$

Only the primary-SOC path of each PT is used for roll-ups. The agents never
invent codes: they pick from search results, and every code they return is
re-validated against this dictionary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rapidfuzz import fuzz, process, utils


@dataclass(frozen=True)
class Term:
    llt_code: str
    llt_name: str
    pt_code: str
    pt_name: str
    hlt_name: str
    hlgt_name: str
    soc_code: str
    soc_name: str


def _read_asc(path: str) -> list[list[str]]:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return [line.rstrip("\r").split("$") for line in text.split("\n") if line.strip()]


def _find(directory: str, name: str) -> str:
    for candidate in (name, name.upper()):
        for sub in ("", "ascii", "MedAscii"):
            path = os.path.join(directory, sub, candidate)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"{name} not found under {directory}")


class MedDRA:
    def __init__(self, directory: str):
        self.version = self._read_version(directory)

        hierarchy: dict[str, list[str]] = {}
        for row in _read_asc(_find(directory, "mdhier.asc")):
            pt_code, primary = row[0], row[11] if len(row) > 11 else "Y"
            if primary == "Y" or pt_code not in hierarchy:
                hierarchy[pt_code] = row

        self.terms: dict[str, Term] = {}
        for row in _read_asc(_find(directory, "llt.asc")):
            llt_code, llt_name, pt_code = row[0], row[1], row[2]
            currency = row[9] if len(row) > 9 else "Y"
            h = hierarchy.get(pt_code)
            if currency != "Y" or h is None:
                continue
            self.terms[llt_code] = Term(llt_code, llt_name, pt_code, h[4], h[5], h[6], h[3], h[7])

        self.pts: dict[str, Term] = {}
        for t in self.terms.values():
            if t.llt_code == t.pt_code:
                self.pts[t.pt_code] = t
        self._codes = list(self.terms)
        self._names = [self.terms[c].llt_name for c in self._codes]

    @staticmethod
    def _read_version(directory: str) -> str | None:
        try:
            row = _read_asc(_find(directory, "meddra_release.asc"))[0]
            return row[0]
        except (FileNotFoundError, IndexError):
            return None

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """Fuzzy-match current LLT names; returns candidates with their PT and SOC."""
        hits = process.extract(
            query, self._names, scorer=fuzz.WRatio, processor=utils.default_process, limit=top_k
        )
        out = []
        for _name, score, idx in hits:
            t = self.terms[self._codes[idx]]
            out.append({
                "llt_code": t.llt_code, "llt_name": t.llt_name,
                "pt_code": t.pt_code, "pt_name": t.pt_name,
                "soc_name": t.soc_name, "match_score": round(score / 100, 2),
            })
        return out

    def get(self, llt_code: str) -> Term | None:
        return self.terms.get(str(llt_code).strip())
