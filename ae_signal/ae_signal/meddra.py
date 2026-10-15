"""MedDRA dictionary loader and LLT candidate search.

Two source formats are supported:

1. The standard MedDRA ASCII distribution (a directory of ``$``-delimited files):

     llt.asc     llt_code$llt_name$pt_code$...$llt_currency$...
     mdhier.asc  pt_code$hlt_code$hlgt_code$soc_code$pt_name$hlt_name$hlgt_name$
                 soc_name$soc_abbrev$null$pt_soc_code$primary_soc_fg$

   Full hierarchy; only the primary-SOC path of each PT is used for roll-ups.

2. A two-column LLT -> PT name table (``.xlsx`` / ``.csv``, columns ``LLT`` and
   ``PT``), e.g. ``MedDRA_28_0_AllPTsLLTs.xlsx``. There are no codes, so term
   names act as identifiers (``llt_code`` = LLT name, ``pt_code`` = PT name) and
   HLT/HLGT/SOC are empty, which disables SOC-level roll-ups.

The agents never invent terms: they pick from search results, and every term
they return is re-validated against this dictionary.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process, utils


@dataclass(frozen=True)
class Term:
    llt_code: str
    llt_name: str
    pt_code: str
    pt_name: str
    hlt_name: str = ""
    hlgt_name: str = ""
    soc_code: str = ""
    soc_name: str = ""


def _key(s: str) -> str:
    return " ".join(str(s).lower().split())


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", utils.default_process(s)) if len(t) > 2]


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
    def __init__(self, source: str):
        if os.path.isdir(source):
            self.version = self._read_version(source)
            terms = self._load_ascii(source)
        else:
            m = re.search(r"(\d{2})[_.](\d)", os.path.basename(source))
            self.version = f"{m.group(1)}.{m.group(2)}" if m else None
            terms = self._load_table(source)

        self.terms: dict[str, Term] = {}  # normalised llt_code -> Term
        self.pts: dict[str, Term] = {}  # normalised pt_code and pt_name -> PT-level Term
        for t in terms:
            self.terms.setdefault(_key(t.llt_code), t)
        for t in terms:
            pt = Term(t.pt_code, t.pt_name, t.pt_code, t.pt_name, t.hlt_name, t.hlgt_name, t.soc_code, t.soc_name)
            self.pts.setdefault(_key(t.pt_code), pt)
            self.pts.setdefault(_key(t.pt_name), pt)
        # Every PT is itself selectable, even where the source lists no same-name LLT.
        for pt in list(self.pts.values()):
            self.terms.setdefault(_key(pt.llt_code), pt)

        self._by_name = {}
        for t in self.terms.values():
            self._by_name.setdefault(_key(t.llt_name), t)

        self.has_hierarchy = any(t.soc_name for t in self.terms.values())
        self._codes = list(self.terms)
        self._names = [self.terms[c].llt_name for c in self._codes]
        # Token index: fuzzy scoring runs only over terms sharing a word with the
        # query, which is ~100x faster than scanning all LLTs.
        self._index: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self._names):
            for tok in set(_tokens(name)):
                self._index[tok].append(i)
        self.search = lru_cache(maxsize=50_000)(self._search)

    @staticmethod
    def _load_ascii(directory: str) -> list[Term]:
        hierarchy: dict[str, list[str]] = {}
        for row in _read_asc(_find(directory, "mdhier.asc")):
            pt_code, primary = row[0], row[11] if len(row) > 11 else "Y"
            if primary == "Y" or pt_code not in hierarchy:
                hierarchy[pt_code] = row
        terms = []
        for row in _read_asc(_find(directory, "llt.asc")):
            llt_code, llt_name, pt_code = row[0], row[1], row[2]
            currency = row[9] if len(row) > 9 else "Y"
            h = hierarchy.get(pt_code)
            if currency == "Y" and h is not None:
                terms.append(Term(llt_code, llt_name, pt_code, h[4], h[5], h[6], h[3], h[7]))
        return terms

    @staticmethod
    def _load_table(path: str) -> list[Term]:
        df = pd.read_excel(path, dtype=str) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
        df.columns = [c.strip().upper() for c in df.columns]
        df = df[["LLT", "PT"]].dropna().apply(lambda s: s.str.strip()).drop_duplicates()
        return [Term(llt, llt, pt, pt) for llt, pt in df.itertuples(index=False)]

    @staticmethod
    def _read_version(directory: str) -> str | None:
        try:
            return _read_asc(_find(directory, "meddra_release.asc"))[0][0]
        except (FileNotFoundError, IndexError):
            return None

    def _search(self, query: str, top_k: int = 15) -> list[dict]:
        """Fuzzy-match LLT names; returns candidates with their PT (and SOC when known)."""
        candidates = sorted({i for tok in _tokens(query) for i in self._index.get(tok, ())})
        if candidates:
            choices = {i: self._names[i] for i in candidates}
        else:  # misspelling or abbreviation: no shared word, so scan everything
            choices = dict(enumerate(self._names))
        hits = process.extract(query, choices, scorer=fuzz.WRatio, processor=utils.default_process, limit=top_k)
        out = []
        for _name, score, idx in hits:
            t = self.terms[self._codes[idx]]
            hit = {"llt_code": t.llt_code, "llt_name": t.llt_name, "pt_code": t.pt_code, "pt_name": t.pt_name}
            if t.soc_name:
                hit["soc_name"] = t.soc_name
            hit["match_score"] = round(score / 100, 2)
            out.append(hit)
        return out

    def get(self, llt_code: str) -> Term | None:
        """Look up an LLT by code, falling back to its exact (case-insensitive) name."""
        k = _key(llt_code)
        return self.terms.get(k) or self._by_name.get(k)

    def to_pt(self, term: str) -> Term | None:
        """Resolve a PT name/code, or an LLT name/code, to its PT."""
        if not term or pd.isna(term):
            return None
        pt = self.pts.get(_key(term))
        if pt is None and (llt := self.get(term)) is not None:
            pt = self.pts.get(_key(llt.pt_code))
        return pt
