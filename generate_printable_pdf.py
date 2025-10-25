#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_printable_pdf.py — context-rich errors

This version adds detailed, human-readable context to ALL exceptions so failures are
self-diagnosable without a debugger:

- Every error includes PHASE, FILE, GROUP, COVER, PAGE, and an actionable HINT when useful.
- Root cause is preserved via "raise ... from e" and included inline as "CAUSE: repr(e)".
- Strict, fail-loud behavior remains identical to the previous version.
- Smaller default font sizes (body/mono) and kept leading & spacing proportional so
  pagination estimates remain correct automatically.
- Cover page is more stylized while remaining ink-light (thin rules, subtle accents).
- Highlight spans (<mark-yellow>...</mark-yellow>) now render as **bold + underlined**
  instead of gray highlighter fills (better for black & white printers).
- Title baseline aligned so “Cover N” sits exactly where intended (no half-line drift).
- Cover shows content page range and groups range in math notation (e.g., pages [2, 21], groups [1, 7]).
 - Group pages titled by rank as “Group i/NumGroups” (the denominator is the number of adjacent groups).
- Underlines now span through spaces for continuous emphasis.
- Each cover page includes:
    1) The Deep Intent (search) query — read from the newest './search_runs/xxxx.jsonl' (first RS record, 'query').
    2) The Ranking (reranker) query — read from './rerank/progress.jsonl' (first JSONL line, 'query').

  These appear inside a styled, two-column “Intent & Ranking” block on the cover:
    - Left: Deep intent (search) query
    - Right: Ranking query (reranker)

  Word-wrapped, ink-light, fail-loud if missing or malformed. Layout flows gracefully
  and preserves the rest of the cover content.

Usage:
    python generate_printable_pdf.py --out ./printable/book.pdf --max-pages 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import glob
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# ---------- Strict FS locations ----------
YELLOW_DIR = Path("./yellow_marker")
ASSETS_DIR = Path("./assets")
FONT_SANS = ASSETS_DIR / "DejaVuSans.ttf"
FONT_MONO = ASSETS_DIR / "DejaVuSansMono.ttf"

# *** NEW: sources for queries ***
SEARCH_RUNS_DIR = Path("./search_runs")
RERANK_PROGRESS_PATH = Path("./rerank/progress.jsonl")
RS_DELIM = "\x1e"  # same record separator used elsewhere in the project

# ---------- Parsing / markers ----------
CODEBLOCK_RE = re.compile(r"```txt\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
FNAME_RE = re.compile(r"^\d{6}\.txt$")
MARK_OPEN = "<mark-yellow>"
MARK_CLOSE = "</mark-yellow>"

# ---------- ReportLab (fail loud if missing) ----------
try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import black
except Exception as e:
    raise RuntimeError(
        "PHASE: import\n"
        "WHERE: reportlab\n"
        "WHAT: Missing or broken ReportLab dependency.\n"
        "HINT: pip install reportlab\n"
        f"CAUSE: {e!r}"
    ) from e

# ---------- Page geometry (A4 portrait) ----------
PAGE_W, PAGE_H = A4
MARGIN_L = 54.0
MARGIN_R = 54.0
MARGIN_T = 60.0
MARGIN_B = 60.0

# ---------- Fonts & sizes ----------
FONT_SANS_NAME = "DejaVuSans"
FONT_MONO_NAME = "DejaVuSansMono"

# Decreased defaults (B/W friendly) — pagination auto-adjusts since leading derives from size.
DEFAULT_BODY_SIZE = 9.4     # was 10.6
DEFAULT_MONO_SIZE = 8.8     # was 9.8
LEADING_FACTOR = 1.24       # slight tightening from 1.28 to match smaller fonts

# ---------- Ink-light styling parameters ----------
# Note: We no longer use gray highlighter rectangles for marked spans.
# Keep grays for hairline rules & accents only.
HILITE_GRAY = 0.85     # kept for cover accents (no big fills)
RULE_GRAY = 0.6
COVER_RULE_GRAY = 0.5

# Faux-bold & underline settings for marked spans (printer-friendly)
FAUX_BOLD_OFFSET = 0.25        # small x-offset to "double-stroke" text for a bolder feel
UNDERLINE_THICKNESS = 0.6      # hairline; readable but not inky
UNDERLINE_OFFSET = 1.0         # distance below baseline (pt)

# Cover title sizing and baseline tweak to align visually with accent bar
COVER_TITLE_SIZE = 28
COVER_TITLE_BASELINE_ADJ = 12.0  # move baseline downward to avoid sitting "half a line" high

# Intent & Ranking panel tuning (cover page)
INTENT_PANEL_LABEL_SIZE = 10.5
INTENT_PANEL_TEXT_SIZE  = 9.6
INTENT_PANEL_GAP_Y      = 10.0
INTENT_PANEL_COL_GAP    = 16.0
INTENT_PANEL_FRAME_GRAY = 0.75
INTENT_PANEL_LABEL_GRAY = 0.35

# ===================== Context & helpers =====================
@dataclass
class Ctx:
    phase: str = ""
    file: str = ""
    group: str = ""
    cover: Optional[int] = None
    page: Optional[int] = None
    detail: str = ""

    def with_(self, **kw) -> "Ctx":
        d = asdict(self)
        d.update(kw)
        return Ctx(**d)

def ctx_raise(ctx: Ctx, message: str, hint: str = "", cause: Optional[BaseException] = None) -> None:
    parts = [
        f"PHASE: {ctx.phase or '-'}",
        f"FILE: {ctx.file or '-'}",
        f"GROUP: {ctx.group or '-'}",
        f"COVER: {ctx.cover if ctx.cover is not None else '-'}",
        f"PAGE: {ctx.page if ctx.page is not None else '-'}",
        f"WHAT: {message}",
    ]
    if ctx.detail:
        parts.append(f"DETAIL: {ctx.detail}")
    if hint:
        parts.append(f"HINT: {hint}")
    if cause is not None:
        parts.append(f"CAUSE: {cause!r}")
    full = "\n".join(parts)
    if cause is not None:
        raise RuntimeError(full) from cause
    raise RuntimeError(full)

# ===================== Data structures =====================
@dataclass
class GroupInput:
    filename: str
    json_rows: List[Dict[str, Any]]
    text: str  # inner of ```txt

@dataclass
class GroupPrepared:
    filename: str
    json_rows: List[Dict[str, Any]]  # evidence removed except top rank
    text: str
    ranks_display: str
    relevant_count: int
    content_pages: int = 0

@dataclass
class Cover:
    index: int
    groups: List[GroupPrepared]
    pages_in_cover: int
    total_relevants: int

# ===================== Fonts =====================
def register_fonts(ctx: Ctx) -> None:
    try:
        if not FONT_SANS.is_file():
            ctx_raise(ctx, f"Missing font file: {FONT_SANS}", "Verify ./assets/DejaVuSans.ttf exists.")
        if not FONT_MONO.is_file():
            ctx_raise(ctx, f"Missing font file: {FONT_MONO}", "Verify ./assets/DejaVuSansMono.ttf exists.")
        pdfmetrics.registerFont(TTFont(FONT_SANS_NAME, str(FONT_SANS)))
        pdfmetrics.registerFont(TTFont(FONT_MONO_NAME, str(FONT_MONO)))
    except Exception as e:
        ctx_raise(ctx, "Failed to register fonts.", "Fonts must be readable and valid TTF files.", e)

# ===================== File discovery & parsing =====================
def list_yellow_files_strict(ctx: Ctx) -> List[Path]:
    try:
        if not YELLOW_DIR.is_dir():
            ctx_raise(ctx, f"Missing directory: {YELLOW_DIR}", "Run yellow_marker.py first.")
        files = sorted(p for p in YELLOW_DIR.iterdir() if p.is_file() and FNAME_RE.match(p.name))
        if not files:
            ctx_raise(ctx, f"No files matching ######.txt under {YELLOW_DIR}", "Expected e.g. 000001.txt, 000002.txt ...")
        return files
    except Exception as e:
        ctx_raise(ctx, "Failed listing yellow_marker files.", "", e)

def parse_one_yellow_file(path: Path, ctx: Ctx) -> GroupInput:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        ctx_raise(ctx.with_(file=path.name, phase="read"), "Failed to read file.", f"Check permissions and encoding.", e)

    m = CODEBLOCK_RE.search(raw)
    if not m:
        snippet = raw[:200].replace("\n", "\\n")
        ctx_raise(
            ctx.with_(file=path.name, phase="parse"),
            "Expected exactly one ```txt code block, but none was found.",
            f"Regex: {CODEBLOCK_RE.pattern}\nFirst 200 chars: {snippet!r}",
        )
    if len(CODEBLOCK_RE.findall(raw)) != 1:
        ctx_raise(
            ctx.with_(file=path.name, phase="parse"),
            "Found more than one ```txt code block.",
            "Yellow files must contain exactly one txt code block.",
        )

    inner = m.group(1)
    prefix = raw[: m.span()[0]]
    if not prefix.lstrip().startswith("["):
        snippet = prefix[:120].replace("\n", "\\n")
        ctx_raise(
            ctx.with_(file=path.name, phase="parse"),
            "Content before code block must start with a JSON list ('[').",
            f"Starts with: {snippet!r}",
        )
    try:
        rows = json.loads(prefix.strip())
    except Exception as e:
        bad = prefix.strip()[:400]
        ctx_raise(
            ctx.with_(file=path.name, phase="parse"),
            "JSON header validation failed.",
            f"First 400 chars of JSON: {bad!r}",
            e,
        )
    if not isinstance(rows, list):
        ctx_raise(ctx.with_(file=path.name, phase="parse"), "JSON header must be a list of objects.")
    return GroupInput(filename=path.name, json_rows=rows, text=inner)

def parse_rank_strict(rank_str: str, ctx: Ctx, file: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", rank_str or "")
    if not m:
        ctx_raise(
            ctx.with_(file=file, phase="parse"),
            f"Invalid relevance_score format: {rank_str!r}",
            "Expected 'X/Y' (e.g., '200/525').",
        )
    return int(m.group(1)), int(m.group(2))

def massage_json_rows_keep_top_evidence(rows: List[Dict[str, Any]], ctx: Ctx, file: str) -> Tuple[List[Dict[str, Any]], str, int]:
    if not rows:
        return rows, "", 0
    parsed: List[Tuple[int, int, int]] = []
    for i, r in enumerate(rows):
        rs = r.get("relevance_score")
        if not isinstance(rs, str):
            ctx_raise(ctx.with_(file=file, phase="parse"), "JSON row missing string 'relevance_score'.", f"Row index: {i}")
        num, den = parse_rank_strict(rs, ctx, file)
        parsed.append((num, den, i))
    parsed.sort(key=lambda t: t[0])
    best_idx = parsed[0][2]
    denom_set = {p[1] for p in parsed}
    denom = parsed[0][1] if len(denom_set) == 1 else None

    rank_nums = [p[0] for p in parsed]
    if denom is not None:
        ranks_disp = " · ".join(f"#{n}" for n in rank_nums)
    else:
        ranks_disp = " · ".join(f"#{n}/{d}" for (n, d, _) in parsed)

    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        d = dict(r)
        if i != best_idx and "evidence_text" in d:
            del d["evidence_text"]
        out_rows.append(d)
    return out_rows, ranks_disp, len(rows)

# ===================== Text wrapping & highlights =====================
def string_width(txt: str, font_name: str, font_size: float) -> float:
    return pdfmetrics.stringWidth(txt, font_name, font_size)

def wrap_mono(lines: List[str], max_width: float, font_size: float) -> List[str]:
    out: List[str] = []
    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        if line == "":
            out.append("")
            continue
        words = re.split(r"(\s+)", line)
        current = ""
        while words:
            token = words.pop(0)
            trial = current + token
            if string_width(trial, FONT_MONO_NAME, font_size) <= max_width:
                current = trial
                continue
            if current:
                out.append(current)
                current = ""
                if string_width(token, FONT_MONO_NAME, font_size) <= max_width:
                    current = token
                else:
                    s = token
                    while s:
                        lo, hi = 0, len(s)
                        cut = 0
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if string_width(s[: mid + 1], FONT_MONO_NAME, font_size) <= max_width:
                                cut = mid + 1
                                lo = mid + 1
                            else:
                                hi = mid
                        if cut == 0:
                            # add context if we somehow can't even place one char (shouldn't happen)
                            raise RuntimeError(f"wrap_mono failed to place any chars at line {idx}.")
                        out.append(s[:cut])
                        s = s[cut:]
            else:
                s = token
                while s:
                    lo, hi = 0, len(s)
                    cut = 0
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if string_width(s[: mid + 1], FONT_MONO_NAME, font_size) <= max_width:
                            cut = mid + 1
                            lo = mid + 1
                        else:
                            hi = mid
                    if cut == 0:
                        cut = 1
                    out.append(s[:cut])
                    s = s[cut:]
        if current != "":
            out.append(current)
    return out

def split_mark_spans(text: str, ctx: Ctx, file: str) -> List[Tuple[str, bool]]:
    """
    Scan the WHOLE text once and return (segment, is_marked) runs.
    Newlines are preserved in segments; we do NOT split per-line here.
    Fail loud on unbalanced or nested markers.
    """
    runs: List[Tuple[str, bool]] = []
    pos = 0
    marked = False
    safety_counter = 0

    while True:
        # Find next relevant token depending on current state
        target = MARK_OPEN if not marked else MARK_CLOSE
        i = text.find(target, pos)
        if i == -1:
            # Remainder (could include newlines)
            seg = text[pos:]
            if seg:
                runs.append((seg, marked))
            break

        # Preceding segment (could include newlines)
        seg = text[pos:i]
        if seg:
            runs.append((seg, marked))

        # Toggle mark state and advance
        marked = not marked
        pos = i + len(target)

        safety_counter += 1
        if safety_counter > 1_000_000:
            ctx_raise(
                ctx.with_(file=file, phase="wrap"),
                "Aborting: excessive mark tag toggles (>1M).",
                "Likely malformed or cyclic marker content.",
            )

    if marked:
        ctx_raise(
            ctx.with_(file=file, phase="wrap"),
            "Unbalanced <mark-yellow> tags detected (missing closing tag).",
            "Ensure yellow_marker produced balanced tags (spans may cross newlines; that's OK).",
        )
    return runs

def wrap_body_with_marks(
    text: str,
    max_width: float,
    font_size: float,
    ctx: Ctx,
    file: str
) -> List[List[Tuple[str, bool]]]:
    """
    Wrap body text into lines while respecting:
      - Highlight spans (from split_mark_spans over the WHOLE text)
      - Hard line breaks on '\n'
    Returns: list of lines; each line is [(segment, is_marked), ...].
    """
    # 1) Get full-text runs with mark state preserved across newlines
    runs = split_mark_spans(text, ctx, file)

    # 2) Tokenize into (token, is_marked, is_newline)
    #    We keep whitespace tokens for accurate width and look.
    tokens: List[Tuple[str, bool, bool]] = []
    for seg, mk in runs:
        if not seg:
            continue
        # Split on '\n' but keep the separators
        parts = re.split(r"(\n)", seg)
        for p in parts:
            if p == "":
                continue
            if p == "\n":
                tokens.append((p, False, True))  # newline forces a break; not highlighted as a char
            else:
                # Further split on whitespace to allow wrapping at spaces, but keep spaces
                subparts = re.split(r"(\s+)", p)
                for sp in subparts:
                    if sp == "":
                        continue
                    tokens.append((sp, mk, False))

    # 3) Greedy wrap by measuring token widths; honor newline tokens as hard breaks
    lines_out: List[List[Tuple[str, bool]]] = []
    current_line: List[Tuple[str, bool]] = []
    current_width = 0.0

    def flush_line():
        nonlocal current_line, current_width
        lines_out.append(current_line)
        current_line = []
        current_width = 0.0

    idx = 0
    safety = 0
    while idx < len(tokens):
        tok, mk, is_nl = tokens[idx]
        if is_nl:
            # Hard line break: flush current line, plus an explicit empty line to reflect the '\n'
            flush_line()
            lines_out.append([])  # blank line (paragraph break)
            idx += 1
            continue

        w = string_width(tok, FONT_SANS_NAME, font_size)
        if current_width + w <= max_width:
            current_line.append((tok, mk))
            current_width += w
            idx += 1
        else:
            if current_line:
                flush_line()
                # retry same token on the next line
            else:
                # Single token too wide: hard-break by binary search on characters
                s = tok
                lo, hi = 0, len(s)
                cut = 0
                while lo < hi:
                    mid = (lo + hi) // 2
                    if string_width(s[: mid + 1], FONT_SANS_NAME, font_size) <= max_width:
                        cut = mid + 1
                        lo = mid + 1
                    else:
                        hi = mid
                if cut == 0:
                    cut = 1
                current_line.append((s[:cut], mk))
                flush_line()
                tokens[idx] = (s[cut:], mk, False)

        safety += 1
        if safety > 3_000_000:
            ctx_raise(
                ctx.with_(file=file, phase="wrap"),
                "Aborting: excessive wrapping iterations (>3M).",
                "Potential degenerate tokenization.",
            )

    # Flush the last line (even if empty we’ll keep output consistent)
    lines_out.append(current_line)
    return lines_out

# Simple word-wrapper for Sans text (no marks, single paragraph)
def wrap_sans_words(text: str, max_width: float, font_size: float) -> List[str]:
    """
    Minimal, predictable word wrap for single-paragraph Sans text (no highlight state).
    Preserves multiple spaces where possible and breaks long tokens character-wise.
    """
    if not text:
        return [""]
    # Normalize newlines to spaces (these queries are single-paragraph)
    text = re.sub(r"\s*\n\s*", " ", text.strip())
    tokens = re.split(r"(\s+)", text)
    out: List[str] = []
    cur = ""
    for tok in tokens:
        if tok == "":
            continue
        trial = (cur + tok) if cur else tok
        if string_width(trial, FONT_SANS_NAME, font_size) <= max_width:
            cur = trial
            continue
        # need to emit cur and then place tok (may be long)
        if cur:
            out.append(cur)
            cur = ""
        if string_width(tok, FONT_SANS_NAME, font_size) <= max_width:
            cur = tok
        else:
            # break the token
            s = tok
            while s:
                lo, hi = 0, len(s)
                cut = 0
                while lo < hi:
                    mid = (lo + hi) // 2
                    if string_width(s[: mid + 1], FONT_SANS_NAME, font_size) <= max_width:
                        cut = mid + 1
                        lo = mid + 1
                    else:
                        hi = mid
                if cut == 0:
                    cut = 1
                out.append(s[:cut])
                s = s[cut:]
    if cur:
        out.append(cur)
    return out

# ===================== Pagination (dry run) =====================
@dataclass
class LayoutConfig:
    body_size: float
    mono_size: float
    leading_body: float
    leading_mono: float
    para_space: float

def page_text_height() -> float:
    return PAGE_H - MARGIN_T - MARGIN_B

def measure_group_pages(g: GroupPrepared, cfg: LayoutConfig, ctx: Ctx) -> int:
    try:
        usable_w = PAGE_W - MARGIN_L - MARGIN_R
        y = PAGE_H - MARGIN_T
        pages = 1

        def need(lines: int, leading: float) -> Tuple[int, float, int]:
            nonlocal y, pages
            for _ in range(lines):
                if y - leading < MARGIN_B:
                    pages += 1
                    y = PAGE_H - MARGIN_T
                y -= leading
            return pages, y, lines

        # title
        if y - cfg.leading_body < MARGIN_B:
            pages += 1
            y = PAGE_H - MARGIN_T
        y -= cfg.leading_body
        y -= cfg.para_space

        # JSON
        json_text = json.dumps(g.json_rows, ensure_ascii=False, indent=2)
        wrapped_mono = wrap_mono(json_text.split("\n"), usable_w, cfg.mono_size)
        pages, y, _ = need(len(wrapped_mono), cfg.leading_mono)
        y -= cfg.para_space

        # Body
        wrapped_body = wrap_body_with_marks(g.text, usable_w, cfg.body_size, ctx.with_(group=g.filename), g.filename)
        pages, y, _ = need(len(wrapped_body), cfg.leading_body)
        y -= cfg.para_space
        return pages
    except Exception as e:
        ctx_raise(ctx.with_(phase="paginate", group=g.filename), "Failed to measure group pages.", "", e)

# ===================== Cover planning =====================
def plan_covers(groups: List[GroupPrepared], max_pages: int, cfg: LayoutConfig, ctx: Ctx) -> List[Cover]:
    covers: List[Cover] = []
    current: List[GroupPrepared] = []
    current_pages = 0
    cover_idx = 1

    for g in groups:
        pages = measure_group_pages(g, cfg, ctx)
        g.content_pages = pages
        if current and current_pages + pages > max_pages:
            total_rels = sum(gr.relevant_count for gr in current)
            covers.append(Cover(index=cover_idx, groups=current, pages_in_cover=current_pages, total_relevants=total_rels))
            cover_idx += 1
            current = [g]
            current_pages = pages
        else:
            current.append(g)
            current_pages += pages

    if current:
        total_rels = sum(gr.relevant_count for gr in current)
        covers.append(Cover(index=cover_idx, groups=current, pages_in_cover=current_pages, total_relevants=total_rels))

    if not covers:
        ctx_raise(ctx.with_(phase="plan"), "Cover planning produced zero covers.", "Check inputs are non-empty.")
    return covers

# ===================== Drawing =====================
def draw_footer(c: rl_canvas.Canvas, page_num: int) -> None:
    c.setFont(FONT_SANS_NAME, 8.6)  # slightly smaller footer for decreased body size
    txt = f"{page_num}"
    w = pdfmetrics.stringWidth(txt, FONT_SANS_NAME, 8.6)
    c.setFillColor(black)
    c.drawString(PAGE_W - MARGIN_R - w, MARGIN_B * 0.5, txt)

def draw_rule(c: rl_canvas.Canvas, y: float, gray: float = RULE_GRAY) -> None:
    c.saveState()
    c.setFillGray(gray)
    c.rect(MARGIN_L, y - 0.5, PAGE_W - MARGIN_L - MARGIN_R, 0.5, stroke=0, fill=1)
    c.restoreState()

def start_new_page(c: rl_canvas.Canvas, page_num: int) -> int:
    if page_num > 0:
        draw_footer(c, page_num)
        c.showPage()
    return page_num + 1

# Readers for queries (strict)
def newest_run_file(ctx: Ctx) -> Path:
    try:
        files = sorted(SEARCH_RUNS_DIR.glob("*.jsonl"))
    except Exception as e:
        ctx_raise(ctx.with_(phase="query"), f"Failed to list {SEARCH_RUNS_DIR}/", "Ensure directory exists.", e)
    if not files:
        ctx_raise(ctx.with_(phase="query"), f"No run files found under {SEARCH_RUNS_DIR}/", "Run deep_search.py first.")
    return files[-1]

def read_first_rs_json(path: Path, ctx: Ctx) -> Dict[str, Any]:
    try:
        blob = path.read_text(encoding="utf-8")
    except Exception as e:
        ctx_raise(ctx.with_(phase="query", file=str(path)), "Failed to read search run file.", "", e)
    parts = [p for p in blob.split(RS_DELIM) if p.strip()]
    if not parts:
        ctx_raise(ctx.with_(phase="query", file=str(path)), "Run file contains no RS-delimited JSON objects.", "Expected 'meta' as first record.")
    try:
        obj = json.loads(parts[0])
    except Exception as e:
        ctx_raise(ctx.with_(phase="query", file=str(path)), "Failed to parse first RS JSON object.", "Check the run file integrity.", e)
    return obj

def read_search_query(ctx: Ctx) -> str:
    run = newest_run_file(ctx)
    meta = read_first_rs_json(run, ctx)
    q = meta.get("query")
    if not isinstance(q, str) or not q.strip():
        ctx_raise(ctx.with_(phase="query", file=str(run)), "Missing 'query' in run meta.", "First RS record must include a non-empty 'query'.")
    return q.strip()

def read_rerank_query(ctx: Ctx) -> str:
    if not RERANK_PROGRESS_PATH.exists():
        ctx_raise(ctx.with_(phase="query", file=str(RERANK_PROGRESS_PATH)), "Missing ./rerank/progress.jsonl", "Run rerank.py first, or provide ranking query.")
    try:
        with RERANK_PROGRESS_PATH.open("r", encoding="utf-8") as f:
            first = f.readline()
    except Exception as e:
        ctx_raise(ctx.with_(phase="query", file=str(RERANK_PROGRESS_PATH)), "Failed to read progress.jsonl", "", e)
    if not first.strip():
        ctx_raise(ctx.with_(phase="query", file=str(RERANK_PROGRESS_PATH)), "progress.jsonl is empty.", "Expected a first 'meta' JSON line with 'query'.")
    try:
        obj = json.loads(first)
    except Exception as e:
        ctx_raise(ctx.with_(phase="query", file=str(RERANK_PROGRESS_PATH)), "Failed to parse first JSON line in progress.jsonl", "", e)
    q = obj.get("query")
    if not isinstance(q, str) or not q.strip():
        ctx_raise(ctx.with_(phase="query", file=str(RERANK_PROGRESS_PATH)), "Missing 'query' in progress.jsonl meta.", "First line must include a non-empty 'query'.")
    return q.strip()

def draw_intent_ranking_block(
    c: rl_canvas.Canvas,
    y_top: float,
    available_height: float,
    ctx: Ctx,
    search_query: str,
    ranking_query: str
) -> Tuple[float, int]:
    """
    Draw a two-column panel:
      - Left: Deep intent (search) query
      - Right: Ranking query (reranker)
    Returns: (y_after, pages_used_delta)
    The block may consume space and continue on the next page if needed.
    """
    # panel geometry
    left_x  = MARGIN_L
    right_x = PAGE_W - MARGIN_R
    total_w = right_x - left_x
    # columns with fixed gap
    gap = INTENT_PANEL_COL_GAP
    col_w = (total_w - gap) / 2.0

    # Internal padding:
    # - frame_pad keeps distance from frame to content area
    # - text_pad keeps distance from column edge to actual text
    frame_pad = 6.0
    text_pad  = 6.0
    # effective width available for wrapping in each column
    wrap_w = max(0.0, col_w - (2 * text_pad))

    # Wrap both columns to lines with the stricter width (prevents touching the frame)
    c.setFont(FONT_SANS_NAME, INTENT_PANEL_TEXT_SIZE)
    left_lines  = wrap_sans_words(search_query, wrap_w, INTENT_PANEL_TEXT_SIZE)
    right_lines = wrap_sans_words(ranking_query, wrap_w, INTENT_PANEL_TEXT_SIZE)

    # label heights
    label_h = INTENT_PANEL_LABEL_SIZE * 1.2
    text_leading = INTENT_PANEL_TEXT_SIZE * LEADING_FACTOR

    # We'll render “panel by panel” possibly across pages:
    y = y_top
    pages_used = 0
    idx_left = 0
    idx_right = 0

    while idx_left < len(left_lines) or idx_right < len(right_lines):
        # Estimate how many text lines can fit below labels + padding
        required_head = frame_pad + label_h + INTENT_PANEL_GAP_Y + frame_pad
        max_text_area = (y - MARGIN_B) - required_head
        if max_text_area < text_leading:
            break

        # how many lines per column fit
        max_lines_this_page = int(max_text_area // text_leading)
        if max_lines_this_page <= 0:
            break

        # slice lines for this page
        left_slice_end  = min(len(left_lines),  idx_left  + max_lines_this_page)
        right_slice_end = min(len(right_lines), idx_right + max_lines_this_page)
        left_slice  = left_lines[idx_left:left_slice_end]
        right_slice = right_lines[idx_right:right_slice_end]
        slice_rows  = max(len(left_slice), len(right_slice))

        # Panel frame height for this slice
        panel_h = frame_pad + label_h + INTENT_PANEL_GAP_Y + slice_rows * text_leading + frame_pad

        # Draw frame
        c.saveState()
        c.setStrokeGray(INTENT_PANEL_FRAME_GRAY)
        c.setLineWidth(0.8)
        c.rect(left_x - 4, y - panel_h, total_w + 8, panel_h, stroke=1, fill=0)
        c.restoreState()

        # Labels
        c.saveState()
        c.setFillGray(INTENT_PANEL_LABEL_GRAY)
        c.setFont(FONT_SANS_NAME, INTENT_PANEL_LABEL_SIZE)
        # align labels with text padding inside each column
        c.drawString(left_x + frame_pad + text_pad, y - frame_pad - (label_h * 0.85), "Deep intent (search) query")
        c.drawString(left_x + col_w + gap + frame_pad + text_pad, y - frame_pad - (label_h * 0.85), "Ranking query (reranker)")
        c.restoreState()

        # Column texts
        c.setFont(FONT_SANS_NAME, INTENT_PANEL_TEXT_SIZE)
        text_y_left = y - frame_pad - label_h - INTENT_PANEL_GAP_Y
        # left column
        for line in left_slice:
            c.drawString(left_x + frame_pad + text_pad, text_y_left - text_leading, line)
            text_y_left -= text_leading
        # right column
        text_y_right = y - frame_pad - label_h - INTENT_PANEL_GAP_Y
        for line in right_slice:
            c.drawString(left_x + col_w + gap + frame_pad + text_pad, text_y_right - text_leading, line)
            text_y_right -= text_leading

        # advance indices & y
        idx_left  = left_slice_end
        idx_right = right_slice_end
        y = y - panel_h - INTENT_PANEL_GAP_Y
        pages_used += 0  # still same page for this slice

        # If we still have lines, the caller will detect no room for more content and start a new page.

    return y, pages_used

def draw_cover_page(
    c: rl_canvas.Canvas,
    cover: Cover,
    cfg: LayoutConfig,
    page_num: int,
    ctx: Ctx,
    content_page_range: Optional[Tuple[int, int]],
    group_idx_range: Optional[Tuple[int, int]],
    # Queries to inject
    search_query: str,
    ranking_query: str
) -> int:
    """
    Stylized, ink-light cover:
      - Large title with a thin accent bar
      - Subtle frame around the content area
      - Compact metrics section and wrapped group listing
      - Shows content page range and groups range in math notation.
      - Two-column 'Intent & Ranking' panel with the search and reranker queries
    """
    try:
        page_num = start_new_page(c, page_num)

        # Frame (very light)
        c.saveState()
        c.setStrokeGray(0.7)
        c.setLineWidth(0.8)
        c.rect(MARGIN_L - 8, MARGIN_B - 8, PAGE_W - (MARGIN_L + MARGIN_R) + 16, PAGE_H - (MARGIN_T + MARGIN_B) + 16, stroke=1, fill=0)
        c.restoreState()

        # Title row (adjusted baseline)
        nominal_top = PAGE_H - MARGIN_T - 6
        title_y = nominal_top - COVER_TITLE_BASELINE_ADJ

        # Left accent bar stays as before (was correct), independent of the baseline tweak
        c.saveState()
        c.setFillGray(0.2)
        c.rect(MARGIN_L - 6, nominal_top - 20, 2, 26, stroke=0, fill=1)
        c.restoreState()

        # Title text
        c.setFont(FONT_SANS_NAME, COVER_TITLE_SIZE)
        c.drawString(MARGIN_L, title_y, f"Cover {cover.index}")

        # Slim rules above/below title
        draw_rule(c, title_y - 6, COVER_RULE_GRAY)
        draw_rule(c, title_y - 12, COVER_RULE_GRAY)

        # Summary metrics, compact layout
        y = title_y - 36
        c.setFont(FONT_SANS_NAME, 12.0)
        # Removed "Groups: N" and "Content pages in this cover: N" to free up space
        c.drawString(MARGIN_L, y, f"Relevant results in this cover: {cover.total_relevants}")
        y -= 18

        # Ranges in math notation
        if content_page_range:
            a, b = content_page_range
            c.drawString(MARGIN_L, y, f"Content pages range: [{a}, {b}]")
            y -= 18
        if group_idx_range:
            a, b = group_idx_range
            c.drawString(MARGIN_L, y, f"Groups range: [{a}, {b}]")
            y -= 22

        draw_rule(c, y + 10, COVER_RULE_GRAY)
        y -= 8

        # Intent & Ranking block
        # Try to draw the panel; if no vertical room, we push to next page (after writing footer)
        panel_start_y = y
        y_after_panel, _ = draw_intent_ranking_block(
            c=c,
            y_top=panel_start_y,
            available_height=panel_start_y - MARGIN_B,
            ctx=ctx.with_(phase="draw-cover", cover=cover.index),
            search_query=search_query,
            ranking_query=ranking_query
        )

        # If the panel function couldn't draw anything (not enough space), force page break and redraw it at top
        if y_after_panel > panel_start_y - 1:  # heuristic: no movement means no draw
            # finish this page footer and go new page
            draw_footer(c, page_num)
            c.showPage()
            page_num += 1
            y = PAGE_H - MARGIN_T
            # Redraw panel at top
            y_after_panel, _ = draw_intent_ranking_block(
                c=c,
                y_top=y,
                available_height=y - MARGIN_B,
                ctx=ctx.with_(phase="draw-cover", cover=cover.index),
                search_query=search_query,
                ranking_query=ranking_query
            )
            y = y_after_panel
        else:
            y = y_after_panel

        # Spacer before groups list
        y -= 6
        draw_rule(c, y + 6, COVER_RULE_GRAY)
        y -= 10

        # List groups with ranks (wrapped)
        c.setFont(FONT_SANS_NAME, 11.0)
        usable_w = PAGE_W - MARGIN_L - MARGIN_R
        for g in cover.groups:
            line1 = f"• {g.filename} — ranks: {g.ranks_display}"
            if pdfmetrics.stringWidth(line1, FONT_SANS_NAME, 11.0) <= usable_w:
                if y < MARGIN_B + 40:
                    draw_footer(c, page_num)
                    c.showPage()
                    page_num += 1
                    y = PAGE_H - MARGIN_T
                    c.setFont(FONT_SANS_NAME, 11.0)
                c.drawString(MARGIN_L, y, line1)
                y -= 16
            else:
                # word-wrapped
                words = line1.split(" ")
                cur = ""
                while words:
                    w0 = words[0]
                    trial = (cur + " " + w0).strip()
                    if pdfmetrics.stringWidth(trial, FONT_SANS_NAME, 11.0) <= usable_w:
                        cur = trial
                        words.pop(0)
                    else:
                        if y < MARGIN_B + 40:
                            draw_footer(c, page_num)
                            c.showPage()
                            page_num += 1
                            y = PAGE_H - MARGIN_T
                            c.setFont(FONT_SANS_NAME, 11.0)
                        c.drawString(MARGIN_L, y, cur)
                        y -= 15
                        cur = ""
                if cur:
                    if y < MARGIN_B + 40:
                        draw_footer(c, page_num)
                        c.showPage()
                        page_num += 1
                        y = PAGE_H - MARGIN_T
                        c.setFont(FONT_SANS_NAME, 11.0)
                    c.drawString(MARGIN_L, y, cur)
                    y -= 16

        draw_footer(c, page_num)
        return page_num
    except Exception as e:
        ctx_raise(ctx.with_(phase="draw-cover", cover=cover.index, page=page_num), "Failed drawing cover page.", "", e)

def draw_lines_mono(c: rl_canvas.Canvas, lines: List[str], y: float, cfg: LayoutConfig, page_num: int, ctx: Ctx, file: str) -> Tuple[float, int]:
    try:
        c.setFont(FONT_MONO_NAME, cfg.mono_size)
        for li, line in enumerate(lines):
            if y - cfg.leading_mono < MARGIN_B:
                draw_footer(c, page_num)
                c.showPage()
                page_num += 1
                y = PAGE_H - MARGIN_T
                c.setFont(FONT_MONO_NAME, cfg.mono_size)
            c.drawString(MARGIN_L, y - cfg.leading_mono, line)
            y -= cfg.leading_mono
        return y, page_num
    except Exception as e:
        ctx_raise(ctx.with_(phase="draw-json", file=file, page=page_num), f"Failed drawing JSON line.", f"Line index: {li}", e)

def draw_lines_body_with_marks(
    c: rl_canvas.Canvas,
    wrapped: List[List[Tuple[str, bool]]],
    y: float,
    cfg: LayoutConfig,
    page_num: int,
    ctx: Ctx,
    file: str
) -> Tuple[float, int]:
    """
    Draw body text. Marked spans are **bold + underlined** (printer-friendly).
    Implementation details:
      - Faux bold: draw the segment twice with a tiny x-offset.
      - Underline: draw a hairline under each marked segment's width.
      - Underlines now also cover whitespace segments to form a continuous line.
    """
    try:
        c.setFont(FONT_SANS_NAME, cfg.body_size)
        for li, parts in enumerate(wrapped):
            if y - cfg.leading_body < MARGIN_B:
                draw_footer(c, page_num)
                c.showPage()
                page_num += 1
                y = PAGE_H - MARGIN_T
                c.setFont(FONT_SANS_NAME, cfg.body_size)

            # First pass: draw text (normal or faux-bold for marked)
            x = MARGIN_L
            for seg, mk in parts:
                w = pdfmetrics.stringWidth(seg, FONT_SANS_NAME, cfg.body_size)
                c.setFillColor(black)
                if mk and seg != "":
                    # Faux bold: draw twice with a minute offset
                    c.drawString(x, y - cfg.leading_body, seg)
                    c.drawString(x + FAUX_BOLD_OFFSET, y - cfg.leading_body, seg)
                else:
                    c.drawString(x, y - cfg.leading_body, seg)
                x += w

            # Second pass: draw underlines for marked segments — include spaces this time
            x = MARGIN_L
            c.saveState()
            c.setStrokeColor(black)
            c.setLineWidth(UNDERLINE_THICKNESS)
            baseline = y - cfg.leading_body
            for seg, mk in parts:
                w = pdfmetrics.stringWidth(seg, FONT_SANS_NAME, cfg.body_size)
                if mk and w > 0:
                    underline_y = baseline - UNDERLINE_OFFSET
                    c.line(x, underline_y, x + w, underline_y)
                x += w
            c.restoreState()

            y -= cfg.leading_body
        return y, page_num
    except Exception as e:
        ctx_raise(ctx.with_(phase="draw-body", file=file, page=page_num), "Failed drawing body line.", f"Line index: {li}", e)

# ---------- Ranges helpers (for cover pages) ----------
def compute_content_page_ranges(covers: List[Cover]) -> Dict[int, Tuple[int, int]]:
    """
    Returns {cover.index: (start, end)} for content pages in PDF numbering.
    Content pages exclude the cover page itself.
    Calculation:
      - Each cover contributes 1 (its cover page) + pages_in_cover (its content)
      - For cover i, with prior total P = sum_{k<i} (1 + pages_in_cover_k):
          cover page = P + 1
          content pages = [P + 2, P + 1 + pages_in_cover_i]
    """
    ranges: Dict[int, Tuple[int, int]] = {}
    prior_total = 0
    for cv in covers:
        if cv.pages_in_cover <= 0:
            ranges[cv.index] = (0, 0)
        else:
            start = prior_total + 2
            end = prior_total + 1 + cv.pages_in_cover
            ranges[cv.index] = (start, end)
        prior_total += 1 + cv.pages_in_cover
    return ranges

def compute_group_index_ranges(covers: List[Cover]) -> Dict[int, Tuple[int, int]]:
    """
    Returns {cover.index: (start_idx, end_idx)} 1-based global group indices.
    """
    ranges: Dict[int, Tuple[int, int]] = {}
    prior_groups = 0
    for cv in covers:
        if len(cv.groups) == 0:
            ranges[cv.index] = (0, 0)
        else:
            start = prior_groups + 1
            end = prior_groups + len(cv.groups)
            ranges[cv.index] = (start, end)
        prior_groups += len(cv.groups)
    return ranges

def render_book(
    covers: List[Cover],
    out_path: Path,
    cfg: LayoutConfig,
    ctx: Ctx,
    search_query: str,
    ranking_query: str
) -> None:
    try:
        c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    except Exception as e:
        ctx_raise(ctx.with_(phase="create-pdf", file=str(out_path)), "Failed to create PDF canvas.", "", e)

    # Precompute ranges for display on cover pages
    content_ranges = compute_content_page_ranges(covers)
    group_ranges = compute_group_index_ranges(covers)
    total_groups = sum(len(cv.groups) for cv in covers)
    current_group_idx = 1  # global 1-based index for “Group i/NumGroups”

    page_num = 0
    try:
        for cover in covers:
            page_num = draw_cover_page(
                c, cover, cfg, page_num, ctx,
                content_page_range=content_ranges.get(cover.index),
                group_idx_range=group_ranges.get(cover.index),
                search_query=search_query,
                ranking_query=ranking_query
            )
            for g in cover.groups:
                page_num = start_new_page(c, page_num)
                y = PAGE_H - MARGIN_T

                # Group title — “Group i/NumGroups”
                try:
                    c.setFont(FONT_SANS_NAME, 13.0)  # slightly reduced to match smaller body
                    denom_for_title = max(1, total_groups)
                    title_txt = f"Group {current_group_idx}/{denom_for_title}"
                    c.drawString(MARGIN_L, y - cfg.leading_body * 0.9, title_txt)
                    draw_rule(c, y - cfg.leading_body * 1.05)
                    y -= (cfg.leading_body * 1.4)
                except Exception as e:
                    ctx_raise(ctx.with_(phase="draw-title", group=g.filename, cover=cover.index, page=page_num),
                              "Failed drawing group title.", "", e)

                # JSON block
                try:
                    json_text = json.dumps(g.json_rows, ensure_ascii=False, indent=2)
                    mono_wrapped = wrap_mono(json_text.split("\n"), PAGE_W - MARGIN_L - MARGIN_R, cfg.mono_size)
                    y, page_num = draw_lines_mono(c, mono_wrapped, y, cfg, page_num, ctx.with_(cover=cover.index), g.filename)
                except Exception as e:
                    ctx_raise(ctx.with_(phase="draw-json", group=g.filename, cover=cover.index, page=page_num),
                              "Failed drawing JSON block.", "", e)

                # space
                y -= cfg.para_space
                if y < MARGIN_B + cfg.leading_body:
                    draw_footer(c, page_num)
                    c.showPage()
                    page_num += 1
                    y = PAGE_H - MARGIN_T

                # Body
                try:
                    wrapped = wrap_body_with_marks(g.text, PAGE_W - MARGIN_L - MARGIN_R, cfg.body_size,
                                                   ctx.with_(group=g.filename, cover=cover.index), g.filename)
                    y, page_num = draw_lines_body_with_marks(c, wrapped, y, cfg, page_num,
                                                             ctx.with_(cover=cover.index), g.filename)
                except Exception as e:
                    ctx_raise(ctx.with_(phase="draw-body", group=g.filename, cover=cover.index, page=page_num),
                              "Failed drawing body block.", "", e)

                # trailing space
                y -= cfg.para_space
                draw_footer(c, page_num)
                current_group_idx += 1

        if page_num == 0:
            ctx_raise(ctx.with_(phase="render"), "No pages were generated.", "Input may be empty.")
        try:
            c.save()
        except Exception as e:
            ctx_raise(ctx.with_(phase="save", file=str(out_path)), "Failed to save PDF.", "", e)
    except Exception:
        # Ensure the canvas gets closed on any failure to avoid a locked file
        try:
            c.restoreState()
        except Exception:
            pass
        try:
            c.save()
        except Exception:
            pass
        raise

# ===================== Main =====================
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable PDF from ./yellow_marker/*.txt (context-rich errors)")
    parser.add_argument("--out", type=Path, default=Path("./printable/printable.pdf"), help="Output PDF path")
    parser.add_argument("--max-pages", type=int, default=20, help="Max content pages per cover (excludes cover page)")
    parser.add_argument("--body-size", type=float, default=DEFAULT_BODY_SIZE, help="Body font size (pt)")
    parser.add_argument("--mono-size", type=float, default=DEFAULT_MONO_SIZE, help="Mono (JSON) font size (pt)")
    args = parser.parse_args()

    ctx = Ctx(phase="init")
    try:
        register_fonts(ctx.with_(phase="fonts"))
        files = list_yellow_files_strict(ctx.with_(phase="discover"))
        # Parse & prepare
        inputs: List[GroupInput] = []
        for p in files:
            inputs.append(parse_one_yellow_file(p, ctx.with_(phase="parse", file=p.name)))
        prepared: List[GroupPrepared] = []
        for gi in inputs:
            rows2, ranks_disp, rel_count = massage_json_rows_keep_top_evidence(gi.json_rows, ctx, gi.filename)
            prepared.append(
                GroupPrepared(
                    filename=gi.filename,
                    json_rows=rows2,
                    text=gi.text,
                    ranks_display=ranks_disp,
                    relevant_count=rel_count,
                )
            )

        cfg = LayoutConfig(
            body_size=args.body_size,
            mono_size=args.mono_size,
            leading_body=args.body_size * LEADING_FACTOR,
            leading_mono=args.mono_size * LEADING_FACTOR,
            para_space=args.body_size * 0.6,
        )

        covers = plan_covers(prepared, args.max_pages, cfg, ctx.with_(phase="plan"))
        if not args.out.parent.exists():
            try:
                args.out.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                ctx_raise(ctx.with_(phase="mkdir", file=str(args.out.parent)), "Failed to create output directory.", "", e)

        # Fetch the queries (fail-loud if anything is unexpected)
        search_query  = read_search_query(ctx.with_(phase="query"))
        ranking_query = read_rerank_query(ctx.with_(phase="query"))

        render_book(
            covers,
            args.out,
            cfg,
            ctx.with_(phase="render", file=str(args.out)),
            search_query=search_query,
            ranking_query=ranking_query
        )
        total_content_pages = sum(cv.pages_in_cover for cv in covers)
        print(
            f"PDF written: {args.out}  | Covers: {len(covers)}  | "
            f"Content pages: {total_content_pages}  | Total pages (incl. covers): {total_content_pages + len(covers)}"
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

if __name__ == "__main__":
    # Let exceptions bubble so you see the rich context string + traceback.
    main()
