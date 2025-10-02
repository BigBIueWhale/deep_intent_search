#!/usr/bin/env python3
"""
Grouped relevant-chunk printer with evidence injection (RS-delimited records).

This version matches the provided example format:
- The scan file is a single text file containing multiple JSON objects,
  delimited by ASCII Record Separator (0x1E, '\x1e') between records.
- It is addressed as: ./search_runs/{RUN:04d}.jsonl  (extension is not relied upon, format is).

Behavior (HAPPY PATH ONLY, no fallbacks):
- Reads judgements from: ./search_runs/{RUN:04d}.jsonl (RS-delimited JSON records).
- Uses ONLY this run to determine which chunks are relevant and to source *positive* evidence for relevant chunks.
- For NON-RELEVANT context chunks at the extremities (the BEFORE and AFTER around each relevant group),
  this version REQUIREs there to be disqualifying "evidence" either in the current run or some previous run,
  and injects the most recent such disqualifier into those context chunks.
- Groups adjacent relevant chunk filenames by consecutive numeric indices.
- For each adjacent group, prints to STDOUT in this order:
    1) BEFORE chunk (one before the first relevant in the group) with
       `"evidence": "<most recent disqualifier>"` and VERY CLEAR "context-only / NOT RELEVANT" labeling
    2) Every RELEVANT chunk in the group, each injected with `"evidence": "<from this run>"`
    3) AFTER chunk (one after the last relevant in the group) with
       `"evidence": "<most recent disqualifier>"` and VERY CLEAR "context-only / NOT RELEVANT" labeling
- The `"evidence"` line is inserted AFTER THE FIRST LINE of each printed chunk.
- Chunk files are read from: ./split/chunks/<filename> using strict UTF-8.

Strict requirements (any deviation raises an uncaught exception with details):
- The scan file exists and is RS (0x1E)–delimited JSON (no JSONL line mode).
- Each relevant judgement has:
    - type == "judgement"
    - is_relevant == true
    - filename == "<zero-padded integer>.txt" (e.g., "000483.txt")
    - evidence key present with a non-empty string value
- Neighbor chunk files (BEFORE and AFTER) exist and are readable.
- For each BEFORE/AFTER (non-relevant) neighbor, there MUST exist a most-recent (<= current run) disqualifying
  judgement (is_relevant == false) with a non-empty 'evidence' string; the script injects that evidence.
- All chunk files use UTF-8 encoding.

CLI:
    python script.py --run 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

RS: str = "\x1e"
SEARCH_RUNS_DIR: Path = Path("./search_runs")
CHUNKS_DIR: Path = Path("./split/chunks")

FILENAME_RE: re.Pattern[str] = re.compile(r"^(?P<num>\d+)\.txt$")


class JudgementRecord(TypedDict, total=False):
    type: str
    is_relevant: bool
    filename: str
    evidence: str


@dataclass(frozen=True)
class ParsedFilename:
    stem: str  # any non-numeric prefix (usually empty)
    num: int
    width: int
    ext: str  # including leading dot, e.g., ".txt"

    def format(self, num: int) -> str:
        if num < 0:
            raise ValueError(f"Negative filename index computed: {num}")
        return f"{self.stem}{num:0{self.width}d}{self.ext}"


def parse_args(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grouped_relevant_printer",
        description="Print adjacent groups of relevant chunks with evidence injection (from an RS-delimited run).",
    )
    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Run number to read from ./search_runs/{RUN:04d}.jsonl (RS-delimited records).",
    )
    ns = parser.parse_args(argv)
    if ns.run < 0:
        raise ValueError(f"--run must be a non-negative integer, got: {ns.run}")
    return int(ns.run)


def read_text_strict(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path!s}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding or "utf-8", e.object, e.start, e.end, f"UTF-8 decode failed for {path!s}: {e.reason}"
        )


def iter_records_rs_json(path: Path) -> Iterable[JudgementRecord]:
    """
    Iterate RS-delimited JSON objects. No line-mode fallback.
    - File MUST contain at least one RS (0x1E).
    - Each RS-separated segment MUST be a JSON object.
    """
    data = read_text_strict(path)
    if RS not in data:
        raise ValueError(
            f"Scan file {path!s} contains no ASCII RS (0x1E) delimiters; "
            f"this tool requires the RS-delimited format."
        )
    parts = [p for p in data.split(RS) if p.strip()]
    if not parts:
        raise ValueError(f"Scan file {path!s} yielded no JSON records after RS split.")

    for idx, raw in enumerate(parts, start=1):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            snippet = raw[:200].replace("\n", "\\n")
            raise ValueError(
                f"JSON parse error in RS record #{idx} from {path!s}: {e.msg} at pos {e.pos}. "
                f"Record starts with: {snippet!r}"
            )
        if not isinstance(obj, dict):
            raise TypeError(f"Record #{idx} is not a JSON object in {path!s}.")
        yield obj  # type: ignore[return-value]


def parse_filename_strict(filename: str) -> ParsedFilename:
    base, ext = os.path.splitext(filename)
    if ext != ".txt":
        raise ValueError(f"Chunk filename must end with .txt, got: {filename!r}")
    m = FILENAME_RE.match(filename)
    if not m:
        # allow potential stems + digits (e.g., 'chunk000123.txt'); enforce trailing digits
        m2 = re.search(r"(?P<num>\d+)\.txt$", filename)
        if not m2:
            raise ValueError(
                f"Chunk filename must contain trailing digits before .txt, got: {filename!r}"
            )
        num_str = m2.group("num")
        width = len(num_str)
        stem = filename[: filename.rfind(num_str) - 4]  # remove ".txt" and digits portion
        return ParsedFilename(stem=stem, num=int(num_str), width=width, ext=".txt")
    num_str = m.group("num")
    return ParsedFilename(stem="", num=int(num_str), width=len(num_str), ext=ext)


def build_relevant_and_evidence(scan_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns:
        ordered_relevant_filenames (dedup, first occurrence order),
        evidence_map (filename -> list of evidence strings from this run)
    Enforces:
        - type == "judgement"
        - is_relevant == True
        - filename conforms to expectations
        - evidence key *present* and non-empty string for relevant judgements
    """
    seen: set[str] = set()
    ordered: List[str] = []
    evidence_map: Dict[str, List[str]] = {}

    for i, rec in enumerate(iter_records_rs_json(scan_path), start=1):
        rec_type = rec.get("type")
        if rec_type != "judgement":
            continue

        fn = rec.get("filename")
        if not isinstance(fn, str) or not fn:
            raise ValueError(f"Record #{i} has invalid or missing 'filename' in {scan_path!s}.")
        # Validate filename format strictly
        _ = parse_filename_strict(fn)

        is_rel = rec.get("is_relevant")
        if is_rel is True:
            if fn not in seen:
                seen.add(fn)
                ordered.append(fn)
            # Evidence is REQUIRED for relevant files in this run
            if "evidence" not in rec:
                raise KeyError(f"Relevant record #{i} for {fn!r} lacks required 'evidence' key in {scan_path!s}.")
            ev = rec.get("evidence")
            if not isinstance(ev, str) or not ev.strip():
                raise ValueError(
                    f"Relevant record #{i} for {fn!r} has non-string or empty 'evidence' in {scan_path!s}."
                )
            evidence_map.setdefault(fn, []).append(ev.strip())

    if not ordered:
        raise ValueError(f"No relevant judgements found in {scan_path!s}.")
    return ordered, evidence_map


def group_adjacent(filenames: Sequence[str]) -> List[List[str]]:
    """
    Group filenames (like "000483.txt") by numeric adjacency:
    consecutive trailing numbers with same width/ext/stem (strict).
    """
    enriched: List[Tuple[ParsedFilename, str]] = []
    for fn in filenames:
        pf = parse_filename_strict(fn)
        enriched.append((pf, fn))

    # Sort by (stem, ext, width, num) to group consistently
    enriched.sort(key=lambda x: (x[0].stem, x[0].ext, x[0].width, x[0].num))

    groups: List[List[str]] = []
    current: List[Tuple[ParsedFilename, str]] = []
    last_pf: Optional[ParsedFilename] = None

    for pf, fn in enriched:
        if not current:
            current = [(pf, fn)]
            last_pf = pf
            continue
        assert last_pf is not None
        last_num = last_pf.num
        same_family = (pf.stem == last_pf.stem) and (pf.ext == last_pf.ext) and (pf.width == last_pf.width)
        if same_family and pf.num == last_num + 1:
            current.append((pf, fn))
        else:
            groups.append([f for _, f in current])
            current = [(pf, fn)]
        last_pf = pf

    if current:
        groups.append([f for _, f in current])

    return groups


def read_chunk_text_strict(filename: str) -> str:
    path = CHUNKS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path!s}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding or "utf-8", e.object, e.start, e.end, f"UTF-8 decode failed for {path!s}: {e.reason}"
        )


def inject_evidence_line_after_header(content: str, evidence: Optional[str]) -> str:
    """
    Insert a line right AFTER the first line of `content`:
        "evidence": "<escaped>"
    or
        "evidence": null
    """
    # Split only on first newline
    if "\n" in content:
        idx = content.find("\n")
        header = content[:idx]
        rest = content[idx + 1 :]
    else:
        header = content
        rest = ""

    if evidence is None:
        ev_line = '"evidence": null'
    else:
        # Single-line JSON-style escaping
        safe = evidence.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        ev_line = f'"evidence": "{safe}"'

    return header + "\n" + ev_line + ("\n" + rest if rest else "")


def join_unique(items: Sequence[str]) -> str:
    seen: set[str] = set()
    out: List[str] = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return " | ".join(out)


def find_most_recent_disqualifier(filename: str, inclusive_run: int) -> Tuple[int, str]:
    """
    Find the most recent run (<= inclusive_run) that contains a disqualifying judgement
    for `filename`:
        - type == "judgement"
        - is_relevant == False
        - evidence present and non-empty string
    Returns (run_number, evidence_string).
    Raises if no such disqualifier is found across 0..inclusive_run.
    """
    last_run: Optional[int] = None
    last_ev: Optional[str] = None

    for r in range(inclusive_run, -1, -1):
        scan_path = SEARCH_RUNS_DIR / f"{r:04d}.jsonl"
        if not scan_path.is_file():
            continue
        # Keep the last seen within this run (later records override earlier ones)
        run_last_ev: Optional[str] = None
        for rec in iter_records_rs_json(scan_path):
            if rec.get("type") != "judgement":
                continue
            if rec.get("filename") != filename:
                continue
            if rec.get("is_relevant") is False:
                ev = rec.get("evidence")
                if isinstance(ev, str) and ev.strip():
                    run_last_ev = ev.strip()
        if run_last_ev:
            last_run = r
            last_ev = run_last_ev
            break  # stop at the first (most recent) run that contains a valid disqualifier

    if last_run is None or last_ev is None:
        raise ValueError(
            f"No disqualifying evidence found for non-relevant neighbor {filename!r} in runs 0..{inclusive_run:04d}."
        )
    return last_run, last_ev


def print_group_to_stdout(group: List[str], evidence_map: Dict[str, List[str]], current_run: int) -> None:
    if not group:
        raise ValueError("Internal error: empty group encountered.")

    first_pf = parse_filename_strict(group[0])
    last_pf = parse_filename_strict(group[-1])

    before_name = first_pf.format(first_pf.num - 1)
    after_name = last_pf.format(last_pf.num + 1)

    # Title
    title = f"{group[0]} .. {group[-1]}" if len(group) > 1 else group[0]
    sep = "=" * 100
    sys.stdout.write(sep + "\n")
    sys.stdout.write(f"ADJACENT GROUP: {title}\n")
    sys.stdout.write(sep + "\n")

    # BEFORE (context-only / NOT RELEVANT) — must have a most-recent disqualifier
    _, before_disq = find_most_recent_disqualifier(before_name, current_run)
    before_text = read_chunk_text_strict(before_name)
    sys.stdout.write(f"\n--- CONTEXT (NOT RELEVANT): BEFORE: {before_name} ---\n")
    sys.stdout.write(inject_evidence_line_after_header(before_text, before_disq) + "\n")

    # RELEVANT(S) — evidence required from this run
    for fn in group:
        ev_items = evidence_map.get(fn)
        if not ev_items:
            raise KeyError(f"Relevant file {fn!r} missing required evidence entries in this run.")
        ev_joined = join_unique(ev_items)
        chunk_text = read_chunk_text_strict(fn)
        sys.stdout.write(f"\n*** RELEVANT: {fn} ***\n")
        sys.stdout.write(inject_evidence_line_after_header(chunk_text, ev_joined) + "\n")

    # AFTER (context-only / NOT RELEVANT) — must have a most-recent disqualifier
    _, after_disq = find_most_recent_disqualifier(after_name, current_run)
    after_text = read_chunk_text_strict(after_name)
    sys.stdout.write(f"\n--- CONTEXT (NOT RELEVANT): AFTER: {after_name} ---\n")
    sys.stdout.write(inject_evidence_line_after_header(after_text, after_disq) + "\n")

    sys.stdout.write("\n")  # spacer between groups


def main() -> None:
    run = parse_args()
    scan_path = SEARCH_RUNS_DIR / f"{run:04d}.jsonl"
    ordered_relevant, evidence_map = build_relevant_and_evidence(scan_path)
    groups = group_adjacent(ordered_relevant)
    if not groups:
        raise ValueError("Internal error: no groups produced from relevant filenames.")
    for g in groups:
        print_group_to_stdout(g, evidence_map, run)


if __name__ == "__main__":
    # Intentionally no broad try/except; deviations from the happy path raise uncaught exceptions.
    main()
