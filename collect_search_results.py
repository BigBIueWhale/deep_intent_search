#!/usr/bin/env python3
"""
Grouped relevant-chunk printer with evidence injection (JSONL + strict validation).

Behavior (HAPPY PATH ONLY, no fallbacks):
- Reads judgements from: ./search_runs/{RUN:04d}.jsonl (JSON Lines; one JSON object per line).
- Uses ONLY this run to determine relevance and to source evidence.
- Groups adjacent relevant chunk filenames by consecutive numeric indices.
- For each adjacent group, prints to STDOUT in this order:
    1) BEFORE chunk (one before the first relevant in the group) with `"evidence": null`
    2) Every RELEVANT chunk in the group, each injected with `"evidence": "<from this run>"`
    3) AFTER chunk (one after the last relevant in the group) with `"evidence": null`
- The `"evidence"` line is inserted AFTER THE FIRST LINE of each printed chunk.
- Chunk files are read from: ./split/chunks/<filename> using strict UTF-8.

Strict requirements (any deviation raises an uncaught exception with details):
- The scan file exists and is newline-delimited JSON (JSONL).
- Each relevant judgement has:
    - type == "judgement"
    - is_relevant == true
    - filename == "<zero-padded integer>.txt" (e.g., "000483.txt")
    - evidence key present with a non-empty string value
- Neighbor chunk files (BEFORE and AFTER) exist and are readable.
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
        description="Print adjacent groups of relevant chunks with evidence injection (from a given JSONL run).",
    )
    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Run number to read from ./search_runs/{RUN:04d}.jsonl",
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


def iter_records_jsonl(path: Path) -> Iterable[JudgementRecord]:
    """
    Iterate JSONL (newline-delimited JSON objects). No fallbacks.
    - Empty lines are not allowed.
    - Each line must be a JSON object.
    """
    data = read_text_strict(path)
    lines = data.splitlines()
    if not lines:
        raise ValueError(f"Scan file {path!s} is empty; expected JSONL records.")

    for ln, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Blank line at {path!s}:{ln}; JSONL must have one JSON object per non-empty line.")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON parse error in {path!s}:{ln}: {e.msg} at pos {e.pos}. "
                f"Line content starts with: {line[:160]!r}"
            )
        if not isinstance(obj, dict):
            raise TypeError(f"Record at {path!s}:{ln} is not a JSON object.")
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
    Enforces (no fallbacks):
        - type == "judgement"
        - is_relevant == True
        - filename conforms to expectations
        - evidence key *present* and non-empty string for relevant judgements
    """
    seen: set[str] = set()
    ordered: List[str] = []
    evidence_map: Dict[str, List[str]] = {}

    for i, rec in enumerate(iter_records_jsonl(scan_path), start=1):
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


def print_group_to_stdout(group: List[str], evidence_map: Dict[str, List[str]]) -> None:
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

    # BEFORE (must exist; evidence null)
    before_text = read_chunk_text_strict(before_name)
    sys.stdout.write(f"\n--- BEFORE: {before_name} ---\n")
    sys.stdout.write(inject_evidence_line_after_header(before_text, None) + "\n")

    # RELEVANT(S) — evidence required from this run
    for fn in group:
        ev_items = evidence_map.get(fn)
        if not ev_items:
            raise KeyError(f"Relevant file {fn!r} missing required evidence entries in this run.")
        ev_joined = join_unique(ev_items)
        chunk_text = read_chunk_text_strict(fn)
        sys.stdout.write(f"\n*** RELEVANT: {fn} ***\n")
        sys.stdout.write(inject_evidence_line_after_header(chunk_text, ev_joined) + "\n")

    # AFTER (must exist; evidence null)
    after_text = read_chunk_text_strict(after_name)
    sys.stdout.write(f"\n--- AFTER: {after_name} ---\n")
    sys.stdout.write(inject_evidence_line_after_header(after_text, None) + "\n")

    sys.stdout.write("\n")  # spacer between groups


def main() -> None:
    run = parse_args()
    scan_path = SEARCH_RUNS_DIR / f"{run:04d}.jsonl"
    ordered_relevant, evidence_map = build_relevant_and_evidence(scan_path)
    groups = group_adjacent(ordered_relevant)
    if not groups:
        raise ValueError("Internal error: no groups produced from relevant filenames.")
    for g in groups:
        print_group_to_stdout(g, evidence_map)


if __name__ == "__main__":
    # Intentionally no broad try/except; deviations from the happy path raise uncaught exceptions.
    main()
