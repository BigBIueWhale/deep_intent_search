#!/usr/bin/env python3
"""
Grouped relevant-chunk printer with evidence injection (RS-delimited records).

This version matches the provided example format:
- The scan file is a single text file containing multiple JSON objects,
  delimited by ASCII Record Separator (0x1E, '\x1e') between records.
- It is addressed as: the *newest* file under ./search_runs (strictly of the form {RUN:04d}.jsonl).

Behavior (STRICT HAPPY PATH, no silent fallbacks):
- Automatically selects the newest RS-delimited run in ./search_runs (filename: {RUN:04d}.jsonl).
- Uses ONLY this run to determine which chunks are relevant and to source *positive* evidence for relevant chunks.
- For NON-RELEVANT context chunks at the extremities (the BEFORE and AFTER around each relevant group),
  this version REQUIREs there to be disqualifying "evidence" either in the current run or some previous run,
  and injects the most recent such disqualifier into those context chunks.
- **Boundary edge case**: If a BEFORE/AFTER neighbor is out-of-bounds (e.g., negative index) or the neighbor
  chunk file does not exist (beginning/end of corpus), that context block is **omitted** without error.
  When a neighbor file *does* exist, absence of disqualifying evidence is treated as a hard error.
- Groups adjacent relevant chunk filenames by consecutive numeric indices.
- Ordering changes:
    * Determine *all* groups first, then order groups by the best (highest-ranked) member
      according to ./rerank/order.csv (rank 1 = best).
    * Within each group, order relevant filenames by their rank in ./rerank/order.csv (not by filename).
    * For each relevant section, inject a `"result_rank": <N>` line at the beginning of its metadata.
- For each adjacent group, prints to STDOUT in this order:
    1) BEFORE chunk (one before the first relevant in the group) with
       `"evidence": "<most recent disqualifier>"` and VERY CLEAR "context-only / NOT RELEVANT" labeling
       (skipped if boundary).
    2) Every RELEVANT chunk in the group (ordered by rank), each injected with:
           "result_rank": <N>
           "evidence":    "<from this run>"
    3) AFTER chunk (one after the last relevant in the group) with
       `"evidence": "<most recent disqualifier>"` and VERY CLEAR "context-only / NOT RELEVANT" labeling
       (skipped if boundary)
- The evidence (and rank for relevant) lines are inserted AFTER THE FIRST LINE of each printed chunk.
- Chunk files are read from: ./split/chunks/<filename> using strict UTF-8.

Strict requirements (any deviation raises an uncaught exception with details):
- ./search_runs exists and contains at least one RS-delimited run named {RUN:04d}.jsonl.
- The newest run's first record is meta with a non-empty query string (consistency).
- ./rerank/order.csv exists and is a single CSV line of integers (original indices), unique.
- Each relevant judgement in the newest run has:
    - type == "judgement"
    - is_relevant == true
    - filename == "<zero-padded integer>.txt" (e.g., "000483.txt")
    - evidence key present with a non-empty string value
- Each relevant filename's numeric index must appear in order.csv.
- Neighbor chunk files (BEFORE and AFTER) that **exist** must be readable, and must have a most-recent
  (<= current run) disqualifying judgement with non-empty evidence. If they do **not exist** due to boundary,
  the context block is omitted.
- All chunk files use UTF-8 encoding.

CLI:
    (none) — always uses newest run found under ./search_runs
"""

from __future__ import annotations

import json
import os
import re
import sys
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

RS: str = "\x1e"
SEARCH_RUNS_DIR: Path = Path("./search_runs")
CHUNKS_DIR: Path = Path("./split/chunks")
RERANK_DIR: Path = Path("./rerank")
ORDER_CSV_PATH: Path = RERANK_DIR / "order.csv"

RUN_FILE_RE: re.Pattern[str] = re.compile(r"^(?P<run>\d{4})\.jsonl$")
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


# ---------- Strict FS + parsing helpers ----------

def newest_run_file() -> Tuple[int, Path]:
    if not SEARCH_RUNS_DIR.is_dir():
        raise FileNotFoundError(f"Missing directory: {SEARCH_RUNS_DIR!s}")
    files = sorted(glob.glob(str(SEARCH_RUNS_DIR / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No run files found under {SEARCH_RUNS_DIR!s}")
    # Enforce naming pattern {RUN:04d}.jsonl and pick the highest RUN
    candidates: List[Tuple[int, Path]] = []
    for f in files:
        name = os.path.basename(f)
        m = RUN_FILE_RE.match(name)
        if not m:
            raise ValueError(f"Run filename does not match required pattern ####.jsonl: {name!r}")
        run_num = int(m.group("run"))
        candidates.append((run_num, Path(f)))
    if not candidates:
        raise FileNotFoundError(f"No properly named run files (####.jsonl) found in {SEARCH_RUNS_DIR!s}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


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


def read_order_csv_strict(path: Path) -> List[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing order CSV: {path!s}")
    content = read_text_strict(path).strip()
    if not content:
        raise ValueError(f"Empty order CSV: {path!s}")
    # Expect a single CSV line of integers
    if "\n" in content.strip():
        raise ValueError(f"order.csv must be a single line of comma-separated integers: {path!s}")
    parts = [p.strip() for p in content.split(",") if p.strip() != ""]
    try:
        nums = [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"order.csv contains non-integer tokens at {path!s}: {e}")
    if len(nums) != len(set(nums)):
        raise ValueError(f"order.csv contains duplicate indices at {path!s}")
    return nums


# ---------- Domain logic ----------

def build_relevant_and_evidence(scan_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns:
        ordered_relevant_filenames (dedup, first occurrence order),
        evidence_map (filename -> list of evidence strings from this run)
    Enforces:
        - first RS record is meta with a non-empty 'query' (consistency check)
        - type == "judgement"
        - is_relevant == True
        - filename conforms to expectations
        - evidence key *present* and non-empty string for relevant judgements
    """
    rec_iter = list(iter_records_rs_json(scan_path))
    if not rec_iter:
        raise ValueError(f"No RS records parsed from {scan_path!s}")

    # meta consistency check
    meta = rec_iter[0]
    if meta.get("type") != "meta":
        raise ValueError(f"First RS record must be 'meta' in {scan_path!s}")
    q = meta.get("query")
    if not isinstance(q, str) or not q.strip():
        raise ValueError(f"Meta record 'query' must be a non-empty string in {scan_path!s}")

    seen: set[str] = set()
    ordered: List[str] = []
    evidence_map: Dict[str, List[str]] = {}

    for i, rec in enumerate(rec_iter[1:], start=2):
        if rec.get("type") != "judgement":
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

    # Sort by (stem, ext, width, num) to group consistently *when forming groups only*
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


def inject_lines_after_header(content: str, injected_lines: List[str]) -> str:
    """
    Insert the provided lines RIGHT AFTER the first line of `content`.
    Each element of `injected_lines` is included as-is (one per new line).
    """
    if "\n" in content:
        idx = content.find("\n")
        header = content[:idx]
        rest = content[idx + 1 :]
    else:
        header = content
        rest = ""
    body = header + "\n" + "\n".join(injected_lines)
    if rest:
        body += "\n" + rest
    return body


def json_escape_one_line(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


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
            break

    if last_run is None or last_ev is None:
        raise ValueError(
            f"No disqualifying evidence found for non-relevant neighbor {filename!r} in runs 0..{inclusive_run:04d}."
        )
    return last_run, last_ev


# ---------- Printing (deferred until *after* all ordering is computed) ----------

def print_group_to_stdout(group: List[str],
                          evidence_map: Dict[str, List[str]],
                          current_run: int,
                          rank_of: Dict[str, int]) -> None:
    """
    Print one adjacent group:
      - BEFORE context (most recent disqualifier) — omitted if boundary (neighbor missing/out-of-bounds)
      - RELEVANT files (ordered by rank_in_csv asc), with "result_rank" + "evidence"
      - AFTER  context (most recent disqualifier) — omitted if boundary (neighbor missing/out-of-bounds)
    """
    if not group:
        raise ValueError("Internal error: empty group encountered.")

    # Determine neighbors using filename adjacency (strict)
    first_pf = parse_filename_strict(group[0])
    last_pf = parse_filename_strict(group[-1])

    # Compute neighbor names; if before would be negative, treat as boundary (omit)
    before_name: Optional[str] = None
    if first_pf.num - 1 >= 0:
        before_name = first_pf.format(first_pf.num - 1)

    after_name: Optional[str] = last_pf.format(last_pf.num + 1)  # may or may not exist as a file

    # Title
    title = f"{group[0]} .. {group[-1]}" if len(group) > 1 else group[0]
    sep = "=" * 100
    sys.stdout.write(sep + "\n")
    sys.stdout.write(f"ADJACENT GROUP: {title}\n")
    sys.stdout.write(sep + "\n")

    # BEFORE (context-only / NOT RELEVANT): only if neighbor chunk file exists
    if before_name is not None:
        before_path = CHUNKS_DIR / before_name
        if before_path.is_file():
            _, before_disq = find_most_recent_disqualifier(before_name, current_run)
            before_text = read_chunk_text_strict(before_name)
            sys.stdout.write(f"\n--- CONTEXT (NOT RELEVANT): BEFORE: {before_name} ---\n")
            sys.stdout.write(inject_lines_after_header(
                before_text,
                [f"\"evidence\": \"{json_escape_one_line(before_disq)}\""]
            ) + "\n")
        else:
            # Boundary at the beginning: omit BEFORE
            pass

    # RELEVANT(S) — order by rank per order.csv, inject result_rank and evidence
    # Validate all members have ranks
    for fn in group:
        if fn not in rank_of:
            raise KeyError(f"Relevant file {fn!r} not present in order.csv mapping.")

    group_sorted = sorted(group, key=lambda fn: rank_of[fn])

    for fn in group_sorted:
        evidences = evidence_map.get(fn)
        if not evidences:
            raise KeyError(f"Relevant file {fn!r} missing required evidence entries in newest run.")
        ev_joined = join_unique(evidences)
        chunk_text = read_chunk_text_strict(fn)
        sys.stdout.write(f"\n*** RELEVANT: {fn} ***\n")
        sys.stdout.write(inject_lines_after_header(
            chunk_text,
            [
                f"\"result_rank\": {rank_of[fn]}",
                f"\"evidence\": \"{json_escape_one_line(ev_joined)}\"",
            ]
        ) + "\n")

    # AFTER (context-only / NOT RELEVANT): only if neighbor chunk file exists
    if after_name is not None:
        after_path = CHUNKS_DIR / after_name
        if after_path.is_file():
            _, after_disq = find_most_recent_disqualifier(after_name, current_run)
            after_text = read_chunk_text_strict(after_name)
            sys.stdout.write(f"\n--- CONTEXT (NOT RELEVANT): AFTER: {after_name} ---\n")
            sys.stdout.write(inject_lines_after_header(
                after_text,
                [f"\"evidence\": \"{json_escape_one_line(after_disq)}\""]
            ) + "\n")
        else:
            # Boundary at the end: omit AFTER
            pass

    sys.stdout.write("\n")  # spacer between groups


# ---------- Main ----------

def main() -> None:
    # 1) Resolve newest run (strict naming ####.jsonl) and basic sanity checks
    run_num, scan_path = newest_run_file()

    # 2) Parse newest run; collect relevant filenames + their positive evidence (strict)
    relevant_files, evidence_map = build_relevant_and_evidence(scan_path)

    # 3) Read order.csv and build mapping from original_index -> rank (1-based)
    order_indices = read_order_csv_strict(ORDER_CSV_PATH)
    index_to_rank: Dict[int, int] = {idx: (i + 1) for i, idx in enumerate(order_indices)}

    # 4) Build groups of adjacent relevant files (by numeric adjacency)
    groups = group_adjacent(relevant_files)
    if not groups:
        raise ValueError("Internal error: no groups produced from relevant filenames.")

    # 5) BEFORE any printing, build filename -> rank mapping and fully determine print order
    filename_rank: Dict[str, int] = {}
    for fn in relevant_files:
        pf = parse_filename_strict(fn)
        idx = pf.num
        if idx not in index_to_rank:
            raise KeyError(
                f"Original index {idx} (from {fn!r}) not present in {ORDER_CSV_PATH!s}. "
                "The rerank/order.csv must include all relevant items."
            )
        filename_rank[fn] = index_to_rank[idx]

    # 6) Order each group’s members by rank ascending (best first)
    groups_sorted_members: List[List[str]] = [
        sorted(g, key=lambda name: filename_rank[name]) for g in groups
    ]

    # 7) Order the groups themselves by the best (minimum) rank among their members
    group_keys: List[Tuple[int, int]] = []  # (min_rank_in_group, original_position) for stability
    for pos, g in enumerate(groups_sorted_members):
        min_rank = min(filename_rank[name] for name in g)
        group_keys.append((min_rank, pos))
    order_of_groups = [groups_sorted_members[pos] for _, pos in sorted(group_keys, key=lambda t: (t[0], t[1]))]

    # 8) Now that *all* ordering is determined, perform the printing with strict neighbor evidence checks
    for g in order_of_groups:
        print_group_to_stdout(g, evidence_map, run_num, filename_rank)


if __name__ == "__main__":
    # Intentionally no broad try/except; deviations from the happy path raise uncaught exceptions.
    main()
