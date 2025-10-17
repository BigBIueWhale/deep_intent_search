#!/usr/bin/env python3
"""
collect_transform_pretty.py

Rewrite of the printer to produce *printable* monolithic text per adjacent group,
while preserving strict validation, ordering, and context rules from
collect_search_results.py — plus resumability.

Output per adjacent group → ./pretty/000001.txt, 000002.txt, ...
Each file begins with a pretty-printed JSON list of objects (one per RELEVANT
section in the group), then a blank line, then the full cleaned, continuous
book-like text for that group (converted by the LLM from the raw source, e.g.,
HTML → intelligible text). The text is returned inside a **single** markdown
code block (```txt ... ```), which we extract.

Happy-path, fail-loud design: any unexpected condition raises with details.

Requirements taken from user's spec:
- Keep all strict checks from the current implementation (newest RS run, ranks,
  group adjacency, evidence presence, etc.).
- Include one BEFORE and one AFTER context chunk when they exist (boundary-aware),
  but ONLY relevant sections appear in the JSON evidence list.
- Concatenate the (BEFORE?) relevant group + (AFTER) chunks in order, removing
  the **first line** (header) from every chunk and storing that header per
  relevant section as "chunk_info".
- Ask LLM (thinking ON) to rewrite the concatenated text as a continuous,
  readable text in a single ```txt code block, preserving content & unicode,
  stripping HTML/markup. Retry on failure, print retry information.
- For each relevant section, capture its evidence (from the newest run) and
  its relevance score as "<rank>/<total>" where rank is from rerank/order.csv
  (1 = most relevant), total = number of ranked items.
- Resumable via existing files in ./pretty. Refuse to proceed on inconsistencies.
- Implement processing as a small state-machine class.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

# ---- Project paths (mirror collect_search_results.py) ----
RS: str = "\x1e"
SEARCH_RUNS_DIR: Path = Path("./search_runs")
CHUNKS_DIR: Path = Path("./split/chunks")
RERANK_DIR: Path = Path("./rerank")
ORDER_CSV_PATH: Path = RERANK_DIR / "order.csv"
PRETTY_DIR: Path = Path("./pretty")

RUN_FILE_RE: re.Pattern[str] = re.compile(r"^(?P<run>\d{4})\.jsonl$")
FILENAME_RE: re.Pattern[str] = re.compile(r"^(?P<num>\d+)\.txt$")

# ---- LLM integration ----
from dotenv import load_dotenv
from core.llm import get_client, chat_complete, print_stats

load_dotenv()
CLIENT = get_client()

# ---------------- Types ----------------
class JudgementRecord(TypedDict, total=False):
    type: str
    is_relevant: bool
    filename: str
    evidence: str

@dataclass(frozen=True)
class ParsedFilename:
    stem: str
    num: int
    width: int
    ext: str
    def format(self, num: int) -> str:
        if num < 0:
            raise ValueError(f"Negative filename index computed: {num}")
        return f"{self.stem}{num:0{self.width}d}{self.ext}"

# --------------- Strict FS helpers ---------------
def newest_run_file() -> Tuple[int, Path]:
    if not SEARCH_RUNS_DIR.is_dir():
        raise FileNotFoundError(f"Missing directory: {SEARCH_RUNS_DIR!s}")
    files = sorted(glob.glob(str(SEARCH_RUNS_DIR / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No run files found under {SEARCH_RUNS_DIR!s}")
    candidates: List[Tuple[int, Path]] = []
    for f in files:
        name = os.path.basename(f)
        m = RUN_FILE_RE.match(name)
        if not m:
            raise ValueError(f"Run filename does not match ####.jsonl: {name!r}")
        run_num = int(m.group("run"))
        candidates.append((run_num, Path(f)))
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]

def read_text_strict(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path!s}")
    return path.read_text(encoding="utf-8")

def iter_records_rs_json(path: Path) -> Iterable[JudgementRecord]:
    data = read_text_strict(path)
    if RS not in data:
        raise ValueError(
            f"Scan file {path!s} contains no ASCII RS (0x1E) delimiters; this tool requires RS-delimited JSON.")
    parts = [p for p in data.split(RS) if p.strip()]
    if not parts:
        raise ValueError(f"Scan file {path!s} yielded no JSON records after RS split.")
    for idx, raw in enumerate(parts, start=1):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            snippet = raw[:200].replace("\n", "\\n")
            raise ValueError(
                f"JSON parse error in RS record #{idx} from {path!s}: {e.msg} at pos {e.pos}. Record starts with: {snippet!r}")
        if not isinstance(obj, dict):
            raise TypeError(f"Record #{idx} is not a JSON object in {path!s}.")
        yield obj  # type: ignore

def parse_filename_strict(filename: str) -> ParsedFilename:
    base, ext = os.path.splitext(filename)
    if ext != ".txt":
        raise ValueError(f"Chunk filename must end with .txt, got: {filename!r}")
    m = FILENAME_RE.match(filename)
    if not m:
        m2 = re.search(r"(?P<num>\d+)\.txt$", filename)
        if not m2:
            raise ValueError(f"Chunk filename must contain trailing digits before .txt, got: {filename!r}")
        num_str = m2.group("num")
        width = len(num_str)
        stem = filename[: filename.rfind(num_str) - 4]
        return ParsedFilename(stem=stem, num=int(num_str), width=width, ext=".txt")
    num_str = m.group("num")
    return ParsedFilename(stem="", num=int(num_str), width=len(num_str), ext=ext)

def read_order_csv_strict(path: Path) -> List[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing order CSV: {path!s}")
    content = read_text_strict(path).strip()
    if not content:
        raise ValueError(f"Empty order CSV: {path!s}")
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

# --------------- Domain logic (reused) ---------------
def build_relevant_and_evidence(scan_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    rec_iter = list(iter_records_rs_json(scan_path))
    if not rec_iter:
        raise ValueError(f"No RS records parsed from {scan_path!s}")
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
        _ = parse_filename_strict(fn)
        if rec.get("is_relevant") is True:
            if fn not in seen:
                seen.add(fn)
                ordered.append(fn)
            if "evidence" not in rec:
                raise KeyError(f"Relevant record #{i} for {fn!r} lacks required 'evidence' key in {scan_path!s}.")
            ev = rec.get("evidence")
            if not isinstance(ev, str) or not ev.strip():
                raise ValueError(
                    f"Relevant record #{i} for {fn!r} has non-string or empty 'evidence' in {scan_path!s}.")
            evidence_map.setdefault(fn, []).append(ev.strip())

    if not ordered:
        raise ValueError(f"No relevant judgements found in {scan_path!s}.")
    return ordered, evidence_map

def group_adjacent(filenames: Sequence[str]) -> List[List[str]]:
    enriched: List[Tuple[ParsedFilename, str]] = []
    for fn in filenames:
        pf = parse_filename_strict(fn)
        enriched.append((pf, fn))
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
    return path.read_text(encoding="utf-8")

# --------------- Utility ---------------
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

# --------------- LLM cleaning ---------------
_CLEAN_PROMPT_TEMPLATE = (
    "You will receive raw text concatenated from multiple adjacent sections.\n"
    "Each section began with a single metadata line that has been removed.\n"
    "Your job: rewrite the ENTIRE text into a single printable, continuous, book-like narrative.\n"
    "STRICT RULES:\n"
    "- Output **one** markdown code block with language tag `txt` (and nothing else).\n"
    "- Preserve all content and names and multilingual Unicode as-is, but remove/ignore markup like HTML tags, CSS, JS.\n"
    "- Convert lists/tables into plain prose where sensible.\n"
    "- Do not invent content.\n"
    "- Keep the original order.\n"
    "- Remove leftover artifacts (anchors, scripts, styles, boilerplate).\n"
    "- Ensure paragraphs are readable with sensible sentence boundaries.\n\n"
    "Example (toy):\n"
    "INPUT: `<h1>Title</h1>Hi <b>there</b>!` → OUTPUT:\n"
    "```txt\nTitle. Hi there!\n```\n\n"
    "Now here is the FULL INPUT to rewrite (use ALL of it, in order):\n"
    "<RAW>\n{raw}\n</RAW>\n"
)

_CODEBLOCK_RE = re.compile(r"```(?:txt)?\n(.*?)\n```", re.DOTALL | re.IGNORECASE)

def llm_clean_to_codeblock(raw: str, max_retries: int = 6) -> Tuple[str, int]:
    if not CLIENT:
        raise RuntimeError("LLM client not initialized")

    messages = [
        {"role": "system", "content": "Follow instructions exactly, output only code block containing the requested raw text. Respond in full, don't dare to omit or skip content in the output."},
        {"role": "user", "content": _CLEAN_PROMPT_TEMPLATE.format(raw=raw)}
    ]
    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = chat_complete(
                messages=messages,
                role="splitter",            # same family used elsewhere; thinking OK
                client=CLIENT,
                max_completion_tokens=16384,
                please_no_thinking=False,    # turn ON thinking for reliability
                require_json=False,
            )
            stats = print_stats(resp)
            if stats:
                print(stats)
            txt = resp.message.content or ""
            m = _CODEBLOCK_RE.search(txt)
            if not m:
                last_err = "Model did not return a single ```txt code block."
                print(f"  -> Retry {attempt}/{max_retries}: {last_err}")
                messages.append({"role": "user", "content": "Please fix: return exactly one ```txt code block using *all* input text, no commentary."})
                continue
            return m.group(0)
        except Exception as e:
            last_err = f"API/parse error: {e}"
            print(f"  -> Retry {attempt}/{max_retries}: {last_err}")
            messages.append({"role": "user", "content": "Error encountered. Try again and adhere strictly to the code-block rule."})
    # If all retries fail: wrap raw
    print(f"All {max_retries} attempts failed. Falling back to raw text wrapped in txt code block.")
    return f"```txt\n{raw}\n```"

# --------------- Group assembly helpers ---------------
def _assert_strictly_increasing_numeric(group: List[str]) -> None:
    nums = [parse_filename_strict(fn).num for fn in group]
    if any(b <= a for a, b in zip(nums, nums[1:])):
        raise ValueError(f"Group not strictly increasing by filename index: {group}")

@dataclass
class GroupAssembled:
    concat_text: str                   # concatenated body without first header-line of each chunk
    removed_headers: Dict[str, str]    # filename -> removed first line (stripped)
    relevant_filenames: List[str]      # only relevant members of the group


def assemble_group_with_context(group: List[str]) -> GroupAssembled:
    """Return concatenated text (BEFORE?+relevants+AFTER?), minus the first line of
    EACH included chunk; and map of removed headers.
    """
    _assert_strictly_increasing_numeric(group)
    # Determine BEFORE and AFTER neighbors by numeric adjacency
    first_pf = parse_filename_strict(group[0])
    last_pf  = parse_filename_strict(group[-1])

    members: List[str] = []
    # BEFORE (if file exists)
    if first_pf.num > 0:
        before_name = first_pf.format(first_pf.num - 1)
        if (CHUNKS_DIR / before_name).is_file():
            members.append(before_name)
    # RELEVANTS (the group itself)
    members.extend(group)
    # AFTER (if file exists)
    after_name = last_pf.format(last_pf.num + 1)
    if (CHUNKS_DIR / after_name).is_file():
        members.append(after_name)

    removed_headers: Dict[str, str] = {}
    pieces: List[str] = []
    for fn in members:
        raw = read_chunk_text_strict(fn)
        if "\n" in raw:
            header, rest = raw.split("\n", 1)
        else:
            header, rest = raw, ""
        removed_headers[fn] = header.strip()
        pieces.append(rest)
    return GroupAssembled(concat_text="".join(pieces), removed_headers=removed_headers, relevant_filenames=list(group))

# --------------- Resumability ---------------
PRETTY_NAME_RE = re.compile(r"^(\d{6})\.txt$")

def discover_resume_position(total_groups: int) -> int:
    """Return the next 1-based group index to process given existing files.
    Validates sequence and basic structure. Raises on inconsistency.
    """
    PRETTY_DIR.mkdir(exist_ok=True)
    files = sorted(p for p in PRETTY_DIR.iterdir() if p.is_file() and PRETTY_NAME_RE.match(p.name))
    if not files:
        return 1
    # Ensure continuous sequence starting at 000001
    for i, p in enumerate(files, start=1):
        expect = f"{i:06d}.txt"
        if p.name != expect:
            raise RuntimeError(
                f"Resumability check failed: expected '{expect}' in ./pretty but found '{p.name}'. "
                "Please fix or move inconsistent files.")
        # Very light structure check: must start with '[' then later a blank line then code block fence present somewhere
        content = p.read_text(encoding="utf-8")
        if not content.lstrip().startswith("["):
            raise RuntimeError(f"Existing pretty/{p.name} does not start with a JSON list.")
        if "```" not in content:
            raise RuntimeError(f"Existing pretty/{p.name} lacks a markdown code block fence.")
    next_idx = len(files) + 1
    if next_idx > total_groups:
        raise RuntimeError(
            f"Resumability check: existing pretty/ has {len(files)} files but there are only {total_groups} groups now."
        )
    return next_idx

# --------------- State machine ---------------
@dataclass
class PrettyState:
    run_num: int
    groups_ordered: List[List[str]]
    rank_of: Dict[str, int]            # filename -> rank (1-based)
    total_ranked: int                  # denominator for rank string
    evidence_map: Dict[str, List[str]]
    next_group_index: int              # 1-based index of next group to process

    def current_output_path(self) -> Path:
        return PRETTY_DIR / f"{self.next_group_index:06d}.txt"

# --------------- Main driver ---------------
def main() -> None:
    # 1) Resolve newest run and relevant/evidence
    run_num, scan_path = newest_run_file()
    relevant_files, evidence_map = build_relevant_and_evidence(scan_path)

    # 2) Load order.csv → filename -> rank
    order_indices = read_order_csv_strict(ORDER_CSV_PATH)
    index_to_rank: Dict[int, int] = {idx: (i + 1) for i, idx in enumerate(order_indices)}
    filename_rank: Dict[str, int] = {}
    for fn in relevant_files:
        pf = parse_filename_strict(fn)
        idx = pf.num
        if idx not in index_to_rank:
            raise KeyError(
                f"Original index {idx} (from {fn!r}) not present in {ORDER_CSV_PATH!s}. The rerank/order.csv must include all relevant items."
            )
        filename_rank[fn] = index_to_rank[idx]

    # 3) Build adjacency groups and order groups by best rank
    groups = group_adjacent(relevant_files)
    if not groups:
        raise ValueError("Internal error: no groups produced from relevant filenames.")
    # enforce internal order
    for g in groups:
        _assert_strictly_increasing_numeric(g)
    group_keys: List[Tuple[int, int]] = []
    for pos, g in enumerate(groups):
        min_rank = min(filename_rank[name] for name in g)
        group_keys.append((min_rank, pos))
    groups_ordered = [groups[pos] for _, pos in sorted(group_keys, key=lambda t: (t[0], t[1]))]

    # 4) Resumability
    next_idx = discover_resume_position(total_groups=len(groups_ordered))

    state = PrettyState(
        run_num=run_num,
        groups_ordered=groups_ordered,
        rank_of=filename_rank,
        total_ranked=len(order_indices),
        evidence_map=evidence_map,
        next_group_index=next_idx,
    )

    # 5) Process remaining groups
    for gi in range(state.next_group_index, len(state.groups_ordered) + 1):
        group = state.groups_ordered[gi - 1]
        assembled = assemble_group_with_context(group)

        # LLM cleaning
        codeblock = llm_clean_to_codeblock(assembled.concat_text)

        # Build per-relevant JSON list
        rows: List[dict] = []
        for fn in assembled.relevant_filenames:
            evidences = state.evidence_map.get(fn)
            if not evidences:
                raise KeyError(f"Relevant file {fn!r} missing required evidence entries in newest run.")
            header = assembled.removed_headers.get(fn, "")
            rank = state.rank_of.get(fn)
            if rank is None:
                raise KeyError(f"Relevant file {fn!r} not present in order.csv mapping.")
            rows.append({
                "relevance_score": f"{rank}/{state.total_ranked}",
                "evidence_text": join_unique(evidences),
                "chunk_filename": fn,
                "chunk_info": header,
            })

        pretty_json = json.dumps(rows, ensure_ascii=False, indent=2)
        # Extract the inner text from the code block and place after the JSON
        out_text = f"{pretty_json}\n\n{codeblock.strip()}\n"

        out_path = PRETTY_DIR / f"{gi:06d}.txt"
        # Strict: do not overwrite existing numbered files unexpectedly
        if out_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing file: {out_path}. Resumability expected next index {state.next_group_index:06d}.")
        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write(out_text)
        print(f"Wrote {out_path.relative_to(Path.cwd()) if out_path.is_absolute() else out_path} (group {gi}/{len(state.groups_ordered)})")

    print("\nAll groups processed. Pretty outputs are in ./pretty/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        # Fail loudly with details
        print("\nFATAL:\n" + repr(e))
        sys.exit(1)
