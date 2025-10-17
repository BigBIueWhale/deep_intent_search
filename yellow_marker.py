#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yellow_marker.py

Takes the outputs at ./pretty/000001.txt (etc) and the search intent from the latest
search run (./search_runs/####.jsonl meta.query), and convinces an LLM to mark the
relevant parts of the text in a "yellow marker" by inserting <mark-yellow>...</mark-yellow>.

STRICT HAPPY PATH:
- Demand files to be exactly where expected and in the exact format produced by collect_transform_pretty.py:
  Each pretty file must contain:
    1) A pretty-printed JSON list at the beginning
    2) Then newline(s)
    3) Then exactly one markdown code block (language tag: txt)
- If anything deviates, raise a detailed exception (no silent fallbacks).

LLM:
- Use the *exact* prompts provided by the user (system + user content).
- Retry multiple times until the model returns exactly one ```txt code block.
- Also require output length >= input (or equal).
- If still invalid after retries, raise a hard exception.

RESUMABLE:
- Outputs go into ./yellow_marker/ with the same filenames and count as ./pretty/.
- If some outputs already exist, verify sequence continuity (000001.txt ...), and resume at next index.
- Each output file preserves the JSON header *verbatim* and replaces only the code block with the highlighted version.

TIP: Don't let newlines get the best of you — we preserve the JSON prefix *exactly as read*
and splice in the returned code block without reformatting the JSON part.

"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, TypedDict

# ---------- Paths & constants ----------
RS: str = "\x1e"
SEARCH_RUNS_DIR: Path = Path("./search_runs")
PRETTY_DIR: Path      = Path("./pretty")
OUT_DIR: Path         = Path("./yellow_marker")

RUN_FILE_RE  = re.compile(r"^(?P<run>\d{4})\.jsonl$")
CODEBLOCK_RE = re.compile(r"```txt\n(.*?)\n```", re.DOTALL | re.IGNORECASE)  # exactly one txt block required
PRETTY_NAME_RE = re.compile(r"^(\d{6})\.txt$")

# ---- LLM integration (same core as other project files) ----
from dotenv import load_dotenv
from core.llm import get_client, chat_complete, print_stats

load_dotenv()
CLIENT = get_client()


# ---------- Types ----------
class JudgementRecord(TypedDict, total=False):
    type: str
    is_relevant: bool
    filename: str
    evidence: str
    query: str


# ---------- Strict FS + parsing helpers ----------
def _assert_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path!s} (expected {name})")

def newest_run_file() -> Tuple[int, Path]:
    _assert_dir(SEARCH_RUNS_DIR, "./search_runs")
    files = sorted(glob.glob(str(SEARCH_RUNS_DIR / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No run files found under {SEARCH_RUNS_DIR!s}")
    candidates: List[Tuple[int, Path]] = []
    for f in files:
        base = os.path.basename(f)
        m = RUN_FILE_RE.match(base)
        if not m:
            raise ValueError(f"Run filename does not match required pattern ####.jsonl: {base!r}")
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
            f"Scan file {path!s} contains no ASCII RS (0x1E) delimiters; this tool requires RS-delimited JSON."
        )
    parts = [p for p in data.split(RS) if p.strip()]
    if not parts:
        raise ValueError(f"Scan file {path!s} yielded no JSON records after RS split.")
    for idx, raw in enumerate(parts, 1):
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


# ---------- Project-specific parsing ----------
@dataclass(frozen=True)
class PrettyParsed:
    json_prefix: str          # the exact JSON (and trailing newlines) as found before the code block
    codeblock_full: str       # the full ```txt ... ``` including fences
    codeblock_inner: str      # the inner TEXT to be highlighted (no fences)

def parse_pretty_file_strict(content: str, filename: str) -> PrettyParsed:
    """
    Demand: file starts with a JSON array (pretty-printed), then blank line(s), then exactly one ```txt code block.
    We keep the JSON prefix *verbatim* and extract the code block.
    """
    m = CODEBLOCK_RE.search(content)
    if not m:
        raise RuntimeError(
            f"{filename}: expected exactly one ```txt code block, but none was found."
        )
    # Ensure there is only one code block
    others = CODEBLOCK_RE.findall(content)
    if len(others) != 1:
        raise RuntimeError(
            f"{filename}: expected exactly one ```txt code block, but found {len(others)}."
        )
    start, end = m.span()
    prefix = content[:start]
    codeblock_full = content[start:end]
    inner = m.group(1)

    # Strong opinion: prefix must begin with a JSON list
    if not prefix.lstrip().startswith("["):
        raise RuntimeError(f"{filename}: content before code block must start with '[' (pretty JSON list).")

    # Validate JSON without altering whitespace; parse the first JSON value only.
    # We attempt to json.loads(prefix.strip()) — if it fails, we alert loudly.
    try:
        json.loads(prefix.strip())
    except Exception as e:
        raise RuntimeError(f"{filename}: JSON header validation failed: {e}")

    return PrettyParsed(json_prefix=prefix, codeblock_full=codeblock_full, codeblock_inner=inner)


# ---------- Search intent ----------
def load_search_intent_from_newest_run() -> str:
    run_num, path = newest_run_file()
    recs = list(iter_records_rs_json(path))
    if not recs:
        raise RuntimeError(f"{path.name}: no RS records.")
    meta = recs[0]
    if meta.get("type") != "meta":
        raise RuntimeError(f"{path.name}: first RS record must be type=='meta'.")
    q = meta.get("query")
    if not isinstance(q, str) or not q.strip():
        raise RuntimeError(f"{path.name}: meta.query must be a non-empty string.")
    return q.strip()


# ---------- LLM highlighter (exact prompts, robust retries) ----------
SYSTEM_PROMPT = (
    "Follow the instructions exactly. Output one markdown code block (language tag: txt) and nothing else. "
    "Keep the input text fully intact except for inserting <mark-yellow>…</mark-yellow>. Do not omit any content."
)

USER_PROMPT_TEMPLATE = (
    "You are a highlighter. Insert <mark-yellow> and </mark-yellow> around the portion of TEXT that is relevant to the SEARCH_INTENT. Do not alter any characters except for inserting these tags.\n\n"
    "RULES:\n"
    "* Span size: choose whatever unit best captures the relevant material—phrase, full sentence, paragraph, or larger—adding surrounding words only if needed for coherence.\n"
    "* Operate ONLY on the TEXT provided below inside a markdown code block (language tag: txt).\n"
    "* Keep original characters, order, spacing, punctuation, and Unicode exactly as-is (only add the tags).\n"
    "* If nothing is relevant, return the TEXT unchanged.\n"
    "* OUTPUT REQUIREMENT: Return exactly one markdown code block (language tag: txt) that contains the full TEXT (with your tag insertions). No other text before or after.\n\n"
    "SEARCH_INTENT (verbatim):\n"
    "{SEARCH_INTENT}\n\n"
    "TEXT (the entire input to be highlighted appears below inside a markdown code block, language tag: txt):\n"
    "{TEXT_CODEBLOCK}\n\n"
    "Now return exactly one markdown code block (language tag: txt) that contains the full TEXT with <mark-yellow>…</mark-yellow> inserted around the relevant portion. No commentary or extra lines outside the code block."
)

def ensure_single_txt_codeblock(s: str) -> Optional[str]:
    m = CODEBLOCK_RE.search(s)
    if not m:
        return None
    # Ensure there is exactly one block
    if len(CODEBLOCK_RE.findall(s)) != 1:
        return None
    return m.group(0)

def run_llm_highlight(text_inner: str, search_intent: str, max_retries: int = 8) -> str:
    """
    Send the *exact* prompts. Require exactly one ```txt code block.
    Also ensure returned length >= input length (or equal).
    """
    if not CLIENT:
        raise RuntimeError("LLM client not initialized (get_client() returned None).")

    # Construct the user content including the TEXT as a code block with language tag txt
    text_codeblock = f"```txt\n{text_inner}\n```"
    user_content = USER_PROMPT_TEMPLATE.format(
        SEARCH_INTENT=search_intent,
        TEXT_CODEBLOCK=text_codeblock
    )

    messages = [
        {"role": "system", "content": "SYSTEM\n" + SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = chat_complete(
                messages=messages,
                role="splitter",            # consistent with other scripts; thinking allowed
                client=CLIENT,
                max_completion_tokens=16384,
                please_no_thinking=False,
                require_json=False,
            )
            stats = print_stats(resp)
            if stats:
                print(stats)

            raw = resp.message.content or ""
            block = ensure_single_txt_codeblock(raw)
            if not block:
                last_err = "Model did not return exactly one ```txt code block."
                print(f"Retry {attempt}/{max_retries}: {last_err}")
                messages.append({"role": "user", "content": "Please follow the OUTPUT REQUIREMENT. Return *exactly one* ```txt code block and nothing else."})
                continue

            # Light validity: length must be >= input (or equal)
            if len(block) < len(text_codeblock):
                last_err = "Returned code block is shorter than input; likely content omitted."
                print(f"Retry {attempt}/{max_retries}: {last_err}")
                messages.append({"role": "user", "content": "Your output must contain the FULL TEXT unchanged except for inserted <mark-yellow> tags. Do not omit anything."})
                continue

            return block
        except Exception as e:
            last_err = f"API/parse error: {e}"
            print(f"Retry {attempt}/{max_retries}: {last_err}")
            messages.append({"role": "user", "content": "An error occurred. Try again, strictly returning one ```txt code block that contains the full TEXT."})

    raise RuntimeError(f"LLM failed to produce a valid single txt code block after {max_retries} attempts. Last error: {last_err}")


# ---------- Resumability ----------
def discover_resume_position(pretty_files: List[Path]) -> int:
    """
    Verify OUT_DIR existing files (if any) are a continuous sequence starting at 000001.txt.
    Return next 1-based index to process.
    """
    OUT_DIR.mkdir(exist_ok=True)
    existing = sorted(p for p in OUT_DIR.iterdir() if p.is_file() and PRETTY_NAME_RE.match(p.name))
    if not existing:
        return 1

    # Ensure continuous sequence and one-to-one name match with the same positions from pretty/
    for i, p in enumerate(existing, start=1):
        expect = f"{i:06d}.txt"
        if p.name != expect:
            raise RuntimeError(
                f"Resumability check failed in ./yellow_marker/: expected '{expect}', found '{p.name}'."
            )
        # Also ensure there is a corresponding pretty file
        if i > len(pretty_files) or pretty_files[i - 1].name != expect:
            raise RuntimeError(
                f"Resumability check failed: yellow_marker/{p.name} has no matching pretty/{expect}."
            )

        # Minimal structure check: should contain a ```txt fence
        content = p.read_text(encoding="utf-8")
        if "```txt" not in content:
            raise RuntimeError(f"Existing yellow_marker/{p.name} lacks a ```txt code block fence.")

    next_idx = len(existing) + 1
    if next_idx > len(pretty_files):
        raise RuntimeError(
            f"Resumability check: yellow_marker/ already has {len(existing)} files but pretty/ has only {len(pretty_files)}."
        )
    return next_idx


# ---------- Driver ----------
def main() -> None:
    _assert_dir(PRETTY_DIR, "./pretty")
    pretty_files = sorted(p for p in PRETTY_DIR.iterdir() if p.is_file() and PRETTY_NAME_RE.match(p.name))
    if not pretty_files:
        raise FileNotFoundError("No files found under ./pretty matching ######.txt")

    # Read the latest search intent (verbatim)
    search_intent = load_search_intent_from_newest_run()
    print(f"Loaded SEARCH_INTENT (verbatim): {search_intent!r}")

    # Resumability: figure where to start
    start_idx = discover_resume_position(pretty_files)

    for idx in range(start_idx, len(pretty_files) + 1):
        in_path = pretty_files[idx - 1]
        out_path = OUT_DIR / in_path.name
        print(f"Processing {in_path.name} → {out_path.name} ({idx}/{len(pretty_files)})")

        # Read input and parse in a strict, opinionated way
        raw = read_text_strict(in_path)
        parsed = parse_pretty_file_strict(raw, in_path.name)

        # Build the prompting TEXT code block exactly as received (we do not trim or normalize)
        # Run LLM (with strong retry discipline)
        highlighted_block = run_llm_highlight(parsed.codeblock_inner, search_intent)

        # As an extra guard: demand output length >= input block length (or equal)
        if len(highlighted_block) < len(parsed.codeblock_full):
            raise RuntimeError(
                f"{in_path.name}: LLM returned a smaller code block than input after validation; refusing to write."
            )

        # Compose output: keep JSON prefix exactly as-is, then append the new code block.
        # We do not touch the prefix newlines; we simply replace the block and add a final newline for POSIX hygiene.
        out_text = f"{parsed.json_prefix}{highlighted_block}\n"

        # Final minimal sanity: out length >= in length (or equal)
        if len(out_text) < len(raw):
            raise RuntimeError(
                f"{in_path.name}: Final output shorter than input (possible content loss). Refusing to write."
            )

        if out_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing file: {out_path}")

        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write(out_text)

        print(f"Wrote {out_path}")

    # Final consistency check: number of files match
    produced = sorted(p for p in OUT_DIR.iterdir() if p.is_file() and PRETTY_NAME_RE.match(p.name))
    if len(produced) != len(pretty_files):
        raise RuntimeError(
            f"Mismatch: produced {len(produced)} files in yellow_marker/ but found {len(pretty_files)} in pretty/."
        )
    print("\nAll files processed. Yellow-marker outputs are in ./yellow_marker/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print("\nFATAL:\n" + repr(e))
        sys.exit(1)
