#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive, deterministic tournament reranker (minimal-state).

- Uses newest RS-delimited deep-search run in ./search_runs/*.jsonl
- Pulls intent from the run's first record (type=meta).
- Candidates are all judgements with is_relevant==true.
- Scheduler chooses next pair dynamically (no precomputed list).
- Only A/B order is pseudo-random (deterministic coin flip).
- progress.jsonl:
    1) First line: {"type":"meta","run_file", "run_created_at","pass_number","query","item_ids":[...]}
    2) Then one {"type":"match","a":oiA,"b":oiB,"winner":"A|B|tie","rationale": "..."} per comparison.
- Elo & counts live in memory and are recomputed by replaying the matches on resume.
- Final order -> ./rerank/order.csv as a single CSV line (most→least relevant original indices).
- No CLI args. Strict “happy-path” with detailed errors; forgiving to LLM failures (5 retries, thinking enabled).
"""

from __future__ import annotations
import os
import json
import math
import glob
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from core.llm import get_client, chat_complete, print_stats

# ---------- Constants & Paths ----------
RS = "\x1e"
SEARCH_RUNS_DIR = "./search_runs"
CHUNKS_DIR     = "./split/chunks"
RERANK_DIR     = "./rerank"
PROGRESS_PATH  = os.path.join(RERANK_DIR, "progress.jsonl")
ORDER_PATH     = os.path.join(RERANK_DIR, "order.csv")

BASE_ELO   = 1500.0
K_SMALL    = 32.0     # n < 8
K_DEFAULT  = 24.0     # n >= 8
PAIR_REPEAT_SOFT_CAP = 1  # avoid same H2H unless needed

# ---------- Data ----------
@dataclass(frozen=True)
class Item:
    original_index: int
    filename: str
    summary: str
    evidence: str
    text: str

@dataclass
class State:
    run_file: str
    run_created_at: str
    pass_number: int
    query: str
    seed_base: int
    items: List[Item]                         # canonical (sorted by original_index)
    ratings: Dict[int, float]                 # oi -> Elo
    matches_played: Dict[int, int]            # oi -> count
    h2h: Dict[Tuple[int,int], int]            # (min_oi,max_oi) -> count
    total_matches: int
    elo_k: float
    budget_per_item: int
    est_total_matches: int

# ---------- Helpers ----------
def ensure_dirs() -> None:
    os.makedirs(RERANK_DIR, exist_ok=True)

def newest_run_file(path: str) -> str:
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not files:
        raise RuntimeError("No run files found under ./search_runs")
    return files[-1]

def read_rs_json(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Run file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read()
    parts = [p for p in blob.split(RS) if p.strip()]
    if not parts:
        raise RuntimeError(f"Run file {os.path.basename(path)} contains no RS-delimited JSON objects.")
    out = []
    for i, raw in enumerate(parts, 1):
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Malformed RS-JSON record #{i} in {os.path.basename(path)}: {e}")
    return out

def read_chunk_text(filename: str) -> str:
    p = os.path.join(CHUNKS_DIR, filename)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing chunk file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def matches_budget_per_item(n: int) -> int:
    if n <= 1: return 0
    if n == 2: return 1
    return int(math.ceil(1.7 * math.log2(n))) + 2

def estimated_total_matches(n: int, per_item: int) -> int:
    return int(math.ceil(n * per_item / 2))

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

def update_elo(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float,float]:
    ea = expected_score(r_a, r_b)
    eb = 1.0 - ea
    return r_a + k * (score_a - ea), r_b + k * ((1.0 - score_a) - eb)

def derive_seed_base(run_file: str, query: str, pass_number: int, item_ids: List[int]) -> int:
    # internal deterministic seed (not stored). hash usage here is fine (not a stored fingerprint).
    h = hashlib.sha256()
    h.update(run_file.encode("utf-8"))
    h.update(b"|")
    h.update(query.encode("utf-8"))
    h.update(b"|")
    h.update(str(pass_number).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(map(str, item_ids)).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)

def det_ab_flip(seed_base: int, match_index: int, oi_a: int, oi_b: int) -> bool:
    s = f"{seed_base}|{match_index}|{min(oi_a,oi_b)}|{max(oi_a,oi_b)}"
    return (hashlib.sha256(s.encode("utf-8")).digest()[0] & 1) == 1

# ---------- Build candidate set ----------
def build_items_from_run(run_path: str) -> Tuple[str, str, int, List[Item]]:
    recs = read_rs_json(run_path)
    meta = recs[0]
    if meta.get("type") != "meta":
        raise RuntimeError("First RS record in run file must be 'meta'.")
    query = meta.get("query", "")
    if not isinstance(query, str) or not query:
        raise RuntimeError("Run meta is missing a non-empty 'query'.")
    run_created_at = meta.get("created_at") or meta.get("created") or meta.get("createdAt")
    if not isinstance(run_created_at, str) or not run_created_at:
        raise RuntimeError("Run meta is missing 'created_at' (string).")
    pass_number = int(meta.get("pass_number", 1))

    judgements = [r for r in recs[1:] if r.get("type") == "judgement" and r.get("is_relevant") is True]
    if not judgements:
        raise RuntimeError("No relevant sections found in newest run.")
    items: List[Item] = []
    for j in judgements:
        fn = j.get("filename")
        if not isinstance(fn, str) or not fn:
            raise RuntimeError("A relevant judgement is missing 'filename'.")
        items.append(Item(
            original_index=int(j["original_index"]),
            filename=fn,
            summary=j.get("summary", "") or "",
            evidence=j.get("evidence", "") or "",
            text=read_chunk_text(fn),
        ))
    items.sort(key=lambda x: x.original_index)
    return os.path.basename(run_path), run_created_at, pass_number, items

# ---------- LLM comparator (with retries & thinking) ----------
def compare_with_llm(query: str, A: Item, B: Item, client, max_retries: int = 5) -> Tuple[str, str]:
    """
    Returns (winner, rationale), where winner in {"A","B","tie"}.

    - PROMPT ORDERING: asks for a concise rationale (evidence-first) and only then the winner.
    - TIES: disallowed in normal operation. We only return "tie" if *all* retries fail
      (e.g., model errors or persistently malformed outputs).
    - Thinking is enabled for robustness (please_no_thinking=False).
    - Summary & evidence are included alongside full section text for both A and B.
    """
    # Build the ranking prompt: evidence/rationale first, then the forced verdict (A or B).
    prompt = f"""
You are ranking two *already relevant* sections by which is **more relevant** to the user's deep intent.

USER INTENT QUERY:
\"\"\"{query}\"\"\"

Judge only by each section's TEXT, its SUMMARY, and its EVIDENCE. Prefer concrete, on-point, specific evidence.

SECTION A
filename: {A.filename}
SUMMARY: {A.summary}
EVIDENCE: {A.evidence}
TEXT:
\"\"\"{A.text}\"\"\"

SECTION B
filename: {B.filename}
SUMMARY: {B.summary}
EVIDENCE: {B.evidence}
TEXT:
\"\"\"{B.text}\"\"\"

First, write a concise rationale comparing A vs B using decisive cues from their TEXT/SUMMARY/EVIDENCE
(cite short key phrases when helpful). Then, **you must pick exactly one winner**. Ties are not allowed.

Respond ONLY with JSON:
{{
  "rationale": "<one dense paragraph grounded in the two sections>",
  "winner": "A" | "B"
}}
""".strip()

    messages = [
        {"role": "system", "content": "Return strictly valid JSON. No preface/suffix."},
        {"role": "user",   "content": prompt},
    ]

    # If the client is missing, we cannot proceed with LLM comparisons.
    # Be forgiving: return a tie only because we cannot even attempt model calls.
    if not client:
        return "tie", "LLM client unavailable; unable to obtain a decision after 0/{} attempts.".format(max_retries)

    last_err: Optional[Exception] = None

    # Retry loop: be tolerant to LLM failures/malformed JSON.
    for attempt in range(1, max_retries + 1):
        try:
            resp = chat_complete(
                messages=messages,
                role="judge",
                client=client,
                max_completion_tokens=512,
                # Thinking explicitly enabled for reliability
                please_no_thinking=False,
                require_json=True,
            )
            stats = print_stats(resp)
            if stats:
                print(stats)

            txt = resp.message.content
            # Extract outermost JSON object defensively
            start = txt.find("{")
            end = txt.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Model output lacked a well-formed JSON object.")

            obj = json.loads(txt[start : end + 1])
            rationale = (obj.get("rationale") or "").strip()
            winner = (obj.get("winner") or "").strip()

            # Enforce "no ties" contract during normal operation.
            if winner not in ("A", "B"):
                raise ValueError(f"Model must choose 'A' or 'B' (no ties). Got: {winner!r}")

            return winner, rationale

        except Exception as e:
            last_err = e
            print(f"  -> Warning (Attempt {attempt}/{max_retries}): {e}")

    # All retries failed → only here do we allow a 'tie' (forgiving behavior).
    return "tie", f"All {max_retries} attempts failed; recorded as tie. Last error: {last_err}"

# ---------- Progress I/O (minimal) ----------
def write_progress_line(obj: dict) -> None:
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

def read_progress_lines() -> List[dict]:
    if not os.path.isfile(PROGRESS_PATH):
        return []
    out: List[dict] = []
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def ensure_meta_first_line(expected_meta: dict) -> List[dict]:
    """
    Ensure progress.jsonl exists and its first line matches expected_meta.
    If file absent -> write expected_meta as first line.
    If present but first line differs -> raise with detailed mismatches.
    Returns the full list of rows for replay.
    """
    rows = read_progress_lines()
    if not rows:
        write_progress_line(expected_meta)
        return [expected_meta]
    first = rows[0]
    if first.get("type") != "meta":
        raise RuntimeError(
            "progress.jsonl is invalid: first line must be a 'meta' object.\n"
            f"Got: {first}"
        )

    mismatches: List[str] = []
    for key in ("run_file", "run_created_at", "pass_number", "query"):
        if first.get(key) != expected_meta.get(key):
            mismatches.append(f"- {key}: expected {expected_meta.get(key)!r}, found {first.get(key)!r}")

    exp_ids = expected_meta.get("item_ids", [])
    got_ids = first.get("item_ids", [])
    if exp_ids != got_ids:
        # Build a small diff summary
        exp_set = set(exp_ids)
        got_set = set(got_ids)
        missing = sorted(exp_set - got_set)[:10]
        extra   = sorted(got_set - exp_set)[:10]
        mismatches.append(
            f"- item_ids differ: expected {len(exp_ids)} ids, found {len(got_ids)} ids"
            + (f"; missing first few: {missing}" if missing else "")
            + (f"; extra first few: {extra}" if extra else "")
        )

    if mismatches:
        raise RuntimeError(
            "progress.jsonl meta does not match the newest deep-search run:\n" +
            "\n".join(mismatches) +
            "\n\nIf you intentionally changed the run, delete ./rerank/progress.jsonl and rerun."
        )
    return rows

# ---------- State init & replay ----------
def fresh_state(run_file: str, run_created_at: str, pass_number: int, query: str, items: List[Item]) -> State:
    item_ids = [it.original_index for it in items]
    seed_base = derive_seed_base(run_file, query, pass_number, item_ids)
    elo_k = K_DEFAULT if len(items) >= 8 else K_SMALL
    budget = matches_budget_per_item(len(items))
    est_total = estimated_total_matches(len(items), budget)
    return State(
        run_file=run_file,
        run_created_at=run_created_at,
        pass_number=pass_number,
        query=query,
        seed_base=seed_base,
        items=items,
        ratings={oi: BASE_ELO for oi in item_ids},
        matches_played={oi: 0 for oi in item_ids},
        h2h={},
        total_matches=0,
        elo_k=elo_k,
        budget_per_item=budget,
        est_total_matches=est_total,
    )

def apply_match(state: State, oi_a: int, oi_b: int, winner: str) -> None:
    score_a = 1.0 if winner == "A" else 0.0 if winner == "B" else 0.5
    rA = state.ratings[oi_a]
    rB = state.ratings[oi_b]
    newA, newB = update_elo(rA, rB, score_a, state.elo_k)
    state.ratings[oi_a] = newA
    state.ratings[oi_b] = newB
    state.matches_played[oi_a] += 1
    state.matches_played[oi_b] += 1
    k = (oi_a, oi_b) if oi_a < oi_b else (oi_b, oi_a)
    state.h2h[k] = state.h2h.get(k, 0) + 1
    state.total_matches += 1

def replay_matches(state: State, rows: List[dict]) -> None:
    for r in rows[1:]:
        if r.get("type") != "match":
            continue
        oi_a = int(r["a"])
        oi_b = int(r["b"])
        winner = r.get("winner", "tie")
        apply_match(state, oi_a, oi_b, winner)

# ---------- Scheduler (adaptive, deterministic) ----------
def choose_next_pair(state: State) -> Optional[Tuple[int,int]]:
    """
    Pick next (oi_a, oi_b) deterministically from current state:
      - Anchor: fewest matches; tie-break by Elo then original_index
      - Opponent: minimize (|E-0.5|, repeats penalty, opp matches, |Δmatches|, |Δelo|, oi)
    """
    items = state.items
    if len(items) < 2:
        return None

    mp = state.matches_played
    r  = state.ratings
    h2h = state.h2h

    anchor = sorted(items, key=lambda it: (mp[it.original_index], r[it.original_index], it.original_index))[0]
    i  = anchor.original_index
    ri = r[i]
    mi = mp[i]

    best = None  # ((score_tuple), j)
    for opp in items:
        j = opp.original_index
        if j == i: continue
        rj = r[j]
        mj = mp[j]
        key = (i, j) if i < j else (j, i)
        repeats = h2h.get(key, 0)

        closeness   = abs(expected_score(ri, rj) - 0.5)
        rep_penalty = 0 if repeats <= PAIR_REPEAT_SOFT_CAP else (repeats - PAIR_REPEAT_SOFT_CAP)
        score = (closeness, rep_penalty, mj, abs(mi - mj), abs(ri - rj), j)
        cand  = (score, j)
        if best is None or cand[0] < best[0]:
            best = cand

    if not best:
        return None
    return (i, best[1])

def stop_condition(state: State) -> bool:
    n = len(state.items)
    if n <= 1: return True
    if n == 2 and state.total_matches >= 1: return True
    if all(state.matches_played[it.original_index] >= state.budget_per_item for it in state.items):
        return True
    return False

# ---------- Output ----------
def print_progress(state: State) -> None:
    done = state.total_matches
    est  = max(state.est_total_matches, 1)
    pct  = min(100.0, 100.0 * done / est)
    print(f"Progress: {done}/{est} (~{pct:.1f}%)")

def write_order_csv(state: State) -> None:
    ordered = sorted([it.original_index for it in state.items], key=lambda oi: (-state.ratings[oi], oi))
    with open(ORDER_PATH, "w", encoding="utf-8") as f:
        f.write(",".join(map(str, ordered)) + "\n")
    print(f"\nWrote ranking ({len(ordered)} items) to {ORDER_PATH}")

# ---------- Main ----------
def main() -> None:
    load_dotenv()
    client = get_client()  # may be None; that's okay (heuristic fallback)

    ensure_dirs()
    run_path = newest_run_file(SEARCH_RUNS_DIR)
    run_file = os.path.basename(run_path)

    run_file2, run_created_at, pass_number, items = build_items_from_run(run_path)
    if run_file2 != run_file:
        # Shouldn't happen, but be explicit if it does.
        raise RuntimeError(f"Internal mismatch: resolved run file {run_file!r} vs parsed {run_file2!r}")

    item_ids = [it.original_index for it in items]
    expected_meta = {
        "type": "meta",
        "run_file": run_file,
        "run_created_at": run_created_at,
        "pass_number": pass_number,
        "query": items and (items[0] and items[0]) and None  # placeholder to keep keys visible below
    }
    # NOTE: query lives in the run meta; fetch it explicitly (not via items)
    recs = read_rs_json(run_path)
    expected_meta["query"] = recs[0].get("query", "")
    expected_meta["item_ids"] = item_ids

    rows = ensure_meta_first_line(expected_meta)

    # Build fresh in-memory state and replay prior matches
    state = fresh_state(run_file, run_created_at, pass_number, expected_meta["query"], items)
    replay_matches(state, rows)
    if state.total_matches:
        print("Resumed from progress.jsonl.")
        print_progress(state)

    # Adaptive loop
    while not stop_condition(state):
        pair = choose_next_pair(state)
        if not pair:
            break
        oi_a, oi_b = pair
        A = next(it for it in state.items if it.original_index == oi_a)
        B = next(it for it in state.items if it.original_index == oi_b)

        # deterministic A/B flip only
        flip = det_ab_flip(state.seed_base, state.total_matches, oi_a, oi_b)
        AA, BB = (B, A) if flip else (A, B)

        winner, rationale = compare_with_llm(state.query, AA, BB, client, max_retries=5)
        if flip:
            winner = {"A": "B", "B": "A", "tie": "tie"}[winner]

        apply_match(state, oi_a, oi_b, winner)
        write_progress_line({"type": "match", "a": oi_a, "b": oi_b, "winner": winner, "rationale": rationale})

        print(f"Compared {A.filename} (oi={oi_a}) vs {B.filename} (oi={oi_b}) → {winner.upper()}")
        print_progress(state)

    # Final ordering
    write_order_csv(state)

if __name__ == "__main__":
    main()
