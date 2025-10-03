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

*** CHANGE: Final ordering is now computed by a path-independent Bradley–Terry
maximum-likelihood fit over all recorded matches (ties count as 0.5 to each side).
Elo is retained only for live scheduling. If BT cannot be produced, the tool
falls back to Elo and prints an explicit reason for the fallback. ***

---------------------------------------------------------------------
*** REWRITE: Swiss + tiny Top-K playoff (no Elo/BT; single-vote matches) ***

- Keep file structure, paths, and progress.jsonl journaling.
- Replace algorithm with:
  (1) Deterministic Swiss for R rounds (R = ceil(log2 n) + 1).
  (2) Deterministic early stop if top-K stabilizes (K=8), else after R rounds.
  (3) Tiny Top-K playoff: complete missing head-to-heads among top-K (single vote).
  (4) Final ranking from pairwise wins using Copeland + schedule tie-breakers.
- LLM verdicts are **single vote** per series. Ties from the LLM are **not allowed**; on judge failure
  after retries, choose a deterministic fallback winner (seeded hash).
- New REQUIRED CLI argument: --query "free-form user query".
  The free-form query replaces reading the query from search-run metadata and is
  persisted into progress.jsonl meta to guarantee deterministic resume.

Additional log records:
- We preserve `type:"match"` lines and include:
    "stage": "swiss"|"playoff",
    "round": <int>,
    "series": {"k":1,"wins_A":0|1,"wins_B":0|1}
- For rare odd-n Swiss rounds, we emit a synthetic bye record:
    {"type":"bye","stage":"swiss","round":r,"a":oiA}
  A bye counts as +0.5 score for that item, affects Buchholz accordingly.

*** Printing / Progress ***
- After every series, print a one-line verdict including filenames, oi, and the winner.
- Maintain and print an estimated progress percentage:
    Progress: {done}/{est} (~{pct:.1f}%)
  where 'done' counts recorded series (matches), and 'est' is a fixed upper-bound
  estimate: R*floor(n/2) + K*(K-1)/2.
---------------------------------------------------------------------
"""

from __future__ import annotations
import os
import sys
import json
import math
import glob
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from dotenv import load_dotenv
from core.llm import get_client, chat_complete, print_stats

# ---------- Constants & Paths ----------
RS = "\x1e"
SEARCH_RUNS_DIR = "./search_runs"
CHUNKS_DIR     = "./split/chunks"
RERANK_DIR     = "./rerank"
PROGRESS_PATH  = os.path.join(RERANK_DIR, "progress.jsonl")
ORDER_PATH     = os.path.join(RERANK_DIR, "order.csv")

# ---------- Swiss/Playoff knobs ----------
SWISS_TOP_K                     = 8
SWISS_MIN_ROUNDS_FOR_STABILITY  = 5

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
    # pairwise results (for final scoring)
    wins: Dict[int, Dict[int, int]]          # wins[i][j] = # of series wins for i over j
    losses: Dict[int, Dict[int, int]]        # losses[i][j] = # series losses for i vs j
    meets: Dict[Tuple[int,int], int]         # number of series between i and j (stage-agnostic)
    # swiss standings (recomputed on replay)
    score: Dict[int, float]                  # W=1, L=0; bye=0.5
    opps: Dict[int, List[int]]               # opponents list (for Buchholz)
    swiss_round: int                         # current round (1-indexed)
    swiss_done: bool
    playoff_done: bool
    total_series: int                        # number of recorded series (matches)

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
    return int.from_bytes(h.digest()[8:16], "big", signed=False)  # slight offset to differ from older seeds

def det_ab_flip(seed_base: int, match_index: int, oi_a: int, oi_b: int) -> bool:
    s = f"{seed_base}|{match_index}|{min(oi_a,oi_b)}|{max(oi_a,oi_b)}"
    return (hashlib.sha256(s.encode("utf-8")).digest()[0] & 1) == 1

def det_fallback_winner(seed_base: int, series_index: int, oi_a: int, oi_b: int) -> str:
    """
    Deterministic fallback winner when the LLM cannot produce a valid decision.
    Returns "A" or "B" based solely on a seeded hash; never "tie".
    """
    s = f"fallback|{seed_base}|{series_index}|{min(oi_a,oi_b)}|{max(oi_a,oi_b)}"
    return "A" if (hashlib.sha256(s.encode("utf-8")).digest()[0] & 1) == 0 else "B"

# ---------- Build candidate set ----------
def build_items_from_run(run_path: str) -> Tuple[str, str, int, List[Item]]:
    recs = read_rs_json(run_path)
    meta = recs[0]
    if meta.get("type") != "meta":
        raise RuntimeError("First RS record in run file must be 'meta'.")
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

# ---------- LLM comparator (single vote; no ties) ----------
def compare_with_llm(query: str, A: Item, B: Item, client, max_retries: int, seed_base: int, series_index: int) -> Tuple[str, str]:
    """
    Returns (winner, rationale), where winner in {"A","B"} ONLY.
    If the client is missing or all retries fail, we return a deterministic fallback winner.
    """
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
        {"role": "system", "content": "Return strictly valid JSON. No preface/suffix. No ties allowed."},
        {"role": "user",   "content": prompt},
    ]

    if not client:
        # Deterministic fallback (never 'tie')
        return det_fallback_winner(seed_base, series_index, 0, 1), "LLM client unavailable; deterministic fallback winner."

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = chat_complete(
                messages=messages,
                role="judge",
                client=client,
                max_completion_tokens=512,
                please_no_thinking=False,
                require_json=True,
            )
            stats = print_stats(resp)
            if stats:
                print(stats)

            txt = resp.message.content
            start = txt.find("{")
            end = txt.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Model output lacked a well-formed JSON object.")

            obj = json.loads(txt[start : end + 1])
            rationale = (obj.get("rationale") or "").strip()
            winner = (obj.get("winner") or "").strip()
            if winner not in ("A", "B"):
                raise ValueError(f"Model must choose 'A' or 'B' (no ties). Got: {winner!r}")
            return winner, rationale
        except Exception as e:
            last_err = e
            print(f"  -> Warning (Attempt {attempt}/{max_retries}): {e}")

    # Deterministic fallback if all retries failed
    fallback = det_fallback_winner(seed_base, series_index, 0, 1)
    return fallback, f"All {max_retries} attempts failed; deterministic fallback winner selected: {fallback}. Last error: {last_err}"

# ---------- Progress I/O ----------
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
        exp_set = set(exp_ids)
        got_set = set(got_ids)
        missing = sorted(exp_set - got_set)[:10]
        extra   = sorted(got_set - exp_set)[:10]
        mismatches.append(
            f"- item_ids differ: expected {len(exp_ids)} ids, found {len(got_ids)} ids"
            + (f"; missing first few: {missing}" if missing else "")
            + (f"; extra first few: {extra}" if extra else "")
        )

    # Algorithm identity must match for deterministic resume
    if first.get("algo") != expected_meta.get("algo"):
        mismatches.append(f"- algo: expected {expected_meta.get('algo')!r}, found {first.get('algo')!r}")

    if mismatches:
        raise RuntimeError(
            "progress.jsonl meta does not match the current configuration:\n" +
            "\n".join(mismatches) +
            "\n\nIf you intentionally changed the run or query, delete ./rerank/progress.jsonl and rerun."
        )
    return rows

# ---------- Replay ----------
def fresh_state(run_file: str, run_created_at: str, pass_number: int, query: str, items: List[Item]) -> State:
    item_ids = [it.original_index for it in items]
    seed_base = derive_seed_base(run_file, query, pass_number, item_ids)
    return State(
        run_file=run_file,
        run_created_at=run_created_at,
        pass_number=pass_number,
        query=query,
        seed_base=seed_base,
        items=items,
        wins=defaultdict(lambda: defaultdict(int)),
        losses=defaultdict(lambda: defaultdict(int)),
        meets=defaultdict(int),
        score={oi: 0.0 for oi in item_ids},
        opps={oi: [] for oi in item_ids},
        swiss_round=0,
        swiss_done=False,
        playoff_done=False,
        total_series=0,
    )

def record_series_result(state: State, oi_a: int, oi_b: int, winner: str) -> None:
    # update meets (use undirected key)
    key = (oi_a, oi_b) if oi_a < oi_b else (oi_b, oi_a)
    state.meets[key] = state.meets.get(key, 0) + 1

    if winner == "A":
        state.wins[oi_a][oi_b] += 1
        state.losses[oi_b][oi_a] += 1
        state.score[oi_a] += 1.0
        state.opps[oi_a].append(oi_b)
        state.opps[oi_b].append(oi_a)
    elif winner == "B":
        state.wins[oi_b][oi_a] += 1
        state.losses[oi_a][oi_b] += 1
        state.score[oi_b] += 1.0
        state.opps[oi_a].append(oi_b)
        state.opps[oi_b].append(oi_a)
    else:
        raise RuntimeError("Invalid winner in record_series_result. Only 'A' or 'B' allowed.")
    state.total_series += 1

def record_bye(state: State, oi_a: int) -> None:
    state.score[oi_a] += 0.5
    # no change to opponents
    # total_series does not include byes

def replay_rows(state: State, rows: List[dict]) -> None:
    # reset standings
    for oi in [it.original_index for it in state.items]:
        state.score[oi] = 0.0
        state.opps[oi] = []
    state.wins.clear()
    state.losses.clear()
    state.meets.clear()
    state.swiss_round = 0
    state.swiss_done = False
    state.playoff_done = False
    state.total_series = 0

    # ingest rows
    for r in rows[1:]:
        t = r.get("type")
        if t == "match":
            oi_a = int(r["a"])
            oi_b = int(r["b"])
            winner = r.get("winner")
            if winner not in ("A","B"):
                # Historical "tie": treat as score 0.5 each; do not increment total_series
                state.score[oi_a] += 0.5
                state.score[oi_b] += 0.5
                state.opps[oi_a].append(oi_b)
                state.opps[oi_b].append(oi_a)
                key = (oi_a, oi_b) if oi_a < oi_b else (oi_b, oi_a)
                state.meets[key] = state.meets.get(key, 0) + 1
                continue
            record_series_result(state, oi_a, oi_b, winner)
            if r.get("stage") == "swiss":
                state.swiss_round = max(state.swiss_round, int(r.get("round", 0)))
        elif t == "bye":
            record_bye(state, int(r["a"]))
            if r.get("stage") == "swiss":
                state.swiss_round = max(state.swiss_round, int(r.get("round", 0)))
        elif t == "meta":
            continue
        else:
            continue

# ---------- Standings utilities ----------
def buchholz(state: State, oi: int) -> float:
    return sum(state.score[opp] for opp in state.opps.get(oi, []))

def median_buchholz(state: State, oi: int) -> float:
    opp_scores = [state.score[opp] for opp in state.opps.get(oi, [])]
    if len(opp_scores) <= 2:
        return sum(opp_scores)
    opp_scores.sort()
    return sum(opp_scores[1:-1])

def has_met(state: State, i: int, j: int) -> int:
    if i == j: return 0
    key = (i, j) if i < j else (j, i)
    return state.meets.get(key, 0)

def swiss_rounds_budget(n: int) -> int:
    return int(math.ceil(math.log2(max(2, n)))) + 1

def sort_key(state: State, oi: int) -> Tuple:
    return (-state.score[oi], -buchholz(state, oi), -median_buchholz(state, oi), oi)

def current_topK(state: State, K: int) -> List[int]:
    ids = [it.original_index for it in state.items]
    ids.sort(key=lambda oi: sort_key(state, oi))
    return ids[:K]

# ---------- Progress printing ----------
def estimated_total_series(n: int, R: int, K: int) -> int:
    # Upper-bound estimate: R * floor(n/2) Swiss series + full K round-robin
    return R * (n // 2) + (K * (K - 1)) // 2

def print_progress(state: State, est_total: int) -> None:
    done = max(0, state.total_series)
    est  = max(1, est_total)
    pct  = min(100.0, 100.0 * done / est)
    print(f"Progress: {done}/{est} (~{pct:.1f}%)")

# ---------- Pairing (deterministic Swiss) ----------
def swiss_pairings(state: State, round_index: int) -> Tuple[List[Tuple[int,int]], Optional[int]]:
    """
    Return:
      - list of tuples (oi_i, oi_j) for this round (single vote per match)
      - optional bye item oi (or None)
    Deterministic:
      - Sort by (-score, -buchholz, -median_buchholz, oi).
      - Group by same score band; pair adjacent within band, skipping prior H2H if possible.
      - If unavoidable, allow a rematch (at most once overall per pair due to sorting evolution).
    """
    ids = [it.original_index for it in state.items]
    ids.sort(key=lambda oi: sort_key(state, oi))

    # Build bands by equal score
    bands: List[List[int]] = []
    cur_band: List[int] = []
    last_score: Optional[float] = None
    for oi in ids:
        s = state.score[oi]
        if last_score is None or s == last_score:
            cur_band.append(oi)
            last_score = s
        else:
            bands.append(cur_band)
            cur_band = [oi]
            last_score = s
    if cur_band:
        bands.append(cur_band)

    pairings: List[Tuple[int,int]] = []
    byes: List[int] = []

    for band in bands:
        used: Set[int] = set()
        i = 0
        L = len(band)
        while i < L:
            if band[i] in used:
                i += 1
                continue
            oi_i = band[i]
            # Find opponent in-band, prefer nearest right neighbor without prior H2H
            opponent = None
            step = 1
            while i + step < L:
                cand = band[i + step]
                if cand in used:
                    step += 1
                    continue
                met = has_met(state, oi_i, cand)
                if met == 0:
                    opponent = cand
                    break
                step += 1
            if opponent is None:
                # fall back to next available (rematch allowed)
                j = i + 1
                while j < L and band[j] in used:
                    j += 1
                if j < L:
                    opponent = band[j]

            if opponent is not None:
                used.add(oi_i)
                used.add(opponent)
                pairings.append((oi_i, opponent))
            i += 1

        leftovers = [x for x in band if x not in used]
        byes.extend(leftovers)

    # If global count is odd, we must produce exactly one bye; otherwise try to pair leftovers cross-bands deterministically
    all_ids = [it.original_index for it in state.items]
    if len(all_ids) % 2 == 1:
        # choose last leftover in final band as bye deterministically
        bye = byes[-1] if byes else ids[-1]
        byes_set = set([bye])
    else:
        byes_set = set()

    # Cross-band greedy pairing for remaining leftovers (excluding any bye)
    leftovers = [x for x in byes if x not in byes_set]
    leftovers.sort()  # deterministic
    while len(leftovers) >= 2:
        a = leftovers.pop(0)
        b = leftovers.pop(0)
        pairings.append((a, b))

    bye_item = next(iter(byes_set)) if byes_set else None
    # Ensure pairings are in deterministic order
    pairings.sort(key=lambda t: (min(t[0], t[1]), max(t[0], t[1])))
    return pairings, bye_item

def swiss_stop_condition(state: State, R: int, K: int, prev_topk: Optional[List[int]]) -> Tuple[bool, Optional[List[int]]]:
    """
    Stop Swiss if:
      A) rounds played == R
      B) round >= SWISS_MIN_ROUNDS_FOR_STABILITY and top-K set identical to previous round
         and no score-tie crosses boundary (i.e., the tuple sort_key differs at K/K+1)
    Returns (stop, current_topk)
    """
    ids_sorted = [it.original_index for it in state.items]
    ids_sorted.sort(key=lambda oi: sort_key(state, oi))
    topk = ids_sorted[:K]
    if state.swiss_round >= R:
        return True, topk
    if state.swiss_round >= SWISS_MIN_ROUNDS_FOR_STABILITY and prev_topk is not None:
        if topk == prev_topk:
            if len(ids_sorted) > K:
                if sort_key(state, ids_sorted[K-1]) > sort_key(state, ids_sorted[K]):
                    return True, topk
            else:
                return True, topk
    return False, topk

# ---------- Match execution ----------
def run_series_and_log(state: State, round_index: int, stage: str, oi_a: int, oi_b: int, client, est_total: int) -> None:
    """
    Run a single-vote series (k=1), log a match line, update state, and print verdict + progress.
    """
    A = next(it for it in state.items if it.original_index == oi_a)
    B = next(it for it in state.items if it.original_index == oi_b)

    base_index = len(read_progress_lines())  # used only to stabilize A/B flip seeds
    flip = det_ab_flip(state.seed_base, base_index, oi_a, oi_b)
    AA, BB = (B, A) if flip else (A, B)
    winner, rationale = compare_with_llm(state.query, AA, BB, client, max_retries=5, seed_base=state.seed_base, series_index=base_index)
    if flip:
        winner = {"A": "B", "B": "A"}[winner]

    record_series_result(state, oi_a, oi_b, winner)
    write_progress_line({
        "type": "match",
        "stage": stage,
        "round": round_index if stage == "swiss" else None,
        "a": oi_a,
        "b": oi_b,
        "winner": winner,
        "rationale": rationale,
        "series": {"k": 1, "wins_A": 1 if winner == "A" else 0, "wins_B": 1 if winner == "B" else 0}
    })

    # ---- Print verdict line and progress ----
    fn_a = A.filename
    fn_b = B.filename
    print(f"Compared {fn_a} (oi={oi_a}) vs {fn_b} (oi={oi_b}) → {winner.upper()}")
    print_progress(state, est_total)

def log_bye(state: State, round_index: int, oi_a: int, est_total: int) -> None:
    record_bye(state, oi_a)
    write_progress_line({
        "type": "bye",
        "stage": "swiss",
        "round": round_index,
        "a": oi_a
    })
    print(f"Round {round_index}: BYE → oi={oi_a} (+0.5 score)")
    print_progress(state, est_total)

# ---------- Final ranking ----------
def copeland_score(state: State, oi: int) -> float:
    opp_ids = [it.original_index for it in state.items if it.original_index != oi]
    wins = sum(1.0 for j in opp_ids if state.wins[oi].get(j, 0) > state.losses[oi].get(j, 0))
    return wins  # no LLM ties recorded

def head_to_head(state: State, a: int, b: int) -> int:
    wa = state.wins[a].get(b, 0)
    wb = state.wins[b].get(a, 0)
    if wa > wb: return -1  # a ahead
    if wb > wa: return 1   # b ahead
    return 0

def final_order(state: State) -> List[int]:
    ids = [it.original_index for it in state.items]
    def key(oi: int):
        return (
            -copeland_score(state, oi),
            -state.score[oi],
            -buchholz(state, oi),
            -median_buchholz(state, oi),
            oi
        )
    ids.sort(key=key)
    # break remaining ties pairwise using head-to-head where applicable (stable)
    i = 0
    while i < len(ids) - 1:
        a, b = ids[i], ids[i+1]
        if key(a) == key(b):
            h2h = head_to_head(state, a, b)
            if h2h > 0:  # b ahead
                ids[i], ids[i+1] = ids[i+1], ids[i]
        i += 1
    return ids

def write_order_csv(state: State) -> None:
    ordered = final_order(state)
    with open(ORDER_PATH, "w", encoding="utf-8") as f:
        f.write(",".join(map(str, ordered)) + "\n")
    print(f"\nWrote ranking (Swiss+TopK, {len(ordered)} items) to {ORDER_PATH}")

# ---------- Main ----------
def main() -> None:
    load_dotenv()

    # --- CLI: require --query ---
    parser = argparse.ArgumentParser(description="Deterministic Swiss + Top-K playoff reranker (single-vote matches)")
    parser.add_argument("--query", type=str, required=True, help="Free-form user query that explains what's important")
    args = parser.parse_args()
    user_query = (args.query or "").strip()
    if not user_query:
        raise SystemExit("ERROR: --query must be a non-empty string.")

    client = get_client()  # may be None; we fall back deterministically

    ensure_dirs()
    run_path = newest_run_file(SEARCH_RUNS_DIR)
    run_file = os.path.basename(run_path)

    run_file2, run_created_at, pass_number, items = build_items_from_run(run_path)
    if run_file2 != run_file:
        raise RuntimeError(f"Internal mismatch: resolved run file {run_file!r} vs parsed {run_file2!r}")

    item_ids = [it.original_index for it in items]
    seed_base = derive_seed_base(run_file, user_query, pass_number, item_ids)
    R = swiss_rounds_budget(len(items))

    expected_meta = {
        "type": "meta",
        "run_file": run_file,
        "run_created_at": run_created_at,
        "pass_number": pass_number,
        "query": user_query,
        "item_ids": item_ids,
        "algo": "swiss+topk",
        "params": {
            "rounds": R,
            "k_top": SWISS_TOP_K,
            "series_k": 1
        }
    }

    rows = ensure_meta_first_line(expected_meta)

    # Build in-memory state and replay prior matches/byes
    state = fresh_state(run_file, run_created_at, pass_number, user_query, items)
    replay_rows(state, rows)

    # Print resume info + initial progress estimate
    est_total = estimated_total_series(n=len(items), R=R, K=SWISS_TOP_K)
    if state.total_series:
        print("Resumed from progress.jsonl.")
    print_progress(state, est_total)

    # --- Swiss phase ---
    prev_topk: Optional[List[int]] = None
    stop, prev_topk = swiss_stop_condition(state, R, SWISS_TOP_K, None)
    while not stop:
        round_index = state.swiss_round + 1
        pairings, bye_item = swiss_pairings(state, round_index)

        for (a, b) in pairings:
            run_series_and_log(state, round_index, "swiss", a, b, client, est_total)

        if bye_item is not None:
            log_bye(state, round_index, bye_item, est_total)

        state.swiss_round = round_index
        stop, curr_topk = swiss_stop_condition(state, R, SWISS_TOP_K, prev_topk)
        prev_topk = curr_topk

    state.swiss_done = True

    # --- Top-K playoff: complete missing pairs among top-K (single vote) ---
    topk = current_topK(state, SWISS_TOP_K)
    missing_pairs: List[Tuple[int,int]] = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            a, b = topk[i], topk[j]
            if has_met(state, a, b) == 0:
                missing_pairs.append((a, b))
    # Deterministic order
    missing_pairs.sort(key=lambda t: (t[0], t[1]))

    for (a, b) in missing_pairs:
        run_series_and_log(state, 0, "playoff", a, b, client, est_total)

    state.playoff_done = True

    # --- Final order ---
    write_order_csv(state)
    # Final progress print
    print_progress(state, est_total)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
