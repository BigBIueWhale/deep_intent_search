#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic 3-step reranker with strict journaling (progress.jsonl) and hard failures.

Preserves:
- CLI: --query "..."
- Paths: ./search_runs, ./split/chunks, ./rerank
- Final output: ./rerank/order.csv (single CSV line, most→least relevant)
- EXACT LLM prompt from the prior rerank.py (no ties; strict JSON)
- Relative-path usage & .env load

Algorithm:
  Step 1) Skeleton (≤120) full round-robin → Bradley–Terry (BT) → skeleton order.
  Step 2) Bracket remaining items into 121 tiers via deterministic binary search (+ configurable confirmations).
  Step 3) Only if Step 2 ran (n > skeleton size):
         Build a "top block" by taking whole tiers from Tier 0 downward, including the skeleton anchors
         *between* those tiers, until adding another tier would exceed 60 items. If Tier 0 alone >60,
         warn and skip Step 3. Otherwise run a fresh full round-robin + BT on that top block and let
         the BT order completely override ordering within that block (anchors included). The rest remains
         tiered; items within those tiers are ordered lexicographically by filename.

Journaling (progress.jsonl):
- meta (strictly matched on resume; includes params and BT hyperparameters)
- skeleton selection
- cmp lines for skeleton and topfull (with filenames)
- bt_summary (skeleton/topfull) with order and scores
- probe/confirm and bracket_decision lines (with filenames)
- topset with explicit topset_ids (and tiers_included, skip flag)

Absolutely no fallbacks: if judge fails after retries, raise with details.

Dependencies:
- choix==0.4.1
- numpy
- python-dotenv
- your core.llm module providing get_client, chat_complete, print_stats
"""

from __future__ import annotations
import os
import sys
import json
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

from dotenv import load_dotenv
import numpy as np
import choix  # choix==0.4.1

from core.llm import get_client, chat_complete, print_stats

# ---------- Constants & Paths ----------
RS = "\x1e"
SEARCH_RUNS_DIR = "./search_runs"
CHUNKS_DIR     = "./split/chunks"
RERANK_DIR     = "./rerank"
PROGRESS_PATH  = os.path.join(RERANK_DIR, "progress.jsonl")
ORDER_PATH     = os.path.join(RERANK_DIR, "order.csv")

# ---------- Algorithm knobs (opinionated) ----------
SKELETON_SIZE          = 120
TOPFULL_SIZE           = 60
RETRIES_PER_JUDGE      = 5
BRACKET_CONFIRMATION   = 2  # number of confirmation comparisons per boundary when possible
BT_ALPHA               = 1e-7
BT_METHOD              = "BFGS"
BT_TOL                 = 1e-10
BT_MAX_ITER            = 1000

ALGO_ID = "skeleton+bracket+topbt.v2"

# ---------- Data ----------
@dataclass(frozen=True)
class Item:
    original_index: int
    filename: str
    summary: str
    evidence: str
    text: str

# ---------- Strict Errors ----------
class RerankError(RuntimeError):
    pass

# ---------- FS helpers ----------
def ensure_dirs() -> None:
    os.makedirs(RERANK_DIR, exist_ok=True)

def newest_run_file(path: str) -> str:
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not files:
        raise RerankError("No run files found under ./search_runs (expected at least one *.jsonl).")
    return files[-1]  # lexicographic newest (0001.jsonl, 0002.jsonl, ...)

def fs_read(path: str) -> str:
    if not os.path.isfile(path):
        raise RerankError(f"Missing file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RerankError(f"Failed to read {path}: {e}")

def fs_write_jsonl_line(path: str, obj: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        raise RerankError(f"Failed to append to {path}: {e}")

def fs_overwrite_text(path: str, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise RerankError(f"Failed to write {path}: {e}")

# ---------- Run file parsing (RS-delimited JSON) ----------
def read_rs_json(path: str) -> List[dict]:
    blob = fs_read(path)
    parts = [p for p in blob.split(RS) if p.strip()]
    if not parts:
        raise RerankError(f"Run file {os.path.basename(path)} contains no RS-delimited JSON objects.")
    out = []
    for i, raw in enumerate(parts, 1):
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError as e:
            raise RerankError(f"Malformed RS-JSON record #{i} in {os.path.basename(path)}: {e}")
    return out

def load_items_from_run(run_path: str) -> Tuple[str, str, int, List[Item]]:
    recs = read_rs_json(run_path)
    meta = recs[0]
    if meta.get("type") != "meta":
        raise RerankError("First RS record in run file must be 'meta'.")
    run_created_at = meta.get("created_at") or meta.get("created") or meta.get("createdAt")
    if not isinstance(run_created_at, str) or not run_created_at:
        raise RerankError("Run meta is missing 'created_at' (string).")
    pass_number = int(meta.get("pass_number", 1))

    judgements = [r for r in recs[1:] if r.get("type") == "judgement" and r.get("is_relevant") is True]
    if not judgements:
        raise RerankError("No relevant sections found in newest run (is_relevant==true).")

    seen_oi: Set[int] = set()
    items: List[Item] = []
    for j in judgements:
        fn = j.get("filename")
        if not isinstance(fn, str) or not fn:
            raise RerankError("A relevant judgement is missing 'filename' (non-empty string required).")
        if "original_index" not in j:
            raise RerankError(f"Judgement for filename {fn!r} is missing 'original_index'.")
        oi = int(j["original_index"])
        if oi in seen_oi:
            raise RerankError(f"Duplicate original_index detected: {oi}")
        seen_oi.add(oi)
        p = os.path.join(CHUNKS_DIR, fn)
        if not os.path.isfile(p):
            raise RerankError(f"Missing chunk file referenced by run: {p}")
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
        items.append(Item(
            original_index=oi,
            filename=fn,
            summary=j.get("summary", "") or "",
            evidence=j.get("evidence", "") or "",
            text=txt,
        ))
    items.sort(key=lambda x: x.original_index)
    return os.path.basename(run_path), run_created_at, pass_number, items

# ---------- Progress I/O ----------
def read_progress_lines() -> List[dict]:
    if not os.path.isfile(PROGRESS_PATH):
        return []
    out: List[dict] = []
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if s:
                try:
                    out.append(json.loads(s))
                except Exception as e:
                    raise RerankError(f"Invalid JSON line #{ln} in progress.jsonl: {e}\nLine: {line[:200]}")
    return out

def assert_meta_or_write(expected_meta: dict) -> List[dict]:
    rows = read_progress_lines()
    if not rows:
        fs_write_jsonl_line(PROGRESS_PATH, expected_meta)
        return [expected_meta]
    first = rows[0]
    if first.get("type") != "meta":
        raise RerankError("progress.jsonl invalid: first line must be a 'meta' object.")
    mismatches: List[str] = []
    # Strict field-by-field equality
    for k in ("algo", "run_file", "run_created_at", "pass_number", "query"):
        if first.get(k) != expected_meta.get(k):
            mismatches.append(f"- {k}: expected {expected_meta.get(k)!r}, found {first.get(k)!r}")
    if first.get("params") != expected_meta.get("params"):
        mismatches.append(f"- params differ.\n  expected: {json.dumps(expected_meta['params'], sort_keys=True)}\n  found:    {json.dumps(first.get('params'), sort_keys=True)}")
    exp_ids = expected_meta.get("item_ids", [])
    got_ids = first.get("item_ids", [])
    if exp_ids != got_ids:
        # show small diff
        exp_set, got_set = set(exp_ids), set(got_ids)
        missing = sorted(exp_set - got_set)[:10]
        extra   = sorted(got_set - exp_set)[:10]
        mismatches.append(f"- item_ids differ: expected {len(exp_ids)} ids, found {len(got_ids)} ids; missing={missing}, extra={extra}")
    if mismatches:
        raise RerankError("progress.jsonl meta does not match current configuration:\n" + "\n".join(mismatches) +
                          "\n\nIf you intentionally changed the run or query, delete ./rerank/progress.jsonl and rerun.")
    return rows

# ---------- Scheduling helpers ----------
def circle_round_robin(ids: List[int]) -> List[Tuple[int,int,int]]:
    """
    Berger-tables style single round-robin schedule.
    Returns list of triples (left,right,round), concatenated round by round.
    Deterministic given 'ids' order. Nearly perfect left/right balance per id.
    """
    n = len(ids)
    if n < 2:
        return []
    arr = list(ids)
    use_dummy = (n % 2 == 1)
    if use_dummy:
        arr.append(None)
    m = len(arr)
    rounds = m - 1
    out: List[Tuple[int,int,int]] = []
    for r in range(rounds):
        for i in range(m // 2):
            a = arr[i]
            b = arr[m - 1 - i]
            if a is None or b is None:
                continue
            # Alternate home/away by round for fairness
            left, right = (a, b) if (r % 2 == 0) else (b, a)
            out.append((left, right, r + 1))
        # rotate (fixed first element)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    # Validate balance with a parity-agnostic tolerance (≤ 2.0 from half)
    cntL: Dict[int,int] = {i:0 for i in ids}
    for l, r, _ in out:
        cntL[l] += 1
    half = (len(ids) - 1) / 2.0
    TOL = 2.0
    for i in ids:
        if abs(cntL[i] - half) > TOL:
            raise RerankError(f"Round-robin left/right balance violated for item {i}: left_count={cntL[i]}, expected≈{half}.")
    return out

# ---------- LLM comparator (EXACT PROMPT; NO FALLBACKS) ----------
def judge_pair_strict(query: str, A: Item, B: Item, client, retries: int) -> Tuple[str, str]:
    """
    Returns ("A" or "B", rationale).
    On any persistent failure after 'retries', raises RerankError.
    PROMPT PRESERVED EXACTLY from the provided rerank.py.
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
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
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
            if winner not in ("A","B"):
                raise ValueError(f"Model must choose 'A' or 'B' (no ties). Got: {winner!r}")
            return winner, rationale
        except Exception as e:
            last_err = e
            print(f"  -> Warning (Attempt {attempt}/{retries}): {e}")
    raise RerankError(f"Judge failed after {retries} attempts; last error: {last_err}")

# ---------- BT helpers ----------
def bt_fit_order(ids: List[int], duels: List[Tuple[int,int]], *, stage: str) -> Tuple[List[int], Dict[int,float]]:
    """
    Fit Bradley–Terry using choix.opt_pairwise with opinionated settings.
    ids: ordered list of item ids (original_index) that the duel indices reference.
    duels: list of (winner_ix, loser_ix) indices referencing 'ids' positions.
    Returns (ordered_ids_desc, score_map).
    """
    try:
        n = len(ids)
        if n == 0:
            return [], {}
        params = choix.opt_pairwise(
            n_items=n,
            data=duels,
            alpha=BT_ALPHA,
            method=BT_METHOD,
            tol=BT_TOL,
            max_iter=BT_MAX_ITER,
        )
        order_ix = list(np.argsort(-params))
        ordered_ids = [ids[i] for i in order_ix]
        scores = {ids[i]: float(params[i]) for i in range(n)}
        return ordered_ids, scores
    except Exception as e:
        raise RerankError(f"BT optimization failed at stage '{stage}': {e}")

# ---------- Journal state ----------
@dataclass
class Journal:
    meta: dict
    skeleton_ids: Optional[List[int]]
    skeleton_cmp: List[dict]
    skeleton_bt_order: Optional[List[int]]
    bracket_probes: Dict[int, List[dict]]          # by item oi
    bracket_confirms: Dict[int, List[dict]]        # by item oi
    bracket_decision: Dict[int, dict]              # by item oi
    topset: Optional[dict]
    topfull_cmp: List[dict]
    topfull_bt_order: Optional[List[int]]

def replay_progress(rows: List[dict]) -> Journal:
    if not rows:
        return Journal({}, None, [], None, {}, {}, {}, None, [], None)
    meta = rows[0]
    skeleton_ids = None
    skeleton_cmp: List[dict] = []
    skeleton_bt_order = None
    bracket_probes: Dict[int, List[dict]] = {}
    bracket_confirms: Dict[int, List[dict]] = {}
    bracket_decision: Dict[int, dict] = {}
    topset = None
    topfull_cmp: List[dict] = []
    topfull_bt_order = None

    for r in rows[1:]:
        t = r.get("type")
        if t == "skeleton":
            if skeleton_ids is not None:
                raise RerankError("progress.jsonl contains multiple 'skeleton' lines.")
            skeleton_ids = list(map(int, r.get("ids", [])))
        elif t == "cmp":
            stage = r.get("stage")
            if stage == "skeleton":
                skeleton_cmp.append(r)
            elif stage == "topfull":
                topfull_cmp.append(r)
            else:
                raise RerankError(f"Unknown cmp stage in progress: {stage!r}")
        elif t == "bt_summary":
            stage = r.get("stage")
            order = r.get("order")
            if not isinstance(order, list):
                raise RerankError("bt_summary missing 'order' list.")
            if stage == "skeleton":
                if skeleton_bt_order is not None:
                    raise RerankError("Duplicate bt_summary for skeleton stage.")
                skeleton_bt_order = list(map(int, order))
            elif stage == "topfull":
                if topfull_bt_order is not None:
                    raise RerankError("Duplicate bt_summary for topfull stage.")
                topfull_bt_order = list(map(int, order))
            else:
                raise RerankError(f"Unknown bt_summary stage: {stage!r}")
        elif t == "probe":
            if r.get("stage") != "bracket":
                raise RerankError("probe with stage != 'bracket' found.")
            oi = int(r["item"])
            bracket_probes.setdefault(oi, []).append(r)
        elif t == "confirm":
            if r.get("stage") != "bracket":
                raise RerankError("confirm with stage != 'bracket' found.")
            oi = int(r["item"])
            bracket_confirms.setdefault(oi, []).append(r)
        elif t == "bracket_decision":
            if r.get("stage") != "bracket":
                raise RerankError("bracket_decision with stage != 'bracket' found.")
            oi = int(r["item"])
            bracket_decision[oi] = r
        elif t == "topset":
            if topset is not None:
                raise RerankError("Duplicate 'topset' line in progress.")
            topset = r
        else:
            raise RerankError(f"Unknown progress line type: {t!r}")

    return Journal(
        meta=meta,
        skeleton_ids=skeleton_ids,
        skeleton_cmp=skeleton_cmp,
        skeleton_bt_order=skeleton_bt_order,
        bracket_probes=bracket_probes,
        bracket_confirms=bracket_confirms,
        bracket_decision=bracket_decision,
        topset=topset,
        topfull_cmp=topfull_cmp,
        topfull_bt_order=topfull_bt_order,
    )

# ---------- Build items & maps ----------
def item_map(items: List[Item]) -> Dict[int, Item]:
    return {it.original_index: it for it in items}

# ---------- Deterministic skeleton selection ----------
def choose_skeleton(all_ids_sorted: List[int], S: int) -> List[int]:
    n = len(all_ids_sorted)
    if n <= S:
        return list(all_ids_sorted)
    out: List[int] = []
    for i in range(S):
        ix = int(round(i * (n - 1) / (S - 1)))
        out.append(all_ids_sorted[ix])
    # ensure uniqueness by forward fill if rounding collided
    if len(set(out)) != len(out):
        seen = set()
        fixed: List[int] = []
        for v in out:
            if v not in seen:
                fixed.append(v); seen.add(v)
        for v in all_ids_sorted:
            if len(fixed) >= S: break
            if v not in seen:
                fixed.append(v); seen.add(v)
        out = fixed
    # strictly increasing check
    if any(out[i] >= out[i+1] for i in range(len(out)-1)):
        raise RerankError("choose_skeleton failed to produce strictly increasing ids.")
    return out

# ---------- Skeleton execution ----------
def run_skeleton_round_robin(j: Journal, items: List[Item], query: str, client) -> List[int]:
    id2item = item_map(items)
    all_ids = [it.original_index for it in items]
    # establish skeleton selection
    if j.skeleton_ids is None:
        skel = choose_skeleton(all_ids, SKELETON_SIZE)
        fs_write_jsonl_line(PROGRESS_PATH, {"type":"skeleton","ids":skel,"count":len(skel)})
    else:
        expected = choose_skeleton(all_ids, SKELETON_SIZE)
        if j.skeleton_ids != expected:
            raise RerankError("Existing 'skeleton' in progress does not match deterministic selection.")
        skel = j.skeleton_ids

    schedule = circle_round_robin(skel)  # (left,right,round)
    need = len(schedule)

    # Validate already recorded comparisons match schedule prefix
    done = len(j.skeleton_cmp)
    if done > need:
        raise RerankError(f"progress has {done} skeleton comparisons but only {need} are required.")
    for k in range(done):
        rec = j.skeleton_cmp[k]
        l, r, rnd = schedule[k]
        if int(rec.get("left")) != l or int(rec.get("right")) != r or int(rec.get("schedule_round")) != rnd:
            raise RerankError("Recorded skeleton comparison does not match deterministic schedule at index "
                              f"{k}: expected (left={l}, right={r}, round={rnd}), got {rec}.")
        if rec.get("winner") not in ("left","right"):
            raise RerankError("Recorded skeleton comparison has invalid 'winner' (expected 'left' or 'right').")

    # If already complete (resume case), print a final line and skip doing work
    if done == need:
        print_progress("skeleton", need, need)
        print_done("skeleton")

    # Execute remaining comparisons
    current = done
    total = need
    for k in range(done, need):
        l, r, rnd = schedule[k]
        A = id2item[l]
        B = id2item[r]
        winner, rationale = judge_pair_strict(query, A, B, client, RETRIES_PER_JUDGE)
        mapped = "left" if winner == "A" else "right"
        fs_write_jsonl_line(PROGRESS_PATH, {
            "type":"cmp",
            "stage":"skeleton",
            "pair_index": k,
            "left": l,
            "left_filename": A.filename,
            "right": r,
            "right_filename": B.filename,
            "winner": mapped,
            "rationale": rationale,
            "schedule_round": rnd
        })
        current += 1
        print_progress("skeleton", current, total)

    # Completed all scheduled comparisons
    print_done("skeleton")

    # Build duels for BT
    rows = read_progress_lines()
    journal = replay_progress(rows)
    index = {oi:i for i,oi in enumerate(skel)}
    duels: List[Tuple[int,int]] = []
    for rec in journal.skeleton_cmp:
        wl = rec["winner"]
        li = index[int(rec["left"])]
        ri = index[int(rec["right"])]
        if wl == "left":
            duels.append((li, ri))
        else:
            duels.append((ri, li))
    order, scores = bt_fit_order(skel, duels, stage="skeleton")
    fs_write_jsonl_line(PROGRESS_PATH, {
        "type":"bt_summary",
        "stage":"skeleton",
        "scores": {str(k): v for k,v in scores.items()},
        "order": order
    })
    return order

# ---------- Bracketing ----------
def run_bracketing(j: Journal, items: List[Item], query: str, client, skeleton_order: List[int]) -> Dict[int, int]:
    """
    Returns tier_map: non-skeleton item oi -> tier index in [0..S].

    Tier convention:
      - Tier 0 contains items **more relevant** than the top skeleton anchor (i.e., above skeleton[0]).
      - Tier 1 contains items between skeleton[0] and skeleton[1] (1st and 2nd most relevant anchors).
      - ...
      - Tier S contains items **less relevant** than the last skeleton anchor (i.e., below skeleton[S-1]).
    """
    id2item = item_map(items)
    S = len(skeleton_order)
    skeleton_set = set(skeleton_order)
    extra_ids = [it.original_index for it in items if it.original_index not in skeleton_set]

    # Build probe/confirm caches from journal
    probe_map: Dict[Tuple[int,int], dict] = {}   # (item,pivot) -> last record
    for oi, plist in j.bracket_probes.items():
        for rec in plist:
            probe_map[(int(rec["item"]), int(rec["pivot"]))] = rec
    confirm_map: Dict[Tuple[int,int], List[dict]] = {}  # (item,against) -> list of recs
    for oi, clist in j.bracket_confirms.items():
        for rec in clist:
            confirm_map.setdefault((int(rec["item"]), int(rec["against"])), []).append(rec)
    tier_map: Dict[int,int] = {int(k): int(v["tier"]) for k,v in j.bracket_decision.items()}

    # For fairness, alternate orientation for each *extra* item sequence deterministically.
    next_left_for_item: Dict[int, bool] = {}
    current_bracket = sum(len(v) for v in j.bracket_probes.values()) + sum(len(v) for v in j.bracket_confirms.values())
    num_extra = len(extra_ids)
    expected_per = math.ceil(math.log2(S + 1)) + 2 * BRACKET_CONFIRMATION
    total_bracket = num_extra * expected_per

    def orient_pair(item_oi: int, a: Item, b: Item) -> Tuple[Item, Item]:
        left_next = next_left_for_item.get(item_oi, True)
        next_left_for_item[item_oi] = not left_next
        return (a, b) if left_next else (b, a)

    def record_probe(item_oi: int, pivot_oi: int, left: Item, right: Item, winner: str, rationale: str, lo: int, hi: int):
        fs_write_jsonl_line(PROGRESS_PATH, {
            "type":"probe",
            "stage":"bracket",
            "item": item_oi,
            "pivot": pivot_oi,
            "left": left.original_index,
            "left_filename": left.filename,
            "right": right.original_index,
            "right_filename": right.filename,
            "winner": winner,
            "rationale": rationale,
            "path_lo": lo,
            "path_hi": hi
        })

    def record_confirm(item_oi: int, against_oi: int, left: Item, right: Item, winner: str, rationale: str):
        fs_write_jsonl_line(PROGRESS_PATH, {
            "type":"confirm",
            "stage":"bracket",
            "item": item_oi,
            "against": against_oi,
            "left": left.original_index,
            "left_filename": left.filename,
            "right": right.original_index,
            "right_filename": right.filename,
            "winner": winner,
            "rationale": rationale
        })

    def record_decision(item_oi: int, tier: int, probes: int, confirms: int, votes_sum: int):
        fs_write_jsonl_line(PROGRESS_PATH, {
            "type":"bracket_decision",
            "stage":"bracket",
            "item": item_oi,
            "tier": tier,
            "probes": probes,
            "confirms": confirms,
            "vote_sum": votes_sum
        })

    # Work remaining items
    for oi in extra_ids:
        if oi in tier_map:
            continue
        x = id2item[oi]
        lo, hi = 0, S
        probes_ct = 0
        votes_sum = 0  # +1 means evidence x > pivot, -1 means x <= pivot

        while lo < hi:
            mid = (lo + hi) // 2
            p_oi = skeleton_order[mid]
            pivot = id2item[p_oi]
            cached = probe_map.get((x.original_index, pivot.original_index))
            if cached:
                # Use exact logged orientation and winner
                l_oi = int(cached["left"]); r_oi = int(cached["right"])
                left = id2item[l_oi]; right = id2item[r_oi]
                win = cached["winner"]
                # Map to direction for votes_sum and lo/hi
                x_on_left = (left.original_index == x.original_index)
                x_won = (win == "left" and x_on_left) or (win == "right" and not x_on_left)
                votes_sum += (1 if x_won else -1)
            else:
                left, right = orient_pair(x.original_index, x, pivot)
                w, rat = judge_pair_strict(query, left, right, client, RETRIES_PER_JUDGE)
                win = "left" if w == "A" else "right"
                record_probe(x.original_index, pivot.original_index, left, right, win, rat, lo, hi)
                probes_ct += 1
                # cache for potential reuse
                probe_map[(x.original_index, pivot.original_index)] = {
                    "left": left.original_index,
                    "right": right.original_index,
                    "winner": win,
                    "rationale": rat,
                    "path_lo": lo,
                    "path_hi": hi
                }
                x_on_left = (left.original_index == x.original_index)
                x_won = (win == "left" and x_on_left) or (win == "right" and not x_on_left)
                votes_sum += (1 if x_won else -1)
                current_bracket += 1
                print_progress("bracketing", current_bracket, total_bracket, approximate=True)

            # Binary decision
            if x_won:
                hi = mid
            else:
                lo = mid + 1

        # Confirmation around boundary 'lo' (0..S)
        confirms_ct = 0
        # We collect up to BRACKET_CONFIRMATION votes per side when available
        def confirm_against(bound_idx: int, expect_x_above: bool) -> int:
            """Return +1 if result supports 'x above boundary' hypothesis, else -1."""
            nonlocal confirms_ct, current_bracket
            if bound_idx < 0 or bound_idx >= S:
                return 0
            y = id2item[skeleton_order[bound_idx]]
            # Run up to BRACKET_CONFIRMATION confirmations (deterministic alternation)
            votes = 0
            prev_list = confirm_map.get((x.original_index, y.original_index), [])
            used = 0
            # Reuse existing confirmations first (in order)
            for rec in prev_list[:BRACKET_CONFIRMATION]:
                l_oi = int(rec["left"]); r_oi = int(rec["right"])
                win = rec["winner"]
                x_on_left = (l_oi == x.original_index)
                x_won_local = (win == "left" and x_on_left) or (win == "right" and not x_on_left)
                votes += (1 if x_won_local else -1)
                used += 1
            # Run remaining confirmations if needed
            for _ in range(used, BRACKET_CONFIRMATION):
                left, right = orient_pair(x.original_index, x, y)
                w, rat = judge_pair_strict(query, left, right, client, RETRIES_PER_JUDGE)
                win = "left" if w == "A" else "right"
                record_confirm(x.original_index, y.original_index, left, right, win, rat)
                confirms_ct += 1
                x_on_left = (left.original_index == x.original_index)
                x_won_local = (win == "left" and x_on_left) or (win == "right" and not x_on_left)
                votes += (1 if x_won_local else -1)
                # update confirm_map to avoid duplicate within-session calls
                confirm_map.setdefault((x.original_index, y.original_index), []).append({
                    "left": left.original_index,
                    "right": right.original_index,
                    "winner": win,
                    "rationale": rat
                })
                current_bracket += 1
                print_progress("bracketing", current_bracket, total_bracket, approximate=True)
            # Interpret votes
            support = (votes > 0)  # strict majority; tie → treat as not supporting move
            return (1 if support == expect_x_above else -1)

        # In-boundary checks
        votes_boundary = 0
        # vs left neighbor (expect x above it)
        votes_boundary += confirm_against(lo - 1, expect_x_above=True)
        # vs right neighbor (expect x NOT above it; i.e., right neighbor >= x)
        votes_boundary += confirm_against(lo, expect_x_above=False)

        # Final deterministic decision from sign of (path votes + boundary votes)
        total_votes = votes_sum + votes_boundary
        tier = lo
        if total_votes < 0 and lo > 0:
            tier = lo - 1
        # Record and store
        record_decision(x.original_index, tier, probes_ct, confirms_ct, total_votes)
        tier_map[x.original_index] = tier

    # All items have recorded bracket_decision; finalize with an explicit 'done' message.
    print_done("bracketing", approximate=True)
    return tier_map

# ---------- Top-block selection & full BT ----------
def build_topblock_ids(tier_map: Dict[int,int], skeleton_order: List[int], top_limit: int) -> Tuple[List[int], List[int], bool, Optional[str]]:
    """
    Build top block by taking whole tiers starting from 0 until adding another tier would exceed 'top_limit',
    and include the skeleton anchors that lie between included tiers.

    Tier convention reminder:
      - Tier 0 is **above** skeleton[0] (more relevant than the most-relevant anchor).
      - Between Tier t and Tier t+1 sits skeleton[t], for t in [0..S-1].

    Returns (topblock_ids, tiers_included, skipped, reason_if_skipped).
    If Tier 0 alone > top_limit -> skipped=True, reason non-null, ids=[].
    """
    S = len(skeleton_order)
    # collect items per tier (non-skeleton)
    tiers: Dict[int, List[int]] = {}
    for oi, t in tier_map.items():
        tiers.setdefault(t, []).append(oi)

    # size of Tier 0 only (no anchors needed if we include only Tier 0)
    if len(tiers.get(0, [])) > top_limit:
        return [], [0], True, f"Tier 0 has {len(tiers.get(0, []))} items (> {top_limit}); skipping Step 3."

    included_consecutive: List[int] = []
    # helper: how many anchors lie *between* tiers 0..t (inclusive)?
    # If we include tiers 0..t consecutively, there are exactly t anchors: skeleton[0..t-1].
    def anchors_between_up_to(t: int) -> int:
        return max(0, min(t, S))

    # Greedily include tiers from 0 up while respecting the cap using the *same* counting as materialization.
    for t in range(0, S + 1):
        total_items = sum(len(tiers.get(k, [])) for k in range(0, t + 1))
        total_anchors = anchors_between_up_to(t)
        projected = total_items + total_anchors
        if projected <= top_limit:
            included_consecutive.append(t)
        else:
            break

    if not included_consecutive:
        included_consecutive = [0]  # sanity; shouldn't happen due to Tier-0 check

    # Materialize: [Tier0] + [skeleton[0]] + [Tier1] + [skeleton[1]] + ... BUT
    # include an anchor *only between* two included tiers.
    top_ids: List[int] = []
    last_t = included_consecutive[-1]
    for t in included_consecutive:
        # items in tier t
        top_ids.extend(sorted(tiers.get(t, [])))
        # anchor after tier t only if the next tier t+1 is also included
        if t < S and (t + 1) in included_consecutive:
            top_ids.append(skeleton_order[t])

    # Safety check: enforce the cap deterministically (should already hold by construction).
    if len(top_ids) > top_limit:
        # Trim from the end (removes last-added elements deterministically).
        while len(top_ids) > top_limit:
            top_ids.pop()

    tiers_included = included_consecutive
    return top_ids, tiers_included, False, None

def run_topfull_bt(j: Journal, items_by_id: Dict[int, Item], query: str, client,
                   top_ids: List[int]) -> Tuple[List[int], Dict[int,float]]:
    if not top_ids:
        return [], {}
    schedule = circle_round_robin(top_ids)
    need = len(schedule)

    # Validate existing comparisons
    done = len(j.topfull_cmp)
    if done > need:
        raise RerankError(f"progress has {done} topfull comparisons but only {need} are required.")
    for k in range(done):
        rec = j.topfull_cmp[k]
        l, r, rnd = schedule[k]
        if int(rec.get("left")) != l or int(rec.get("right")) != r or int(rec.get("schedule_round")) != rnd:
            raise RerankError("Recorded topfull comparison does not match deterministic schedule at index "
                              f"{k}: expected (left={l}, right={r}, round={rnd}), got {rec}.")
        if rec.get("winner") not in ("left","right"):
            raise RerankError("Recorded topfull comparison has invalid 'winner' (expected 'left' or 'right').")

    if done == need:
        print_progress("topfull", need, need)
        print_done("topfull")

    # Execute remaining pairs
    current = done
    total = need
    for k in range(done, need):
        l, r, rnd = schedule[k]
        A = items_by_id[l]
        B = items_by_id[r]
        winner, rationale = judge_pair_strict(query, A, B, client, RETRIES_PER_JUDGE)
        mapped = "left" if winner == "A" else "right"
        fs_write_jsonl_line(PROGRESS_PATH, {
            "type":"cmp",
            "stage":"topfull",
            "pair_index": k,
            "left": l,
            "left_filename": A.filename,
            "right": r,
            "right_filename": B.filename,
            "winner": mapped,
            "rationale": rationale,
            "schedule_round": rnd
        })
        current += 1
        print_progress("topfull", current, total)

    # Completed all scheduled comparisons for topfull
    print_done("topfull")

    # Build duels for BT
    rows = read_progress_lines()
    journal = replay_progress(rows)
    index = {oi:i for i,oi in enumerate(top_ids)}
    duels: List[Tuple[int,int]] = []
    for rec in journal.topfull_cmp:
        wl = rec["winner"]
        li = index[int(rec["left"])]
        ri = index[int(rec["right"])]
        if wl == "left":
            duels.append((li, ri))
        else:
            duels.append((ri, li))
    order, scores = bt_fit_order(top_ids, duels, stage="topfull")
    fs_write_jsonl_line(PROGRESS_PATH, {
        "type":"bt_summary",
        "stage":"topfull",
        "scores": {str(k): v for k,v in scores.items()},
        "order": order
    })
    return order, scores

# ---------- Final assembly ----------
def write_final_order_csv(items: List[Item],
                          skeleton_order: List[int],
                          tier_map: Dict[int,int],
                          top_block_ids: List[int],
                          top_block_bt_order: Optional[List[int]]) -> None:
    """
    Final order:
      - If top_block_bt_order is provided (Step 3 ran), we emit that entire block first (most→least) and exclude those IDs from the remainder.
      - The remainder is tiered relative to the skeleton: Tier t (lexicographic by filename) then skeleton[t], for t over the tiers not covered by the top block.
      - If Step 3 did not run (or was skipped), everything is tiered with lexicographic inside tiers; skeleton anchors keep their Step-1 BT order.
      - If n <= skeleton size, Step 2/3 don't run; write skeleton order directly.
    """
    id2item = item_map(items)
    S = len(skeleton_order)

    all_ids = [it.original_index for it in items]
    in_top_block = set(top_block_bt_order or [])
    out: List[int] = []

    if top_block_bt_order:
        out.extend(top_block_bt_order)

    # Build tiers for remaining items (non-top-block and non-skeleton? we must rebuild consistently)
    tiers: Dict[int, List[int]] = {}
    for oi, t in tier_map.items():
        if oi in in_top_block:
            continue
        tiers.setdefault(t, []).append(oi)

    # Emit the remaining tiers + anchors, skipping any anchor that is already included in the top block
    for t in range(0, S + 1):
        # tier items
        rest = tiers.get(t, [])
        rest_sorted = sorted(rest, key=lambda x: (id2item[x].filename, x))
        out.extend(rest_sorted)
        # skeleton anchor between Tier t and t+1
        if t < S:
            anchor = skeleton_order[t]
            if anchor not in in_top_block:
                out.append(anchor)

    # Sanity: must be a permutation of all items
    if sorted(out) != sorted(all_ids):
        missing = sorted(set(all_ids) - set(out))
        extra   = sorted(set(out) - set(all_ids))
        raise RerankError(f"Final order assembly mismatch. missing={missing[:10]}, extra={extra[:10]}")

    fs_overwrite_text(ORDER_PATH, ",".join(map(str, out)) + "\n")
    print(f"\nWrote ranking ({len(out)} items) to {ORDER_PATH}")

# ---------- Main ----------
def print_progress(stage: str, current: int, total: int, approximate: bool = False) -> None:
    if total == 0:
        return
    percent = (current / total) * 100
    percent_str = f"~{percent:.1f}" if approximate else f"{percent:.1f}"
    print(f"{stage.capitalize()} progress: ({percent_str}%, {current}/{total})\n")

def print_done(stage: str, approximate: bool = False) -> None:
    if approximate:
        # Explicitly call out that the % was an estimate
        print(f"{stage.capitalize()} done: progress was an estimate, so the printed percent may not have reached exactly 100%.\n")
    else:
        print(f"{stage.capitalize()} done: 100.0% (all tasks complete).\n")

def main() -> None:
    load_dotenv()
    # CLI
    parser = argparse.ArgumentParser(description="Deterministic 3-step reranker (skeleton+bracket+topBT)")
    parser.add_argument("--query", type=str, required=True, help="Free-form user query that explains what's important")
    args = parser.parse_args()
    user_query = (args.query or "").strip()
    if not user_query:
        raise RerankError("ERROR: --query must be a non-empty string.")

    client = get_client()
    if client is None:
        raise RerankError("LLM client unavailable (core.llm.get_client() returned None). Configure .env and Ollama models.")

    ensure_dirs()
    run_path = newest_run_file(SEARCH_RUNS_DIR)
    run_file = os.path.basename(run_path)

    run_file2, run_created_at, pass_number, items = load_items_from_run(run_path)
    if run_file2 != run_file:
        raise RerankError(f"Internal mismatch: resolved run file {run_file!r} vs parsed {run_file2!r}")

    all_item_ids = [it.original_index for it in items]

    expected_meta = {
        "type": "meta",
        "algo": ALGO_ID,
        "run_file": run_file,
        "run_created_at": run_created_at,
        "pass_number": pass_number,
        "query": user_query,
        "item_ids": all_item_ids,
        "params": {
            "skeleton_size": SKELETON_SIZE,
            "topfull_size": TOPFULL_SIZE,
            "retries_per_judge": RETRIES_PER_JUDGE,
            "bracket_confirmation": BRACKET_CONFIRMATION,
            "bt_alpha": BT_ALPHA,
            "bt_method": BT_METHOD,
            "bt_tol": BT_TOL,
            "bt_max_iter": BT_MAX_ITER
        }
    }
    rows = assert_meta_or_write(expected_meta)
    journal = replay_progress(rows)

    # STEP 1: Skeleton full round-robin + BT
    if journal.skeleton_bt_order is None:
        skeleton_order = run_skeleton_round_robin(journal, items, user_query, client)
        journal = replay_progress(read_progress_lines())
    else:
        skeleton_order = journal.skeleton_bt_order

    # If total n <= SKELETON_SIZE → DONE (no Step 2/3)
    if len(items) <= SKELETON_SIZE:
        fs_overwrite_text(ORDER_PATH, ",".join(map(str, skeleton_order)) + "\n")
        print(f"\nWrote ranking ({len(skeleton_order)} items) to {ORDER_PATH}")
        return

    # STEP 2: Bracketing
    tier_map = run_bracketing(journal, items, user_query, client, skeleton_order)
    journal = replay_progress(read_progress_lines())

    # STEP 3: Top block only if Step 2 ran (we're here => n>SKELETON_SIZE)
    items_by_id = item_map(items)
    top_ids, tiers_included, skipped, reason = build_topblock_ids(tier_map, skeleton_order, TOPFULL_SIZE)

    # log topset deterministically (with concrete ids)
    if journal.topset is None:
        fs_write_jsonl_line(PROGRESS_PATH, {
            "type":"topset",
            "tiers_included": tiers_included,
            "topset_ids": top_ids,
            "count": len(top_ids),
            "skipped_full_bt": skipped,
            "reason_if_skipped": reason
        })
        journal = replay_progress(read_progress_lines())
    else:
        # Validate consistency
        if journal.topset.get("tiers_included") != tiers_included:
            raise RerankError("Existing 'topset' tiers_included differ from deterministic selection.")
        if journal.topset.get("topset_ids") != top_ids:
            raise RerankError("Existing 'topset' ids differ from deterministic selection.")
        if bool(journal.topset.get("skipped_full_bt")) != bool(skipped):
            raise RerankError("Existing 'topset' skipped_full_bt flag differs from deterministic decision.")

    top_block_bt_order: Optional[List[int]] = None
    if skipped:
        print(f"Warning: {reason}")
        print_done("topfull")  # Trivial 'done' to mirror other stages
    else:
        # Run or reuse topfull BT
        if journal.topfull_bt_order is None:
            order, scores = run_topfull_bt(journal, items_by_id, user_query, client, top_ids)
            journal = replay_progress(read_progress_lines())
            top_block_bt_order = order
        else:
            # ensure last bt_summary exists
            bt_line = None
            for r in reversed(read_progress_lines()):
                if r.get("type") == "bt_summary" and r.get("stage") == "topfull":
                    bt_line = r; break
            if bt_line is None:
                raise RerankError("progress has topfull_bt_order but no corresponding bt_summary.")
            top_block_bt_order = journal.topfull_bt_order

    # FINAL: assemble & write order.csv
    write_final_order_csv(items, skeleton_order, tier_map, top_ids, top_block_bt_order)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except RerankError as e:
        print(f"\nFATAL: {e}")
        sys.exit(1)
