import os
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import datetime
from bisect import bisect_right

from dotenv import load_dotenv
from core.tokens import count_tokens
from core.llm import get_client, print_stats, chat_complete

# --- Setup ---
# Load environment variables from a .env file for security.
load_dotenv()

# Initialize the Ollama client (no raw HTTP; library only).
CLIENT = get_client()

# Get context window size from environment variable. This governs how much *surrounding text*
# we feed around the chunk of interest, not the model's num_ctx.
CONTEXT_WINDOW_SIZE_TOKENS = int(os.environ["CONTEXT_WINDOW_SIZE_TOKENS"])

RECORD_DELIM = "\x1e"

# --- Data Structure ---

@dataclass
class RunPlan:
    mode: str  # 'fresh' | 'refine' | 'resume'
    run_path: str
    selected_indices: List[int]          # original indices in the run's selection
    positions_to_process: List[int]      # run-relative positions that still need processing
    run_total: int                       # total positions in this run
    pass_number: int
    meta: Optional[Dict[str, Any]] = None  # meta to write at start (None for resume)

@dataclass
class Chunk:
    """
    A dataclass to hold the content of a text chunk, its filename,
    its pre-calculated token count, and its original index.
    """
    original_index: int
    filename: str
    content: str
    token_count: int

# --- Helper Functions ---

def load_chunks_from_disk(directory: str) -> List[Chunk]:
    """
    Loads all .txt files from a directory into a list of Chunk objects.
    Pre-calculates token counts and stores the original index for each chunk.
    """
    print("Loading and tokenizing all chunks from disk...")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return []

    chunk_files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    chunks: List[Chunk] = []

    for i, filepath in enumerate(chunk_files):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            token_count = count_tokens(content)
            filename = os.path.basename(filepath)
            # Store the original index 'i' in the Chunk object.
            chunks.append(Chunk(original_index=i + 1, filename=filename, content=content, token_count=token_count))
        except IOError as e:
            print(f"Warning: Could not read file {filepath}: {e}")

    total_tokens = sum(c.token_count for c in chunks)
    print(f"Loaded {len(chunks)} chunks ({total_tokens:,} tokens).")
    return chunks

def build_token_prefix_sums(all_chunks: List[Chunk]) -> List[int]:
    """
    Build token prefix sums over the chunks in their on-disk order.

    Let P be a list of length N+1 where:
      - P[0] = 0
      - P[i] = total tokens in chunks [0 .. i-1]
      - chunk i spans token positions [P[i], P[i+1])

    This allows constant-time mapping between chunk index and token-space positions,
    and logarithmic-time mapping from token positions back to chunk boundaries.
    """
    P = [0]
    total = 0
    for c in all_chunks:
        total += int(c.token_count)
        P.append(total)
    return P

def round_div_half_away_from_zero(numer: int, denom: int) -> int:
    """
    Deterministic rounding for rational numer/denom where denom > 0.
    Half values round away from zero.
    """
    if denom <= 0:
        raise ValueError("denom must be > 0")
    if numer >= 0:
        return (numer + (denom // 2)) // denom
    return -((-numer + (denom // 2)) // denom)

def end_boundary_from_start(prefix_sums: List[int], start_chunk: int, max_tokens: int) -> int:
    """
    Given a start chunk index s, choose the maximal end boundary e (a boundary index, not a chunk index)
    such that the total tokens in chunks [s .. e-1] is <= max_tokens.

    This is defined by the prefix sums:
      prefix_sums[e] - prefix_sums[s] <= max_tokens

    The chosen end is maximal (packs as many chunks as possible under budget).
    """
    max_tok = prefix_sums[start_chunk] + max_tokens
    e = bisect_right(prefix_sums, max_tok) - 1
    return max(start_chunk + 1, min(e, len(prefix_sums) - 1))

def select_stable_window_bounds(
    all_chunks: List[Chunk],
    prefix_sums: List[int],
    current_index: int,
    max_tokens: int
) -> Tuple[int, int]:
    """
    Deterministically choose a contiguous window [start_chunk, end_boundary) around current_index.

    Requirements:
      - Stateless: the result depends only on (chunk sizes, current_index, max_tokens).
        Killing the process and resuming produces the same (start,end) for each chunk.
      - The chunk of interest is intended to sit at 50% of the window, and is allowed to drift
        between 30% and 70% of the window.
      - Cache-friendly: the first chunk of the window is made stable on purpose.

    The cache story in plain terms:
      - Prefix caching is only as good as prompt-prefix stability.
      - The earliest place where prompts differ determines how many tokens can be reused.
      - A window that changes its first chunk frequently forces frequent cache misses.
      - Stabilizing the first chunk stabilizes a large prompt prefix.
      - Increasing max_tokens makes the stability stronger, because the "allowed drift"
        grows linearly with max_tokens, so it takes more chunks to cross the threshold.

    How the window is chosen:
      1) Convert the chunk index into token-space using prefix sums.
      2) Compute the midpoint token position of the chunk-of-interest.
      3) Compute the ideal left edge that would put that midpoint at 50% of the max_tokens budget.
      4) Snap that ideal left edge onto a fixed token grid whose spacing equals the allowed drift width.
         With a 30%-70% band, the band width is 40% of the budget.
      5) Convert the snapped left edge back to a chunk boundary, then pack to the right up to the budget.

    Snapping is the key to stability:
      - With budget W and allowed band [0.3W, 0.7W], the band width is 0.4W.
      - The snapped start changes only when the ideal left edge crosses a multiple of 0.4W.
      - As W grows, 0.4W grows, so the snapped start changes less often.
    """
    n = len(all_chunks)
    if n == 0:
        return 0, 0
    if current_index < 0 or current_index >= n:
        raise IndexError("current_index out of range")

    W = int(max_tokens)
    if W <= 0:
        raise ValueError("max_tokens must be > 0")

    # Band endpoints relative to the token budget W.
    # Allowed midpoint offset is [30% of W, 70% of W].
    min_ok = (3 * W) // 10
    max_ok = (7 * W) // 10
    min_ok2 = 2 * min_ok
    max_ok2 = 2 * max_ok

    # Grid spacing equals band width: (70% - 30%) = 40% of W.
    # This drives the "longer context -> fewer misses" behavior.
    step = max(1, max_ok - min_ok)

    # Use doubled token coordinates to keep exact midpoints without floats.
    # mid2 = 2 * midpoint_token_position
    mid2 = prefix_sums[current_index] + prefix_sums[current_index + 1]

    # ideal_left = midpoint - W/2
    # With doubled coordinates: ideal_left = (mid2 - W) / 2
    # To snap onto multiples of step: k = round( ideal_left / step )
    # => k = round( (mid2 - W) / (2*step) )
    k = round_div_half_away_from_zero(mid2 - W, 2 * step)
    snapped_left = k * step

    # Convert snapped_left (token position) to a candidate start chunk boundary.
    # Two adjacent boundaries are considered to compensate for boundary quantization.
    s0 = bisect_right(prefix_sums, snapped_left) - 1
    s0 = max(0, min(s0, n - 1))
    candidates = [s0]
    if s0 + 1 <= n - 1:
        candidates.append(s0 + 1)

    def score_start(s: int) -> Tuple[int, int, int, int, int]:
        e = end_boundary_from_start(prefix_sums, s, W)

        # Inclusion is non-negotiable: current_index must be inside [s, e).
        inclusion_penalty = 0 if (s <= current_index < e) else 1

        # Midpoint offset from window start in doubled-token units.
        off2 = mid2 - 2 * prefix_sums[s]

        band_violation = 0
        if off2 < min_ok2:
            band_violation = min_ok2 - off2
        elif off2 > max_ok2:
            band_violation = off2 - max_ok2

        # Prefer starts closer to snapped_left in token space.
        dist_to_snapped = abs(prefix_sums[s] - snapped_left)

        # Prefer longer realized windows (more chunks under budget).
        realized = prefix_sums[e] - prefix_sums[s]
        realized_penalty = W - realized  # smaller is better; can be >0 near document end

        # Deterministic tie-break: smaller start index.
        return (inclusion_penalty, band_violation, realized_penalty, dist_to_snapped, s)

    start_chunk = min(candidates, key=score_start)
    end_boundary = end_boundary_from_start(prefix_sums, start_chunk, W)

    if not (start_chunk <= current_index < end_boundary):
        start_chunk = current_index
        end_boundary = end_boundary_from_start(prefix_sums, start_chunk, W)

    return start_chunk, end_boundary

def build_context_from_bounds(all_chunks: List[Chunk], start_chunk: int, end_boundary: int) -> str:
    """
    Build the contiguous context window as text, preserving on-disk chunk order.

    The window is [start_chunk, end_boundary) in chunk indices.
    """
    parts = [all_chunks[i].content for i in range(start_chunk, end_boundary)]
    return "\n---\n".join(parts)

def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parses a JSON string that might be embedded in other text.
    """
    try:
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return None

def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO8601-like format with 'Z' suffix."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def check_relevance_with_llm(
    context: str,
    chunk_of_interest: str,
    chunk_filename: str,
    ctx: str,
    query: str,
    max_retries: int = 5
) -> Dict[str, Any]:
    if not CLIENT:
        print("LLM client not initialized. Assuming relevance as a fallback.")
        return {
            "summary": "LLM unavailable. Fallback path used. Treated as relevant.",
            "evidence": "No model call; defaulted to relevant for continuity.",
            "is_relevant": True,
        }

    prompt = f"""
--- BACKGROUND KNOWLEDGE BEGINS ---
{ctx}
--- BACKGROUND KNOWLEDGE ENDS ---

You are a highly focused research assistant. Your task is to determine if a specific, section of text is relevant to a user's query.

I will provide you with a large context of surrounding text to help you understand the overall topic. However, your final judgment must be based ONLY on the meaning of the contents in **"SECTION OF INTEREST"** provided at the very end. The context is for reasoning and to understand the actual significance of the information portrayed.

--- USER QUERY BEGINS ---
{query}
--- USER QUERY ENDS ---

--- CONTEXT BEGINS ---
{context}
--- CONTEXT ENDS ---

Focusing on the following text, does the meaning of this text or anything within it fit the user's query **as it is written**?

The user wants you to understand the section relative to the context given. No false positives, and more importantly if the section strictly speaking matches the criteria, don't miss it!

--- SECTION OF INTEREST BEGINS ({chunk_filename}) ---
{chunk_of_interest}
--- SECTION OF INTEREST ENDS ---

Respond with a JSON object in the following format and nothing else:
{{
  "summary": "<Consice ENGLISH overview of the contents of SECTION OF INTEREST (only)\nString should be one paragraphp containing EXACTLY seven short sentences.\nExtremely concise, information-dense and explanatory.>",
  "evidence": "<Summarize the decisive cues from the SECTION OF INTEREST only (quote key words/phrases when useful).\nIf NOT relevant, state the key reason(s) it fails; if relevant, state the key reason(s) it matches\nString should be one paragraphp containing EXACTLY five short sentences.\nExtremely concise and information-dense.>",
  "is_relevant": <true or false>
}}
"""

    messages = [
        {"role": "system", "content": "Adhere to the instructions as they are written, respond only in JSON."},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(max_retries):
        try:
            response = chat_complete(
                messages=messages,
                role="smartest",
                client=CLIENT,
                # Fail fast on infinite generations, only
                # has effect for models that don't emit <think> tag.
                max_completion_tokens=2048,
                # Do think (if supported).
                # It's important to get a reliable judgement
                please_no_thinking=False,
                require_json=True
            )
            stats = print_stats(response)
            if stats:
                print(stats)
            response_text = response.message.content
            thinking_text = response.message.thinking

            parsed_json = safe_json_loads(response_text)
            if parsed_json and "is_relevant" in parsed_json:
                relevance = parsed_json["is_relevant"]
                print(f"  -> LLM decision: {'Relevant' if relevance else 'Not Relevant'}")
                evidence = parsed_json.get("evidence", "").strip()
                if evidence:
                    print(f"  -> Evidence: {evidence}")
                summary = parsed_json.get("summary", "").strip()
                if summary:
                    print(f"  -> Summary: {summary}")
                return {
                    "summary": summary,
                    "evidence": evidence,
                    "is_relevant": bool(relevance),
                }
            else:
                print(f"  -> Warning (Attempt {attempt + 1}/{max_retries}): LLM response was malformed. Response: {response_text}")
                # Aid debugging for thinking models
                if thinking_text:
                    print("\n" + "-" * 80)
                    print("THINKING (debug):")
                    print("-" * 80)
                    print(thinking_text)
                    print("-" * 80 + "\n")

        except Exception as e:
            print(f"  -> An error occurred on attempt {attempt + 1}/{max_retries}: {e}")

    print(f"  -> All {max_retries} retries failed. Assuming relevance for {chunk_filename}.")
    return {
        "summary": "All retries failed; assumed relevant.",
        "evidence": "Fallback path due to repeated errors.",
        "is_relevant": True,
    }

def run_search_pass(
    all_chunks: List[Chunk],
    prefix_sums: List[int],
    ctx: str,
    query: str,
    plan: RunPlan
) -> None:
    """
    Unified pass that handles 'fresh', 'refine', and 'resume' modes.
    - Writes metadata if provided (fresh/refine).
    - Streams judgements record-by-record (durable progress).
    - Only processes plan.positions_to_process (resume safety).
    Returns the list of *relevant* chunks processed in this pass.
    """
    # If we are starting a new run file (fresh/refine), write meta first.
    if plan.meta is not None:
        write_json_record_append(plan.run_path, plan.meta)

    # Build the run's selection once, then map positions => chunks
    chunks_in_run = build_selected_chunks(all_chunks, plan.selected_indices)
    position_to_chunk = {pos + 1: chunks_in_run[pos] for pos in range(len(chunks_in_run))}

    last_window_key: Optional[Tuple[int, int]] = None
    last_window_text: str = ""
    last_start_chunk: Optional[int] = None
    start_cache_misses = 0

    for pos in plan.positions_to_process:
        chunk = position_to_chunk[pos]
        print(f"\nAnalyzing chunk position {pos}/{plan.run_total} ('{chunk.filename}')...")

        current_index = chunk.original_index - 1
        start_chunk, end_boundary = select_stable_window_bounds(
            all_chunks=all_chunks,
            prefix_sums=prefix_sums,
            current_index=current_index,
            max_tokens=CONTEXT_WINDOW_SIZE_TOKENS
        )

        if last_start_chunk is not None and start_chunk != last_start_chunk:
            start_cache_misses += 1
        last_start_chunk = start_chunk

        window_key = (start_chunk, end_boundary)
        if window_key != last_window_key:
            last_window_text = build_context_from_bounds(all_chunks, start_chunk, end_boundary)
            last_window_key = window_key

        judgement = check_relevance_with_llm(
            last_window_text, chunk.content, chunk.filename, ctx, query
        )

        record = {
            "type": "judgement",
            "position_in_run": pos,
            "original_index": chunk.original_index,
            "filename": chunk.filename,
            "is_relevant": judgement["is_relevant"],
            "summary": judgement["summary"],
            "evidence": judgement["evidence"],
        }
        write_json_record_append(plan.run_path, record)

    end_label = {
        "fresh": "initial pass",
        "refine": "refinement pass",
        "resume": "resumed run",
    }.get(plan.mode, "pass")

    print(f"\nContext-window start cache misses in this pass: {start_cache_misses}/{len(plan.positions_to_process)}")
    print(f"\n--- Deep Search Complete ({end_label}) ---")

# --- Run file helpers ---

def ensure_runs_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def list_run_files(path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(path, "*.jsonl")))

def newest_run_file(path: str) -> Optional[str]:
    files = list_run_files(path)
    return files[-1] if files else None

def next_run_filename(path: str) -> str:
    files = list_run_files(path)
    if not files:
        return os.path.join(path, "0001.jsonl")
    stem = os.path.basename(files[-1]).split(".")[0]
    try:
        n = int(stem)
    except ValueError:
        n = len(files)
    return os.path.join(path, f"{n+1:04d}.jsonl")

def read_jsonl_records(filepath: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            blob = f.read()
    except FileNotFoundError:
        return data

    for segment in blob.split(RECORD_DELIM):
        segment = segment.strip()
        if not segment:
            continue
        try:
            obj = json.loads(segment)
            data.append(obj)
        except json.JSONDecodeError:
            print(f"Warning: Skipping a malformed JSON record in {os.path.basename(filepath)}")
            continue
    return data

def write_json_record_append(filepath: str, obj: Dict[str, Any]) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write(RECORD_DELIM)
        f.write("\n")
        f.flush()                 # flush Python buffers
        os.fsync(f.fileno())      # flush OS buffers (works on Windows & Linux)

def assert_compatible_with_current_chunks(meta: Dict[str, Any], current_total_chunks: int) -> None:
    meta_orig = meta.get("original_total_chunks")
    if meta_orig is None:
        raise RuntimeError(
            "The newest run file is missing 'original_total_chunks' in metadata.\n"
            "This file cannot be resumed or refined safely. Please delete the './search_runs' folder and re-run."
        )
    if int(meta_orig) != int(current_total_chunks):
        raise RuntimeError(
            "The newest run file refers to a different number of original chunks than what is currently loaded.\n"
            f"- File says original_total_chunks = {meta_orig}\n"
            f"- Currently loaded chunks       = {current_total_chunks}\n\n"
            "This usually means your 'split/chunks' directory changed since the run was created.\n"
            "To proceed safely, delete the './search_runs' folder and run again so a new series can be created."
        )

def summarize_run_status(run_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not run_records:
        raise RuntimeError("Run file is empty; cannot determine metadata.")
    meta = run_records[0]
    if meta.get("type") != "meta":
        raise RuntimeError("First record in run file is not metadata; file is invalid.")
    judgements = [r for r in run_records[1:] if r.get("type") == "judgement"]
    return meta, judgements

def build_selected_chunks(all_chunks: List[Chunk], selected_indices: List[int]) -> List[Chunk]:
    index_map = {c.original_index: c for c in all_chunks}
    result = []
    for oi in selected_indices:
        if oi not in index_map:
            raise RuntimeError(f"Selected index {oi} not found in current chunks.")
        result.append(index_map[oi])
    return result

# --- Main Execution ---

def main():
    """
    Main function to parse arguments and orchestrate the iterative search process.
    """
    parser = argparse.ArgumentParser(
        description="Perform a deep, contextual search through text chunks using an LLM."
    )
    parser.add_argument("--ctx", type=str, required=True, help="The search context.")
    parser.add_argument("--query", type=str, required=True, help="The search query for this run.")
    parser.add_argument("--dir", type=str, default="split/chunks", help="The directory containing the text chunks (default: 'split/chunks').")
    parser.add_argument("--outdir", type=str, default='search_runs', help="Directory for JSONL run files (default: './search_runs').")
    args = parser.parse_args()

    # Load all chunks once at the start. This list will be used for context.
    all_chunks = load_chunks_from_disk(args.dir)
    if not all_chunks:
        print("No chunk files found or loaded. Exiting.")
        return

    prefix_sums = build_token_prefix_sums(all_chunks)

    ensure_runs_dir(args.outdir)

    try:
        newest = newest_run_file(args.outdir)

        # ========== Case 1: Fresh start ==========
        if newest is None:
            print("\nNo existing runs found. Starting a new run over all chunks.")
            selected_indices = [c.original_index for c in all_chunks]
            run_total = len(selected_indices)
            run_path = next_run_filename(args.outdir)
            print(f"Creating new run file: {os.path.basename(run_path)}")

            meta = {
                "type": "meta",
                "created_at": utc_now_iso(),
                "query": args.query,
                "original_total_chunks": len(all_chunks),
                "run_total_chunks": run_total,
                "selected_indices": selected_indices,
                "pass_number": 1,
                "delimiter": "\\u001e",
            }

            plan = RunPlan(
                mode="fresh",
                run_path=run_path,
                selected_indices=selected_indices,
                positions_to_process=list(range(1, run_total + 1)),
                run_total=run_total,
                pass_number=1,
                meta=meta,
            )

            run_search_pass(all_chunks, prefix_sums, args.ctx, args.query, plan)
            return

        # ========== Cases 2 & 3: Existing series ==========
        print(f"\nFound existing runs. Newest: {os.path.basename(newest)}")
        records = read_jsonl_records(newest)
        meta, judgements = summarize_run_status(records)
        assert_compatible_with_current_chunks(meta, len(all_chunks))

        selected_indices = meta.get("selected_indices", [])
        if not isinstance(selected_indices, list) or not all(isinstance(x, int) for x in selected_indices):
            raise RuntimeError("Metadata 'selected_indices' is missing or invalid; cannot resume/refine safely.")

        run_total = int(meta.get("run_total_chunks", len(selected_indices)))
        pass_number = int(meta.get("pass_number", 1))

        # Completed -> Refinement pass
        if len(judgements) >= run_total:
            print("Newest run is complete. Starting a refinement round focusing on positive judgements.")

            positive_original_indices = [j["original_index"] for j in judgements if j.get("is_relevant") is True]
            if not positive_original_indices:
                print("No positively judged chunks to refine. Nothing more to do.")
                print("\n--- Deep Search Complete (no further refinement) ---")
                return

            run_path = next_run_filename(args.outdir)
            print(f"Creating new refinement run file: {os.path.basename(run_path)}")

            new_meta = {
                "type": "meta",
                "created_at": utc_now_iso(),
                "query": args.query,
                "original_total_chunks": len(all_chunks),
                "run_total_chunks": len(positive_original_indices),
                "selected_indices": positive_original_indices,
                "pass_number": pass_number + 1,
                "delimiter": "\\u001e",
            }

            plan = RunPlan(
                mode="refine",
                run_path=run_path,
                selected_indices=positive_original_indices,
                positions_to_process=list(range(1, len(positive_original_indices) + 1)),
                run_total=len(positive_original_indices),
                pass_number=pass_number + 1,
                meta=new_meta,
            )

            run_search_pass(all_chunks, prefix_sums, args.ctx, args.query, plan)
            return

        # Partially complete -> Resume
        print("Newest run is partially complete. Resuming...")
        already_done_positions = {int(j.get("position_in_run", -1)) for j in judgements}
        total_positions = list(range(1, run_total + 1))
        remaining_positions = [p for p in total_positions if p not in already_done_positions]

        if not remaining_positions:
            print("Nothing to resume; however file appears incomplete. Exiting to avoid inconsistency.")
            return

        plan = RunPlan(
            mode="resume",
            run_path=newest,                 # append to the SAME file
            selected_indices=selected_indices,
            positions_to_process=remaining_positions,
            run_total=run_total,
            pass_number=pass_number,
            meta=None,                       # don't rewrite meta on resume
        )

        run_search_pass(all_chunks, prefix_sums, args.ctx, args.query, plan)
        return

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt received. Gracefully stopping current run.")
        print("All completed judgements up to this point have been saved.")
    except RuntimeError as e:
        print("\n" + "="*80)
        print("FATAL: Cannot resume or refine the newest run.\n")
        print(str(e))
        print("="*80 + "\n")
    except Exception as e:
        print("\nAn unexpected error occurred:")
        print(repr(e))


if __name__ == "__main__":
    if not CLIENT:
        print("Exiting due to initialization failure.")
    else:
        main()
