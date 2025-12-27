import os
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import datetime

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

def get_dynamic_context_window(
    all_chunks: List[Chunk],
    current_index: int,
    max_tokens: int
) -> str:
    """
    Builds a context window around a central chunk, dynamically expanding
    outwards until the token limit is reached.

    Args:
        all_chunks: The list of all chunk objects.
        current_index: The index of the chunk of interest.
        max_tokens: The maximum number of tokens the context window can have.

    Returns:
        A single string containing the concatenated text of the context window.
    """
    chunk_of_interest = all_chunks[current_index]

    # Start with the chunk of interest
    context_parts = [chunk_of_interest.content]
    current_tokens = chunk_of_interest.token_count

    # Pointers for expanding left and right
    left_ptr = current_index - 1
    right_ptr = current_index + 1

    # Alternate adding from left and right until we can't anymore
    while left_ptr >= 0 or right_ptr < len(all_chunks):
        # Try adding from the right
        if right_ptr < len(all_chunks):
            next_chunk = all_chunks[right_ptr]
            if current_tokens + next_chunk.token_count <= max_tokens:
                context_parts.append(next_chunk.content)
                current_tokens += next_chunk.token_count
                right_ptr += 1
            else:
                # Can't add more on the right, stop trying
                right_ptr = len(all_chunks)

        # Try adding from the left
        if left_ptr >= 0:
            next_chunk = all_chunks[left_ptr]
            if current_tokens + next_chunk.token_count <= max_tokens:
                context_parts.insert(0, next_chunk.content)
                current_tokens += next_chunk.token_count
                left_ptr -= 1
            else:
                # Can't add more on the left, stop trying
                left_ptr = -1

    return "\n---\n".join(context_parts)

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

You are a highly focused research assistant. Your task is to determine if a specific, isolated section of text is relevant to a user's query.

I will provide you with a large context of surrounding text to help you understand the overall topic. However, your final judgment must be based ONLY on the content of the **"SECTION OF INTEREST"** provided at the very end. The context is for reasoning, but the decision must be about the specific section.

--- USER QUERY BEGINS ---
{query}
--- USER QUERY ENDS ---

--- CONTEXT BEGINS ---
{context}
--- CONTEXT ENDS ---

Now, focusing exclusively on the following text, is there anything in this specific section
that fits the user's query **as it is written**?

--- SECTION OF INTEREST BEGINS ({chunk_filename}) ---
{chunk_of_interest}
--- SECTION OF INTEREST ENDS ---

Respond with a JSON object in the following format and nothing else:
{{
  "summary": "<Consice ENGLISH overview of the contents of SECTION OF INTEREST (only)\nString should be one paragraphp containing EXACTLY three short sentences.\nExtremely concise and information-dense.>",
  "evidence": "<Summarize the decisive cues from the SECTION OF INTEREST only (quote key words/phrases when useful).\nIf NOT relevant, state the key reason(s) it fails; if relevant, state the key reason(s) it matches\nString should be one paragraphp containing EXACTLY three short sentences.\nExtremely concise and information-dense.>",
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

    for pos in plan.positions_to_process:
        chunk = position_to_chunk[pos]
        print(f"\nAnalyzing chunk position {pos}/{plan.run_total} ('{chunk.filename}')...")

        context_window = get_dynamic_context_window(
            all_chunks, chunk.original_index - 1, CONTEXT_WINDOW_SIZE_TOKENS
        )
        judgement = check_relevance_with_llm(
            context_window, chunk.content, chunk.filename, ctx, query
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

            run_search_pass(all_chunks, args.ctx, args.query, plan)
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

            run_search_pass(all_chunks, args.ctx, args.query, plan)
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

        run_search_pass(all_chunks, args.ctx, args.query, plan)
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
