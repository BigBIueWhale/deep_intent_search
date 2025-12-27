import os
import json
import argparse
import datetime
from dotenv import load_dotenv

from core.tokens import count_tokens
from core.llm import get_client, print_stats, chat_complete

# --- Setup ---
# Load environment variables from a .env file for security
load_dotenv()

# Initialize the Ollama client (no raw HTTP; library only).
client = get_client()

# Get context window size from environment variable, with a default.
SEMANTIC_SPLITTER_CONTEXT_WINDOW = 8192
MAX_TOKENS_PER_CHUNK = 1024

# For durable JSONL segments (same delimiter convention as deep_search.py)
RECORD_DELIM = "\x1e"

# --- Helper Functions ---

def get_bounded_anchor_preview(anchor: str, maxlen: int = 140) -> str:
    """
    Returns the exact bounded preview string that print_bounded_anchor prints,
    so we can feed it into the LLM conversation without altering the logs.
    """
    s = anchor or ""
    preview = (
        s.replace("\\", "\\\\")
         .replace("\n", r"\n")
         .replace("\r", r"\r")
         .replace("\t", r"\t")
    )
    if len(preview) > maxlen:
        half = maxlen // 2
        preview = f"{preview[:half]} … {preview[-half:]}"
    return f"[BEGIN_ANCHOR len={len(s)}]{preview}[END_ANCHOR]"

# For debug only
def print_bounded_anchor(anchor: str, maxlen: int = 140) -> None:
    """
    Prints a compact, single-line, clearly bounded preview of an anchor string.
    Escapes control characters and trims while keeping both ends visible.
    """
    print(get_bounded_anchor_preview(anchor, maxlen))

def custom_span_tokenize(text: str) -> list[tuple[int, int]]:
    """
    A custom implementation that mimics NLTK's span_tokenize but uses
    a simpler, deterministic approach based on characters.

    It first attempts to split the text by newline characters ('\n').
    If this results in fewer than 3 spans, it falls back to splitting
    by any whitespace character (' ', '\t', '\n').

    Args:
        text: The input string to split.

    Returns:
        A list of (start, end) tuples indicating the spans of the segments.
        The segments themselves do not include the delimiter.
    """
    # First, try splitting by newlines.
    spans = []
    start = 0
    delimiters = ['\n', '\r']
    for i, char in enumerate(text):
        if char in delimiters:
            if i > start:  # Ensure we don't create empty spans from consecutive delimiters
                spans.append((start, i))
            start = i + 1
    if start < len(text):  # Add the final segment after the last delimiter
        spans.append((start, len(text)))

    # If we got too few splits (e.g., a single long line of text),
    # fall back to splitting by any whitespace.
    if len(spans) < 3:
        spans = []  # Reset for the new strategy
        start = 0
        delimiters = [' ', '\t', '\n', '\r']
        for i, char in enumerate(text):
            if char in delimiters:
                if i > start:  # Ensure we don't create empty spans
                    spans.append((start, i))
                start = i + 1
        if start < len(text):  # Add the final segment
            spans.append((start, len(text)))

    return spans

def truncate_text_to_window(text: str, window_size: int) -> str:
    """
    Truncates a single string to fit within the window_size by removing characters
    from both ends, preserving the center of the string. This is a last resort
    if a single sentence or line is too long.

    Args:
        text: The text to be truncated.
        window_size: The maximum number of tokens for the output text.

    Returns:
        The truncated, centered text.
    """
    if count_tokens(text) <= window_size:
        return text

    # Binary search for the number of characters to keep in the middle
    low = 0
    # The number of characters to keep can't exceed the total length
    high = len(text)
    best_text = ""

    while low <= high:
        k = (low + high) // 2 # k is the number of chars to keep in the middle

        # To keep the center, we calculate start and end points
        text_len = len(text)
        start = (text_len - k) // 2
        end = start + k
        candidate_text = text[start:end]

        if count_tokens(candidate_text) <= window_size:
            # This is a valid candidate, see if we can make it larger
            best_text = candidate_text
            low = k + 1
        else:
            # This candidate is too large, we must make it smaller
            high = k - 1

    return best_text


def create_llm_window_from_center(text: str, window_size: int) -> str:
    """
    Extracts a window of text from the center of the input using the custom span tokenizer.
    This preserves the original text, including whitespace. It uses a binary search
    to find the largest central chunk of text that fits the token limit.

    Args:
        text: The text to be windowed.
        window_size: The maximum number of tokens for the window.

    Returns:
        The central text window, sliced directly from the original text.
    """
    if count_tokens(text) <= window_size:
        return text

    # Use the custom span_tokenize to get segment start/end indices.
    sentence_spans = custom_span_tokenize(text)
    num_sentences = len(sentence_spans)

    if num_sentences <= 1:
        # If there's only one segment, and it's too long, we must truncate it.
        return truncate_text_to_window(text, window_size)

    # Find the middle segment(s) which will be the anchor for our window
    center_start_idx = (num_sentences - 1) // 2
    center_end_idx = num_sentences // 2

    # Binary search for the optimal number of segments (k) to expand on each side of the center
    low = 0
    high = num_sentences // 2

    # Start with the center segment(s) as the best guess
    start_char = sentence_spans[center_start_idx][0]
    end_char = sentence_spans[center_end_idx][1]
    best_window_text = text[start_char:end_char]

    # Check if even the center segment(s) are too large.
    if count_tokens(best_window_text) > window_size:
        # If the combined center segments are too big, try just the single middle one.
        single_middle_span = sentence_spans[num_sentences // 2]
        single_middle_sentence = text[single_middle_span[0]:single_middle_span[1]]
        if count_tokens(single_middle_sentence) > window_size:
            # If even the single middle segment is too big, it must be truncated.
            return truncate_text_to_window(single_middle_sentence, window_size)
        else:
            return single_middle_sentence

    while low <= high:
        k = (low + high) // 2
        start_idx = max(0, center_start_idx - k)
        end_idx = min(num_sentences - 1, center_end_idx + k)

        # Get the text from the original string using the calculated spans
        start_char = sentence_spans[start_idx][0]
        end_char = sentence_spans[end_idx][1]
        candidate_text = text[start_char:end_char]

        if count_tokens(candidate_text) <= window_size:
            # This window is valid; store it and try to expand it further
            best_window_text = candidate_text
            low = k + 1
        else:
            # This window is too large; we need to try a smaller one
            high = k - 1

    return best_window_text

def fallback_split_by_delimiter(text: str) -> int:
    """
    Finds a split point in the text that is closest to the middle using the custom span tokenizer.

    Args:
        text: The text to be split.

    Returns:
        The index at which to split the text.
    """
    sentence_spans = custom_span_tokenize(text)

    best_split_point = -1

    if len(sentence_spans) <= 1:
        # If there's one or no segments, we can't split by this method.
        pass
    else:
        target_length = len(text) // 2
        min_distance = float('inf')

        # Find the segment end that's closest to the middle of the text
        for start, end in sentence_spans:
            # We don't want to split at the very end of the text, so we skip the last boundary
            if end == len(text):
                continue

            distance = abs(end - target_length)
            if distance < min_distance:
                min_distance = distance
                best_split_point = end

    if best_split_point != -1:
        percentage = (best_split_point / len(text)) * 100 if len(text) > 0 else 0
        print(f"Using fallback: splitting text by delimiter boundary. Split {len(text)} chars at {best_split_point} ({percentage:.1f}%).")
    else:
        print("Using fallback: splitting text by delimiter, but no suitable split point was found.")

    # Note: Might return -1
    return best_split_point

def load_files_from_list_file(list_file_path: str) -> list[str]:
    """
    Load file paths from a list file, one path per line.
    Empty/whitespace-only lines are ignored.
    """
    try:
        with open(list_file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file list '{list_file_path}' was not found")

    paths: list[str] = []
    for line in lines:
        p = line.strip()
        if not p:
            continue
        paths.append(p)
    return paths


# -------------------- Progress JSONL utilities --------------------

def write_json_record_append(filepath: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write(RECORD_DELIM)
        f.write("\n")
        f.flush()                 # flush Python buffers
        os.fsync(f.fileno())      # flush OS buffers (works on Windows & Linux)

def read_progress_file(filepath: str) -> tuple[dict | None, list[int]]:
    """
    Robustly parse the per-file cuts JSONL (delimiter-separated).
    Returns (meta, cuts). Skips malformed trailing fragments.
    """
    meta = None
    cuts: list[int] = []
    if not os.path.exists(filepath):
        return (None, cuts)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            blob = f.read()
    except Exception as e:
        print(f"Warning: Could not read progress file '{os.path.basename(filepath)}': {e}")
        return (None, cuts)

    for seg in blob.split(RECORD_DELIM):
        seg = seg.strip()
        if not seg:
            continue
        try:
            rec = json.loads(seg)
        except json.JSONDecodeError:
            # Most likely an interrupted last line; skip silently.
            continue
        t = rec.get("type")
        if t == "meta" and meta is None:
            meta = rec
        elif t == "cut":
            pos = rec.get("pos")
            if isinstance(pos, int):
                cuts.append(pos)
        else:
            # ignore unknown types; keep the format forward-compatible
            pass
    return (meta, cuts)

def ensure_meta(progress_path: str, file_index: int, path: str, size_chars: int) -> dict:
    """
    If progress file exists: return and validate its meta.
    Else: create meta (no manifest/hashes).
    """
    existing_meta, _ = read_progress_file(progress_path)
    if existing_meta is None:
        meta = {
            "type": "meta",
            "file_index": file_index,
            "path": path,
            "size_chars": size_chars,
            "max_tokens": MAX_TOKENS_PER_CHUNK,
            "context_window": SEMANTIC_SPLITTER_CONTEXT_WINDOW,
        }
        write_json_record_append(progress_path, meta)
        return meta

    # Validate minimal invariants to keep resume foolproof.
    errs = []
    if existing_meta.get("file_index") != file_index:
        errs.append(f"- file_index mismatch: progress={existing_meta.get('file_index')} vs expected={file_index}")
    if existing_meta.get("path") != path:
        errs.append(f"- path mismatch: progress='{existing_meta.get('path')}' vs expected='{path}'")
    if existing_meta.get("size_chars") != size_chars:
        errs.append(f"- size_chars mismatch: progress={existing_meta.get('size_chars')} vs current={size_chars}")
    if existing_meta.get("max_tokens") != MAX_TOKENS_PER_CHUNK:
        errs.append(f"- max_tokens mismatch: progress={existing_meta.get('max_tokens')} vs current={MAX_TOKENS_PER_CHUNK}")
    if existing_meta.get("context_window") != SEMANTIC_SPLITTER_CONTEXT_WINDOW:
        errs.append(f"- context_window mismatch: progress={existing_meta.get('context_window')} vs current={SEMANTIC_SPLITTER_CONTEXT_WINDOW}")

    if errs:
        msg = "Refusing to resume due to incompatible progress file:\n" + "\n".join(errs) + \
              f"\nProgress file: {progress_path}\nTip: move/delete this file or re-run with a stable file list."
        raise RuntimeError(msg)

    return existing_meta

def build_boundaries(text_len: int, cuts: list[int]) -> list[int]:
    uniq = sorted({p for p in cuts if isinstance(p, int) and 0 < p < text_len})
    return [0] + uniq + [text_len]

def first_oversized_segment(text: str, boundaries: list[int]) -> tuple[int, int] | None:
    for i in range(len(boundaries)-1):
        lo, hi = boundaries[i], boundaries[i+1]
        if count_tokens(text[lo:hi]) > MAX_TOKENS_PER_CHUNK:
            return (lo, hi)
    return None

# -------------------- One-split proposer (segment-scoped) --------------------

def propose_split_index_for_segment(full_text: str, seg_lo: int, seg_hi: int, filename: str) -> tuple[int, bool]:
    """
    Run the same LLM cascade as your recursive splitter, but restricted to [seg_lo:seg_hi].
    Returns (absolute split index in the full_text, used_fallback_flag). If no valid split
    can be obtained from the LLM, falls back to algorithmic methods and marks used_fallback=True.
    """
    segment = full_text[seg_lo:seg_hi]
    # If the segment is larger than the window, use a central window for the LLM.
    seg_token_count = count_tokens(segment)
    if seg_token_count > SEMANTIC_SPLITTER_CONTEXT_WINDOW:
        print(f"Text token count ({seg_token_count}) exceeds window size ({SEMANTIC_SPLITTER_CONTEXT_WINDOW}). Creating a central window for the LLM.")
        section_text = create_llm_window_from_center(segment, SEMANTIC_SPLITTER_CONTEXT_WINDOW)
    else:
        section_text = segment

    prompt = f"""I will provide a large block of text.
The task is: split the large block of text into exactly 2 blocks of text.
The purpose is choosing a starting position for the second section that makes sense- **structurally**.

Output JSON in the format:
{{ "begin_second_section": "text\ngoes here. Always ends with double quote (as JSON requires)" }}

Output only enough text in the JSON field to be uniquely identifiable (3-5 words).
Output the text **exactly** as it appears in the raw input to allow for a naive str.find() approach to work.

Full text:
```{filename}
{section_text}
```"""

    split_index = -1
    used_fallback = False  # Track if algorithmic fallback was used
    max_chat_attempts = 6
    attempts_per_chat = 3
    max_retries = attempts_per_chat * max_chat_attempts

    for attempt_chat_idx in range(max_chat_attempts):
        messages = [
            {"role": "system", "content": "Adhere to the instructions as they are written, respond only in JSON."},
            {"role": "user", "content": prompt},
        ]
        for attempt_in_chat in range(attempts_per_chat):
            already_in_plea1 = len(messages) > 2
            # Flag to apply angry prompt on next iteration of the loop
            be_draconian = already_in_plea1
            attempt_idx = attempt_chat_idx * attempts_per_chat + attempt_in_chat
            try:
                response = chat_complete(
                    messages=messages,
                    role="hybrid",
                    client=client,
                    max_completion_tokens=256,
                    please_no_thinking=attempt_chat_idx < 2,
                    require_json=True
                )
                print("", end='\n')
                stats = print_stats(response)
                if stats:
                    print(stats)
                response_text = response.message.content
                thinking_length = len(response.message.thinking) if response.message.thinking else 0
                if response.ran_out_of_tokens and thinking_length > 10:
                    print(f"Run away. The data is poisonous. Our poor model is going into infinite generations. seg_lo: {seg_lo}, seg_hi: {seg_hi}")
                    attempt_chat_idx = max_chat_attempts
                    break

                json_start = response_text.find('{')
                if json_start != -1:
                    response_text = response_text[json_start:]
                json_end = response_text.rfind('}')
                if json_end != -1:
                    response_text = response_text[:json_end + 1]

                split_data = json.loads(response_text)
                split_string = split_data.get("begin_second_section")

                if split_string:
                    # Search INSIDE THE SEGMENT ONLY
                    found_rel = segment.find(split_string)
                    if found_rel != -1:
                        text_len = len(segment)
                        # Reject gross imbalance near edges of THIS segment.
                        if text_len > 0 and (found_rel < text_len * 0.02 or found_rel > text_len * 0.98):
                            percentage = (found_rel / text_len) * 100
                            warn_line = f"Warning (Attempt {attempt_idx + 1}): LLM proposed a highly imbalanced split within segment ({percentage:.2f}%) of {text_len} chars. Discarding."
                            print(warn_line)
                            plea1 = "Fix it by changing `begin_second_section` to somewhere in the **middle** of the text"
                            plea2 = """DRACONIAN OVERRIDE:
- Your last split was grossly imbalanced. Choose `begin_second_section` whose FIRST CHARACTER INDEX lies strictly within [45%, 55%] of the provided text length. Anything outside this band is invalid.
- Prefer a natural boundary at the nearest delimiter (newline, sentence end, or whitespace), but DO NOT move outside [45%, 55%].
- The JSON value must be an exact, verbatim substring from the provided text, consisting of 3-5 words that uniquely identify the location. If your phrase is not unique, change it until it is unique (still 3-5 words).
- Preserve precise origin language and specific unicode letters used.
- If uncertain, take the 3-5 word phrase starting EXACTLY at the midpoint character of the provided text and expand rightward until uniqueness is achieved (max 5 words).
Return the JSON now."""
                            plea = plea2 if be_draconian else plea1
                            messages.append({"role": "user", "content": f"{warn_line}\n{plea}"})
                        else:
                            abs_index = seg_lo + found_rel
                            percentage = (found_rel / text_len) * 100 if text_len > 0 else 0
                            print(f"LLM identified a valid split point within segment. Split {text_len} chars at {found_rel} ({percentage:.1f}%).")
                            split_index = abs_index
                            break
                    else:
                        warn_line = f"Warning (Attempt {attempt_idx + 1}): LLM-suggested string not found in segment."
                        print(warn_line)
                        print_bounded_anchor(split_string)
                        bounded = get_bounded_anchor_preview(split_string)
                        plea1 = "Fix it by changing `begin_second_section` to a short string that can be found and searched easily"
                        plea2 = """PRECISION MODE (DRACONIAN):
- Set "begin_second_section" to a SHORT, SINGLE-LINE, EXACT substring (3-5 words) from the text.
- Copy characters exactly: space(s), punctuation, case, diacritics, dashes. No normalization or spelling fixes.
- No newlines/tabs. Prefer a snippet without JSON escapes (\", \\\\, \\n, \\t); if present, pick a nearby clean snippet.
- Must be UNIQUE in the provided text; extend rightward (max 5 words) or shift slightly until unique.
- No leading/trailing spaces; don't wrap in quotes/backticks unless those exact characters exist in the source.
- Preserve original text artifacts such as double spaces or possible spelling mistakes
- Sanity check: segment.find(snippet) >= 0; first and last 3 chars match the source.
- Output ONLY JSON with the single key \"begin_second_section\". Nothing else."""
                        plea = plea2 if be_draconian else plea1
                        messages.append({"role": "user", "content": f"{warn_line}\n{bounded}\n{plea}"})
                else:
                    print(f"Warning (Attempt {attempt_idx + 1}): LLM response did not contain 'begin_second_section'.")
            except (json.JSONDecodeError, AttributeError, Exception) as e:
                print(f"Warning (Attempt {attempt_idx + 1}): An API or JSON parsing error occurred: {e}.")

            if split_index != -1:
                break
            if attempt_idx < max_retries:
                print("Retrying LLM call...")
        else:
            # inner didn't break → (possibly) keep outer going
            if split_index == -1:
                continue
        break # inner did break → break outer too

    # Fallbacks restricted to segment:
    if split_index == -1:
        print("LLM splitting failed after all retries.")
        rel = fallback_split_by_delimiter(segment)
        if rel == -1:
            print("Delimiter splitting failed. Reverting to naive middle split.")
            rel = len(segment) // 2
        split_index = seg_lo + rel
        used_fallback = True

    return (split_index, used_fallback)

# -------------------- Driver that uses per-file cut indexes --------------------

def process_file_with_cuts(file_index: int, input_filename: str, progress_dir: str) -> bool:
    """
    Happy path behavior:
      - Starting anew: creates progress file with meta, then iteratively appends cuts.
      - In-progress: replays cuts, continues from leftmost oversized segment.
      - Already done: returns immediately.

    Returns True if the file is complete (all segments <= MAX), else False.
    """
    progress_path = os.path.join(progress_dir, f"{str(file_index).zfill(6)}.cuts.jsonl")

    try:
        with open(input_filename, "r", encoding="utf-8", errors='ignore') as fr:
            file_contents = fr.read()
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return True  # treat as "nothing to do" to let the loop continue

    meta = ensure_meta(progress_path, file_index, input_filename, len(file_contents))

    # Replay existing progress
    _, cuts = read_progress_file(progress_path)

    while True:
        boundaries = build_boundaries(len(file_contents), cuts)
        seg = first_oversized_segment(file_contents, boundaries)
        if seg is None:
            # file done
            return True

        lo, hi = seg
        # Propose a split inside [lo:hi)
        split_abs, used_fallback = propose_split_index_for_segment(file_contents, lo, hi, input_filename)
        # Guard against duplicates and near-edges
        if split_abs <= lo or split_abs >= hi:
            print(f"Warning: Proposed split {split_abs} outside segment ({lo},{hi}). Forcing midpoint.")
            split_abs = lo + (hi - lo) // 2
            used_fallback = True
        if split_abs in cuts:
            # Very rare, but keep loop progressing
            print("Info: Proposed split already exists; nudging by +1.")
            split_abs = min(hi-1, split_abs + 1)
            if split_abs in cuts or split_abs <= lo or split_abs >= hi:
                # fallback hard midpoint if still invalid
                split_abs = lo + (hi - lo) // 2
                used_fallback = True

        # Append the new cut (durable)
        write_json_record_append(progress_path, {
            "type": "cut",
            "pos": int(split_abs),
            # Mark if this position came from an algorithmic fallback
            "algorithmic_fallback": bool(used_fallback),
        })
        cuts.append(int(split_abs))
        # loop continues to re-evaluate from the leftmost oversized segment


def all_files_complete(file_paths: list[str], progress_dir: str) -> bool:
    for idx, path in enumerate(file_paths, start=1):
        p = os.path.join(progress_dir, f"{str(idx).zfill(6)}.cuts.jsonl")
        try:
            with open(path, "r", encoding="utf-8", errors='ignore') as fr:
                text = fr.read()
        except FileNotFoundError:
            return False
        _, cuts = read_progress_file(p)
        boundaries = build_boundaries(len(text), cuts)
        if first_oversized_segment(text, boundaries) is not None:
            return False
    return True


def finalize_chunks(file_paths: list[str], output_dir: str, progress_dir: str) -> None:
    """
    Finalization pass: derive segments from cuts and materialize into ./split/chunks/
    with global numbering. Uses temp+rename for each file.

    Each chunk file starts with a single-line header, e.g.:
    [Chunk 7/52 | File 2/5 | Source: /path/to/file.txt | chars:12345-16789 | tokens:987]
    """
    chunks_dir = os.path.join(output_dir, "chunks")
    tmp_dir = chunks_dir + ".tmp"
    if os.path.exists(chunks_dir):
        raise FileExistsError(f"Output chunks directory already exists: {chunks_dir}. Refusing to overwrite.")
    os.makedirs(tmp_dir, exist_ok=True)

    global_index = 0

    for file_idx, path in enumerate(file_paths, start=1):
        progress_path = os.path.join(progress_dir, f"{str(file_idx).zfill(6)}.cuts.jsonl")
        with open(path, "r", encoding="utf-8", errors='ignore') as fr:
            text = fr.read()
        _, cuts = read_progress_file(progress_path)
        boundaries = build_boundaries(len(text), cuts)

        # Sanity check: ensure all segments are compliant
        for i in range(len(boundaries)-1):
            lo, hi = boundaries[i], boundaries[i+1]
            if count_tokens(text[lo:hi]) > MAX_TOKENS_PER_CHUNK:
                raise RuntimeError(f"Finalization aborted: file {file_idx} still has an oversized segment ({lo},{hi}).")

        # Per-file counters
        total_in_file = len(boundaries) - 1
        local_index = 0

        # Emit chunks in order
        for i in range(len(boundaries)-1):
            lo, hi = boundaries[i], boundaries[i+1]
            local_index += 1
            global_index += 1

            # Compute tokens first so the header can include it
            content = text[lo:hi]
            tok = count_tokens(content)

            out_tmp = os.path.join(tmp_dir, f"{str(global_index).zfill(6)}.txt.tmp")
            out_final = os.path.join(tmp_dir, f"{str(global_index).zfill(6)}.txt")

            # Single-line, nicely formatted header
            header_line = (
                f"[Chunk {local_index}/{total_in_file} | "
                f"Source: {path} | "
                f"chars:{lo}-{hi} | "
                f"tokens:{tok}]\n"
            )

            with open(out_tmp, "w", encoding="utf-8") as fw:
                fw.write(header_line)
                fw.write(content)
            os.replace(out_tmp, out_final)

            # Mirror the same clean summary in the console
            print(
                f"Saved {header_line.strip()} → '{out_final}'"
            )

    # Atomically move tmp → final location
    os.replace(tmp_dir, chunks_dir)
    
    print(f"--- All chunks saved successfully in the '{chunks_dir}/' directory. ---")


if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Split one or more large text files into smaller, semantically coherent chunks."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--files",
        type=str,
        nargs='+',  # Accept one or more file arguments
        help="Path(s) to the input text file(s) to be split."
    )
    input_group.add_argument(
        "--files-list",
        type=str,
        help="Path to a text file containing the list of input file paths (one path per line)."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="split",
        help="The directory to save the output chunks. Defaults to 'split'."
    )
    args = parser.parse_args()

    if args.files_list:
        file_paths = load_files_from_list_file(args.files_list)
    else:
        file_paths = args.files or []

    if not file_paths:
        raise ValueError(
            "Error: No input files specified. Please provide at least one file using --files or --files-list."
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    progress_dir = os.path.join(output_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)

    # Happy path #1: Starting anew (no progress files yet) → meta lines created lazily.

    # Happy path #2/#3 + edge cases:
    # - If progress files exist and are "done" (all segments small enough) → finalize.
    # - If progress files exist and are in-progress → resume from the leftmost oversized segment.
    # - If progress files contain an invalid trailing record → ignored during replay.

    # Additional guard: refuse if there are stale extra progress files not matching CLI length.
    progress_files = sorted([f for f in os.listdir(progress_dir) if f.endswith(".cuts.jsonl")])
    max_allowed = len(file_paths)
    for fname in progress_files:
        try:
            idx = int(os.path.splitext(os.path.splitext(fname)[0])[0])
        except Exception:
            raise RuntimeError(f"Unexpected file in progress dir: {fname}")
        if idx < 1 or idx > max_allowed:
            raise RuntimeError(
                f"Progress file '{fname}' has index {idx} which is outside the current file list (1..{max_allowed}).\n"
                f"Refusing to proceed to avoid mismatched resumes. Move/delete the stale file to continue."
            )

    # Check if all complete already
    if all_files_complete(file_paths, progress_dir):
        print("--- Progress indicates all files are already split. Finalizing into chunks... ---")
        finalize_chunks(file_paths, output_dir, progress_dir)
        raise SystemExit(0)

    # --- File Processing Loop (sequential, resumable) ---
    for file_idx, input_filename in enumerate(file_paths, start=1):
        print(f"--- Splitting file: {input_filename} ---")
        try:
            with open(input_filename, "r", encoding="utf-8", errors='ignore') as fr:
                file_contents = fr.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{input_filename}' was not found")

        print(f"Original token count: {count_tokens(file_contents)}")
        print(f"Max tokens per chunk: {MAX_TOKENS_PER_CHUNK}\n")

        done = process_file_with_cuts(file_idx, input_filename, progress_dir)
        if done:
            print(f"--- File '{input_filename}' is complete. ---\n")
        else:
            print(f"--- File '{input_filename}' in progress. ---\n")

    # --- Final Summary / Finalization ---
    if all_files_complete(file_paths, progress_dir):
        print(f"--- Completed: All {len(file_paths)} file(s) have compliant segments. Materializing chunks... ---")
        finalize_chunks(file_paths, output_dir, progress_dir)
    else:
        print("\n--- Progress saved. Resume later to continue splitting. ---")
