import os
import json
import argparse
from dotenv import load_dotenv

from core.tokens import count_tokens
from core.llm import get_client, print_stats, chat_complete

# --- Setup ---
# Load environment variables from a .env file for security
load_dotenv()

# Initialize the Ollama client (no raw HTTP; library only).
client = get_client()

# Get context window size from environment variable, with a default.
CONTEXT_WINDOW_SIZE_TOKENS = int(os.environ.get("CONTEXT_WINDOW_SIZE_TOKENS", 8192))
MAX_TOKENS_PER_CHUNK = 1024

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


# --- Core Recursive Function ---

def semantic_split(
    text: str,
    filename: str,
    chunk_max_size_tokens: int = MAX_TOKENS_PER_CHUNK,
    window_size: int = CONTEXT_WINDOW_SIZE_TOKENS,
) -> list[str]:
    """
    Recursively splits a text into semantically coherent chunks based on LLM suggestions.
    Retries the LLM call before using a delimiter-based fallback.

    Args:
        text: The block of text to be split.
        filename: The name of the file the text belongs to (for use in the prompt).
        chunk_max_size_tokens: The target maximum token size for a chunk.
        window_size: Max tokens to expose to the LLM from the center of the text.

    Returns:
        A list of text chunks, each smaller than the token limit.
    """
    # Base Case: If the text is already within the desired token limit, return it as a single chunk.
    if count_tokens(text) <= chunk_max_size_tokens:
        return [text]

    # --- Recursive Step ---

    # If the text is larger than the window size, create a smaller, central window
    # of the text to pass to the LLM. This is only for the LLM prompt.
    text_token_count = count_tokens(text)
    if text_token_count > window_size:
        print(f"Text token count ({text_token_count}) exceeds window size ({window_size}). Creating a central window for the LLM.")
        section_text = create_llm_window_from_center(text, window_size)
    else:
        section_text = text

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
    max_groups = 3
    attempts_per_group = 3
    max_retries = attempts_per_group * max_groups

    # We will run the groups. Each group starts a fresh conversation
    # (system + initial user prompt). Within a group, we append feedback
    # messages containing the exact warning lines that are already printed.
    for group_idx in range(max_groups):
        messages = [
            {"role": "system", "content": "Adhere to the instructions as they are written, respond only in JSON."},
            {"role": "user", "content": prompt},
        ]

        for attempt_in_group in range(attempts_per_group):
            attempt_idx = group_idx * attempts_per_group + attempt_in_group
            try:
                response = chat_complete(messages=messages, role="splitter", client=client, require_json=True)
                print("", end='\n')
                stats = print_stats(response)
                if stats:
                    print(stats)
                response_text = response.message.content

                json_start = response_text.find('{')
                if json_start != -1:
                    response_text = response_text[json_start:]
                json_end = response_text.rfind('}')
                if json_end != -1:
                    response_text = response_text[:json_end + 1]

                split_data = json.loads(response_text)
                split_string = split_data.get("begin_second_section")

                if split_string:
                    # IMPORTANT: We search for the returned string in the original, full-length text,
                    # not in the potentially smaller 'section_text' window.
                    found_index = text.find(split_string)
                    if found_index != -1:
                        text_len = len(text)
                        # Check for highly imbalanced splits. If one section has less than
                        # 2% of the text, discard the LLM's suggestion.
                        if text_len > 0 and (found_index < text_len * 0.02 or found_index > text_len * 0.98):
                            percentage = (found_index / text_len) * 100
                            warn_line = f"Warning (Attempt {attempt_idx + 1}): LLM proposed a highly imbalanced split ({percentage:.2f}%) of {text_len} chars. Discarding."
                            print(warn_line)
                            plea = "Fix it by changing `begin_second_section` to somewhere in the **middle** of the text"
                            # Feed the *exact* warning line back as a new user turn
                            messages.append({"role": "user", "content": f"{warn_line}\n{plea}"})
                            # Continue to next attempt (within same group until triplet is exhausted)
                            pass
                        else:
                            split_index = found_index
                            percentage = (split_index / text_len) * 100 if text_len > 0 else 0
                            print(f"LLM identified a valid split point. Split {text_len} chars at {split_index} ({percentage:.1f}%).")
                            break  # Success, exit the inner loop (and later the outer loop)
                    else:
                        warn_line = f"Warning (Attempt {attempt_idx + 1}): LLM-suggested string not found in text."
                        print(warn_line)
                        # Keep the console output identical:
                        print_bounded_anchor(split_string)
                        # Feed both the warning and the exact bounded-anchor text back as the next user turn
                        bounded = get_bounded_anchor_preview(split_string)
                        plea = "Fix it by changing `begin_second_section` to a short string that can be found and searched easily"
                        messages.append({"role": "user", "content": f"{warn_line}\n{bounded}\n{plea}"})
                else:
                    print(f"Warning (Attempt {attempt_idx + 1}): LLM response did not contain 'begin_second_section'.")

            except (json.JSONDecodeError, AttributeError, Exception) as e:
                print(f"Warning (Attempt {attempt_idx + 1}): An API or JSON parsing error occurred: {e}.")

            # If succeeded, break out to avoid printing "Retrying..."
            if split_index != -1:
                break

            if attempt_idx < max_retries:
                print("Retrying LLM call...")

        # If succeeded inside this group, break out of outer loop too
        if split_index != -1:
            break
        # Otherwise, group resets implicitly by reinitializing `messages` on next loop iteration.

    # 2. Fallback: If LLM fails after all retries, use a delimiter-based split.
    if split_index == -1:
        print("LLM splitting failed after all retries.")
        split_index = fallback_split_by_delimiter(text)
        # Final safety net if delimiter splitting also fails
        if split_index == -1:
            print("Delimiter splitting failed. Reverting to naive middle split.")
            split_index = len(text) // 2

    # 3. Perform the split and recurse on both halves.
    part1 = text[:split_index]
    part2 = text[split_index:]

    # The first part always touches the original chunk's start.
    # The second part always touches the original chunk's end.
    results1 = semantic_split(part1, filename, chunk_max_size_tokens)
    results2 = semantic_split(part2, filename, chunk_max_size_tokens)

    return results1 + results2

if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Split one or more large text files into smaller, semantically coherent chunks."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        nargs='+',  # Accept one or more file arguments
        help="Path(s) to the input text file(s) to be split."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="split",
        help="The directory to save the output chunks. Defaults to 'split'. The script will exit if this directory already exists."
    )
    args = parser.parse_args()

    if not args.file:
        raise ValueError(
            "Error: No input files specified. Please provide at least one file using the --file argument."
        )

    output_dir = args.output_dir

    # Check if the output directory already exists.
    if os.path.exists(output_dir):
        # Raising an error is more Pythonic and provides a clearer traceback.
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists. "
            "Please specify a new directory with --output-dir or remove the existing one."
        )

    # Create the output directory
    os.makedirs(output_dir)

    global_chunk_counter = 0

    # --- File Processing Loop ---
    for input_filename in args.file:
        try:
            with open(input_filename, "r", encoding="utf-8", errors='ignore') as fr:
                file_contents = fr.read()
        except FileNotFoundError:
            print(f"Error: The file '{input_filename}' was not found. Skipping.")
            continue  # Skip to the next file in the list

        print(f"--- Splitting file: {input_filename} ---")
        print(f"Original token count: {count_tokens(file_contents)}")
        print(f"Max tokens per chunk: {MAX_TOKENS_PER_CHUNK}\n")

        # Call the main function to start the splitting process for the current file.
        chunks = semantic_split(text=file_contents, filename=input_filename)

        num_chunks_for_this_file = len(chunks)
        if num_chunks_for_this_file > 0:
            print(f"--- File '{input_filename}' split into {num_chunks_for_this_file} chunks. Saving... ---\n")

        # --- Saving Output Chunks for the current file ---
        for i, chunk in enumerate(chunks):
            global_chunk_counter += 1
            chunk_num_for_this_file = i + 1

            # Format the filename with a global, incrementing counter
            output_filename = os.path.join(output_dir, f"{str(global_chunk_counter).zfill(6)}.txt")

            # Create the header with the file-specific chunk count
            header = f"SOURCE {input_filename} (Chunk {chunk_num_for_this_file}/{num_chunks_for_this_file})\n"

            with open(output_filename, "w", encoding="utf-8") as fw:
                # Prepend the new header
                fw.write(header)
                fw.write(chunk)

            chunk_token_count = count_tokens(chunk)
            print(f"Saved chunk {chunk_num_for_this_file}/{num_chunks_for_this_file} to '{output_filename}' (from '{input_filename}', Tokens: {chunk_token_count})")

        # Add a newline for better visual separation between file processing in the console
        if num_chunks_for_this_file > 0:
            print("\n")

    # --- Final Summary ---
    if global_chunk_counter == 0:
        print("\n--- No chunks were generated. ---")
    else:
        print(f"--- Completed: Split {len(args.file)} file(s) into a total of {global_chunk_counter} chunks. ---")
        print(f"--- All chunks saved successfully in the '{output_dir}/' directory. ---")
