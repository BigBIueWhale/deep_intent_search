import os
import json
import glob
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# NOTE: Centralized utilities (as requested)
from dotenv import load_dotenv
from core.tokens import count_tokens  # centralized token counting (Qwen tokenizer)
from core.llm import get_client, get_model_name, get_ollama_options  # centralized LLM access

# --- Setup ---
# Load environment variables from a .env file for security.
load_dotenv()

# Initialize the Ollama client (no raw HTTP; library only).
CLIENT = get_client()

# Get context window size from environment variable, with a default.
# (This governs how much *surrounding text* we feed around the chunk of interest, not the model's num_ctx.)
CONTEXT_WINDOW_SIZE_TOKENS = int(os.environ.get("CONTEXT_WINDOW_SIZE_TOKENS", 8192))

# --- Data Structure ---

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
            chunks.append(Chunk(original_index=i, filename=filename, content=content, token_count=token_count))
        except IOError as e:
            print(f"Warning: Could not read file {filepath}: {e}")

    total_tokens = sum(c.token_count for c in chunks)
    print(f"Loaded {len(chunks)} chunks ({sum(c.token_count for c in chunks):,} tokens).")
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

def check_relevance_with_llm(
    context: str,
    chunk_of_interest: str,
    chunk_filename: str,
    query: str,
    max_retries: int = 5
) -> bool:
    if not CLIENT:
        print("LLM client not initialized. Assuming relevance as a fallback.")
        return True

    prompt = f"""
You are a highly focused research assistant. Your task is to determine if a specific, isolated section of text is relevant to a user's query.

I will provide you with a large context of surrounding text to help you understand the overall topic. However, your final judgment must be based ONLY on the content of the **"SECTION OF INTEREST"** provided at the very end. The context is for reasoning, but the decision must be about the specific section.

The user's query is: "{query}"

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

    options = get_ollama_options()
    model = get_model_name()

    for attempt in range(max_retries):
        try:
            # Use Ollama client, request JSON-formatted output on every request.
            response = CLIENT.chat(
                model=model,
                messages=messages,
                # Don't use format="json" because that disables thinking
                # format="json",
                options=options,
                stream=False,
                think=True,
            )
            response_text = response.message.content

            thinking_text = response.message.thinking
            thinking_tokens = count_tokens(thinking_text) if thinking_text else 0

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
                print(f"  -> Thinking tokens: {thinking_tokens:,}")
                return bool(relevance)
            else:
                print(f"  -> Warning (Attempt {attempt + 1}/{max_retries}): LLM response was malformed. Response: {response_text}")
                print(f"  -> Thinking tokens: {thinking_tokens:,}")
                # Nicely print the entire thinking output to aid debugging.
                if thinking_text:
                    print("\n" + "-" * 80)
                    print("THINKING (debug):")
                    print("-" * 80)
                    print(thinking_text)
                    print("-" * 80 + "\n")

        except Exception as e:
            print(f"  -> An error occurred on attempt {attempt + 1}/{max_retries}: {e}")

        time.sleep(2)

    print(f"  -> All {max_retries} retries failed. Assuming relevance for {chunk_filename}.")
    return True

def run_search_pass(
    all_chunks: List[Chunk],
    chunks_to_search: List[Chunk],
    query: str
) -> List[Chunk]:
    """
    Runs a single search pass, iterating through `chunks_to_search` and
    checking relevance, using `all_chunks` to build the context window.
    """
    relevant_chunks: List[Chunk] = []

    for i, chunk in enumerate(chunks_to_search):
        # The progress report now shows progress through the current search set.
        print(f"\nAnalyzing chunk {i + 1}/{len(chunks_to_search)} ('{chunk.filename}')...")

        # The context window is always built from the complete original set of chunks
        # using the chunk's stored original_index.
        context_window = get_dynamic_context_window(all_chunks, chunk.original_index, CONTEXT_WINDOW_SIZE_TOKENS)

        if check_relevance_with_llm(context_window, chunk.content, chunk.filename, query):
            relevant_chunks.append(chunk)

    return relevant_chunks

def display_results(relevant_chunks: List[Chunk]):
    """
    Prints the filenames of the relevant chunks to the console.
    """
    print("\n" + "="*80)
    print(f"Found {len(relevant_chunks)} relevant section(s):")
    print("="*80 + "\n")

    if not relevant_chunks:
        print("No relevant sections were found for your query.")
        return

    # Only print the filename and token count, not the content.
    for chunk in relevant_chunks:
        print(f"- {chunk.filename} (Tokens: {chunk.token_count})")

# --- Main Execution ---

def main():
    """
    Main function to parse arguments and orchestrate the iterative search process.
    """
    parser = argparse.ArgumentParser(
        description="Perform a deep, contextual search through text chunks using an LLM."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The initial search query."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="split",
        help="The directory containing the text chunks (default: 'split')."
    )
    args = parser.parse_args()

    # Load all chunks once at the start. This list will be used for context.
    all_chunks = load_chunks_from_disk(args.dir)
    if not all_chunks:
        print("No chunk files found or loaded. Exiting.")
        return

    # This list will be updated after each successful pass.
    active_results = all_chunks
    query = args.query
    pass_number = 1

    # Main refinement loop
    while True:
        print(f"\n--- Starting Deep Search: Pass {pass_number} ({len(active_results)} sections to search, Context window: {CONTEXT_WINDOW_SIZE_TOKENS} tokens) ---")

        # Always use `all_chunks` for context and `active_results` for the items to search.
        pass_results = run_search_pass(all_chunks, active_results, query)

        if not pass_results:
            print("\n" + "="*80)
            print("⚠️ Your query returned 0 results.")
            print("   The search set remains unchanged from the previous pass.")
            # Do not update active_results, allowing the next search to use the old set.
        else:
            # Update the active set to the new, narrower results.
            active_results = pass_results
            display_results(active_results)

        # Prompt for the next query
        print("\n" + "="*80)
        print("You can now enter a new query to search within these results.")
        print("Press Enter to exit.")
        print("="*80)

        pass_number += 1
        query = input(f"Refinement Query (Pass {pass_number}) > ").strip()

        if not query:
            break

    print("\n--- Deep Search Complete ---")


if __name__ == "__main__":
    if not CLIENT:
        print("Exiting due to initialization failure.")
    else:
        main()
