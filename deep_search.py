import os
import json
import glob
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- Setup ---
# Load environment variables from a .env file for security.
load_dotenv()

# Initialize the Google Generative AI client. Handle missing API key.
try:
    api_key = os.environ.get("GOOGLE_AISTUDIO_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_AISTUDIO_API_KEY environment variable not set.")
    # Create a client instance, as requested.
    CLIENT = genai.Client(api_key=api_key)
except (ValueError, ImportError) as e:
    print(f"Error initializing Google Generative AI Client: {e}")
    CLIENT = None

# Get context window size from environment variable, with a default.
CONTEXT_WINDOW_SIZE_TOKENS = int(os.environ.get("CONTEXT_WINDOW_SIZE_TOKENS", 8192))

# --- Data Structure ---

@dataclass
class Chunk:
    """
    A dataclass to hold the content of a text chunk, its filename,
    and its pre-calculated token count.
    """
    filename: str
    content: str
    token_count: int

# --- Helper Functions ---

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text string using the API.
    """
    if not CLIENT:
        # Fallback for when the client isn't available.
        return len(text) // 4
    try:
        # The 'models/' prefix is required for the count_tokens method.
        response = CLIENT.models.count_tokens(
            model='models/gemini-2.5-flash',
            contents=[text]
        )
        return response.total_tokens
    except Exception as e:
        print(f"Could not count tokens due to an API error: {e}. Falling back to an estimate.")
        return len(text) // 4

def load_chunks_from_disk(directory: str) -> List[Chunk]:
    """
    Loads all .txt files from a directory into a list of Chunk objects.
    Pre-calculates token counts for each chunk.
    """
    print("Loading and tokenizing all chunks from disk...")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return []

    chunk_files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    chunks: List[Chunk] = []
    
    for filepath in chunk_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            token_count = count_tokens(content)
            filename = os.path.basename(filepath)
            chunks.append(Chunk(filename=filename, content=content, token_count=token_count))
        except IOError as e:
            print(f"Warning: Could not read file {filepath}: {e}")
    
    print(f"Successfully loaded {len(chunks)} chunks into memory.")
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
    max_retries: int = 3
) -> bool:
    """
    Asks the LLM to determine if a chunk is relevant to the query.
    """
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

Now, focusing exclusively on the following text, is there anything in this specific section that is even **remotely** relevant to the user's query?

--- SECTION OF INTEREST ({chunk_filename}) ---
{chunk_of_interest}
--- END OF SECTION ---

Respond with a JSON object in the following format and nothing else:
{{
  "is_relevant": <true or false>
}}
"""

    for attempt in range(max_retries):
        try:
            response = CLIENT.models.generate_content(
                contents=prompt,
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                    response_mime_type='application/json',
                ),
            )
            
            response_text = response.text
            parsed_json = safe_json_loads(response_text)

            if parsed_json and "is_relevant" in parsed_json:
                relevance = parsed_json["is_relevant"]
                print(f"  -> LLM decision: {'Relevant' if relevance else 'Not Relevant'}")
                return bool(relevance)
            else:
                print(f"  -> Warning (Attempt {attempt + 1}/{max_retries}): LLM response was malformed. Response: {response_text}")

        except Exception as e:
            print(f"  -> An error occurred on attempt {attempt + 1}/{max_retries}: {e}")
        
        time.sleep(2)

    print(f"  -> All {max_retries} retries failed. Assuming relevance for {chunk_filename}.")
    return True

def run_search_pass(chunks_to_search: List[Chunk], query: str) -> List[Chunk]:
    """
    Runs a single pass of the search, iterating through chunks and checking relevance.
    """
    relevant_chunks: List[Chunk] = []
    total_chunks = len(chunks_to_search)

    for i, chunk in enumerate(chunks_to_search):
        print(f"\nAnalyzing chunk {i + 1}/{total_chunks} ('{chunk.filename}')...")
        
        context_window = get_dynamic_context_window(chunks_to_search, i, CONTEXT_WINDOW_SIZE_TOKENS)

        if check_relevance_with_llm(context_window, chunk.content, chunk.filename, query):
            relevant_chunks.append(chunk)

    return relevant_chunks

def display_results(relevant_chunks: List[Chunk]):
    """
    Prints the content of the relevant chunks to the console.
    """
    print("\n" + "="*80)
    print(f"Found {len(relevant_chunks)} relevant section(s):")
    print("="*80 + "\n")

    if not relevant_chunks:
        print("No relevant sections were found for your query.")
        return

    for chunk in relevant_chunks:
        print(f"--- BEGIN: {chunk.filename} (Tokens: {chunk.token_count}) ---")
        print(chunk.content)
        print(f"--- END: {chunk.filename} ---\n")

# --- Main Execution ---

def main():
    """
    Main function to parse arguments and orchestrate the search process.
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

    all_chunks = load_chunks_from_disk(args.dir)
    if not all_chunks:
        print("No chunk files found or loaded. Exiting.")
        return

    # --- Pass 1: Initial Broad Search ---
    print(f"\n--- Starting Deep Search: Pass 1 (Context window: {CONTEXT_WINDOW_SIZE_TOKENS} tokens) ---")
    first_pass_results = run_search_pass(all_chunks, args.query)

    if not first_pass_results:
        print("\n--- No relevant sections found in the first pass. ---")
        return

    # --- Display Intermediate Results ---
    display_results(first_pass_results)

    # --- Pass 2: Refinement Search ---
    print("\n" + "="*80)
    print("You can now enter a new query to search within these results.")
    print("Press Enter to exit without a second pass.")
    print("="*80)
    
    second_query = input("Refinement Query > ").strip()

    if not second_query:
        print("Exiting.")
        return

    print("\n--- Starting Deep Search: Pass 2 (Refinement) ---")
    second_pass_results = run_search_pass(first_pass_results, second_query)
    
    # --- Display Final Results ---
    display_results(second_pass_results)
    print("\n--- Deep Search Complete ---")


if __name__ == "__main__":
    if not CLIENT:
        print("Exiting due to initialization failure.")
    else:
        main()
