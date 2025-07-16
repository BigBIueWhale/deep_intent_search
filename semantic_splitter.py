import os
import json
import nltk
from google import genai
from google.genai import types
import argparse
from dotenv import load_dotenv

# --- Setup ---
# Load environment variables from a .env file for security
load_dotenv()

# Initialize the Google Generative AI client. Handle missing API key.
api_key = os.environ.get("GOOGLE_AISTUDIO_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_AISTUDIO_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

# Download and load NLTK's sentence tokenizer model.
try:
    # This tokenizer is used for its `span_tokenize` method to get sentence indices.
    punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
except LookupError:
    print("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt')
    punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print("NLTK 'punkt' sentence tokenizer loaded.")

# Get context window size from environment variable, with a default.
CONTEXT_WINDOW_SIZE_TOKENS = int(os.environ.get("CONTEXT_WINDOW_SIZE_TOKENS", 8192))
MAX_TOKENS_PER_CHUNK = 1024

# --- Helper Functions ---

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text string using the
    integrated tokenizer of the Google Generative AI SDK.

    Args:
        text: The text to be tokenized.

    Returns:
        The number of tokens in the text.
    """
    # The model name must match the one used for generation for consistent token counting.
    # The 'models/' prefix is required for the count_tokens method.
    try:
        response = client.models.count_tokens(
            model='models/gemini-2.5-flash',
            contents=[text]
        )
        return response.total_tokens
    except Exception as e:
        print(f"Could not count tokens due to an API error: {e}. Falling back to an estimate.")
        # Fallback to a character-based estimate if the API call fails.
        return len(text) // 4

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

    print("Warning: A single text segment is too large. Truncating it to fit the window.")
    
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
    Extracts a window of text from the center of the input using NLTK sentence spans.
    This preserves the original text, including whitespace. It uses a binary search
    to find the largest central chunk of sentences that fits the token limit.

    Args:
        text: The text to be windowed.
        window_size: The maximum number of tokens for the window.

    Returns:
        The central text window, sliced directly from the original text.
    """
    if count_tokens(text) <= window_size:
        return text

    # Use span_tokenize to get sentence start/end indices, preserving original text.
    sentence_spans = list(punkt_tokenizer.span_tokenize(text))
    num_sentences = len(sentence_spans)
    
    if num_sentences <= 1:
        # If there's only one sentence, and it's too long, we must truncate it.
        return truncate_text_to_window(text, window_size)

    # Find the middle sentence(s) which will be the anchor for our window
    center_start_idx = (num_sentences - 1) // 2
    center_end_idx = num_sentences // 2

    # Binary search for the optimal number of sentences (k) to expand on each side of the center
    low = 0
    high = num_sentences // 2
    
    # Start with the center sentence(s) as the best guess
    start_char = sentence_spans[center_start_idx][0]
    end_char = sentence_spans[center_end_idx][1]
    best_window_text = text[start_char:end_char]

    # Check if even the center sentence(s) are too large.
    if count_tokens(best_window_text) > window_size:
        # If the combined center sentences are too big, try just the single middle one.
        single_middle_span = sentence_spans[num_sentences // 2]
        single_middle_sentence = text[single_middle_span[0]:single_middle_span[1]]
        if count_tokens(single_middle_sentence) > window_size:
            # If even the single middle sentence is too big, it must be truncated.
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

def fallback_split_by_sentence(text: str) -> int:
    """
    Finds a split point in the text that is closest to the middle using NLTK sentence spans.
    This is much more robust than string searching.

    Args:
        text: The text to be split.

    Returns:
        The index at which to split the text.
    """
    print("Using fallback: splitting text by sentence boundary.")
    sentence_spans = list(punkt_tokenizer.span_tokenize(text))
    
    if len(sentence_spans) <= 1:
        # If there's one or no sentences, we can't split by sentence.
        # Fallback to the original naive middle split.
        return len(text) // 2

    target_length = len(text) // 2
    best_split_point = -1
    min_distance = float('inf')

    # Find the sentence end that's closest to the middle of the text
    for start, end in sentence_spans:
        # We don't want to split at the very end of the text, so we skip the last sentence boundary
        if end == len(text):
            continue
            
        distance = abs(end - target_length)
        if distance < min_distance:
            min_distance = distance
            best_split_point = end
            
    # Note: Might return -1
    return best_split_point


# --- Core Recursive Function ---

def semantic_split(
    text: str,
    filename: str,
    touching_file_begin: bool = True,
    touching_file_end: bool = True,
    chunk_max_size_tokens: int = MAX_TOKENS_PER_CHUNK,
    window_size: int = CONTEXT_WINDOW_SIZE_TOKENS,
) -> list[str]:
    """
    Recursively splits a text into semantically coherent chunks based on LLM suggestions.
    Retries the LLM call before using a sentence-based fallback.

    Args:
        text: The block of text to be split.
        filename: The name of the file the text belongs to (for use in the prompt).
        chunk_max_size_tokens: The target maximum token size for a chunk.
        touching_file_begin: Flag indicating if the text chunk is at the absolute beginning of the file.
        touching_file_end: Flag indicating if the text chunk is at the absolute end of the file.
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
        section_text = text.strip()

    prompt = f"""I will provide a large block of text.
The task is: split the large block of text into exactly 2 blocks of text.
The purpose is choosing a starting position for the second section that makes sense- **structurally**.

Output JSON in the format:
{{ "begin_second_section": "text\ngoes here. Always ends with double quote (as JSON requires)" }}

Output enough text in the JSON field to be uniquely identifiable (3-5 words).
Output the text **exactly** as it appears in the raw input to allow for a naive str.find() approach to work.

Full text:
```{filename}
{section_text}
```"""

    messages = [{"role": "user", "content": prompt}]
    split_index = -1
    max_retries = 3

    # 1. Call the LLM to find the optimal split point.
    for attempt in range(max_retries):
        try:
            # Generate content using the Gemini model
            response = client.models.generate_content(
                contents=prompt,
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                    response_mime_type='application/json',
                ),
            )
            response_text = response.text
            
            # qwen3 would require removing think tag
            # if response_text.strip().startswith('<think>'):
            #     response_text = response_text.split('</think>', 1)[-1]
            json_start = response_text.find('{')
            if json_start != -1:
                response_text = response_text[json_start:]
            json_end = response_text.find('}')
            if json_end != -1:
                response_text = response_text[:json_end + 1]

            try:
                split_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                # The LLM often gets confused when the contents themselves contain closing curly braces '}'
                if "unterminated string" in e.msg.lower():
                    response_text += '"}'
                    split_data = json.loads(response_text)
                else:
                    raise
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
                        print(f"Warning (Attempt {attempt + 1}): LLM proposed a highly imbalanced split ({percentage:.2f}%) of {text_len} chars. Discarding.")
                        # By continuing, we treat this as a failed attempt, allowing retries or the fallback to trigger.
                        continue
                    else:
                        split_index = found_index
                        percentage = (split_index / text_len) * 100 if text_len > 0 else 0
                        print(f"LLM identified a valid split point. Split {text_len} chars at {split_index} ({percentage:.1f}%).")
                        break  # Success, exit the retry loop
                else:
                    print(f"Warning (Attempt {attempt + 1}): LLM-suggested string not found in text.")
            else:
                print(f"Warning (Attempt {attempt + 1}): LLM response did not contain 'begin_second_section'.")

        except (json.JSONDecodeError, AttributeError, Exception) as e:
            print(f"Warning (Attempt {attempt + 1}): An API or JSON parsing error occurred: {e}.")
        
        if attempt < max_retries - 1:
            print("Retrying LLM call...")

    # 2. Fallback: If LLM fails after all retries, use a sentence-based split.
    if split_index == -1:
        print("LLM splitting failed after all retries.")
        split_index = fallback_split_by_sentence(text)
        # Final safety net if sentence splitting also fails
        if split_index == -1:
            print("Sentence splitting failed. Reverting to naive middle split.")
            split_index = len(text) // 2


    # 3. Perform the split and recurse on both halves.
    part1 = text[:split_index].strip()
    part2 = text[split_index:].strip()

    # The first part always touches the original chunk's start.
    # The second part always touches the original chunk's end.
    results1 = semantic_split(part1, filename, touching_file_begin, False, chunk_max_size_tokens)
    results2 = semantic_split(part2, filename, False, touching_file_end, chunk_max_size_tokens)

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
            with open(input_filename, "r", encoding="utf-8") as fr:
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
            chunk_num_for_file = i + 1
            
            # Format the filename with a global, incrementing counter
            output_filename = os.path.join(output_dir, f"{str(global_chunk_counter).zfill(6)}.txt")
            
            # Create the header with the file-specific chunk count
            header = f"SOURCE {input_filename} (Chunk {chunk_num_for_file}/{num_chunks_for_this_file})\n"

            with open(output_filename, "w", encoding="utf-8") as fw:
                # Prepend the new header
                fw.write(header)
                fw.write(chunk)
            
            chunk_token_count = count_tokens(chunk)
            print(f"Saved chunk {chunk_num_for_file}/{num_chunks_for_this_file} to '{output_filename}' (from '{input_filename}', Tokens: {chunk_token_count})")
        
        # Add a newline for better visual separation between file processing in the console
        if num_chunks_for_this_file > 0:
            print("\n")

    # --- Final Summary ---
    if global_chunk_counter == 0:
        print("\n--- No chunks were generated. ---")
    else:
        print(f"--- Completed: Split {len(args.file)} file(s) into a total of {global_chunk_counter} chunks. ---")
        print(f"--- All chunks saved successfully in the '{output_dir}/' directory. ---")
