# Deep Intent Search
Search algorithm that rivals the accuracy of a human reading through an article and highlighting relevant sections with a yellow marker

## Introduction

Existing solutions:

- 📏 Long context-length LLMs

- 🌲 Utilizing embedding LLMs to create a vector database.

- 🔢 Classic keyword-based lexical search algorithms

None of the aforementioned solutions are as thorough as manually reading a book or a series of articles chapter by chapter, and marking relevant information with a yellow marker 🖍️.

I can't afford missing **any** relevant information in the series of text documents I present, so I introduce `Deep Intent Search`.

For more information, see [search process](#search-through-the-chunks).

## Create .env

Create [.env](./.env) file containing:

```md
GOOGLE_AISTUDIO_API_KEY=XX123XX123_XXX123XXXXX123XXXXXXX123XXXX
CONTEXT_WINDOW_SIZE_TOKENS=8192
```

## Split the file(s)

The input to the search are text file(s). Say you have a PDF- you'll have to first convert that into a text file.

Use [semantic_splitter.py](./semantic_splitter.py) utility to create a folder at [./split](./split/).
A group of files with naming convention `[000001.txt, 000002.txt, ...]` will be created.

The splitting logic is done by utilizing structured output from `Gemini 2.5 Flash` in a recursive approach, to split the file(s) into small chunks as the LLM sees fit.

```powershell
PS C:\Users\user\Downloads\deep_intent_search> python semantic_splitter.py --file "C:\Users\user\Downloads\tanakh\Prophets\Amos\Hebrew\Tanach with Text Only.txt" "C:\Users\user\Downloads\tanakh\Prophets\Ezekiel\Hebrew\Tanach with Text Only.txt" [...total 39 files]
--- Splitting file: C:\Users\user\Downloads\tanakh\Prophets\Amos\Hebrew\Tanach with Text Only.txt ---
Original token count: 4848
Max tokens per chunk: 1024

Warning (Attempt 1): LLM-suggested string not found in text.
Retrying LLM call...
Warning (Attempt 2): LLM proposed a highly imbalanced split (0.71%) of 10281 chars. Discarding.
LLM identified a valid split point. Split 10281 chars at 2261 (22.0%).
LLM identified a valid split point. Split 2261 chars at 1152 (51.0%).
Warning (Attempt 1): LLM-suggested string not found in text.
Retrying LLM call...
LLM identified a valid split point. Split 8020 chars at 1015 (12.7%).
LLM identified a valid split point. Split 7005 chars at 1088 (15.5%).
LLM identified a valid split point. Split 5917 chars at 2480 (41.9%).
--- File 'C:\Users\user\Downloads\tanakh\Prophets\Amos\Hebrew\Tanach with Text Only.txt' split into 9 chunks. Saving... ---
--- Splitting file: C:\Users\user\Downloads\tanakh\Prophets\Ezekiel\Hebrew\Tanach with Text Only.txt ---
Original token count: 45281
Max tokens per chunk: 1024

Text token count (45281) exceeds window size (8192). Creating a central window for the LLM.
LLM identified a valid split point. Split 95022 chars at 50254 (52.9%).
Text token count (24024) exceeds window size (8192). Creating a central window for the LLM.
LLM identified a valid split point. Split 50254 chars at 18444 (36.7%).
[...collapsed 2811 lines]
--- Completed: Split 39 file(s) into a total of 1127 chunks. ---
--- All chunks saved successfully in the 'split/' directory. ---
PS C:\Users\user\Downloads\deep_intent_search>
```

## Search through the chunks
Once you have a folder [./split](./split/) containing ordered chunks of the files (or a single large file), it's time to perform the deep search.

We'll use [deep_search.py](./deep_search.py) utility to search based on intent, just like a human reading through a book.

The algorithm goes through each chunk (<1024 tokens) while providing up to `CONTEXT_WINDOW_SIZE_TOKENS` surrounding that chunk.

For example, if the current chunk is `000019.txt`, then the contents of surrounding adjacent chunks such as `[000016.txt, ..., 000019.txt, ..., 000022.txt]` will be included. This is important to give the LLM context regarding the meaning and significance of the (current) chunk of interest.

We want to avoid missing any relevant information, so the script makes a new and separate `Gemini 2.5 Flash` LLM completion request focusing on each chunk- meaning each chunk will be fed into the LLM multiple times in practice.

This is the most expensive possible way to search, and it's very unlikely to miss any relevant information about the search query.

Run this command:
```powershell
PS C:\Users\user\Downloads\deep_semantic_chunking> python deep_search.py --query "Interested finding a fully-spelled-out direct explanation of why we should believe in god. I want to find an explicitly stated logical argument- and crucially I'm exclusively interested in an argument that directly addresses the concern of no proof being available for his existence"
Loading and tokenizing all chunks from disk...
Successfully loaded 1127 chunks into memory.

--- Starting Deep Search: Pass 1 (1127 sections to search, Context window: 8192 tokens) ---

Analyzing chunk 1/1127 ('000001.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 2/1127 ('000002.txt')...
  -> LLM decision: Not Relevant
[...collapsed 3373 lines]
Analyzing chunk 1127/1127 ('001127.txt')...
  -> LLM decision: Not Relevant

================================================================================
Found 47 relevant section(s):
================================================================================

- 000021.txt (Tokens: 933)
- 000034.txt (Tokens: 806)
[...collapsed 44 lines]
- 001049.txt (Tokens: 844)

================================================================================
You can now enter a new query to search within these results.
Press Enter to exit.
================================================================================
Refinement Query (Pass 2) > Looking for arguments of god existing that might still be relevant today, and not dependent on specific ancient people who may or may not have witnessed specific miracles more than 2000 years ago.  

--- Starting Deep Search: Pass 2 (47 sections to search, Context window: 8192 tokens) ---

Analyzing chunk 1/47 ('000021.txt')...
  -> LLM decision: Not Relevant

Analyzing chunk 2/47 ('000034.txt')...
  -> LLM decision: Relevant

[...collapsed 133 lines]
Analyzing chunk 47/47 ('001049.txt')...
  -> LLM decision: Relevant

================================================================================
Found 14 relevant section(s):
================================================================================

- 000034.txt (Tokens: 806)
- 000043.txt (Tokens: 818)
- 000058.txt (Tokens: 820)
- 000062.txt (Tokens: 1033)
- 000305.txt (Tokens: 933)
- 000307.txt (Tokens: 824)
- 000308.txt (Tokens: 1023)
- 000309.txt (Tokens: 866)
- 000310.txt (Tokens: 423)
- 000312.txt (Tokens: 688)
- 000488.txt (Tokens: 908)
- 000798.txt (Tokens: 829)
- 000805.txt (Tokens: 1038)
- 001049.txt (Tokens: 844)

================================================================================
You can now enter a new query to search within these results.
Press Enter to exit.
================================================================================
Refinement Query (Pass 3) >
```

We've found 14 relevant sections, which are 11,853 tokens combined.
I've uploaded [the final search results](./tanakh_search_results.txt) of this example.
