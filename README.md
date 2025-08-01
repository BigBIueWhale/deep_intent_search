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
[...total ~3000 lines]
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
PS C:\Users\user\Downloads\deep_semantic_chunking> python deep_search.py --query "Interested only in occurences of god himself directly explaining why we should believe in him using an actual logical argument- and crucially I'm exclusively interested in instances where he directly addresses the concern of no proof being available for his existence"
```
