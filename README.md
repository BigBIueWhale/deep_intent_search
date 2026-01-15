# Deep Intent Search
Search algorithm that rivals the accuracy of a human reading through an article and highlighting relevant sections with a yellow marker

## Introduction

### My Solution
I can't afford missing **any** relevant information in the series of text documents I present, so I introduce `Deep Intent Search`.

My [search process](#search-through-the-chunks) takes advantage of the accurate language understanding that LLMs exhibit at short context lengths (<1024 tokens).

## Evidence-First Verdicts (Why This Helps)

LLMs are human-preferenced trained, they tend to "put on a show". In strict relevance tasks that creates a bias toward **marking borderline text as relevant** just to avoid disappointing the user.

To redirect that impulse productively, Deep Intent Search **requires evidence** alongside every verdict. For each chunk, the model must output:
- `is_relevant`: `true` or `false`
- `evidence`: **exactly three short sentences**, extremely concise and information-dense, drawn from the section of interest.

Instead of overselling relevance, the model "performs" by writing a high-quality micro-justification we can ignore operationally but use for quick spot-checks.

### Existing Alternatives

- 📏 Long context-length LLMs

- 🌲 Utilizing embedding LLMs to create a vector database.

- 🔢 Classic keyword-based lexical search algorithms

## Normal Flow

1. [semantic_splitter.py](./semantic_splitter.py):
    - **Input (choose exactly one):**
      - `--files path1 path2 ...`
      - `--files-list /path/to/files_list.txt`
    - **Final output:** [./split/chunks](./split/chunks/)
2. [deep_search.py](./deep_search.py):
    - **Input:**
      1. [./split/chunks](./split/chunks/)
      2. `--ctx "We're searching thoroughly through ..."`
      3. `--query "Interested in content that ..."`
      4. Newest `search_runs/xxxx.jsonl` if exists.
    - **Output:** [./search_runs/0001.jsonl](./search_runs/0001.jsonl)
3. [rerank.py](./rerank.py):
    - **Input:**
      1. [./split/chunks](./split/chunks/)
      2. Newest `search_runs/xxxx.jsonl`
      3. `--query "Interested in content that... Priority to content that..."`
    - **Final output:** [./rerank/order.csv](./rerank/order.csv)
4. [collect_transform_pretty.py](./collect_transform_pretty.py)
    - **Input:**
      1. [./split/chunks](./split/chunks/)
      2. Newest `search_runs/xxxx.jsonl`
      3. [./rerank/order.csv](./rerank/order.csv)
    - **Final output:** [./pretty/000001.txt](./pretty/000001.txt) (and `000002.txt` etc)
5. [yellow_marker.py](./yellow_marker.py)
    - **Input:**
      1. [./pretty/000001.txt](./pretty/000001.txt) (and `000002.txt` etc)
      2. Newest `search_runs/xxxx.jsonl`- take "query" from first line of jsonl, and use "evidence" as a hint for highlighting purposes.
    - **Final output:** [./yellow_marker/000001.txt](./pretty/000001.txt) (and `000002.txt` etc)
6. [generate_printable_pdf.py](./generate_printable_pdf.py)
    - **Input:**
      1. [./yellow_marker/000001.txt](./yellow_marker/000001.txt) (and `000002.txt` etc)
      2. Newest `search_runs/xxxx.jsonl`- take "query" from first line of jsonl
      3. [./rerank/progress.jsonl](./rerank/progress.jsonl)- take "query" from first line of jsonl
    - **Output:** [./printable/printable.pdf](./printable/printable.pdf)

## Create .env

Create `./.env` with:

```sh
# Optional: point to a non-standard Ollama address (IP+Port as a single string).
# If omitted, defaults to 127.0.0.1:11434
#OLLAMA_HOST=172.17.0.1:11434

# Optional: Customize timeout value in seconds for each Ollama request
# Defaults to eight minutes.
# This functionality is required to avoid Ollama bug where it sometimes
# just forgets about our request, and leaves our script hanging forever.
#OLLAMA_TIMEOUT_SECONDS=480

# Controls how many tokens of surrounding text are given to the LLM around each 1024 token chunk, during deep_intent_search.py.
# There's an algorithm "select_stable_window_bounds()" that makes sure that
# consecutive search prompts will mostly be cached by Ollama,
# so you can increase this number to hundreds of thousands of tokens
# while experiencing minimal performance hit. Number chosen here has to
# fit within OLLAMA_MODEL_SMARTEST context window.
# Normally 8192 is the smallest number to choose here.
CONTEXT_WINDOW_SIZE_TOKENS=100000

# Model selection (required)
# You must set both variables explicitly. There is no fallback.
# Allowed values:
#   - qwen3-vl:32b-instruct
#   - qwen3-vl:32b-thinking
#   - qwen3:32b
#   - milkey/Seed-OSS-36B-Instruct:q4_K_M
#   - qwen3:30b-a3b-thinking-2507-q4_K_M
#   - qwen3:30b-a3b-instruct-2507-q4_K_M
#   - JollyLlama/GLM-4-32B-0414-Q4_K_M
#   - gemma3:27b
#   - gpt-oss:20b

# OLLAMA_MODEL_HYBRID_REASONING:
#   Used by semantic_splitter.py. Needs to support both fast instruction following
#   and deep self-correction (reasoning) when splitting fails.
OLLAMA_MODEL_HYBRID_REASONING=qwen3:32b

# OLLAMA_MODEL_LONG_CONTEXT:
#   Used by collect_transform_pretty.py, and yellow_marker.py (primary).
#   Needs a massive context window to ingest large chunks and history.
OLLAMA_MODEL_LONG_CONTEXT=qwen3:32b

# OLLAMA_MODEL_ANOTHER_LONG_CONTEXT:
#   Used by yellow_marker.py as an escalation fallback.
#   Needs long context and different architecture to solve persistent failures.
OLLAMA_MODEL_ANOTHER_LONG_CONTEXT=qwen3-vl:32b-instruct

# OLLAMA_MODEL_SMARTEST:
#   Used by deep_search.py, rerank.py. The highest quality model that fits on our GPU.
OLLAMA_MODEL_SMARTEST=qwen3:30b-a3b-thinking-2507-q4_K_M

# OLLAMA_MODEL_VISION
#   Used by convert_pdf_to_md_vl.py
OLLAMA_MODEL_VISION=qwen3-vl:32b-thinking
```

> Pull models once:
> ```bash
> ollama pull qwen3-vl:32b-instruct
> ollama pull qwen3-vl:32b-thinking
> ollama pull qwen3:32b
> ollama pull milkey/Seed-OSS-36B-Instruct:q4_K_M
> ollama pull qwen3:30b-a3b-thinking-2507-q4_K_M
> ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
> ollama pull JollyLlama/GLM-4-32B-0414-Q4_K_M
> ollama pull gemma3:27b
> ```

The LLM is called with the metaparameters described in [./core/llm.py](./core/llm.py) on **every request**

## Per-model options

- `milkey/Seed-OSS-36B-Instruct:q4_K_M` can be trusted as a judge. Supports a "hybrid reasoning" mode where it can be switched between fast instruct-only responses and deep thinking responses. This makes it ideal for `OLLAMA_MODEL_HYBRID_REASONING` and `OLLAMA_MODEL_SMARTEST`. It has a context limit of \~15k tokens on 32 GB VRAM.

- `qwen3:32b` is highly recommended for all tasks in this project. In its `/no_think` variable, it's a model of average reliability, probably even less reliable than `qwen3:30b-a3b-instruct-2507-q4_K_M`. But, since it's a hybrid reasoning model- upon multiple consecurive failures in [semantic_splitter.py](./semantic_splitter.py) we simply turn on thinking mode on the fly! Without having to unload and then load a different model into VRAM!

- `qwen3-vl:32b-instruct`- Recommended for all judgement tasks. Seems to have avoided the issue that plagued the non-thinking version of `qwen3:32b`- namely getting confused between sections. Definitely seems to find more results than `qwen3:32b` (with thinking), which is incredible! Definitely faster too (obviously, because it's non-thinking). Has lots of false-positives though.

- `qwen3-vl:32b-thinking`- Thinks at least twice as much as `qwen3:32b`. Not that great, and dangerous due to its high VRAM consumption (all those thinking tokens come dangerously close to the context length).

- `gpt-oss:20b` is supported because it's really not that bad (on newer versions of Ollama since they fixed format issues), and gives us 128k context window within only ~17GB of VRAM.

- `JollyLlama/GLM-4-32B-0414-Q4_K_M` was an attempt to find a dense non-thinking model to be an `OLLAMA_MODEL_SPLITTER`, and it's definitely the most capable non-thinking model shown here that can fit on a single consumer GPU. Its high multilingual capabilities, and general world knowledge might prove useful.

- `gemma3:27b` is extremely eager and that makes it unusable as a judge model. `qwen3:32b` (with thinking) might find 1 result in 1500+ sections, and `gemma3:27b` has been observed to find hundreds of "results" given the same prompt. Its vision capabilities and SoTA multilingual capabilities might prove useful.

- `qwen3:30b-a3b-instruct-2507-q4_K_M` is the fastest non-thinking model (that actually works and is useful) to be able to fit on a consumer GPU. The attempt was to use it for `semantic_splitter.py` because it doesn't possess hybrid thinking capabilities, `qwen3:30b-a3b-thinking-2507-q4_K_M` is a different model, which means it would require unloading from GPU and loading a different model. Its speed and reliability (as much as a non-thinking model can be reliable) might prove crucial for speeding-up operations in this project.

- `qwen3:30b-a3b-thinking-2507-q4_K_M` does so much thinking, to compensate for its measly 3 billion active parameters. The 200 tok/sec generation speed (on RTX 5090) might prove useful. However, it usually ends up taking longer than `qwen3:32b` just because it thinks so much. Oh, and `qwen3:32b` is definitely smarter and more reliable. **However** this model has excellend `RULER` benchmark scores so it might literally be the key to solving the "I don't understand this chunk in isloation" problem that plauges `deep_search.py`!

## Search through the chunks
First use [semantic_splitter.py](#split-the-files). Then once you have a folder [./split](./split/) containing ordered chunks of all the files, it's time to perform the deep search.

We'll use [deep_search.py](./deep_search.py) utility to search based on intent, just like a human reading through a book.

The algorithm goes through each chunk (<1024 tokens) while providing up to `CONTEXT_WINDOW_SIZE_TOKENS` surrounding that chunk.

For example, if the current chunk is `000019.txt`, then the contents of surrounding adjacent chunks such as `[000016.txt, ..., 000019.txt, ..., 000022.txt]` will be included. This is important to give the LLM context regarding the meaning and significance of the (current) chunk of interest.

We want to avoid missing any relevant information, so the script makes a new and separate LLM completion request focusing on each chunk—meaning each chunk will be fed into the LLM multiple times in practice.

This is the most expensive possible way to search, and it's very unlikely to miss any relevant information about the search query.

Run this command:
```powershell
PS C:\Users\user\Downloads\deep_semantic_chunking> python deep_search.py --ctx "We're searching thoroughly through all seven books of Harry Potter" --query "Looking for **solid** evidence of the fact that if you say Voldemort's explicit name, he will find you. I'm **not** interested in theories or rumors that are mentioned in the book. I'm **not** interested in hearing someone say He who must not be named, just because they're afraid, again, due to these rumors. I'm only looking for instances where this actually happened. And where Voldemort **actually** somehow seems to gain information from somebody saying his name."
Loading and tokenizing all chunks from disk...
Loaded 2790 chunks (1,646,744 tokens).

--- Starting Deep Search: Pass 1 (2790 sections to search, Context window: 8192 tokens) ---

Analyzing chunk 1/2790 ('000001.txt')...
  -> LLM decision: Not Relevant
  -> Evidence: ...

Analyzing chunk 2/2790 ('000002.txt')...
  -> LLM decision: Not Relevant
  -> Evidence: ...

[...collapsed 5000+ lines]

Analyzing chunk 2544/2790 ('002544.txt')...
  -> LLM decision: Relevant
  -> Evidence: "the name’s been jinxed, Harry, that’s how they track people!" "Using his name breaks protective enchantments, it’s how they found us in Tottenham Court Road!" "they’ve put a Taboo on it, anyone who says it is trackable — quick-and-easy way to find Order members!"

[...collapsed 100+ lines]

Analyzing chunk 2587/2790 ('002587.txt')...
  -> LLM decision: Relevant
  -> Evidence: The text states, "the name’s been Tabooed" and that "a few Order members have been tracked that way." This indicates that saying Voldemort's name can lead to being tracked by his followers. It does not show Voldemort actively finding someone in this section, but it presents the Taboo as a mechanism for discovery.

[...collapsed 800+ lines]

================================================================================
Found 2 relevant section(s):
================================================================================

- 002544.txt (Tokens: 475)
- 002587.txt (Tokens: 624)

================================================================================
You can now enter a new query to search within these results.
Press Enter to exit.
================================================================================
Refinement Query (Pass 2) >
```

The only issue is- `GPT-5-nano` is an idiot. It not the best at instruction following, and often confuses nearby sections- ignoring the base requirement to only look at the "SECTION OF INTEREST".

Disclaimer: In the above Harry Potter search, `GPT-5-nano` actually marked 9 additional results as relevant. All nearby (before and after) `002587.txt`. This caused 9 false-positives. Entirely due to the lack of instruction following displayed by `GPT-5-nano`. I had to manually edit the results to remove those false positives to avoid confusing the readers of this readme.

## Split the file(s)

The input to the search are text file(s). Say you have a PDF- you'll have to first convert that into a text file.

Use [convert_pdf_to_md_vl.py](./convert_pdf_to_md_vl.py) for this task (works great if the language is English). Get outputs at [./conversion_vl](./conversion_vl/)

Alternatively, you can use [convert_all_pdfs.py](./convert_all_pdfs.py) to batch-run the conversion logic, and get outputs at [./all_pdfs](./all_pdfs)

Use [semantic_splitter.py](./semantic_splitter.py) utility to create a folder at [./split](./split/).
A group of files with naming convention `[000001.txt, 000002.txt, ...]` will be created.

The splitting logic is done by utilizing structured output from the LLM in a recursive approach, to split the file(s) into small chunks as the LLM sees fit.

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

## Future Research

😱 Running the tool is very expensive and slow. Estimated price $3.65 (GPT-5-nano pricing) for every search pass on the 7 books of Harry Potter- and that's not including the `semantic_splitter.py` step.

Given that, future project goals could be-
- Generate an extremely high-quality dataset of search results, will require generating synthetic intent-oriented search queries. Even if the synthetic intent-oriented search queries aren't extremely high-quality, the search results are the outcome of so much processing power (especially with thinking turned on), that the generated dataset will contain concentrated intelligence traces.

- Add **ELO tournament** scoring for the purpose of ordering the results from most relevant to least relevant.

- Generate high-quality search results dataset **as a benchmark** to test the long context quality in LLM research.

- Generate huge (potentially limitless) high-quality search results dataset to train an LLM-like model to be **natively** good at long-context search, which could be offered as an affordable, scalable and fast alternative to this expensive and rather brute-force approach.

## Usecases

Regardless of the high cost of this brute force approach, there are some legitimate usecases where `Deep Intent Search` can be a gamechanger:

- **RAG**- allows a smart LLM to get the precise information it needs for in-context learning to actually work, as opposed to current RAG implementations that fail at having a model truly learn what it needs to know.

- **Law**- can do the menial job of searching through a large book of laws (or precedents) to find specific sections that are relevant to the case of interest.

- **Police / Intelligence Agencies**- can be a game-changing tool for professional detectives that might read through boring logs and reports, 99% of which are not relevant to the case being researched.

- **Cybersecurity**- can read through network logs to identify suspicious activity.

- **Messaging Apps**- make Telegram search actually work.

- **Image Search**- This technology in essense should work with other media formats other than text.

## Fundamental Limitation

What if the LLM doesn't fully understand the text?

When a human reads text, we might learn new concepts that enter our long-term memory, then upon the second read we will actually be able to better-understand the text and more accurately mark relevant sections with a yellow marker.

Thanks to human's long-term learning, we can also link two distant parts of the text to gain additional insights and context. Classic example is learning new terminology from context in one part of a report, to understand another part of the report much more accurately.

Current LLM technology only has in-context learning, which starts degrading in quality when surpassing ~4096 tokens.\
Training an LLM on a piece of text will not help it learn the specific information at all, because LLMs can only learn from being presented the same information in multiple different contexts, wordings and formats.

For example- the reason most LLMs can accurately describe specific plot points in Harry Potter is hardly the result of the LLM training on the English text of Harry Potter. It's actually thanks to the discussion forums, hints in cultural references, summaries, book reports, and literary analyses that all help the LLM memorize the plot line details of Harry Potter. The information has to exist in multiple contexts, wordings and formats for the LLM to actually learn and not just overfit.

That's why current LLMs cannot learn any specific information by training, only in-context learning.

This can be observed in humans. A human who tries to memorize a sentence without thinking about the meaning, will not memorize the meaning of that sentence. They might be able to repeat the sentence, but definitely not use its contents as part of a thought process.

Memorization of a piece of text into long-term memory on-demand will always require a human to make a conscious effort to find strong mental connections to existing brain circuits. A human might think about the information in a diverse set of ways, ask questions about the information in their internal dialogue, and consciously / subconsciously find connections to existing knowledge- In LLM lingo this might be called "synthetic information expansion for self-training".

Once we solve that problem, we will have AGI- In current LLM architectures the weights are static. Imagine if LLMs were constantly trained on the output of the LLM itself during pretraining, instead of being trained on the raw text they can be trained on the LLM's thoughts regarding that text. It can also be like humans with a mix of internal dialogue and experiences. Somehow train the LLM to truly memorize and understand pieces of information by optimally blabbering about the topic, A.K.A "synthetic information expansion for self-training". The problem is we need to find a robust training methodology for the LLM to have an incentive to optimally think about text while learning.
