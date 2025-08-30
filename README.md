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

## Create .env

Create [./.env](./.env) file containing:

```sh
# Optional: point to a non-standard Ollama address (IP+Port as a *single* string).
# If omitted, defaults to 127.0.0.1:11434
OLLAMA_HOST=172.17.0.1:11434

# Controls how many tokens of *surrounding* text are given to the LLM around each 1024 token chunk
CONTEXT_WINDOW_SIZE_TOKENS=8192

# Model selection (optional)
# Allowed values:
#   - qwen3:32b   (default if unset/empty)
#   - qwen3:30b-a3b-thinking-2507
OLLAMA_MODEL=qwen3:32b
```

> **Model**: This project uses **qwen3** locally via **Ollama v0.11.7** (no cloud keys required).
> Pull once with:
> ```bash
> ollama pull qwen3:32b # Default
> ollama pull qwen3:30b-a3b-thinking-2507-q4_K_M
> ```

The LLM is called with the following advanced options on **every request**:
- Context Length `num_ctx = 24000`
- `num_predict = -1`
- `temperature = 0.6`
- `top_k = 20`
- `top_p = 0.95`
- `min_p = 0`
- `repeat_penalty = 1`
- `num_gpu = 65`

## Search through the chunks
First use [semantic_splitter.py](#split-the-files). Then once you have a folder [./split](./split/) containing ordered chunks of all the files, it's time to perform the deep search.

We'll use [deep_search.py](./deep_search.py) utility to search based on intent, just like a human reading through a book.

The algorithm goes through each chunk (<1024 tokens) while providing up to `CONTEXT_WINDOW_SIZE_TOKENS` surrounding that chunk.

For example, if the current chunk is `000019.txt`, then the contents of surrounding adjacent chunks such as `[000016.txt, ..., 000019.txt, ..., 000022.txt]` will be included. This is important to give the LLM context regarding the meaning and significance of the (current) chunk of interest.

We want to avoid missing any relevant information, so the script makes a new and separate LLM completion request focusing on each chunk—meaning each chunk will be fed into the LLM multiple times in practice.

This is the most expensive possible way to search, and it's very unlikely to miss any relevant information about the search query.

Run this command:
```powershell
PS C:\Users\user\Downloads\deep_semantic_chunking> python deep_search.py --query "Looking for **solid** evidence of the fact that if you say Voldemort's explicit name, he will find you. I'm **not** interested in theories or rumors that are mentioned in the book. I'm **not** interested in hearing someone say He who must not be named, just because they're afraid, again, due to these rumors. I'm only looking for instances where this actually happened. And where Voldemort **actually** somehow seems to gain information from somebody saying his name."
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
- Generate an extremely high-quality dataset of search results, will require generating synthetic intent-oriented search queries. Even if the synthetic intent-oriented search queries aren't extremely high-quality, the search results are the outcome of so much processing power `ThinkingConfig(thinking_budget=-1)` that the generated dataset will contain concentrated intelligence traces.

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
