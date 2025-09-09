Below is a self-contained plan you can hand to an engineer and they can build it. It states the problem and edge cases, why we chose the embedding model, the exact math to score split points, and the exact way to pick them while *always* staying at or under 1024 tokens (by the Qwen3 tokenizer). Words are simple, steps are ordered, nothing is assumed.

---

# Problem we’re solving

We must split very large text files into chunks where:

* Each chunk is **at most 1024 tokens** measured by the **Qwen3 tokenizer**.
* It must work for **any text**, including:

  * normal prose (e.g., Talmud in English, Hebrew/Aramaic passages),
  * code,
  * OCR mess,
  * files that are **one single very long line**,
  * files that have **no spaces** (for example, long base64 strings).
* Splits should prefer **meaningful places** when meaning exists (topic shifts, speaker changes, paragraph-like transitions).
* When **no meaning** is present (for example, uniform base64), the method must **still** produce clean, legal chunks without guessing.

We want a **deterministic** method (same input → same result), with no reliance on a chat model “judging” split points.

---

# Why we choose **Qwen3-Embedding-8B**

* It produces **one vector of numbers** for a piece of text that reflects its meaning across languages and styles (good for Gemara + commentaries + English notes, and also okay for code).
* It is **recent and strong**, and fits your stack.
* Its vectors are stable for **longer windows**, which lowers noise and gives clearer “this is the same” vs “this is different.”

We will use **only** this embedding model to measure similarity. Everything else is plain math.

---

# Core idea in one line

We slide overlapping windows over the text, get an embedding vector for each window, and build a **score** at many candidate positions that says, “left side holds together, right side holds together, left and right do not look alike.” High score = good cut. Then we pick a **set** of cuts that maximizes total score while **never** making any chunk longer than 1024 tokens.

---

# Exact procedure (ready to implement)

## 0) Notation and helper functions

* Let `cos(u, v)` be **cosine similarity** of two vectors `u` and `v`.
  Implementation detail: normalize each embedding to unit length; then `cos(u, v)` is the dot product.

* Let `avg(S)` mean the simple average (sum divided by count) of a set `S` of numbers.

* Small constant: `delta = 1e-6` to avoid division by zero.

* All lengths and positions below are measured in **Qwen tokens** (not characters).

## 1) Tokenize once

Input: full text `text`.

* Run the **Qwen3 tokenizer** with offset mapping:

  * `tokens[0..L-1]` are token ids (we only need their positions for counting).
  * `offsets[i] = (char_start_i, char_end_i)` tells where token `i` lives in the original string.
* We will cut on token indices, and later map back to character positions using `offsets`.

## 2) Build two passes of overlapping windows and embed

We want a **coarse view** (stable) and a **fine view** (precise).

* **Coarse windows**

  * length `Wc = 384` tokens
  * stride `Sc = 128` tokens
  * For each coarse window `k`, record:

    * `c_span_k = [start_token, end_token)`
    * `c_center_k = floor((start_token + end_token)/2)`
    * `c_emb_k = embedding(text[c_span_k])`  (normalize this vector to unit length)

* **Fine windows**

  * length `Wf = 176` tokens (anything in `160..192` is fine)
  * stride `Sf = 64` tokens
  * For each fine window `j`, record:

    * `f_span_j`, `f_center_j`, `f_emb_j` (normalized)

**Streaming note:** For huge files, you can stream: move a token cursor, detokenize that slice via offsets to get the bytes for each window, embed, discard old windows when they fall far behind.

## 3) Build a grid of candidate cut positions

* Grid step `g = 16` tokens.
* Candidate set: `G = {0, 16, 32, …, L}` (make sure `L` is included even if it is not a multiple of `g`).

We will compute a **boundary strength** `B(t)` for each `t` in `G`, except `t = 0`.

## 4) Compute boundary strength at each candidate `t`

This score is “big” when content on the left is internally similar, content on the right is internally similar, and the two sides are not similar to each other. We combine a **block view** (coarse) and a **local view** (fine).

### 4.1) Coarse “block view” around `t`

* Choose `h = 5` coarse windows **left** of `t`: the five windows with centers `< t` that are closest to `t`. If fewer exist, use what you have (skip if zero).
* Choose `h = 5` coarse windows **right** of `t`: centers `≥ t`, closest to `t`. (Skip if zero.)

Let `Left = {c_emb_a}` and `Right = {c_emb_b}` be those two small sets of vectors.

Compute three averages of cosine similarity:

* **Within left**
  `InLeft(t) = avg( { cos(u, v) for all unordered pairs u, v in Left, u != v } )`
  If `Left` has size 1, define `InLeft(t) = 1.0` (a single item is perfectly “coherent” with itself).

* **Within right**
  Same formula on `Right`. If size 1, use `1.0`.

* **Across left and right**
  `Cross(t) = avg( { cos(u, v) for all u in Left and v in Right } )`
  If one side is empty, skip this `t` (cannot form a boundary at extreme ends).

Define a **novelty** value:

```
N(t) = ((InLeft(t) + InRight(t)) / 2) − Cross(t)
if N(t) < 0, set N(t) = 0
```

Intuition: big `N(t)` means “each side hangs together, but they do not look like each other”.

### 4.2) Fine “local view” around `t`

Pick a **small band** of fine windows near `t`. A simple choice:

* Take `k = 4` fine windows with centers `< t` that are closest to `t`, call them `L1, L2, L3, L4` (if fewer, use what you have).
* Take `k = 4` fine windows with centers `≥ t` that are closest, call them `R1, R2, R3, R4`.

Compute:

* **Left cohesion**
  `C_left(t) = avg( { cos(Li, Lj) for i and j that are neighbors in order } )`
  A simple “neighbors” rule: `(L1,L2), (L2,L3), (L3,L4)` for as many as exist. If only one left window exists, define `C_left(t) = 1.0`.

* **Right cohesion**
  Same on the right: `(R1,R2), (R2,R3), (R3,R4)`. If only one exists, set to `1.0`.

* **Cross bleed**
  Pair the closest windows across the cut: `(L1,R1), (L2,R2), (L3,R3), (L4,R4)` for as many pairs as exist.
  `X(t) = avg( cos(Li, Ri) over existing pairs )`. If no pair exists, skip this `t`.

### 4.3) One number per candidate

Combine the parts:

```
B(t) = N(t) * ( C_left(t) * C_right(t) ) / ( X(t) + delta )
```

* If `B(t)` is undefined due to missing windows, set `B(t) = 0`.

This value is higher when each side is smooth on its own, the sides are different from each other, and there is little “bleed” across the would-be cut.

## 5) Pick cut positions with a left-to-right table (always ≤ 1024 tokens)

Now we choose a **set** of cuts that maximizes total `B(t)` while respecting the size rule.

Set constants:

* `M = 1024` (maximum allowed chunk length, hard rule)
* `m = 320`  (minimum helpful length; avoids tiny scraps)
* `tau = 900` (soft target length; helps fill space when `B` is flat)
* `alpha = 0.002` (weight for the soft length penalty)

We fill a table over the grid:

* `best_score_at[t]`: the best total score we can get up to position `t`.
* `prev_cut_at[t]`: the previous cut position `p` that produced that best score.

Initialize:

```
best_score_at[0] = 0
prev_cut_at[0] = None
For all other t in G: best_score_at[t] = -infinity (or a very small number)
```

For each `t` in `G` in increasing order (skip `t=0`):

1. Consider earlier grid points `p` in `G` with `m <= (t - p) <= M`.
2. For each such `p`, compute a trial score:

   ```
   trial = best_score_at[p] + B(t) - alpha * abs( (t - p) - tau )
   ```
3. If `trial > best_score_at[t]`, update:

   ```
   best_score_at[t] = trial
   prev_cut_at[t]   = p
   ```

At the end, treat `L` (the last grid point) as the end position. Backtrack:

```
cuts = []
t = L
while t is not None and t > 0:
    cuts.append(t)
    t = prev_cut_at[t]
cuts.append(0)
reverse(cuts)   # now cuts are [0, t1, t2, ..., L]
```

Every adjacent pair `(cuts[i], cuts[i+1])` is a chunk in **token units**, and each one is guaranteed to be `<= 1024` tokens by construction.

## 6) Small, safe refinements

**(a) Nudge to a nearby stronger point.**
For each internal cut `t_i` (not 0, not L), look at neighbors within `±32` tokens on the grid.
If there is a neighbor `t'` with larger `B(t')`, and both neighbor chunk lengths remain `<= 1024`, move the cut to `t'`.

**(b) Merge a tiny tail.**
If the final chunk length `< m`, and merging with the previous chunk keeps length `<= 1024`, merge them.

These steps never break the size rule.

## 7) Map cuts back to characters and write chunks

For each token cut `t`, use `offsets[t]` to find the character position to cut the original string.
Write each chunk as `text[char_start_of_token(cuts[i]) : char_start_of_token(cuts[i+1])]`.

**Optional overlap for retrieval:** if you want overlap, pick `overlap_tokens = 64`. When **writing** chunks, include the next 64 tokens at the end of each chunk (do not change the logical cut list).

---

# Why this cannot fail

* **When structure exists:** embeddings on the left and right differ, while each side is stable; `B(t)` rises right where humans would cut. The table prefers those peaks while keeping each chunk ≤ 1024 tokens.
* **When structure does not exist (base64, one mega line, no spaces):** embeddings barely change; `B(t)` stays low and flat; the table then fills the file with near-target lengths that never exceed 1024. No guessing, no crashes.
* **Always legal:** the length rule is enforced inside the chooser itself, not as a later fix. So no chunk can ever be over 1024 tokens.
* **Deterministic:** same file → same windows → same vectors → same scores → same cuts.

---

# Reference values (good defaults)

* Grid step: `g = 16`
* Coarse windows: `Wc = 384`, `Sc = 128`, `h = 5`
* Fine windows: `Wf = 176`, `Sf = 64`, `k = 4`
* Limits: `M = 1024`, `m = 320`, `tau = 900`, `alpha = 0.002`
* Nudge radius: `32`
* Overlap when writing (optional): `64`
* `delta = 1e-6` for divisions

---

# Pseudocode you can copy

```python
# Required:
# - qwen3_tokenize(text) -> tokens, offsets  (offsets: list of (char_start, char_end))
# - embed(text_slice) -> vector (normalize to unit length)

def build_windows(tokens, offsets, W, S):
    windows = []
    t = 0
    L = len(tokens)
    while t < L:
        start = t
        end = min(L, t + W)
        # map token span to character slice
        char_start = offsets[start][0]
        char_end   = offsets[end-1][1] if end > start else char_start
        windows.append({
            "t_start": start,
            "t_end": end,
            "t_center": (start + end) // 2,
            "char_start": char_start,
            "char_end": char_end
        })
        t += S
    return windows

def add_embeddings(text, windows, embed_fn):
    for w in windows:
        seg = text[w["char_start"]:w["char_end"]]
        w["emb"] = normalize(embed_fn(seg))  # L2 normalize to unit length

def cosine(u, v):
    # assumes u and v are already unit length
    return float((u * v).sum())

def boundary_scores(tokens, coarse, fine, g=16, h=5, k=4, delta=1e-6):
    L = len(tokens)
    # index windows by center position for fast nearest lookups
    c_centers = [w["t_center"] for w in coarse]
    f_centers = [w["t_center"] for w in fine]
    B = {}  # map token index -> boundary strength

    # helper to fetch closest n windows left/right of t from a given list
    def closest_lr(windows, centers, t, n):
        left_idx = [i for i,c in enumerate(centers) if c < t]
        right_idx = [i for i,c in enumerate(centers) if c >= t]
        left = [windows[i] for i in left_idx[-n:]]   # last n
        right = [windows[i] for i in right_idx[:n]]  # first n
        return left, right

    # neighbors utility for cohesion
    def neighbor_pairs(ws):
        pairs = []
        for i in range(len(ws)-1):
            pairs.append((ws[i]["emb"], ws[i+1]["emb"]))
        return pairs

    # pair averages
    def avg_cos_pairs(vecs):
        if len(vecs) <= 1: return 1.0
        s, cnt = 0.0, 0
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                s += cosine(vecs[i], vecs[j]); cnt += 1
        return s/cnt if cnt else 1.0

    for t in range(g, L+1, g):
        # Coarse block view
        Lc, Rc = closest_lr(coarse, c_centers, t, h)
        if len(Lc) == 0 or len(Rc) == 0:
            B[t] = 0.0
            continue
        InLeft  = avg_cos_pairs([w["emb"] for w in Lc])
        InRight = avg_cos_pairs([w["emb"] for w in Rc])
        # cross average
        cross_vals = [cosine(a["emb"], b["emb"]) for a in Lc for b in Rc]
        Cross = sum(cross_vals)/len(cross_vals)

        N = max(0.0, ((InLeft + InRight)/2.0) - Cross)

        # Fine local view
        Lf, Rf = closest_lr(fine, f_centers, t, k)
        # cohesion left/right
        left_pairs  = neighbor_pairs(Lf)
        right_pairs = neighbor_pairs(Rf)
        C_left  = (sum(cosine(a,b) for a,b in left_pairs) / len(left_pairs)) if left_pairs else 1.0
        C_right = (sum(cosine(a,b) for a,b in right_pairs) / len(right_pairs)) if right_pairs else 1.0
        # cross bleed pairs (pair by rank)
        cross_pairs = []
        m = min(len(Lf), len(Rf))
        for i in range(m):
            cross_pairs.append((Lf[-m+i]["emb"], Rf[i]["emb"]))
        if not cross_pairs:
            B[t] = 0.0
            continue
        X = sum(cosine(a,b) for a,b in cross_pairs) / len(cross_pairs)

        B[t] = N * (C_left * C_right) / (X + delta)

    return B  # dictionary from token index to boundary strength

def choose_cuts(L, B, g=16, M=1024, m=320, tau=900, alpha=0.002):
    # dynamic table
    G = list(range(0, L+1, g))
    if G[-1] != L: G.append(L)  # ensure L included
    best = {t: float("-inf") for t in G}
    prev = {t: None for t in G}
    best[0] = 0.0

    for ti in range(1, len(G)):
        t = G[ti]
        for pj in range(ti-1, -1, -1):
            p = G[pj]
            span = t - p
            if span < m: continue
            if span > M: break  # earlier p will only be farther
            b = B.get(t, 0.0)
            trial = best[p] + b - alpha * abs(span - tau)
            if trial > best[t]:
                best[t] = trial
                prev[t] = p

    # backtrack
    cuts = []
    t = G[-1]
    # if prev[t] is None because file shorter than m, still handle single chunk
    while t is not None and t > 0:
        cuts.append(t)
        t = prev[t]
    cuts.append(0)
    cuts.sort()
    return cuts  # token indices including 0 and L

def nudge_and_merge(cuts, B, L, g=16, radius=32, M=1024, m=320):
    # Nudge to local maxima of B within radius
    # Work on internal cuts only
    new = cuts[:]
    for idx in range(1, len(new)-1):
        t = new[idx]
        low = max(0, t - radius)
        high = min(L, t + radius)
        # snap to grid
        low  = (low // g) * g
        high = (high // g) * g
        best_t = t; best_b = B.get(t, 0.0)
        c_lo = new[idx-1]; c_hi = new[idx+1]
        tt = low
        while tt <= high:
            left_len  = tt - c_lo
            right_len = c_hi - tt
            if 0 < left_len <= M and 0 < right_len <= M:
                b = B.get(tt, 0.0)
                if b > best_b:
                    best_b, best_t = b, tt
            tt += g
        new[idx] = best_t

    # Merge tiny tail if possible
    if len(new) >= 2:
        last_len = new[-1] - new[-2]
        if last_len < m and len(new) >= 3:
            # try merge with previous
            prev_len = new[-2] - new[-3]
            if prev_len + last_len <= M:
                # drop the penultimate cut
                new.pop(-2)
    return new
```

Usage outline:

```python
tokens, offsets = qwen3_tokenize(text)

coarse = build_windows(tokens, offsets, W=384, S=128)
add_embeddings(text, coarse, embed_fn=qwen3_embed_8b)

fine   = build_windows(tokens, offsets, W=176, S=64)
add_embeddings(text, fine,   embed_fn=qwen3_embed_8b)

B = boundary_scores(tokens, coarse, fine, g=16, h=5, k=4, delta=1e-6)

cuts = choose_cuts(L=len(tokens), B=B, g=16, M=1024, m=320, tau=900, alpha=0.002)
cuts = nudge_and_merge(cuts, B, L=len(tokens), g=16, radius=32, M=1024, m=320)

# Map token cuts back to character offsets and write chunks
for i in range(len(cuts)-1):
    t0, t1 = cuts[i], cuts[i+1]
    char0 = offsets[t0][0]
    char1 = offsets[t1-1][1] if t1 > t0 else offsets[t0][0]
    chunk_text = text[char0:char1]
    # (Optional) add 64-token overlap when writing if desired
```

---

## Closing checks (so the implementer trusts it)

* **Edge cases:**

  * If the file is shorter than 1024 tokens, you get one chunk (the table backtrack will naturally return `[0, L]`).
  * If content is flat (base64), `B(t)` stays near zero; `choose_cuts` will space cuts near the soft target and never exceed 1024.
  * Single long line or no spaces does not matter; windows are by **tokens**, not by lines.
  * Code works without parsers; when blocks change, embeddings change; when they do not, the method falls back to size.

* **Determinism:** No randomness. Same inputs produce the same embeddings and the same cuts.

* **Speed:** Grid step of 16 means each “look back” checks at most `1024/16 = 64` positions. The scorer uses tiny sets (`h=5`, `k=4`) so cosine work is constant time per candidate. This scales to very large books.

* **Most important promise:** the chooser **enforces** the 1024-token maximum in the selection step itself. There is no way to output an illegal chunk.

This is everything needed to build it, reason about it, and be confident it will behave well in both rich and hostile text.
