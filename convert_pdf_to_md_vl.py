# convert_pdf_to_md_vl.py
#
# Opinionated PDF -> 4K page PNGs -> qwen3-vl:32b-thinking (Ollama) -> Markdown
#
# Usage:
#   python convert_pdf_to_md_vl.py /path/to/input.pdf
#
# Outputs (hard-coded):
#   conversion/output.md
#   conversion/progress.jsonl
#   conversion/images/page_0001.png, ...
#
# Dependencies:
#   pip install pymupdf pillow python-dotenv httpx
#
# Repo dependency:
#   from core.llm import get_client, chat_complete, print_stats

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'pymupdf'. Install with: pip install pymupdf\n"
        f"Import error: {e}"
    )

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'Pillow'. Install with: pip install pillow\n"
        f"Import error: {e}"
    )

from core.llm import get_client, chat_complete, print_stats


# -----------------------------
# Opinionated constants
# -----------------------------

SCHEMA_VERSION = 1

CONVERSION_DIR = Path("conversion_vl")
PROGRESS_PATH = CONVERSION_DIR / "progress.jsonl"
OUTPUT_MD_PATH = CONVERSION_DIR / "output.md"
IMAGES_DIR = CONVERSION_DIR / "images"

TARGET_LONG_SIDE_PX = 4096
SUPERSAMPLE_FACTOR = 2.0  # render 2x, then downsample for crisp text/lines

IMAGE_FMT = "PNG"
IMAGE_EXT = ".png"
PNG_COMPRESS_LEVEL = 6  # balanced: not too slow, still smaller

MAX_LLM_ATTEMPTS = 3

# Sanity bounds (runaway / nonsense detection)
MAX_MD_CHARS_PER_PAGE = 400_000


MD_START_RE = re.compile(r"<<<MD_START\s+page=(\d{4})>>>")
MD_END_RE = re.compile(r"<<<MD_END\s+page=(\d{4})>>>")


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()

def atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", errors="strict")
    os.replace(tmp, path)

def ensure_dirs() -> None:
    CONVERSION_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def b64_of_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")

def _parse_jsonl_line(line: str, lineno: int) -> dict:
    try:
        obj = json.loads(line)
    except Exception as e:
        raise RuntimeError(f"progress.jsonl parse error at line {lineno}: {e}\nLine: {line[:2000]}")
    if not isinstance(obj, dict):
        raise RuntimeError(f"progress.jsonl line {lineno} is not a JSON object.")
    return obj


# -----------------------------
# Progress model / strict parsing
# -----------------------------

@dataclass(frozen=True)
class Meta:
    schema_version: int
    created_utc: str
    input_pdf_path: str
    input_pdf_sha256: str
    input_pdf_bytes: int
    page_count: int
    target_long_side_px: int
    supersample_factor: float
    image_format: str
    ollama_host: str | None

@dataclass(frozen=True)
class PageDone:
    page_index: int
    page_number: int
    image_path: str
    image_sha256: str
    image_w: int
    image_h: int
    md_sha256: str
    md: str
    attempts: int
    utc: str


def write_initial_meta(meta: Meta) -> None:
    ensure_dirs()
    meta_obj = {
        "type": "meta",
        "schema_version": meta.schema_version,
        "created_utc": meta.created_utc,
        "input_pdf": {
            "path": meta.input_pdf_path,
            "sha256": meta.input_pdf_sha256,
            "bytes": meta.input_pdf_bytes,
            "page_count": meta.page_count,
        },
        "rendering": {
            "target_long_side_px": meta.target_long_side_px,
            "supersample_factor": meta.supersample_factor,
            "image_format": meta.image_format,
        },
        "ollama": {
            "host": meta.ollama_host,
        },
    }
    with PROGRESS_PATH.open("w", encoding="utf-8", errors="strict") as f:
        f.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_progress_or_init(meta_expected: Meta) -> Tuple[Meta, Dict[int, PageDone]]:
    """
    Returns:
      - validated meta
      - completed pages mapping: page_index -> PageDone
    """
    completed: Dict[int, PageDone] = {}

    if not PROGRESS_PATH.exists():
        write_initial_meta(meta_expected)
        return meta_expected, completed

    lines = PROGRESS_PATH.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines:
        raise RuntimeError("progress.jsonl exists but is empty. Refusing to proceed.")

    meta_line = _parse_jsonl_line(lines[0], 1)
    if meta_line.get("type") != "meta":
        raise RuntimeError("progress.jsonl first line must be meta object with type='meta'.")

    def require(obj: dict, key: str, where: str) -> Any:
        if key not in obj:
            raise RuntimeError(f"progress.jsonl meta missing key '{key}' at {where}.")
        return obj[key]

    schema_version = require(meta_line, "schema_version", "root")
    if schema_version != meta_expected.schema_version:
        raise RuntimeError(f"schema_version mismatch: file={schema_version}, expected={meta_expected.schema_version}")

    in_pdf = require(meta_line, "input_pdf", "root")
    rendering = require(meta_line, "rendering", "root")
    llm = require(meta_line, "llm", "root")

    file_path = require(in_pdf, "path", "input_pdf")
    file_sha = require(in_pdf, "sha256", "input_pdf")
    file_bytes = require(in_pdf, "bytes", "input_pdf")
    file_pages = require(in_pdf, "page_count", "input_pdf")

    tlong = require(rendering, "target_long_side_px", "rendering")
    ss = require(rendering, "supersample_factor", "rendering")
    imgfmt = require(rendering, "image_format", "rendering")

    mismatches = []
    if file_path != meta_expected.input_pdf_path:
        mismatches.append(f"input_pdf.path file='{file_path}' expected='{meta_expected.input_pdf_path}'")
    if file_sha != meta_expected.input_pdf_sha256:
        mismatches.append(f"input_pdf.sha256 file='{file_sha}' expected='{meta_expected.input_pdf_sha256}'")
    if file_bytes != meta_expected.input_pdf_bytes:
        mismatches.append(f"input_pdf.bytes file='{file_bytes}' expected='{meta_expected.input_pdf_bytes}'")
    if file_pages != meta_expected.page_count:
        mismatches.append(f"input_pdf.page_count file='{file_pages}' expected='{meta_expected.page_count}'")
    if int(tlong) != meta_expected.target_long_side_px:
        mismatches.append(f"rendering.target_long_side_px file='{tlong}' expected='{meta_expected.target_long_side_px}'")
    if float(ss) != float(meta_expected.supersample_factor):
        mismatches.append(f"rendering.supersample_factor file='{ss}' expected='{meta_expected.supersample_factor}'")
    if imgfmt != meta_expected.image_format:
        mismatches.append(f"rendering.image_format file='{imgfmt}' expected='{meta_expected.image_format}'")

    if mismatches:
        raise RuntimeError("progress.jsonl meta mismatch:\n- " + "\n- ".join(mismatches))

    # Parse page_done entries
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        obj = _parse_jsonl_line(line, lineno)
        if obj.get("type") != "page_done":
            raise RuntimeError(f"Unexpected event type at line {lineno}: {obj.get('type')} (only page_done allowed).")

        for k in ["page_index", "page_number", "image", "md", "md_sha256", "attempts", "utc"]:
            if k not in obj:
                raise RuntimeError(f"progress.jsonl line {lineno} page_done missing key '{k}'.")

        img = obj["image"]
        if not isinstance(img, dict):
            raise RuntimeError(f"progress.jsonl line {lineno} image must be an object.")
        for k in ["path", "sha256", "w", "h"]:
            if k not in img:
                raise RuntimeError(f"progress.jsonl line {lineno} image missing key '{k}'.")

        page_index = int(obj["page_index"])
        if page_index in completed:
            raise RuntimeError(f"Duplicate page_done entry for page_index={page_index} (line {lineno}).")

        pd = PageDone(
            page_index=page_index,
            page_number=int(obj["page_number"]),
            image_path=str(img["path"]),
            image_sha256=str(img["sha256"]),
            image_w=int(img["w"]),
            image_h=int(img["h"]),
            md_sha256=str(obj["md_sha256"]),
            md=str(obj["md"]),
            attempts=int(obj["attempts"]),
            utc=str(obj["utc"]),
        )
        completed[page_index] = pd

    # Validate referenced images and hashes
    for pi, pd in completed.items():
        if pi < 0 or pi >= meta_expected.page_count:
            raise RuntimeError(f"progress.jsonl contains page_index out of range: {pi} (page_count={meta_expected.page_count})")
        img_path = Path(pd.image_path)
        if not img_path.exists():
            raise RuntimeError(f"progress.jsonl references missing image file: {img_path}")
        with Image.open(img_path) as im:
            w, h = im.size
        if (w, h) != (pd.image_w, pd.image_h):
            raise RuntimeError(
                f"Image dimension mismatch for {img_path}: progress has ({pd.image_w},{pd.image_h}) "
                f"but file is ({w},{h})"
            )
        if max(w, h) != TARGET_LONG_SIDE_PX:
            raise RuntimeError(f"Image {img_path} does not have long side {TARGET_LONG_SIDE_PX}px. Got {w}x{h}.")
        actual_hash = sha256_file(img_path)
        if actual_hash != pd.image_sha256:
            raise RuntimeError(
                f"Image sha256 mismatch for {img_path}:\n"
                f"progress={pd.image_sha256}\nactual={actual_hash}"
            )

        if sha256_text(pd.md) != pd.md_sha256:
            raise RuntimeError(f"Markdown sha256 mismatch for page_index={pi} (progress does not match content).")

    # Ensure output.md is derived from progress (rebuild deterministically)
    rebuilt = build_output_md(meta_expected, completed)
    if OUTPUT_MD_PATH.exists():
        existing = OUTPUT_MD_PATH.read_text(encoding="utf-8", errors="strict")
        if existing != rebuilt:
            atomic_write_text(OUTPUT_MD_PATH, rebuilt)
    else:
        atomic_write_text(OUTPUT_MD_PATH, rebuilt)

    return meta_expected, completed


def append_progress_page_done(pd: PageDone) -> None:
    obj = {
        "type": "page_done",
        "page_index": pd.page_index,
        "page_number": pd.page_number,
        "image": {
            "path": pd.image_path,
            "sha256": pd.image_sha256,
            "w": pd.image_w,
            "h": pd.image_h,
        },
        "md": pd.md,
        "md_sha256": pd.md_sha256,
        "attempts": pd.attempts,
        "utc": pd.utc,
    }
    with PROGRESS_PATH.open("a", encoding="utf-8", errors="strict") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_output_md(meta: Meta, completed: Dict[int, PageDone]) -> str:
    parts: List[str] = []
    parts.append(
        "<!--\n"
        "  GENERATED FILE — DO NOT EDIT BY HAND\n"
        f"  input_pdf_sha256: {meta.input_pdf_sha256}\n"
        f"  page_count: {meta.page_count}\n"
        f"  renderer: long_side={meta.target_long_side_px}px supersample={meta.supersample_factor}\n"
        f"  images: {meta.image_format}\n"
        "-->\n"
    )
    for page_index in sorted(completed.keys()):
        pd = completed[page_index]
        parts.append(f"\n\n<!-- BEGIN_PAGE {pd.page_number:04d} -->\n")
        parts.append(pd.md.rstrip() + "\n")
        parts.append(f"<!-- END_PAGE {pd.page_number:04d} -->\n")
    return "".join(parts)


# -----------------------------
# Rendering (PDF page -> crisp 4K PNG)
# -----------------------------

def render_page_to_4k_png(doc: fitz.Document, page_index: int, out_path: Path) -> Tuple[int, int]:
    page = doc.load_page(page_index)
    rect = page.rect  # points (1/72 inch)
    long_pt = max(rect.width, rect.height)
    if long_pt <= 0:
        raise RuntimeError(f"Page {page_index} has invalid dimensions: {rect}")

    # Render at (4096 * supersample) on the long side, then downsample to exactly 4096.
    target_super = int(round(TARGET_LONG_SIDE_PX * SUPERSAMPLE_FACTOR))
    zoom = target_super / long_pt
    mat = fitz.Matrix(zoom, zoom)

    # Render RGB, no alpha.
    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
    if pix.width <= 0 or pix.height <= 0:
        raise RuntimeError(f"Rendered pixmap invalid for page {page_index}: {pix.width}x{pix.height}")

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    w, h = img.size
    long_px = max(w, h)
    if long_px != TARGET_LONG_SIDE_PX:
        scale = TARGET_LONG_SIDE_PX / float(long_px)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w <= 0 or new_h <= 0:
            raise RuntimeError(f"Invalid resize target for page {page_index}: {new_w}x{new_h}")
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    img.save(out_path, format=IMAGE_FMT, compress_level=PNG_COMPRESS_LEVEL, optimize=True)

    with Image.open(out_path) as verify:
        vw, vh = verify.size
    if max(vw, vh) != TARGET_LONG_SIDE_PX:
        raise RuntimeError(
            f"Post-save verification failed for {out_path}: got {vw}x{vh}, expected long side {TARGET_LONG_SIDE_PX}px"
        )
    return vw, vh


def ensure_page_image(doc: fitz.Document, page_index: int) -> Tuple[Path, str, int, int]:
    img_path = IMAGES_DIR / f"page_{page_index+1:04d}{IMAGE_EXT}"

    if img_path.exists():
        with Image.open(img_path) as im:
            w, h = im.size
        if max(w, h) != TARGET_LONG_SIDE_PX:
            raise RuntimeError(f"Existing image has wrong size: {img_path} ({w}x{h})")
        img_hash = sha256_file(img_path)
        return img_path, img_hash, w, h

    w, h = render_page_to_4k_png(doc, page_index, img_path)
    img_hash = sha256_file(img_path)
    return img_path, img_hash, w, h


# -----------------------------
# LLM prompting / parsing
# -----------------------------

def system_prompt() -> str:
    # Minimal, per your preference. Do not rely on system prompt behavior for qwen3-vl.
    return "Convert the given page image into Markdown."

def user_prompt(page_number_4d: str, attempt: int) -> str:
    # Tight, page-scoped, marker-framed output.
    # Attempt 2/3 slightly tighten behavior without adding “options”.
    extra = ""
    if attempt == 2:
        extra = (
            "\nCRITICAL:\n"
            "- Output NOTHING outside the markers.\n"
            "- Do not include any extra commentary.\n"
        )
    elif attempt >= 3:
        extra = (
            "\nCRITICAL:\n"
            "- Output NOTHING outside the markers.\n"
            "- If text is unreadable, write [illegible]. Do not refuse.\n"
            "- Do not repeat content.\n"
        )

    return (
        f"Convert this single PAGE IMAGE into Markdown.\n"
        f"Page number: {page_number_4d}\n\n"
        "Rules:\n"
        "- Preserve reading order.\n"
        "- Preserve headings, lists, tables (Markdown tables), and code blocks.\n"
        "- Do not invent content not visible on the page.\n"
        "- Describe images/figures on the page in great detail:\n"
        "  * Insert: ![ALT TEXT](#) where the figure appears.\n"
        "  * Immediately after, add:\n"
        "    > **Image description:** ...\n"
        "- If the page is blank, output exactly:\n"
        "  <!-- PAGE BLANK -->\n\n"
        "Output format (MUST match exactly):\n"
        f"<<<MD_START page={page_number_4d}>>>\n"
        "(Markdown for this page only)\n"
        f"<<<MD_END page={page_number_4d}>>>\n"
        + extra
    )

def extract_md(response_text: str, page_number_4d: str) -> str:
    starts = [m for m in MD_START_RE.finditer(response_text) if m.group(1) == page_number_4d]
    ends = [m for m in MD_END_RE.finditer(response_text) if m.group(1) == page_number_4d]

    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            f"Expected exactly 1 MD_START and 1 MD_END for page={page_number_4d}, "
            f"got starts={len(starts)} ends={len(ends)}"
        )

    s = starts[0]
    e = ends[0]
    if e.start() <= s.end():
        raise ValueError(f"Markers out of order for page={page_number_4d}.")

    md = response_text[s.end():e.start()].strip("\n")
    return md.strip()

def md_sanity_check(md: str) -> None:
    if len(md) > MAX_MD_CHARS_PER_PAGE:
        raise ValueError(f"Markdown too large for a single page ({len(md)} chars). Likely runaway output.")
    # Accept blank marker
    if md.strip() == "<!-- PAGE BLANK -->":
        return
    # Otherwise require at least *something*.
    if not md.strip():
        raise ValueError("Empty markdown for a non-blank page.")

def llm_convert_page(client_http, image_b64: str, page_number_4d: str) -> Tuple[str, int]:
    """
    Returns (md, attempts_used).
    Retries on marker/validation failures or truncation (done_reason=length).
    """
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        messages = [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": user_prompt(page_number_4d, attempt),
                "images_b64": [image_b64],
            },
        ]

        resp = chat_complete(
            messages=messages,
            role="vision",
            client=client_http,
            max_completion_tokens=34000, # ignored for thinking models; harmless
        )

        stats = print_stats(resp)
        if stats is not None:
            print(stats)
        else:
            print("[warn] stats unavailable (missing duration/count fields)")

        if getattr(resp, "ran_out_of_tokens", False):
            if attempt >= MAX_LLM_ATTEMPTS:
                raise RuntimeError("LLM repeatedly hit done_reason=length. Refusing to proceed.")
            print("[warn] LLM hit done_reason=length; retrying.")
            continue

        text = resp.message.content or ""
        try:
            md = extract_md(text, page_number_4d)
            md_sanity_check(md)
            return md, attempt
        except Exception as e:
            if attempt >= MAX_LLM_ATTEMPTS:
                raise RuntimeError(
                    f"LLM output failed validation after {MAX_LLM_ATTEMPTS} attempts.\n"
                    f"Last error: {e}\n"
                    f"--- Raw output (first 2000 chars) ---\n{text[:2000]}"
                )
            print(f"[warn] Output validation failed (attempt {attempt}/{MAX_LLM_ATTEMPTS}): {e}")
            continue

    raise RuntimeError("Unreachable: retry loop exhausted incorrectly.")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    load_dotenv()

    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python convert_pdf_to_md_vl.py /path/to/input.pdf")

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise RuntimeError(f"Input PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise RuntimeError(f"Input must be a .pdf file. Got: {pdf_path}")

    ensure_dirs()

    client_http = get_client()
    if client_http is None:
        raise RuntimeError("get_client() returned None (unexpected).")

    pdf_bytes = pdf_path.stat().st_size
    pdf_hash = sha256_file(pdf_path)

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    if page_count <= 0:
        raise RuntimeError(f"PDF has no pages: {pdf_path}")

    meta_expected = Meta(
        schema_version=SCHEMA_VERSION,
        created_utc=utc_now_iso(),
        input_pdf_path=str(pdf_path),
        input_pdf_sha256=pdf_hash,
        input_pdf_bytes=pdf_bytes,
        page_count=page_count,
        target_long_side_px=TARGET_LONG_SIDE_PX,
        supersample_factor=SUPERSAMPLE_FACTOR,
        image_format=IMAGE_FMT,
        ollama_host=os.environ.get("OLLAMA_HOST"),
    )

    meta, completed = load_progress_or_init(meta_expected)

    done_pages = set(completed.keys())
    print(f"[info] Input: {pdf_path}")
    print(f"[info] Pages: {page_count}")
    print(f"[info] Resuming: {len(done_pages)} pages already completed")
    print(f"[info] Image format: {IMAGE_FMT} (lossless)")
    print(f"[info] Output: {OUTPUT_MD_PATH}")

    for page_index in range(page_count):
        if page_index in done_pages:
            continue

        page_number = page_index + 1
        page_4d = f"{page_number:04d}"
        print(f"\n[info] Page {page_number}/{page_count}")

        img_path, img_hash, w, h = ensure_page_image(doc, page_index)
        print(f"[info] Image: {img_path} ({w}x{h}) sha256={img_hash[:16]}...")

        img_b64 = b64_of_file(img_path)
        md, attempts_used = llm_convert_page(client_http, img_b64, page_4d)

        pd = PageDone(
            page_index=page_index,
            page_number=page_number,
            image_path=str(img_path),
            image_sha256=img_hash,
            image_w=w,
            image_h=h,
            md_sha256=sha256_text(md),
            md=md,
            attempts=attempts_used,
            utc=utc_now_iso(),
        )

        # Persist progress first (source of truth)
        append_progress_page_done(pd)

        # Update in-memory and rebuild output.md deterministically
        completed[page_index] = pd
        rebuilt = build_output_md(meta, completed)
        atomic_write_text(OUTPUT_MD_PATH, rebuilt)

        print(f"[info] Wrote: {OUTPUT_MD_PATH} (completed {len(completed)}/{page_count})")

    print("\n[done] Conversion complete.")
    print(f"[done] Output: {OUTPUT_MD_PATH}")
    print(f"[done] Progress: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
