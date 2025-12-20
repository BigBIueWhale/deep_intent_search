#!/usr/bin/env python3
"""
Generate an A4 printable PDF with line numbers from ./yellow_marker/000000.txt, 000001.txt, ...
- Mixed Hebrew/English rendering with simple built-in BiDi reordering
- <mark-yellow>...</mark-yellow> sections are rendered in bold (tags removed)
Output: ./printable.pdf
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ----------------------------
# Config
# ----------------------------

# Try FreeMono first (good Hebrew coverage in many Linux distros). Fallback to DejaVuSansMono.
FONT_CANDIDATES = [
    ("FreeMono", "FreeMonoBold",
     "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
     "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"),
    ("DejaVuSansMono", "DejaVuSansMono-Bold",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
]

MARK_TAG_RE = re.compile(r"<\s*(/?)\s*mark-yellow\s*>", flags=re.IGNORECASE)

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
LATIN_RE = re.compile(r"[A-Za-z]")


# ----------------------------
# Minimal BiDi reordering (Hebrew/English + digits)
# ----------------------------

STRONG = {"L", "R", "AL"}
NEUTRALS = {"WS", "ON", "B", "S", "BN"}  # include BN
MIRROR_MAP = {
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
    "{": "}",
    "}": "{",
    "<": ">",
    ">": "<",
}

@lru_cache(maxsize=4096)
def bidi_class(ch: str) -> str:
    bc = unicodedata.bidirectional(ch)
    return bc if bc else "L"

def _resolve_types(types: List[str], base_dir: str) -> List[str]:
    # W1: NSM -> previous type
    prev = "L" if base_dir == "L" else "R"
    for i, t in enumerate(types):
        if t == "NSM":
            types[i] = prev
        else:
            if t in STRONG or t in {"EN", "AN"}:
                prev = "R" if t == "AL" else t

    # W2: EN after AL -> AN
    prev_strong = "L" if base_dir == "L" else "R"
    for i, t in enumerate(types):
        if t in STRONG:
            prev_strong = "R" if t == "AL" else t
        elif t == "EN" and prev_strong == "AL":
            types[i] = "AN"

    # W3: AL -> R
    types = ["R" if t == "AL" else t for t in types]

    # W4: ES/CS in number runs
    for i, t in enumerate(types):
        if t == "ES":
            if 0 < i < len(types) - 1 and types[i - 1] == "EN" and types[i + 1] == "EN":
                types[i] = "EN"
        elif t == "CS":
            if 0 < i < len(types) - 1:
                if types[i - 1] == "EN" and types[i + 1] == "EN":
                    types[i] = "EN"
                elif types[i - 1] == "AN" and types[i + 1] == "AN":
                    types[i] = "AN"

    # W5: ET adjacent to EN -> EN
    i = 0
    while i < len(types):
        if types[i] == "ET":
            start = i
            while i < len(types) and types[i] == "ET":
                i += 1
            left = start - 1
            right = i
            if (left >= 0 and types[left] == "EN") or (right < len(types) and types[right] == "EN"):
                for j in range(start, i):
                    types[j] = "EN"
        else:
            i += 1

    # W6: remaining ES/ET/CS -> ON
    types = ["ON" if t in {"ES", "ET", "CS"} else t for t in types]

    # W7: EN with previous strong L -> L
    prev_strong = "L" if base_dir == "L" else "R"
    for i, t in enumerate(types):
        if t in {"L", "R"}:
            prev_strong = t
        elif t == "EN" and prev_strong == "L":
            types[i] = "L"

    return types

def _resolve_neutrals(types: List[str], base_dir: str) -> List[str]:
    base_type = "L" if base_dir == "L" else "R"
    n = len(types)
    i = 0
    while i < n:
        if types[i] in NEUTRALS or types[i] == "ON":
            start = i
            while i < n and (types[i] in NEUTRALS or types[i] == "ON"):
                i += 1
            end = i

            # Find left/right strong types (L/R)
            left = start - 1
            while left >= 0 and types[left] not in {"L", "R"}:
                left -= 1

            right = end
            while right < n and types[right] not in {"L", "R"}:
                right += 1

            left_type = types[left] if left >= 0 else base_type
            right_type = types[right] if right < n else base_type

            resolved = left_type if left_type == right_type else base_type
            for j in range(start, end):
                types[j] = resolved
        else:
            i += 1
    return types

def _assign_levels(types: List[str], base_dir: str) -> List[int]:
    base_level = 0 if base_dir == "L" else 1
    levels = [base_level] * len(types)
    for i, t in enumerate(types):
        if base_level % 2 == 0:  # L base
            if t == "R":
                levels[i] = base_level + 1
            elif t in {"AN", "EN"}:
                levels[i] = base_level + 2
            else:
                levels[i] = base_level
        else:  # R base
            if t == "L":
                levels[i] = base_level + 1
            elif t in {"AN", "EN"}:
                levels[i] = base_level + 1
            else:
                levels[i] = base_level
    return levels

def bidi_display(s: str, base_dir: str) -> str:
    """Return visually-ordered string for simple Hebrew/English lines."""
    if not s:
        return s
    chars = list(s)
    types = [bidi_class(ch) for ch in chars]
    types = _resolve_types(types, base_dir)
    types = _resolve_neutrals(types, base_dir)
    levels = _assign_levels(types, base_dir)

    # Mirror brackets on RTL levels
    for i, ch in enumerate(chars):
        if levels[i] % 2 == 1 and ch in MIRROR_MAP:
            chars[i] = MIRROR_MAP[ch]

    max_level = max(levels) if levels else 0
    # Reverse runs from max down to 1
    for lvl in range(max_level, 0, -1):
        i = 0
        while i < len(chars):
            if levels[i] >= lvl:
                start = i
                i += 1
                while i < len(chars) and levels[i] >= lvl:
                    i += 1
                chars[start:i] = chars[start:i][::-1]
                levels[start:i] = levels[start:i][::-1]
            else:
                i += 1

    return "".join(chars)


# ----------------------------
# Parsing <mark-yellow> to bold segments
# ----------------------------

Segment = Tuple[str, bool]  # (text, is_bold)

def parse_mark_segments(line: str, mark_level: int) -> Tuple[List[Segment], int]:
    """Parse a single source line; remove tags and return bold segments; update mark_level."""
    segs: List[Segment] = []
    last = 0
    for m in MARK_TAG_RE.finditer(line):
        if m.start() > last:
            t = line[last:m.start()].replace("\t", "    ")
            if t:
                segs.append((t, mark_level > 0))
        if m.group(1) == "/":
            mark_level = max(0, mark_level - 1)
        else:
            mark_level += 1
        last = m.end()

    if last < len(line):
        t = line[last:].replace("\t", "    ")
        if t:
            segs.append((t, mark_level > 0))

    return segs, mark_level

def wrap_segments(segs: List[Segment], width_chars: int) -> List[List[Segment]]:
    """Wrap by fixed character count (monospace assumption), preserving bold segments."""
    out: List[List[Segment]] = []
    cur: List[Segment] = []
    cur_len = 0

    for text, bold in segs:
        pos = 0
        while pos < len(text):
            space = width_chars - cur_len
            if space <= 0:
                out.append(cur)
                cur = []
                cur_len = 0
                space = width_chars

            take = text[pos:pos + space]
            if take:
                if cur and cur[-1][1] == bold:
                    cur[-1] = (cur[-1][0] + take, bold)
                else:
                    cur.append((take, bold))
                cur_len += len(take)

            pos += len(take)

    if cur or not out:
        out.append(cur)
    return out


# Toggle marker to preserve bold boundaries through bidi reordering.
# Use a Private Use Area code point very unlikely to occur in your text.
TOGGLE = "\uE000"

def make_toggled_string(line_segs: List[Segment]) -> str:
    out: List[str] = []
    bold = False
    for t, is_bold in line_segs:
        if not t:
            continue
        if is_bold != bold:
            out.append(TOGGLE)
            bold = is_bold
        out.append(t)
    if bold:
        out.append(TOGGLE)
    return "".join(out)

def parse_visual_segments(visual: str) -> List[Segment]:
    segs: List[Segment] = []
    bold = False
    buf: List[str] = []
    for ch in visual:
        if ch == TOGGLE:
            if buf:
                segs.append(("".join(buf), bold))
                buf = []
            bold = not bold
        else:
            buf.append(ch)
    if buf:
        segs.append(("".join(buf), bold))
    return segs


# ----------------------------
# Direction detection (same heuristic as before)
# ----------------------------

def classify_dir(s: str) -> str:
    """Return 'R' for RTL paragraph, else 'L'."""
    if not s:
        return "L"
    heb = len(HEBREW_RE.findall(s))
    lat = len(LATIN_RE.findall(s))
    m = re.search(r"[\u0590-\u05FFA-Za-z]", s)
    first = m.group(0) if m else ""
    first_is_hebrew = bool(first and HEBREW_RE.match(first))
    if heb > 0 and (first_is_hebrew or (heb >= max(3, int(0.35 * (heb + lat + 1))))):
        return "R"
    return "L"


# ----------------------------
# PDF rendering
# ----------------------------

@dataclass
class PdfStyle:
    font_reg: str
    font_bold: str
    font_size: float = 9.3
    leading: float = 11.4
    left_margin: float = 16 * mm
    right_margin: float = 14 * mm
    top_margin: float = 16 * mm
    bottom_margin: float = 16 * mm
    header_h: float = 10 * mm
    footer_h: float = 8 * mm

def register_fonts() -> Tuple[str, str]:
    for reg_name, bold_name, reg_path, bold_path in FONT_CANDIDATES:
        if Path(reg_path).exists() and Path(bold_path).exists():
            if reg_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(reg_name, reg_path))
            if bold_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            return reg_name, bold_name
    raise RuntimeError(
        "Could not find a Hebrew-capable monospace font on this system.\n"
        "Tried:\n"
        + "\n".join([f"  - {r} / {b}" for _, _, r, b in FONT_CANDIDATES])
        + "\nInstall FreeMono (freefont) or DejaVuSansMono, or edit FONT_CANDIDATES."
    )

def seg_width(text: str, bold: bool, style: PdfStyle) -> float:
    return pdfmetrics.stringWidth(text, style.font_bold if bold else style.font_reg, style.font_size)

def draw_header(c: canvas.Canvas, style: PdfStyle, title: str, stamp: str, page_num: int, page_w: float, page_h: float):
    c.saveState()
    c.setFont(style.font_reg, 8.6)
    c.drawString(style.left_margin, page_h - style.top_margin + 2, f"{title}  •  Generated {stamp}")
    c.drawRightString(page_w - style.right_margin, page_h - style.top_margin + 2, f"Page {page_num}")
    c.setLineWidth(0.3)
    c.line(style.left_margin, page_h - style.top_margin - 1, page_w - style.right_margin, page_h - style.top_margin - 1)
    c.restoreState()

def draw_footer(c: canvas.Canvas, style: PdfStyle, page_num: int, page_w: float):
    c.saveState()
    c.setFont(style.font_reg, 8.2)
    c.drawCentredString(page_w / 2, style.bottom_margin + 2, f"{page_num}")
    c.restoreState()

def draw_line_number(c: canvas.Canvas, style: PdfStyle, digits: int, char_w: float, n: int, y: float):
    c.saveState()
    c.setFont(style.font_reg, style.font_size)
    c.setFillGray(0.45)
    c.drawRightString(style.left_margin + (digits * char_w), y, str(n).rjust(digits))
    c.setFillGray(0.65)
    c.drawString(style.left_margin + (digits * char_w), y, " | ")
    c.restoreState()

def draw_continuation_gutter(c: canvas.Canvas, style: PdfStyle, digits: int, char_w: float, y: float):
    c.saveState()
    c.setFont(style.font_reg, style.font_size)
    c.setFillGray(0.85)
    c.drawString(style.left_margin + (digits * char_w), y, " | ")
    c.restoreState()

def draw_visual_line(
    c: canvas.Canvas,
    style: PdfStyle,
    visual_segs: List[Segment],
    dir_: str,
    y: float,
    content_x: float,
    content_right_x: float,
):
    total_w = sum(seg_width(t, b, style) for t, b in visual_segs)
    x = content_x if dir_ == "L" else (content_right_x - total_w)

    for t, b in visual_segs:
        if not t:
            continue
        c.setFont(style.font_bold if b else style.font_reg, style.font_size)
        c.drawString(x, y, t)
        x += seg_width(t, b, style)


# ----------------------------
# File collection
# ----------------------------

def collect_yellow_marker_files(folder: Path) -> List[Path]:
    # Prefer exactly 6-digit names (000000.txt, ...)
    numeric = sorted(folder.glob("[0-9][0-9][0-9][0-9][0-9][0-9].txt"))
    if numeric:
        return numeric
    # Fallback: all .txt sorted
    return sorted(folder.glob("*.txt"))

def read_all_lines(files: Sequence[Path]) -> List[str]:
    lines: List[str] = []
    for p in files:
        txt = p.read_text(encoding="utf-8", errors="replace")
        lines.extend(txt.splitlines())
    return lines


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="./yellow_marker", help="Folder containing 000000.txt, 000001.txt, ...")
    ap.add_argument("--out", default="./printable.pdf", help="Output PDF path")
    args = ap.parse_args()

    folder = Path(args.folder)
    out_pdf = Path(args.out)

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    files = collect_yellow_marker_files(folder)
    if not files:
        raise SystemExit(f"No .txt files found in: {folder}")

    # Pre-read all lines so we can compute line-number width (digits)
    src_lines = read_all_lines(files)
    total_lines = len(src_lines)
    digits = len(str(total_lines))

    # Fonts
    font_reg, font_bold = register_fonts()
    style = PdfStyle(font_reg=font_reg, font_bold=font_bold)

    # Page geometry
    page_w, page_h = A4
    body_top_y = page_h - style.top_margin - style.header_h
    body_bottom_y = style.bottom_margin + style.footer_h

    # Wrapping (monospace estimate)
    char_w = pdfmetrics.stringWidth("M", style.font_reg, style.font_size)
    ln_col_chars = digits + 3  # digits + " | "
    ln_col_w = ln_col_chars * char_w

    content_x = style.left_margin + ln_col_w
    content_right_x = page_w - style.right_margin
    content_w = content_right_x - content_x
    max_chars = max(10, int(content_w // char_w))

    # Title
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"{folder.as_posix()} ({len(files)} files)"

    # Render
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    page_num = 1
    draw_header(c, style, title, stamp, page_num, page_w, page_h)
    y = body_top_y

    mark_level = 0

    for line_no, src_line in enumerate(src_lines, start=1):
        segs, mark_level = parse_mark_segments(src_line, mark_level)
        clean_text = "".join(t for t, _ in segs)
        dir_ = classify_dir(clean_text)

        wrapped = wrap_segments(segs, max_chars)
        for j, line_segs in enumerate(wrapped):
            if y < body_bottom_y:
                draw_footer(c, style, page_num, page_w)
                c.showPage()
                page_num += 1
                draw_header(c, style, title, stamp, page_num, page_w, page_h)
                y = body_top_y

            if j == 0:
                draw_line_number(c, style, digits, char_w, line_no, y)
            else:
                draw_continuation_gutter(c, style, digits, char_w, y)

            # Blank (still consumes a line visually)
            if not line_segs:
                y -= style.leading
                continue

            toggled = make_toggled_string(line_segs)
            visual = bidi_display(toggled, "R" if dir_ == "R" else "L")
            visual_segs = parse_visual_segments(visual)

            draw_visual_line(c, style, visual_segs, dir_, y, content_x, content_right_x)
            y -= style.leading

    draw_footer(c, style, page_num, page_w)
    c.save()

    print(f"Wrote: {out_pdf.resolve()}  (source lines: {total_lines}, pages: {page_num})")


if __name__ == "__main__":
    main()
