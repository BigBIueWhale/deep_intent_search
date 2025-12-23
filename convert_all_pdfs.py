#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_all_pdfs.py

Generic, resumable batch runner for convert_pdf_to_md_vl.py.

Key behavior (kept intentionally strict + robust):
- You provide a REQUIRED base folder (absolute or relative).
- The script recursively finds **visible** PDF files under that base folder,
  in a deterministic order, then runs:

    python3 convert_pdf_to_md_vl.py "/abs/path/to/file.pdf"

  for each PDF, sequentially, streaming combined stdout/stderr to your console
  exactly like a manual run, and also capturing it to a per-PDF log file.

- For EACH PDF:
  1) Delete ./conversion_vl (converter output folder) before running.
  2) Run the converter with streaming output.
  3) On success, move ./conversion_vl -> ./all_pdfs/converted/<PDF_BASENAME>/
     so that folder contains the contents of conversion_vl.

- Resumability via a dedicated file: ./all_pdfs/progress.jsonl (append-only).
  - If you Ctrl+C, the current converter process is terminated, a "cancelled"
    entry is recorded, and the script exits (resume later).
  - On re-run, PDFs whose latest status is "success" are skipped.

- Safety / correctness:
  - The set of PDFs is snapshotted in a manifest entry in progress.jsonl.
    If the scanned PDFs differ from the original manifest, the script refuses
    to resume (to avoid silently converting a different set).
  - If progress.jsonl exists from an older version (no manifest entry),
    the script tries to bootstrap the manifest from existing "pdf" entries.
    If it can't do that safely, it refuses to resume.

- Hidden/visible policy:
  - On macOS/Linux: skips any file/dir where ANY path segment starts with '.'
    (e.g. ".git", ".cache", ".hidden.pdf").
  - On Windows: also skips items marked Hidden/System (best effort).
  - (You said Ubuntu 24.04; this still stays cross-platform and strict.)

Run:
  python3 convert_all_pdfs.py /home/user/Desktop/g_force
  python3 convert_all_pdfs.py /home/user/Desktop/g_force --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ----------------------------
# Errors
# ----------------------------
@dataclass
class ConvertAllError(Exception):
    message: str
    path: Optional[Path] = None
    original: Optional[BaseException] = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.path is not None:
            parts.append(f"Path: {self.path}")
        if self.original is not None:
            parts.append(f"Original: {type(self.original).__name__}: {self.original}")
            err_no = getattr(self.original, "errno", None)
            if err_no is not None:
                parts.append(f"Errno: {err_no}")
        return "\n".join(parts)


# ----------------------------
# Time helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------
# Filesystem helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ConvertAllError("Failed to create directory.", path=p, original=e)


def rmtree_strict(p: Path) -> None:
    """
    Remove a directory tree, failing with a detailed error on the first issue.
    """
    if not p.exists():
        return

    # IMPORTANT: refuse to delete a symlink path (even if it points to a dir).
    # This is a safety guard; converter output folder should be a real dir.
    if p.is_symlink():
        raise ConvertAllError("Refusing to delete: expected a real directory, but found a symlink.", path=p)

    if not p.is_dir():
        raise ConvertAllError("Expected a directory to delete, but found a non-directory.", path=p)

    def _onerror(func, path, exc_info):
        exc = exc_info[1]
        raise ConvertAllError(
            f"Failed while deleting directory tree (operation={getattr(func, '__name__', str(func))}).",
            path=Path(path),
            original=exc,
        )

    try:
        shutil.rmtree(p, onerror=_onerror)
    except ConvertAllError:
        raise
    except Exception as e:
        raise ConvertAllError("Unexpected error while deleting directory tree.", path=p, original=e)


# ----------------------------
# Visibility rules (skip hidden)
# ----------------------------
def _is_hidden_posix(relative_path: Path) -> bool:
    # Treat any dot-segment as hidden (".git", ".DS_Store", etc.)
    return any(part.startswith(".") and part not in (".", "..") for part in relative_path.parts)


def _is_hidden_windows(path: Path, relative_path: Path) -> bool:
    # Also treat dot-segments as hidden on Windows
    if _is_hidden_posix(relative_path):
        return True

    # If we can read Windows file attributes, respect Hidden/System
    try:
        import ctypes  # stdlib

        FILE_ATTRIBUTE_HIDDEN = 0x2
        FILE_ATTRIBUTE_SYSTEM = 0x4

        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == 0xFFFFFFFF:  # INVALID_FILE_ATTRIBUTES
            raise OSError("GetFileAttributesW returned INVALID_FILE_ATTRIBUTES")
        return bool(attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM))
    except Exception:
        # If attributes are unavailable for some reason, fall back to dot-segment logic only.
        return False


def is_visible(base_dir: Path, path: Path) -> bool:
    """
    Returns True if the path is considered "visible" under base_dir.
    (Hidden items are skipped.)
    """
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        # Should never happen because we only generate descendants of base_dir
        raise ConvertAllError("Internal error: encountered path outside base directory.", path=path)

    if os.name == "nt":
        return not _is_hidden_windows(path, rel)
    return not _is_hidden_posix(rel)


# ----------------------------
# Deterministic PDF scan
# ----------------------------
def scan_visible_pdfs(
    base_dir: Path,
    *,
    follow_symlink_dirs: bool = False,
) -> List[Path]:
    """
    Deterministically scans base_dir recursively and returns absolute PDF paths.

    - Deterministic traversal:
      - Directories and entries are sorted by casefolded name, then name.
      - Final PDF list is sorted by relative path (casefolded), then relative path.

    - Strictness:
      - Permission / IO errors raise immediately.
      - Symlinked directories are refused by default (can be allowed via flag).
      - Broken symlinks or special files raise immediately.

    - "Visible" means no dot-segments (and Windows hidden/system) in the path.
    """
    if not base_dir.exists():
        raise ConvertAllError("Base directory does not exist.", path=base_dir)
    if not base_dir.is_dir():
        raise ConvertAllError("Base path exists but is not a directory.", path=base_dir)

    pdfs: List[Path] = []
    stack: List[Path] = [base_dir]

    while stack:
        current_dir = stack.pop()

        # Skip hidden directories (except the root base_dir itself)
        if current_dir != base_dir and not is_visible(base_dir, current_dir):
            continue

        try:
            with os.scandir(current_dir) as it:
                entries = sorted(it, key=lambda e: (e.name.casefold(), e.name))
        except OSError as e:
            raise ConvertAllError("Failed to read directory (permission or IO error).", path=current_dir, original=e)

        for entry in entries:
            p = Path(entry.path)

            # Apply visibility check early (for both files and directories)
            if p != base_dir and not is_visible(base_dir, p):
                continue

            try:
                is_symlink = entry.is_symlink()
            except OSError as e:
                raise ConvertAllError("Failed to determine whether entry is a symlink.", path=p, original=e)

            if is_symlink:
                # Identify what the symlink points to without following recursively by default.
                try:
                    is_dir_nofollow = entry.is_dir(follow_symlinks=False)
                    is_file_nofollow = entry.is_file(follow_symlinks=False)
                except OSError as e:
                    raise ConvertAllError("Failed to stat symlink (might be broken).", path=p, original=e)

                if is_dir_nofollow:
                    if not follow_symlink_dirs:
                        raise ConvertAllError(
                            "Encountered a symlinked directory. Refusing to traverse it by default.",
                            path=p,
                        )
                    stack.append(p)
                    continue

                if is_file_nofollow:
                    if p.suffix.lower() == ".pdf":
                        pdfs.append(p.resolve())
                    continue

                raise ConvertAllError(
                    "Encountered a symlink that is neither a file nor a directory (broken or special).",
                    path=p,
                )

            # Non-symlink paths
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(p)
                    continue

                if entry.is_file(follow_symlinks=False):
                    if p.suffix.lower() == ".pdf":
                        pdfs.append(p.resolve())
                    continue

            except OSError as e:
                raise ConvertAllError("Failed while checking entry type.", path=p, original=e)

            # Special files: sockets, FIFOs, devices, etc.
            raise ConvertAllError(
                "Encountered an unexpected non-file, non-directory entry (special file).",
                path=p,
            )

    # Deterministic final order by RELATIVE path (case-insensitive primary sort)
    def _rel_sort_key(abs_path: Path) -> Tuple[str, str]:
        rel = abs_path.relative_to(base_dir.resolve())
        s = str(rel)
        return (s.casefold(), s)

    return sorted(pdfs, key=_rel_sort_key)


# ----------------------------
# Progress + manifest (JSONL)
# ----------------------------
def append_progress(progress_jsonl: Path, entry: dict) -> None:
    """
    Append one JSON object line to progress.jsonl, fsync for durability.
    """
    try:
        with progress_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        raise ConvertAllError("Failed to append to progress.jsonl.", path=progress_jsonl, original=e)


def load_progress_lines(progress_jsonl: Path) -> List[dict]:
    """
    Loads all JSONL entries. Fail-fast on invalid JSON or malformed lines.
    """
    if not progress_jsonl.exists():
        return []
    if not progress_jsonl.is_file():
        raise ConvertAllError("progress.jsonl exists but is not a file.", path=progress_jsonl)

    out: List[dict] = []
    try:
        with progress_jsonl.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ConvertAllError(f"Invalid JSON in progress.jsonl at line {i}.", path=progress_jsonl, original=e)
                if not isinstance(obj, dict):
                    raise ConvertAllError(f"Invalid progress entry type at line {i} (expected JSON object).", path=progress_jsonl)
                out.append(obj)
    except OSError as e:
        raise ConvertAllError("Failed to read progress.jsonl.", path=progress_jsonl, original=e)

    return out


def load_latest_status(progress_entries: List[dict]) -> Dict[str, str]:
    """
    Returns mapping: pdf_abs_path -> latest_status
    Ignores non-conversion entries (e.g. manifest).
    """
    latest: Dict[str, Tuple[int, str]] = {}  # pdf -> (line_index, status)
    for i, obj in enumerate(progress_entries, start=1):
        # Conversion entries have "pdf" and "status". Manifest uses type="manifest".
        if obj.get("type") == "manifest":
            continue
        pdf = obj.get("pdf")
        status = obj.get("status")
        if pdf is None or status is None:
            continue
        if not isinstance(pdf, str) or not isinstance(status, str):
            raise ConvertAllError(
                f"Malformed progress entry at logical line {i} (expected string fields: pdf, status).",
                path=None,
            )
        latest[pdf] = (i, status)

    return {pdf: status for pdf, (_, status) in latest.items()}


def compute_manifest(base_dir_abs: Path, pdfs_abs: List[Path]) -> dict:
    """
    Creates a manifest record that binds:
      - the base directory
      - the exact ordered list of PDFs (absolute)
      - a sha256 fingerprint
    """
    pdf_strs = [str(p) for p in pdfs_abs]
    payload = "\n".join(pdf_strs).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()

    return {
        "type": "manifest",
        "ts_utc": utc_now_iso(),
        "base_dir": str(base_dir_abs),
        "pdf_count": len(pdf_strs),
        "pdfs_sha256": sha,
        "pdfs": pdf_strs,  # store the exact list for precise mismatch diagnostics
        "script": "convert_all_pdfs.py",
        "manifest_version": 1,
    }


def find_manifest(progress_entries: List[dict]) -> Optional[dict]:
    """
    Returns the FIRST manifest entry if present.
    (First is considered canonical; later runs should not change it.)
    """
    for obj in progress_entries:
        if obj.get("type") == "manifest":
            return obj
    return None


def infer_manifest_from_progress(progress_entries: List[dict]) -> Optional[List[str]]:
    """
    Best-effort bootstrap if progress.jsonl was created by an older script version
    (no manifest line). We infer the intended PDF set from the unique "pdf" fields
    present in progress entries.

    Returns a sorted list of absolute PDF path strings, or None if not possible.
    """
    seen: Set[str] = set()
    for obj in progress_entries:
        if obj.get("type") == "manifest":
            continue
        pdf = obj.get("pdf")
        if isinstance(pdf, str) and pdf.strip():
            seen.add(pdf.strip())

    if not seen:
        return None

    # Deterministic order: lexicographic casefold, then original
    return sorted(seen, key=lambda s: (s.casefold(), s))


def diff_lists(expected: List[str], actual: List[str], limit: int = 20) -> str:
    """
    Returns a human-friendly diff summary (limited).
    """
    exp_set = set(expected)
    act_set = set(actual)
    missing = sorted(exp_set - act_set, key=lambda s: (s.casefold(), s))
    extra = sorted(act_set - exp_set, key=lambda s: (s.casefold(), s))

    lines = []
    lines.append(f"Expected count: {len(expected)}; Actual count: {len(actual)}")
    if missing:
        lines.append(f"Missing (up to {limit}):")
        lines.extend(f"  - {p}" for p in missing[:limit])
    if extra:
        lines.append(f"Extra (up to {limit}):")
        lines.extend(f"  + {p}" for p in extra[:limit])
    return "\n".join(lines)


# ----------------------------
# Logging helpers
# ----------------------------
def safe_log_name(pdf_path: Path) -> str:
    """
    Create a filesystem-friendly, collision-resistant log name.

    We keep it readable while preventing collisions for same-stem PDFs by adding
    a short sha256 suffix derived from the absolute path.
    """
    stem = pdf_path.stem.replace("/", "_").replace("\x00", "_")
    h = hashlib.sha256(str(pdf_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}__{h}.log"


def stream_process_output_to_console_and_log(proc: subprocess.Popen, log_fp) -> bytes:
    """
    Streams combined stdout/stderr (stderr redirected to stdout) to:
      - terminal (stdout)
      - a log file
    Returns the complete captured bytes (for failure diagnostics / tail).

    NOTE: We intentionally do binary streaming to preserve any progress bars,
    mixed encodings, and non-line-buffered output.
    """
    captured = bytearray()
    assert proc.stdout is not None

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            captured += chunk

            # terminal
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except Exception:
                # If terminal write fails, we still keep logging and fail later if needed.
                pass

            # log
            log_fp.write(chunk)
            log_fp.flush()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise ConvertAllError("Error while streaming converter output.", original=e)

    return bytes(captured)


def tail_bytes(data: bytes, max_bytes: int = 4000) -> str:
    """
    Return a UTF-8 tail snippet (best-effort decode) for quick diagnostics in progress.jsonl.
    """
    tail = data[-max_bytes:]
    try:
        return tail.decode("utf-8", errors="replace")
    except Exception:
        return repr(tail)


# ----------------------------
# Core conversion routine
# ----------------------------
def run_one_pdf(
    *,
    repo_root: Path,
    converter_script: Path,
    converter_output_dir: Path,
    converted_root: Path,
    logs_root: Path,
    progress_jsonl: Path,
    pdf_abs: Path,
    dry_run: bool,
) -> None:
    if not pdf_abs.is_absolute():
        raise ConvertAllError("PDF path is not absolute (unexpected).", path=pdf_abs)
    if not pdf_abs.exists():
        raise ConvertAllError("PDF does not exist.", path=pdf_abs)
    if not pdf_abs.is_file():
        raise ConvertAllError("PDF path exists but is not a file.", path=pdf_abs)
    if pdf_abs.suffix.lower() != ".pdf":
        raise ConvertAllError("Input path does not end with .pdf (unexpected).", path=pdf_abs)

    # Output destination policy (as requested): folder named by PDF basename/stem
    dest_dir = converted_root / pdf_abs.stem
    log_path = logs_root / safe_log_name(pdf_abs)

    if dry_run:
        print(f"[DRY RUN] Would delete: {converter_output_dir}")
        print(f"[DRY RUN] Would run: {sys.executable} {converter_script} {pdf_abs}")
        print(f"[DRY RUN] Would move: {converter_output_dir} -> {dest_dir}")
        return

    # Always delete converter output dir before running
    rmtree_strict(converter_output_dir)

    # Ensure output parents exist
    ensure_dir(converted_root)
    ensure_dir(logs_root)

    start = time.time()
    entry_base = {
        "ts_utc": utc_now_iso(),
        "pdf": str(pdf_abs),
        "converter": str(converter_script),
        "cwd": str(repo_root),
        "log_path": str(log_path),
    }

    # Open log file in binary mode to mirror terminal output exactly
    try:
        with log_path.open("ab") as log_fp:
            log_fp.write(f"\n\n===== RUN @ {utc_now_iso()} =====\n".encode("utf-8"))

            cmd = [sys.executable, str(converter_script), str(pdf_abs)]
            proc: Optional[subprocess.Popen] = None
            captured: bytes = b""

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,  # unbuffered pipe from the child to us (we still stream carefully)
                )

                captured = stream_process_output_to_console_and_log(proc, log_fp)
                rc = proc.wait()

            except KeyboardInterrupt:
                # Graceful cancellation:
                # - Terminate the converter process
                # - Record "cancelled"
                # - Exit 130 (standard SIGINT-ish exit code)
                if proc is not None:
                    try:
                        log_fp.write(b"\n\n[convert_all_pdfs] KeyboardInterrupt received. Terminating converter...\n")
                        log_fp.flush()

                        # Try gentle termination first
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            log_fp.write(b"[convert_all_pdfs] Converter did not exit after SIGTERM. Killing...\n")
                            log_fp.flush()
                            proc.kill()
                            proc.wait(timeout=10)
                    except Exception as e:
                        append_progress(
                            progress_jsonl,
                            {
                                **entry_base,
                                "status": "cancelled",
                                "duration_s": round(time.time() - start, 3),
                                "note": "KeyboardInterrupt during termination handling.",
                                "termination_error": f"{type(e).__name__}: {e}",
                                "output_tail": tail_bytes(captured),
                            },
                        )
                        raise

                append_progress(
                    progress_jsonl,
                    {
                        **entry_base,
                        "status": "cancelled",
                        "duration_s": round(time.time() - start, 3),
                        "output_tail": tail_bytes(captured),
                    },
                )
                raise SystemExit(130)

            duration = round(time.time() - start, 3)

            if rc != 0:
                append_progress(
                    progress_jsonl,
                    {
                        **entry_base,
                        "status": "failed",
                        "return_code": rc,
                        "duration_s": duration,
                        "output_tail": tail_bytes(captured),
                    },
                )
                raise ConvertAllError(
                    f"Converter exited with non-zero return code: {rc}. See log for details.",
                    path=log_path,
                )

            # Success: verify converter output exists
            if not converter_output_dir.exists():
                append_progress(
                    progress_jsonl,
                    {
                        **entry_base,
                        "status": "failed",
                        "return_code": rc,
                        "duration_s": duration,
                        "error": "Converter reported success but ./conversion_vl does not exist.",
                        "output_tail": tail_bytes(captured),
                    },
                )
                raise ConvertAllError(
                    "Converter reported success but output folder ./conversion_vl is missing.",
                    path=converter_output_dir,
                )

            if converter_output_dir.is_symlink():
                append_progress(
                    progress_jsonl,
                    {
                        **entry_base,
                        "status": "failed",
                        "return_code": rc,
                        "duration_s": duration,
                        "error": "Converter output path is a symlink (unexpected/safety refusal).",
                        "output_tail": tail_bytes(captured),
                    },
                )
                raise ConvertAllError("Converter output path is a symlink (unexpected).", path=converter_output_dir)

            if not converter_output_dir.is_dir():
                append_progress(
                    progress_jsonl,
                    {
                        **entry_base,
                        "status": "failed",
                        "return_code": rc,
                        "duration_s": duration,
                        "error": "Converter output path exists but is not a directory.",
                        "output_tail": tail_bytes(captured),
                    },
                )
                raise ConvertAllError("Converter output path exists but is not a directory.", path=converter_output_dir)

            # Prepare destination: remove any existing dest (e.g., from partial runs)
            if dest_dir.exists():
                rmtree_strict(dest_dir)

            ensure_dir(dest_dir.parent)

            # Move conversion_vl -> dest_dir (so dest_dir contains the contents of conversion_vl)
            try:
                shutil.move(str(converter_output_dir), str(dest_dir))
            except Exception as e:
                append_progress(
                    progress_jsonl,
                    {
                        **entry_base,
                        "status": "failed",
                        "return_code": rc,
                        "duration_s": duration,
                        "error": "Failed to move conversion_vl into converted output folder.",
                        "move_src": str(converter_output_dir),
                        "move_dst": str(dest_dir),
                        "output_tail": tail_bytes(captured),
                    },
                )
                raise ConvertAllError("Failed to move conversion_vl into destination folder.", path=dest_dir, original=e)

            append_progress(
                progress_jsonl,
                {
                    **entry_base,
                    "status": "success",
                    "return_code": rc,
                    "duration_s": duration,
                    "converted_output_dir": str(dest_dir),
                },
            )

    except OSError as e:
        # Log IO issues also get recorded before fail-fast
        append_progress(
            progress_jsonl,
            {
                **entry_base,
                "status": "failed",
                "duration_s": round(time.time() - start, 3),
                "error": f"OSError while preparing/running conversion: {type(e).__name__}: {e}",
            },
        )
        raise ConvertAllError("Filesystem error while preparing/running conversion.", path=pdf_abs, original=e)


# ----------------------------
# Main
# ----------------------------
def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Convert all visible PDFs under a base folder via convert_pdf_to_md_vl.py (resumable; strict manifest)."
    )
    parser.add_argument(
        "base_dir",
        help="Base folder to recursively scan for PDFs (visible only). Example: /home/user/Desktop/g_force",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions but do not run conversions or modify/delete anything.",
    )
    parser.add_argument(
        "--follow-symlink-dirs",
        action="store_true",
        help="Allow traversing symlinked directories (OFF by default; may cause unexpected recursion).",
    )
    args = parser.parse_args(list(argv))

    # Script lives next to convert_pdf_to_md_vl.py and conversion_vl
    repo_root = Path(__file__).resolve().parent
    converter_script = repo_root / "convert_pdf_to_md_vl.py"
    converter_output_dir = repo_root / "conversion_vl"

    # Dedicated outputs
    all_pdfs_root = repo_root / "all_pdfs"
    progress_jsonl = all_pdfs_root / "progress.jsonl"
    converted_root = all_pdfs_root / "converted"
    logs_root = all_pdfs_root / "logs"

    # Fail-fast sanity checks
    if not converter_script.exists():
        raise ConvertAllError("Missing convert_pdf_to_md_vl.py (expected it next to this script).", path=converter_script)
    if not converter_script.is_file():
        raise ConvertAllError("convert_pdf_to_md_vl.py exists but is not a file.", path=converter_script)

    base_dir_abs = Path(args.base_dir).expanduser().resolve()

    # Scan PDFs deterministically (visible only)
    pdfs_abs = scan_visible_pdfs(base_dir_abs, follow_symlink_dirs=args.follow_symlink_dirs)

    if not pdfs_abs:
        print("No visible PDFs found under base directory. Nothing to do.")
        return 0

    # Detect basename collisions (destination folders are based on stem only).
    # We refuse rather than silently overwriting.
    stems: Dict[str, List[Path]] = {}
    for p in pdfs_abs:
        stems.setdefault(p.stem, []).append(p)
    collisions = {stem: paths for stem, paths in stems.items() if len(paths) > 1}
    if collisions:
        msg_lines = [
            "Refusing to proceed: multiple PDFs share the same basename/stem, which would collide in ./all_pdfs/converted/<stem>/",
            "Collisions (stem -> files):",
        ]
        for stem, paths in sorted(collisions.items(), key=lambda kv: (kv[0].casefold(), kv[0])):
            msg_lines.append(f"- {stem}:")
            for pp in sorted(paths, key=lambda x: (str(x).casefold(), str(x))):
                msg_lines.append(f"    {pp}")
        raise ConvertAllError("\n".join(msg_lines))

    # Ensure all_pdfs exists (unless dry-run)
    if not args.dry_run:
        ensure_dir(all_pdfs_root)

    # Load progress + enforce/establish manifest
    progress_entries = load_progress_lines(progress_jsonl)

    current_manifest = compute_manifest(base_dir_abs, pdfs_abs)
    existing_manifest = find_manifest(progress_entries)

    if existing_manifest is None and progress_entries:
        # Older progress file without manifest: attempt safe bootstrap.
        inferred = infer_manifest_from_progress(progress_entries)
        if inferred is None:
            raise ConvertAllError(
                "progress.jsonl exists but has no manifest and no usable pdf entries to infer the intended set. "
                "Refusing to resume.",
                path=progress_jsonl,
            )

        scanned_list = current_manifest["pdfs"]
        if inferred != scanned_list:
            detail = diff_lists(inferred, scanned_list)
            raise ConvertAllError(
                "Refusing to resume: scanned PDFs do not match PDFs inferred from existing progress.jsonl.\n"
                + detail,
                path=progress_jsonl,
            )

        # If dry-run, don't mutate progress; otherwise append a manifest.
        if not args.dry_run:
            bootstrap_manifest = {
                **current_manifest,
                "note": "Bootstrapped manifest because progress.jsonl existed without one.",
                "bootstrapped_from_progress": True,
            }
            append_progress(progress_jsonl, bootstrap_manifest)
            progress_entries.append(bootstrap_manifest)
            existing_manifest = bootstrap_manifest

    if existing_manifest is None:
        # First run (no progress yet): write a manifest.
        if args.dry_run:
            print("[DRY RUN] Would write manifest to ./all_pdfs/progress.jsonl")
        else:
            append_progress(progress_jsonl, current_manifest)
            progress_entries.append(current_manifest)
            existing_manifest = current_manifest

    # Enforce manifest equality to refuse resuming a different set.
    assert existing_manifest is not None
    expected_base = existing_manifest.get("base_dir")
    expected_pdfs = existing_manifest.get("pdfs")
    expected_sha = existing_manifest.get("pdfs_sha256")

    if not isinstance(expected_base, str) or not isinstance(expected_pdfs, list) or not isinstance(expected_sha, str):
        raise ConvertAllError("Existing manifest in progress.jsonl is malformed. Refusing to proceed.", path=progress_jsonl)

    # Compare base dir and the exact list/hash.
    if expected_base != str(base_dir_abs):
        raise ConvertAllError(
            "Refusing to resume: base_dir differs from the existing manifest.\n"
            f"Manifest base_dir: {expected_base}\n"
            f"Current base_dir:  {base_dir_abs}",
            path=progress_jsonl,
        )

    if expected_sha != current_manifest["pdfs_sha256"] or expected_pdfs != current_manifest["pdfs"]:
        detail = diff_lists([str(x) for x in expected_pdfs], current_manifest["pdfs"])
        raise ConvertAllError(
            "Refusing to resume: scanned PDFs do not match the existing manifest.\n" + detail,
            path=progress_jsonl,
        )

    # Determine which ones are done
    latest_status = load_latest_status(progress_entries)

    # Process sequentially, skipping successes
    total = len(pdfs_abs)
    for idx, pdf in enumerate(pdfs_abs, start=1):
        status = latest_status.get(str(pdf))
        if status == "success":
            print(f"[{idx}/{total}] SKIP (already success): {pdf}")
            continue

        print(f"\n[{idx}/{total}] CONVERT: {pdf}")
        run_one_pdf(
            repo_root=repo_root,
            converter_script=converter_script,
            converter_output_dir=converter_output_dir,
            converted_root=converted_root,
            logs_root=logs_root,
            progress_jsonl=progress_jsonl,
            pdf_abs=pdf,
            dry_run=args.dry_run,
        )

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ConvertAllError as e:
        print(f"\nERROR:\n{e}", file=sys.stderr)
        raise SystemExit(2)
