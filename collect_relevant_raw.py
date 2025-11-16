#!/usr/bin/env python3
import os
import json
import glob
import argparse
from typing import Dict, Any, List, Optional

RECORD_DELIM = "\x1e"


def read_jsonl_records(filepath: str) -> List[Dict[str, Any]]:
    """
    Reads the custom-delimited JSONL records from a deep_search run file.

    Uses the same RECORD_DELIM ("\x1e") convention as deep_search.py.
    """
    data: List[Dict[str, Any]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            blob = f.read()
    except FileNotFoundError:
        return data

    for segment in blob.split(RECORD_DELIM):
        segment = segment.strip()
        if not segment:
            continue
        try:
            obj = json.loads(segment)
            data.append(obj)
        except json.JSONDecodeError:
            print(f"Warning: Skipping a malformed JSON record in {os.path.basename(filepath)}")
            continue
    return data


def newest_run_file(path: str) -> Optional[str]:
    """
    Returns the newest run file (last in sorted order) from a directory.
    """
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    return files[-1] if files else None


def load_chunk_map(directory: str) -> Dict[int, str]:
    """
    Loads all .txt files from the given directory and returns a map:
        original_index (1-based, matching deep_search) -> file contents
    The file ordering is the same as deep_search.py:
        sorted(glob.glob(os.path.join(directory, "*.txt")))
    """
    if not os.path.isdir(directory):
        raise RuntimeError(f"Chunks directory not found: {directory}")

    chunk_files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    if not chunk_files:
        raise RuntimeError(f"No chunk files (*.txt) found in directory: {directory}")

    index_to_content: Dict[int, str] = {}
    for i, filepath in enumerate(chunk_files):
        original_index = i + 1  # deep_search.py uses 1-based index
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            index_to_content[original_index] = content
        except OSError as e:
            print(f"Warning: Could not read chunk file {filepath}: {e}")

    return index_to_content


def is_record_relevant(rec: Dict[str, Any]) -> bool:
    """
    Returns True if a judgement record is marked as relevant.
    Handles bool and string 'true' just in case.
    """
    val = rec.get("is_relevant")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all relevant chunks from the newest deep_search run into ./relevant_raw.txt\n"
            "For each judgement with is_relevant == true, this writes:\n"
            "  <JSON line>\n"
            "  <chunk contents>\n"
            "  \\n\\n"
        )
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="split/chunks",
        help="Directory containing the text chunks (default: 'split/chunks').",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="search_runs",
        help="Directory containing deep_search run files (default: 'search_runs').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="relevant_raw.txt",
        help="Output file to write relevant JSON + raw contents (default: './relevant_raw.txt').",
    )

    args = parser.parse_args()

    runs_dir = args.runs
    chunks_dir = args.dir
    output_path = args.output

    # 1. Find newest run file
    newest = newest_run_file(runs_dir)
    if newest is None:
        print(f"No run files found in '{runs_dir}'. Nothing to do.")
        return

    print(f"Using newest run file: {os.path.basename(newest)}")

    # 2. Load all records from newest run
    records = read_jsonl_records(newest)
    if not records:
        print(f"Run file '{newest}' is empty or invalid. Nothing to do.")
        return

    # First record is expected to be meta; keep it for sanity checks but don't output it.
    meta = records[0] if isinstance(records[0], dict) else {}
    if meta.get("type") != "meta":
        print("Warning: First record does not look like metadata. Proceeding anyway.")

    judgements = [r for r in records if isinstance(r, dict) and r.get("type") == "judgement"]
    if not judgements:
        print("No judgement records found in run file. Nothing to do.")
        return

    # 3. Load all chunks so we can map original_index -> content
    chunk_map = load_chunk_map(chunks_dir)

    # Optional sanity check against metadata (if present)
    meta_total = meta.get("original_total_chunks")
    if meta_total is not None:
        try:
            meta_total_int = int(meta_total)
            if meta_total_int != len(chunk_map):
                print(
                    f"Warning: Metadata original_total_chunks={meta_total_int}, "
                    f"but {len(chunk_map)} chunk files were found in '{chunks_dir}'."
                )
        except (TypeError, ValueError):
            print("Warning: Could not interpret 'original_total_chunks' from metadata as an integer.")

    # 4. Filter relevant judgements
    relevant_judgements = [j for j in judgements if is_record_relevant(j)]

    if not relevant_judgements:
        print("No judgements are marked as is_relevant == true. Output will be empty.")
    else:
        print(f"Found {len(relevant_judgements)} relevant judgements.")

    # 5. Write output: JSON line, then chunk contents, then blank line, repeated
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for j in relevant_judgements:
            original_index = j.get("original_index")
            if original_index is None:
                print("Warning: Skipping a judgement without 'original_index'.")
                continue

            try:
                original_index_int = int(original_index)
            except (TypeError, ValueError):
                print(f"Warning: Skipping judgement with non-integer original_index={original_index!r}.")
                continue

            chunk_content = chunk_map.get(original_index_int)
            if chunk_content is None:
                print(
                    f"Warning: No chunk content found for original_index={original_index_int}. "
                    "Maybe 'split/chunks' has changed since the run."
                )
                continue

            # JSON line
            out_f.write(json.dumps(j, ensure_ascii=False))
            out_f.write("\n")

            # Raw chunk contents
            out_f.write(chunk_content)
            out_f.write("\n\n")  # double newline between entries

            written += 1

    print(f"Wrote {written} relevant entries to '{output_path}'.")
    if written == 0:
        print("Note: File exists but contains no entries because nothing was relevant.")


if __name__ == "__main__":
    main()
