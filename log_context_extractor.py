#!/usr/bin/env python3
"""
Log Context Extractor

Description:
- Scans a target folder (non-recursive by default) for all files by default (regardless of extension)
- Reads a search string from a properties file (default: search.properties)
- Finds occurrences of the search string in each file
- For each occurrence, extracts 100 lines above and 100 lines below (configurable) as context
- Optionally filters matches to a date/time window provided in properties (e.g., 24-Dec-2025 11:00am - 11:30am, or time-of-day only: 12:00 PM - 12:30 PM)
- Writes results per search term to per-log text files named after the source log with term and timestamp (e.g., server1.log -> server1_ERROR_20250101_120305_context.txt)
- If no results across all logs and terms, writes a single file with "No relevant data found" (timestamped)
- Prompts interactively for source log directory, pipe-separated search terms, optional date/time window, and output directory when not provided via CLI/properties

Properties file format (Java .properties):
- search_string=term1|term2|term3     # use '|' to provide multiple search terms when regex=false; searched sequentially; for a single term, just provide it
- case_sensitive=false        # optional, default false
- regex=false                 # optional, default false
- above_lines=100             # optional, default 100
- below_lines=100             # optional, default 100
- time_window=24-Dec-2025 11:00am - 11:30am   # optional; restrict matches to this window (date once, start - end time)
- time_window=12:00 PM - 12:30 PM             # optional; time-of-day only; applies across all dates
- timestamp_regex=(\d{2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2}:\d{2},\d{3})   # optional; regex with one capturing group for the datetime substring in your logs
- timestamp_format=%d-%b-%Y %H:%M:%S,%f       # optional; strptime format matching the captured datetime

CLI Usage:
  python utilities/log_context_extractor.py \\
      --dir "path/to/log/folder" \\
      --properties "path/to/search.properties" \\
      --above 100 --below 100 \\
      --recursive

Notes:
- CLI arguments override properties file values if provided.
- If values are not provided via CLI/properties, the script will prompt for source log directory, pipe-separated search terms, optional date/time window, and output directory.
"""

import argparse
import os
import re
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Tuple, Optional


DEFAULT_EXTENSIONS = [".out", ".log", ".txt"]


def load_properties(path: Path) -> Dict[str, str]:
    props: Dict[str, str] = {}
    if not path or not path.exists():
        return props
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                # simple key=value parser; does not support escaped '='
                if "=" in line:
                    key, val = line.split("=", 1)
                    props[key.strip()] = val.strip()
    except Exception as e:
        print(f"Warning: Failed to read properties file '{path}': {e}", file=sys.stderr)
    return props


def parse_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def parse_int(val: Optional[str], default: int) -> int:
    if val is None:
        return default
    try:
        return int(val.strip())
    except Exception:
        return default


def parse_search_terms(search: str, use_regex: bool) -> List[str]:
    """
    Returns a list of search terms.
    - When regex is False: split on '|' and trim whitespace (supports multiple terms sequentially)
    - When regex is True: treat the entire search string as a single regex pattern
    """
    if not search:
        return []
    if use_regex:
        return [search]
    return [s.strip() for s in search.split("|") if s.strip()]


# Default timestamp parse formats and regexes for extracting datetimes from log lines
DEFAULT_TS_FORMATS: List[str] = [
    "%d-%b-%Y %H:%M:%S,%f",
    "%d-%b-%Y %H:%M:%S.%f",
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y %H:%M",
    "%d-%b-%Y %I:%M:%S%p",
    "%d-%b-%Y %I:%M%p",
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d %b %Y;%H:%M:%S,%f",
    "%d %b %Y;%H:%M:%S.%f",
    "%d %b %Y;%H:%M:%S",
    "%d %B %Y;%H:%M:%S,%f",
    "%d %B %Y;%H:%M:%S.%f",
    "%d %B %Y;%H:%M:%S",
]

DEFAULT_TS_REGEXES: List[re.Pattern] = [
    re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2}:\d{2}[.,]\d{1,6})"),
    re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2}:\d{2})"),
    re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2}\s?[AaPp][Mm])"),
    re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2})"),
    re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2}[.,]\d{1,6})"),
    re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2})"),
    re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})"),
    re.compile(r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4};\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)"),
]


def try_parse_dt(text: str, fmts: List[str]) -> Optional[datetime]:
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def extract_line_dt(line: str, ts_regex: Optional[re.Pattern], ts_format: Optional[str]) -> Optional[datetime]:
    """
    Attempt to extract a datetime from a log line.
    - If ts_regex is provided, the first capturing group is parsed; if ts_format is provided, it's used; otherwise default formats are tried.
    - Otherwise, try built-in regexes and formats.
    """
    if ts_regex is not None:
        m = ts_regex.search(line)
        if not m:
            return None
        dt_str = m.group(1) if m.groups() else m.group(0)
        if ts_format:
            try:
                return datetime.strptime(dt_str, ts_format)
            except Exception:
                return None
        return try_parse_dt(dt_str, DEFAULT_TS_FORMATS)

    for rx in DEFAULT_TS_REGEXES:
        m = rx.search(line)
        if m:
            dt_str = m.group(1)
            dt = try_parse_dt(dt_str, DEFAULT_TS_FORMATS)
            if dt:
                return dt
    return None


def parse_time_window(window_str: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Parses strings like:
      '24-Dec-2025 11:00am - 11:30am'
      '24-Dec-2025 11:00 - 11:30'
      '24-Dec-2025 11:00:00 - 11:30:00'
    Returns (start_dt, end_dt) or None if parse fails.
    """
    if not window_str:
        return None
    # Extract date and the two times
    m = re.match(r"\s*(\d{1,2}-[A-Za-z]{3}-\d{4})\s+([^-\n]+)-\s*(.+)\s*$", window_str)
    if not m:
        return None
    date_part = m.group(1).strip()
    start_time_part = m.group(2).strip()
    end_time_part = m.group(3).strip()
    try:
        base_date = datetime.strptime(date_part, "%d-%b-%Y").date()
    except Exception:
        return None

    def combine(date_obj, time_str) -> Optional[datetime]:
        cand_formats = ["%I:%M%p", "%I:%M:%S%p", "%H:%M", "%H:%M:%S"]
        for tf in cand_formats:
            try:
                t = datetime.strptime(time_str.replace(" ", ""), tf)  # allow '11:00 am'
                return datetime(
                    date_obj.year, date_obj.month, date_obj.day,
                    t.hour, t.minute, t.second, t.microsecond
                )
            except Exception:
                continue
        return None

    start_dt = combine(base_date, start_time_part)
    end_dt = combine(base_date, end_time_part)
    if start_dt is None or end_dt is None:
        return None
    # Handle window crossing midnight by rolling end forward a day
    if end_dt < start_dt:
        from datetime import timedelta
        end_dt = end_dt + timedelta(days=1)
    return (start_dt, end_dt)


def parse_time_of_day_window(window_str: str) -> Optional[Tuple[dt_time, dt_time]]:
    """
    Parses time-of-day windows like:
      '12:00 PM - 12:30 PM'
      '11:00 - 11:30'
      '11:00:00 - 11:30:00'
    Returns (start_time, end_time) or None if parse fails.
    """
    if not window_str:
        return None
    m = re.match(r"\s*([^-\n]+)-\s*(.+)\s*$", window_str)
    if not m:
        return None
    start_time_part = m.group(1).strip()
    end_time_part = m.group(2).strip()

    cand_formats = ["%I:%M%p", "%I:%M:%S%p", "%H:%M", "%H:%M:%S"]

    def parse_t(s: str) -> Optional[dt_time]:
        for tf in cand_formats:
            try:
                t = datetime.strptime(s.replace(" ", ""), tf)
                return dt_time(hour=t.hour, minute=t.minute, second=t.second, microsecond=t.microsecond)
            except Exception:
                continue
        return None

    start_t = parse_t(start_time_part)
    end_t = parse_t(end_time_part)
    if start_t is None or end_t is None:
        return None
    return (start_t, end_t)


def filter_indices_by_time_of_day(
    lines: List[str],
    indices: List[int],
    start_t: dt_time,
    end_t: dt_time,
    ts_regex: Optional[re.Pattern],
    ts_format: Optional[str],
) -> List[int]:
    filtered: List[int] = []
    for i in indices:
        dt = extract_line_dt(lines[i], ts_regex, ts_format)
        if dt is None:
            continue
        tt = dt.time()
        if start_t <= end_t:
            if start_t <= tt <= end_t:
                filtered.append(i)
        else:
            # crosses midnight
            if tt >= start_t or tt <= end_t:
                filtered.append(i)
    return filtered


def filter_indices_by_time(
    lines: List[str],
    indices: List[int],
    start_dt: datetime,
    end_dt: datetime,
    ts_regex: Optional[re.Pattern],
    ts_format: Optional[str],
) -> List[int]:
    filtered: List[int] = []
    for i in indices:
        dt = extract_line_dt(lines[i], ts_regex, ts_format)
        if dt is not None and start_dt <= dt <= end_dt:
            filtered.append(i)
    return filtered


def ask_output_dir(default_dir: Path) -> Path:
    print("")
    print("Enter output directory for result txt files.")
    print(f"Press Enter to use default: {default_dir}")
    out = input("Output directory path: ").strip()
    if not out:
        out_dir = default_dir
    else:
        out_dir = Path(out).expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: cannot create output directory '{out_dir}': {e}", file=sys.stderr)
        sys.exit(2)
    return out_dir


def ask_scan_dir(default_dir: Path) -> Path:
    print("")
    print("Enter source log directory to scan.")
    print(f"Press Enter to use default: {default_dir}")
    src = input("Source directory path: ").strip()
    scan_dir = (default_dir if not src else Path(src).expanduser().resolve())
    if not scan_dir.exists() or not scan_dir.is_dir():
        print(f"Error: directory does not exist or is not a directory: {scan_dir}", file=sys.stderr)
        sys.exit(2)
    return scan_dir


def ask_time_window_interactive() -> Optional[str]:
    print("")
    print("Restrict search by time window? (y/N)")
    ans = input("Choice: ").strip().lower()
    if ans not in ("y", "yes"):
        return None
    date_str = input("Date (e.g., 24-Dec-2025) or leave blank to apply to all dates: ").strip()
    start_time = input("Start time (e.g., 12:00 PM or 12:00): ").strip()
    end_time = input("End time (e.g., 12:30 PM or 12:30): ").strip()
    if not start_time or not end_time:
        return None
    if date_str:
        return f"{date_str} {start_time} - {end_time}"
    return f"{start_time} - {end_time}"


def read_lines_with_fallback(p: Path) -> List[str]:
    # Try UTF-8 first, then fall back to latin-1 to be robust for various log encodings.
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        try:
            return p.read_text(encoding="latin-1", errors="replace").splitlines()
        except Exception as e:
            print(f"Warning: failed to read file '{p}': {e}", file=sys.stderr)
            return []


def find_match_indices(
    lines: List[str],
    needle: str,
    case_sensitive: bool,
    use_regex: bool,
) -> List[int]:
    indices: List[int] = []
    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(needle, flags)
        except re.error as e:
            print(f"Error: invalid regex pattern '{needle}': {e}", file=sys.stderr)
            return []
        for i, line in enumerate(lines):
            if pattern.search(line) is not None:
                indices.append(i)
    else:
        if case_sensitive:
            for i, line in enumerate(lines):
                if needle in line:
                    indices.append(i)
        else:
            needle_lower = needle.lower()
            for i, line in enumerate(lines):
                if needle_lower in line.lower():
                    indices.append(i)
    return indices


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = ranges[0]
    for s, e in ranges[1:]:
        if s <= cur_end + 1:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


def build_context_blocks(
    match_indices: List[int],
    total_lines: int,
    above: int,
    below: int,
) -> List[Tuple[int, int]]:
    ranges = []
    for idx in match_indices:
        start = max(0, idx - above)
        end = min(total_lines - 1, idx + below)
        ranges.append((start, end))
    return merge_ranges(ranges)


def sanitize_filename(name: str) -> str:
    # Windows-incompatible characters: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def write_no_results_file(
    out_dir: Path,
    search: str,
    above: int,
    below: int,
    case_sensitive: bool,
    use_regex: bool,
    scan_dir: Path,
) -> Path:
    out_file = out_dir / f"no_relevant_data_found_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = [
        f"Log Context Extractor - No Results",
        f"Timestamp: {ts}",
        f"Scan directory: {scan_dir}",
        f"Search: {search}",
        f"Case sensitive: {case_sensitive}",
        f"Regex: {use_regex}",
        f"Context: above={above}, below={below}",
        "",
        "No relevant data found.",
    ]
    out_file.write_text("\n".join(content), encoding="utf-8")
    return out_file


def write_results_for_file(
    source_file: Path,
    out_dir: Path,
    lines: List[str],
    blocks: List[Tuple[int, int]],
    match_indices: List[int],
    search: str,
    above: int,
    below: int,
    case_sensitive: bool,
    use_regex: bool,
    run_ts: str,
    time_window: Optional[str] = None,
) -> Path:
    base = sanitize_filename(source_file.name)
    safe_search = sanitize_filename(search) if search else "search"
    out_file = out_dir / f"{base}_{safe_search}_{run_ts}_context.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    match_set = set(match_indices)

    header = [
        f"Log Context Extractor - Results",
        f"Timestamp: {ts}",
        f"Source log: {source_file}",
        f"Search: {search}",
        f"Case sensitive: {case_sensitive}",
        f"Regex: {use_regex}",
        f"Context: above={above}, below={below}",
    ]
    if time_window:
        header.append(f"Time window: {time_window}")
    header.extend([
        f"Total matches: {len(match_indices)}",
        "",
    ])

    body: List[str] = []
    for bi, (start, end) in enumerate(blocks, start=1):
        body.append(f"----- Context block {bi}: lines {start+1}..{end+1} -----")
        for i in range(start, end + 1):
            prefix = ">> " if i in match_set else "   "
            # include 1-based line numbers aligned to width
            ln = f"{i+1:7d}"
            body.append(f"{prefix}{ln}: {lines[i]}")
        body.append("")

    out_file.write_text("\n".join(header + body), encoding="utf-8")
    return out_file


def collect_files(scan_dir: Path, recursive: bool, extensions: Optional[List[str]]) -> List[Path]:
    files: List[Path] = []
    if recursive:
        it = scan_dir.rglob("*")
    else:
        it = scan_dir.iterdir()
    for p in it:
        if p.is_file():
            if extensions is None:
                files.append(p)
            else:
                if p.suffix.lower() in extensions:
                    files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract +/-N lines around a search string across log files.")
    parser.add_argument("--dir", "-d", dest="scan_dir", default=None, help="Directory to scan for logs (overrides properties)")
    parser.add_argument("--properties", "-p", dest="properties", default=None, help="Path to properties file (default: ./search.properties if exists)")
    parser.add_argument("--above", type=int, default=None, help="Lines above each match (overrides properties)")
    parser.add_argument("--below", type=int, default=None, help="Lines below each match (overrides properties)")
    parser.add_argument("--recursive", action="store_true", help="Scan recursively (default: false)")
    parser.add_argument("--extensions", default="*", help="Comma-separated list of extensions (include dot), or '*' to include all files (default)")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Output directory for result files (if omitted, will prompt at runtime)")
    args = parser.parse_args()

    # Determine properties path first (prefer explicit, otherwise cwd/search.properties)
    if args.properties:
        properties_path = Path(args.properties).expanduser().resolve()
    else:
        default_props = Path.cwd() / "search.properties"
        properties_path = default_props if default_props.exists() else None

    props = load_properties(properties_path) if properties_path else {}

    # Resolve scan directory (prompt if not provided)
    provided_dir = False
    if args.scan_dir is not None:
        scan_dir = Path(args.scan_dir).expanduser().resolve()
        provided_dir = True
    else:
        prop_dir_str = props.get("log_dir") or props.get("logs_dir") or props.get("scan_dir") or props.get("dir")
        if prop_dir_str:
            base = properties_path.parent if properties_path else Path.cwd()
            candidate = Path(prop_dir_str)
            scan_dir = (base / candidate).expanduser().resolve() if not candidate.is_absolute() else candidate.expanduser().resolve()
            provided_dir = True
        else:
            scan_dir = Path.cwd()
            provided_dir = False

    if not provided_dir or not scan_dir.exists() or not scan_dir.is_dir():
        scan_dir = ask_scan_dir(scan_dir)

    # Resolve search string
    search = props.get("search_string") or props.get("search.string") or ""
    if not search:
        # As fallback, try first non-comment, non-empty line as the search string
        # if a properties file was provided but no key was found.
        if properties_path and properties_path.exists():
            try:
                with properties_path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        s = line.strip()
                        if s and not s.startswith("#") and not s.startswith(";") and "=" not in s:
                            search = s
                            break
            except Exception:
                pass

    # If still empty, prompt the user for the search string interactively
    if not search:
        print("Enter pipe-separated search terms (required). Example: ERROR|ORA-|Exception")
        search = input("Search terms: ").rstrip("\n")
        if not search:
            print("Error: search string is required.", file=sys.stderr)
            sys.exit(2)

    # Resolve flags and context sizes
    case_sensitive = parse_bool(props.get("case_sensitive"), default=False)
    use_regex = parse_bool(props.get("regex"), default=False)

    above_default = parse_int(props.get("above_lines"), 100)
    below_default = parse_int(props.get("below_lines"), 100)
    above = args.above if args.above is not None else above_default
    below = args.below if args.below is not None else below_default

    # Optional time window filtering (properties or interactive)
    time_window_str = props.get("time_window") or props.get("time.window")
    ts_regex_str = props.get("timestamp_regex") or props.get("timestamp.regex")
    ts_format = props.get("timestamp_format") or props.get("timestamp.format")

    if ts_regex_str:
        try:
            ts_regex = re.compile(ts_regex_str)
        except re.error as e:
            print(f"Warning: invalid timestamp_regex '{ts_regex_str}': {e}", file=sys.stderr)
            ts_regex = None
    else:
        ts_regex = None

    if not time_window_str:
        time_window_str = ask_time_window_interactive()

    time_window = parse_time_window(time_window_str) if time_window_str else None
    time_window_tod = None
    if (not time_window) and time_window_str:
        time_window_tod = parse_time_of_day_window(time_window_str)

    # Output directory - prompt if not provided
    default_output_dir = scan_dir / "context_output"
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: cannot create output directory '{out_dir}': {e}", file=sys.stderr)
            sys.exit(2)
    else:
        out_dir = ask_output_dir(default_output_dir)

    # Extensions: '*' or 'ALL' means include all files
    ext_arg = (args.extensions or "*").strip()
    if ext_arg in ("*", "ALL", "all"):
        ext_list = None
    else:
        ext_list = [e.strip().lower() for e in ext_arg.split(",") if e.strip()]
        if not ext_list:
            ext_list = None

    files = collect_files(scan_dir, args.recursive, ext_list)

    if not files:
        print(f"No files found in {scan_dir} (recursive={args.recursive}).")
        # Even if no files, still create a "no results" marker file.
        marker = write_no_results_file(out_dir, search, above, below, case_sensitive, use_regex, scan_dir)
        print(f"Wrote: {marker}")
        sys.exit(0)

    total_files = len(files)
    overall_matches = 0
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine search terms (sequential search support with '|' separated when regex=false)
    search_terms = parse_search_terms(search, use_regex)
    if not search_terms:
        search_terms = [search] if search else []

    for term in search_terms:
        print("")
        print(f"=== Searching for: {term} ===")
        files_with_matches = 0
        out_files: List[Path] = []

        for f in files:
            lines = read_lines_with_fallback(f)
            if not lines:
                continue
            term_use_regex = use_regex
            match_indices = find_match_indices(lines, term, case_sensitive, term_use_regex)
            if time_window:
                match_indices = filter_indices_by_time(
                    lines, match_indices, time_window[0], time_window[1], ts_regex, ts_format
                )
            elif time_window_tod:
                match_indices = filter_indices_by_time_of_day(
                    lines, match_indices, time_window_tod[0], time_window_tod[1], ts_regex, ts_format
                )
            if not match_indices:
                continue

            blocks = build_context_blocks(match_indices, len(lines), above, below)
            out_file = write_results_for_file(
                source_file=f,
                out_dir=out_dir,
                lines=lines,
                blocks=blocks,
                match_indices=match_indices,
                search=term,
                above=above,
                below=below,
                case_sensitive=case_sensitive,
                use_regex=term_use_regex,
                run_ts=run_ts,
                time_window=time_window_str if (time_window or time_window_tod) else None,
            )
            files_with_matches += 1
            overall_matches += 1
            out_files.append(out_file)
            print(f"Matches in '{f.name}': {len(match_indices)} -> wrote '{out_file.name}'")

        if files_with_matches == 0:
            print(f"No relevant data found for search: {term}")
        else:
            print("")
            print(f"Scanned {total_files} files. Matches in {files_with_matches} file(s) for search: {term}.")
            print(f"Output directory: {out_dir}")
            for p in out_files:
                print(f" - {p.name}")

    if overall_matches == 0:
        marker = write_no_results_file(out_dir, " | ".join(search_terms) if search_terms else search, above, below, case_sensitive, use_regex, scan_dir)
        print(f"No relevant data found. Wrote: {marker}")


if __name__ == "__main__":
    main()
