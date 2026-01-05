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
- If
