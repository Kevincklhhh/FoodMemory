#!/usr/bin/env python3
"""
Format VLM Logs for Readability

Converts JSON VLM logs into human-readable text format with proper newlines.

Usage:
    python format_vlm_logs.py                          # Format all logs in default dir
    python format_vlm_logs.py --input ../outputs/food_graph/vlm_logs
    python format_vlm_logs.py --file block_001_transactions.json
"""

import json
import argparse
from pathlib import Path


def format_log(log_data: dict) -> str:
    """Format a single VLM log entry into readable text"""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"VIDEO: {log_data.get('video_id', 'unknown')}")
    lines.append(f"BLOCK: {log_data.get('block_id', '?')}")
    lines.append(f"TYPE: {log_data.get('call_type', 'unknown')}")
    if log_data.get('has_video'):
        lines.append(f"VIDEO CLIP: {log_data.get('video_clip', 'none')}")
    lines.append("=" * 80)

    # Combined Prompt
    lines.append("")
    lines.append("-" * 40)
    lines.append("PROMPT")
    lines.append("-" * 40)

    # System prompt (instructions)
    system_prompt = log_data.get('system_prompt', '')
    if system_prompt:
        lines.append("")
        lines.append("[SYSTEM INSTRUCTIONS]")
        lines.append("")
        lines.append(system_prompt)

    # User prompt (the actual task)
    user_prompt = log_data.get('user_prompt', '')
    if user_prompt:
        lines.append("")
        lines.append("[TASK]")
        lines.append("")
        lines.append(user_prompt)

    # Response
    lines.append("")
    lines.append("-" * 40)
    lines.append("RESPONSE")
    lines.append("-" * 40)
    lines.append("")

    raw_response = log_data.get('raw_response', '')
    if raw_response:
        lines.append(raw_response)

    # Parsed response (formatted JSON)
    parsed = log_data.get('parsed_response')
    if parsed:
        lines.append("")
        lines.append("[PARSED]")
        lines.append(json.dumps(parsed, indent=2))

        # Check for errors (only for dict responses)
        if isinstance(parsed, dict) and parsed.get('parse_error'):
            lines.append("")
            lines.append("[PARSE ERROR]")
            lines.append(parsed['parse_error'])

    # Warnings from transaction execution
    warnings = log_data.get('warnings', [])
    if warnings:
        lines.append("")
        lines.append("[EXECUTION WARNINGS]")
        for warn in warnings:
            lines.append(f"  - {warn}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def format_log_file(json_path: Path, output_dir: Path = None) -> Path:
    """Format a single JSON log file to readable text"""
    with open(json_path, 'r') as f:
        log_data = json.load(f)

    formatted = format_log(log_data)

    # Output path
    if output_dir:
        output_path = output_dir / f"{json_path.stem}.txt"
    else:
        output_path = json_path.with_suffix('.txt')

    with open(output_path, 'w') as f:
        f.write(formatted)

    return output_path


def format_all_logs(input_dir: Path, output_dir: Path = None):
    """Format all JSON logs in a directory"""
    if output_dir is None:
        output_dir = input_dir / 'formatted'
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob('*.json'))

    print(f"Found {len(json_files)} JSON log files")
    print(f"Output directory: {output_dir}")
    print()

    for json_path in json_files:
        output_path = format_log_file(json_path, output_dir)
        print(f"  {json_path.name} -> {output_path.name}")

    # Also create a combined file with all logs
    combined_path = output_dir / '_all_logs.txt'
    with open(combined_path, 'w') as f:
        for json_path in json_files:
            with open(json_path, 'r') as jf:
                log_data = json.load(jf)
            f.write(format_log(log_data))
            f.write("\n\n")

    print()
    print(f"Combined log: {combined_path}")
    print(f"Total: {len(json_files)} files formatted")


def main():
    parser = argparse.ArgumentParser(description="Format VLM logs for readability")
    parser.add_argument(
        '--input', '-i',
        default='../outputs/food_graph/vlm_logs',
        help='Input directory with JSON logs (or single file with --file)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (default: input_dir/formatted)'
    )
    parser.add_argument(
        '--file', '-f',
        default=None,
        help='Format a single file instead of directory'
    )

    args = parser.parse_args()

    if args.file:
        # Single file mode
        json_path = Path(args.input) / args.file if not Path(args.file).is_absolute() else Path(args.file)
        if not json_path.exists():
            json_path = Path(args.file)

        output_path = format_log_file(json_path)
        print(f"Formatted: {output_path}")
    else:
        # Directory mode
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.output else None

        if not input_dir.exists():
            print(f"ERROR: Input directory not found: {input_dir}")
            return

        format_all_logs(input_dir, output_dir)


if __name__ == '__main__':
    main()
