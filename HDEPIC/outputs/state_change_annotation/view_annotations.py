#!/usr/bin/env python3
"""
Visualization Launcher for HD-EPIC State Change Annotations

Starts a local HTTP server and opens the visualization interface in a web browser.
This allows the HTML file to load video clips, frame images, and grounding masks
without CORS restrictions.

Usage:
    python view_annotations.py [--port PORT] [--no-browser]

    --port PORT      Port number for HTTP server (default: 8000)
    --no-browser     Don't automatically open browser
"""

import argparse
import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path
import signal


def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def main():
    parser = argparse.ArgumentParser(
        description='Launch visualization server for state change annotations'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port number for HTTP server (default: 8000)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help="Don't automatically open browser"
    )
    args = parser.parse_args()

    # Script is already in the output directory
    output_dir = Path(__file__).parent.resolve()

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    # Verify visualization.html exists
    viz_html = output_dir / 'visualization.html'
    if not viz_html.exists():
        print(f"Error: visualization.html not found in {output_dir}")
        sys.exit(1)

    os.chdir(output_dir)
    print(f"Serving files from: {output_dir}")

    # Find available port
    try:
        port = find_available_port(args.port)
        if port != args.port:
            print(f"Port {args.port} unavailable, using port {port} instead")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create HTTP server
    Handler = http.server.SimpleHTTPRequestHandler

    # Suppress logging for cleaner output
    class QuietHandler(Handler):
        def log_message(self, format, *args):
            # Only log errors
            if args[1] != '200':
                super().log_message(format, *args)

    try:
        with socketserver.TCPServer(("", port), QuietHandler) as httpd:
            url = f"http://localhost:{port}/visualization.html"

            print("\n" + "=" * 70)
            print("HD-EPIC State Change Annotation Visualization")
            print("=" * 70)
            print(f"\nServer running at: {url}")
            print("\nInstructions:")
            print("  1. Upload your *_annotation_tasks.json file")
            print("  2. Video, frames, and masks will load automatically")
            print("  3. Use the timeline to navigate")
            print("  4. Toggle 'Show Masks' to overlay grounding masks")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 70 + "\n")

            # Open browser
            if not args.no_browser:
                print("Opening browser...")
                webbrowser.open(url)

            # Handle Ctrl+C gracefully
            def signal_handler(signum, frame):
                del signum, frame  # Unused but required by signal API
                print("\n\nShutting down server...")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            # Start serving
            httpd.serve_forever()

    except OSError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
