#!/usr/bin/env python3
"""
Compatibility wrapper for dataset filtering and filename cleanup.

Prefer:
    python dataset_cli.py filter-ready ...
"""

from dataset_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["filter-ready", *(__import__("sys").argv[1:])]))
