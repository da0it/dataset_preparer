#!/usr/bin/env python3
"""
Compatibility wrapper for label normalization.

Prefer:
    python dataset_cli.py normalize-labels ...
"""

from dataset_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["normalize-labels", *(__import__("sys").argv[1:])]))
