#!/usr/bin/env python3
"""
Compatibility wrapper for noise phrase cleanup.

Prefer:
    python dataset_cli.py clean-noise ...
"""

from dataset_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["clean-noise", *(__import__("sys").argv[1:])]))
