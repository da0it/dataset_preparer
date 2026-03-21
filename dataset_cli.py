#!/usr/bin/env python3
"""Compatibility wrapper for dataset CLI."""

from dataset_tools.dataset_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
