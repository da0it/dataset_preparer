#!/usr/bin/env python3
"""Compatibility wrapper for the main training entrypoint."""

from training_tools.train_advanced import main


if __name__ == "__main__":
    raise SystemExit(main())
