"""CLI wrapper for building dataset manifests."""

from __future__ import annotations

from .dataset_loader import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
