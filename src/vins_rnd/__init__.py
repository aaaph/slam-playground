"""
Top-level package for the vins-rnd project.

This package exists mainly to provide a stable console-script entrypoint
(`vins-rnd = vins_rnd:main`) and to give Hatchling a package that matches the
project name.
"""

from __future__ import annotations


def main() -> int:
    """Console entrypoint for the project."""
    # CLI entrypoint: stdout is the intended output channel.
    print(  # noqa: T201
        "vins-rnd: installed successfully.\n"
        "This project primarily exposes library packages: core/, dataset/, visualizer/, logger/.\n"
        "See `pipeline/README.md` and `examples/` for runnable demos.",
    )
    return 0
