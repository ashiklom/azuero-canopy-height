"""Command-line entry point: `azuero <step> [--profile small|full] [step args...]`.

Each step maps to a `main(argv)` function in a module; steps are imported lazily so that
optional heavy dependencies are only needed for the steps that use them.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys

STEPS: dict[str, str] = {
    "aoi": "azuero.aoi:cli",
    "lidar": "azuero.io_lidar:cli",
    "alphaearth": "azuero.io_alphaearth:cli",
    "dem": "azuero.io_dem:cli",
    "hls": "azuero.io_hls:cli",
    "reference": "azuero.io_reference:cli",
}


def register(name: str, target: str) -> None:
    STEPS[name] = target


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="azuero", description=__doc__)
    p.add_argument("step", choices=sorted(STEPS))
    p.add_argument("--profile", choices=["small", "full"], default=None,
                   help="resource profile (overrides config / AZUERO_PROFILE)")
    p.add_argument("-v", "--verbose", action="store_true")
    args, rest = p.parse_known_args(argv)
    if args.profile:
        os.environ["AZUERO_PROFILE"] = args.profile
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    mod, fn = STEPS[args.step].split(":")
    getattr(importlib.import_module(mod), fn)(rest)


if __name__ == "__main__":
    main()
