from __future__ import annotations

"""
Optional helper module.

This file is included for repositories that want a clean place to centralize
stage names. It does not change behavior by itself. The sweep script can be
used directly without modifying main.py:

    python scripts/run_surrogate_sweep.py --dataset unsw_nb15 --core-only

If you want to expose a main.py stage later, map:
    surrogate_sweep -> scripts/run_surrogate_sweep.py
"""

SUPPORTED_EXTRA_STAGES = {"surrogate_sweep"}
