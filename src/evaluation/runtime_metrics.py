from __future__ import annotations

import time
from typing import Any, Callable


def measure_runtime(func: Callable[..., Any], *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    elapsed = time.time() - start
    return out, elapsed
