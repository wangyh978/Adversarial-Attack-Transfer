from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "torch",
    "xgboost",
    "yaml",
    "pyarrow",
]

OPTIONAL_PACKAGES = [
    "pytorch_tabnet",
]


def check_package(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as e:
        return False, str(e)


def main() -> None:
    print("=== Python Environment Check ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    print("\n[Required]")
    for pkg in REQUIRED_PACKAGES:
        ok, info = check_package(pkg)
        print(f"{pkg:20s} {'OK' if ok else 'FAIL'} | {info}")

    print("\n[Optional]")
    for pkg in OPTIONAL_PACKAGES:
        ok, info = check_package(pkg)
        print(f"{pkg:20s} {'OK' if ok else 'WARN'} | {info}")

    try:
        import torch
        print("\n[Torch]")
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
    except Exception as e:
        print("Torch check failed:", e)


if __name__ == "__main__":
    main()
