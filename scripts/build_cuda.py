"""Compatibility build entry: build the single installed extension, never JIT-load."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root), "--no-build-isolation"],
        check=True,
    )


if __name__ == "__main__":
    main()
