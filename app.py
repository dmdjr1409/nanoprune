#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to sys.path automatically
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nanoprune.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["app", "--dir", str(Path(__file__).parent / "sample_data" / "medical")])
    sys.exit(main())
