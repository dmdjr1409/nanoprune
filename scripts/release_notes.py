"""Print the GitHub release notes of a version, after checking that the version is consistent.

Usage: python scripts/release_notes.py 0.7.0 > release-notes.md

Fails when the version is malformed, differs from pyproject.toml or
nanoprune.__version__, or has no section in CHANGELOG.md. Used by
.github/workflows/release.yml.
"""
import os
import re
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def package_versions() -> Dict[str, str]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    init = (ROOT / "src" / "nanoprune" / "__init__.py").read_text(encoding="utf-8")
    found = {
        "pyproject.toml": re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M),
        "nanoprune/__init__.py": re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M),
    }
    return {source: match.group(1) if match else "?" for source, match in found.items()}


def changelog_section(version: str, changelog: str) -> str:
    match = re.search(rf"^## {re.escape(version)}(?=[ \t]|$)[^\n]*\n(.*?)(?=^## |\Z)", changelog, re.M | re.S)
    if not match or not match.group(1).strip():
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    return match.group(1).strip()


def release_notes(version: str) -> str:
    if not VERSION_RE.match(version):
        raise ValueError(f"'{version}' is not a version number such as 0.7.0")
    for source, found in package_versions().items():
        if found != version:
            raise ValueError(f"{source} says {found}, not {version}")
    section = changelog_section(version, (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    repo = os.environ.get("GITHUB_REPOSITORY", "dmdjr1409/nanoprune")
    wheel = f"https://github.com/{repo}/releases/download/v{version}/nanoprune-{version}-py3-none-any.whl"
    readme = f"https://github.com/{repo}/blob/v{version}/README.md"
    return f"""{section}

### Install

```bash
pip install "nanoprune[semantic] @ {wheel}"
nanoprune download --dense multilingual-e5-large   # semantic model, once (about 550 MB)
nanoprune app                                      # local search app in the browser
```

Which scorer to use, and how each one compares with keyword matching on the
evaluation suites: see the status table of the [README]({readme}).
"""


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        sys.stdout.write(release_notes(argv[1]))
    except ValueError as exc:
        print(f"release_notes.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
