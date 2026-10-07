"""Build an anonymized copy of the repository for review.

Copies the working tree without git history, removes the files that only make sense on GitHub,
replaces the few places that name the author or the account, and then scans the result for
identifying strings. Fails if any are left.

Usage: python3 scripts/make_anonymous_supplement.py OUTPUT.zip
"""
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".github", "__pycache__", ".venv", "venv", ".ruff_cache", ".pytest_cache"}
SKIP_FILES = {".DS_Store", "make_anonymous_supplement.py", "CODEOWNERS"}
IDENTIFYING = re.compile(
    r"amethystani|animesh|mishra|khetarpaul|krishang|shiv nadar|snu\.edu|prakhar|zghal|cyu\.fr", re.I)


def strip_repo_switch(text: str) -> str:
    """Keep only the anonymous footnote: drop the camera-ready switch and its real URL."""
    text = re.sub(r"% Footnote on the first page\..*?\\showrepofalse\n\n", "", text, flags=re.S)
    return re.sub(r"\\thanks\{\\ifshowrepo .*?\\else (.*?)\\fi\}", r"\\thanks{\1}", text, flags=re.S)


def scrub(path: Path) -> None:
    text = path.read_text()
    name = path.name
    if name == "LICENSE":
        text = text.replace("Animesh Mishra", "Anonymous authors")
    elif name in ("CODE_OF_CONDUCT.md", "SECURITY.md"):
        text = re.sub(r"[\w.+-]+@[\w.-]+", "[contact withheld for review]", text)
    elif name == "README.md" and path.parent == ROOT_COPY:
        text = "\n".join(line for line in text.split("\n") if "github.com/" not in line)
    elif name == "main.tex":
        text = strip_repo_switch(text)
    path.write_text(text)


def main() -> None:
    global ROOT_COPY
    out = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        ROOT_COPY = Path(tmp) / "tessera-supplement"
        shutil.copytree(ROOT, ROOT_COPY,
                        ignore=lambda d, names: [n for n in names if n in SKIP_DIRS | SKIP_FILES])
        for f in ROOT_COPY.rglob("*"):
            if f.is_file() and f.suffix in {".md", ".tex", ".txt", ".yml", ".py", ""}:
                try:
                    scrub(f)
                except UnicodeDecodeError:
                    pass
        hits = []
        for f in ROOT_COPY.rglob("*"):
            if f.is_file():
                try:
                    if IDENTIFYING.search(f.read_text(errors="ignore")):
                        hits.append(str(f.relative_to(ROOT_COPY)))
                except OSError:
                    pass
        if hits:
            sys.exit("identifying strings remain in: " + ", ".join(hits))
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(ROOT_COPY.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(ROOT_COPY.parent))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
