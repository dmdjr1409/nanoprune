import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..errors import OperationCancelled
from .extractors import SUPPORTED_SUFFIXES, MissingDependencyError, extract_text

# progress(done, total) and should_stop() callbacks of long operations.
ProgressCallback = Callable[[int, int], None]
StopCallback = Callable[[], bool]

# Directories never worth indexing (VCS metadata, dependencies, caches).
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "site-packages",
}

# A segment ends after sentence punctuation followed by whitespace, or at a line break.
_SEGMENT_RE = re.compile(r"[^\n]*?(?:[.?!](?=\s)|\n|$)")
_HAS_WORD_RE = re.compile(r"\w", re.UNICODE)


class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        file_path: str,
        file_name: str,
        text: str,
        line_start: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        line_end: Optional[int] = None,
        body: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.file_name = file_name
        # ``text`` (with the "[Document title]" prefix) is what gets scored;
        # ``body`` is the original passage, shown to users.
        self.text = text.strip()
        self.body = (body if body is not None else text).strip()
        self.line_start = line_start
        self.line_end = line_end if line_end is not None else line_start
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "text": self.text,
            "body": self.body,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "metadata": self.metadata,
        }


def _paragraphs(text: str) -> Iterator[Tuple[int, int, str]]:
    """Yield (first line, last line, paragraph) for blocks separated by blank lines."""
    lines = text.splitlines()
    block: List[str] = []
    start = 0
    for number, line in enumerate(lines, 1):
        if line.strip():
            if not block:
                start = number
            block.append(line)
        elif block:
            yield start, number - 1, "\n".join(block)
            block = []
    if block:
        yield start, len(lines), "\n".join(block)


def _segments(paragraph: str) -> List[Tuple[int, str]]:
    """Split a paragraph into (character offset, sentence or line) segments."""
    result = []
    for match in _SEGMENT_RE.finditer(paragraph):
        segment = match.group(0).strip()
        if segment:
            result.append((match.start() + len(match.group(0)) - len(match.group(0).lstrip()), segment))
    return result


class LocalDocumentIndexer:
    """
    Fast, local, zero-cloud document indexer.
    Parses directories and splits documents into overlapping passages.
    """
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_file_bytes: int = 25 * 1024 * 1024,
        max_files: int = 5000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.chunks: List[DocumentChunk] = []
        self.version = 0
        self.last_report: Dict[str, Any] = {}

    def clear(self) -> None:
        self.chunks.clear()
        self.version += 1

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> int:
        """Index every supported file under ``dir_path``; returns the number of files indexed.

        Hidden directories and dependency/cache folders are skipped, files larger
        than ``max_file_bytes`` are ignored and at most ``max_files`` files are read.
        Details are stored in ``last_report``.

        ``progress(files_done, files_total)`` is called after each file, and
        ``should_stop()`` is polled between files: when it returns True,
        ``OperationCancelled`` is raised (passages already added stay indexed).
        """
        target = Path(dir_path).expanduser()
        if not target.exists() or not target.is_dir():
            raise ValueError(f"Directory does not exist: {dir_path}")

        # List first, so that progress can be reported against a total.
        paths: List[Path] = []
        for path in self._walk(target, recursive):
            if should_stop is not None and should_stop():
                raise OperationCancelled("Indexing cancelled")
            paths.append(path)
            if len(paths) > self.max_files:
                break
        report: Dict[str, Any] = {"files_indexed": 0, "skipped": [], "truncated": len(paths) > self.max_files}
        paths = paths[:self.max_files]
        self.last_report = report
        if progress is not None:
            progress(0, len(paths))

        for done, path in enumerate(paths, 1):
            if should_stop is not None and should_stop():
                raise OperationCancelled("Indexing cancelled")
            try:
                if path.stat().st_size > self.max_file_bytes:
                    report["skipped"].append((str(path), "file too large"))
                else:
                    rel = path.relative_to(target).as_posix()
                    self.index_file(str(path), chunk_prefix=rel)
                    report["files_indexed"] += 1
            except MissingDependencyError as exc:
                report["skipped"].append((str(path), str(exc)))
            except (OSError, ValueError, KeyError) as exc:
                report["skipped"].append((str(path), f"unreadable: {exc}"))
            if progress is not None:
                progress(done, len(paths))
        return report["files_indexed"]

    @staticmethod
    def _walk(root: Path, recursive: bool) -> Iterator[Path]:
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
            for name in sorted(files):
                path = Path(current) / name
                if not name.startswith(".") and path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                    yield path
            if not recursive:
                break

    def index_file(self, file_path: str, chunk_prefix: Optional[str] = None) -> List[DocumentChunk]:
        """Index one file. Raises MissingDependencyError for PDFs without pypdf."""
        path = Path(file_path)
        try:
            content = extract_text(path)
        except MissingDependencyError:
            raise
        except (OSError, ValueError, KeyError):
            return []
        return self.index_text(content, path.name, str(path), chunk_prefix=chunk_prefix)

    def index_text(
        self,
        text: str,
        file_name: str,
        file_path: Optional[str] = None,
        chunk_prefix: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk raw text and add the passages to the index."""
        chunks = self._chunk_content(text, file_path or file_name, file_name, chunk_prefix=chunk_prefix)
        self.chunks.extend(chunks)
        self.version += 1
        return chunks

    def _chunk_content(
        self,
        text: str,
        file_path: str,
        file_name: str,
        chunk_prefix: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Split text into passages (does not add them to the index)."""
        doc_title = Path(file_name).stem.replace("_", " ").title()
        context_prefix = f"[{doc_title}] "
        id_prefix = chunk_prefix or file_name
        budget = max(50, self.chunk_size - len(context_prefix))
        result: List[DocumentChunk] = []

        def join(units: List[Tuple[int, str]]) -> str:
            # Units from the same line are sentences; a new line keeps its line break.
            parts = []
            for i, (line, segment) in enumerate(units):
                if i:
                    parts.append(" " if line == units[i - 1][0] else "\n")
                parts.append(segment)
            return "".join(parts)

        def emit(body: str, line_start: int, line_end: int, shared: int = 0) -> None:
            result.append(DocumentChunk(
                chunk_id=f"{id_prefix}#{len(result)}",
                file_path=file_path,
                file_name=file_name,
                text=f"{context_prefix}{body}",
                body=body,
                line_start=line_start,
                line_end=line_end,
                # Characters at the start of the body repeated from the previous passage.
                metadata={"shared_prev": shared} if shared else None,
            ))

        for first_line, last_line, para in _paragraphs(text):
            para = para.strip()
            if not _HAS_WORD_RE.search(para):
                continue  # separators such as "---"
            if len(para) <= budget:
                emit(para, first_line, last_line)
                continue

            # Long paragraph: pack sentences/lines into passages with overlap.
            units: List[Tuple[int, str]] = []
            for offset, segment in _segments(para):
                line = first_line + para.count("\n", 0, offset)
                while len(segment) > budget:
                    cut = segment.rfind(" ", 0, budget)
                    cut = cut if cut > budget // 2 else budget
                    units.append((line, segment[:cut].strip()))
                    segment = segment[cut:].strip()
                if segment:
                    units.append((line, segment))

            current: List[Tuple[int, str]] = []
            shared = 0
            for unit in units:
                length = sum(len(s) + 1 for _, s in current)
                if current and length + len(unit[1]) > budget:
                    emit(join(current), current[0][0], current[-1][0], shared)
                    # Carry trailing units into the next passage, up to chunk_overlap characters.
                    overlap: List[Tuple[int, str]] = []
                    carried = 0
                    for prev in reversed(current[1:]):
                        if carried + len(prev[1]) > self.chunk_overlap:
                            break
                        overlap.insert(0, prev)
                        carried += len(prev[1]) + 1
                    current = overlap
                    shared = len(join(overlap))
                current.append(unit)
            if current:
                emit(join(current), current[0][0], current[-1][0], shared)

        return result
