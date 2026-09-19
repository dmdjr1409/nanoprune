from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re

class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        file_path: str,
        file_name: str,
        text: str,
        line_start: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.file_name = file_name
        self.text = text.strip()
        self.line_start = line_start
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "text": self.text,
            "line_start": self.line_start,
            "metadata": self.metadata,
        }


class LocalDocumentIndexer:
    """
    Fast, local, zero-cloud document indexer.
    Parses directories and generates semantic chunk indices.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []

    def index_directory(self, dir_path: str, recursive: bool = True) -> int:
        target = Path(dir_path)
        if not target.exists() or not target.is_dir():
            raise ValueError(f"Directory does not exist: {dir_path}")

        supported = {".txt", ".md", ".json", ".csv", ".log"}
        pattern = "**/*" if recursive else "*"

        count = 0
        for file in target.glob(pattern):
            if file.is_file() and file.suffix.lower() in supported:
                self.index_file(str(file))
                count += 1

        return count

    def index_file(self, file_path: str) -> List[DocumentChunk]:
        path = Path(file_path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        file_chunks = self._chunk_content(content, str(path), path.name)
        self.chunks.extend(file_chunks)
        return file_chunks

    def _chunk_content(self, text: str, file_path: str, file_name: str) -> List[DocumentChunk]:
        paragraphs = re.split(r"\n\s*\n", text)
        result = []
        chunk_idx = 0

        # Context header derived from filename and top heading
        doc_title = Path(file_name).stem.replace("_", " ").title()
        context_prefix = f"[{doc_title}] "

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            full_para = f"{context_prefix}{para}" if not para.startswith("[") else para

            if len(full_para) <= self.chunk_size:
                result.append(
                    DocumentChunk(
                        chunk_id=f"{file_name}#{chunk_idx}",
                        file_path=file_path,
                        file_name=file_name,
                        text=full_para,
                    )
                )
                chunk_idx += 1
            else:
                sentences = re.split(r"(?<=[.?!])\s+", para)
                current = []
                current_len = len(context_prefix)
                for s in sentences:
                    if current_len + len(s) > self.chunk_size and current:
                        result.append(
                            DocumentChunk(
                                chunk_id=f"{file_name}#{chunk_idx}",
                                file_path=file_path,
                                file_name=file_name,
                                text=f"{context_prefix}{' '.join(current)}",
                            )
                        )
                        chunk_idx += 1
                        current = []
                        current_len = len(context_prefix)
                    current.append(s)
                    current_len += len(s)

                if current:
                    result.append(
                        DocumentChunk(
                            chunk_id=f"{file_name}#{chunk_idx}",
                            file_path=file_path,
                            file_name=file_name,
                            text=f"{context_prefix}{' '.join(current)}",
                        )
                    )
                    chunk_idx += 1

        return result
