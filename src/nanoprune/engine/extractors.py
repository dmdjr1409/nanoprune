"""Plain-text extraction for the document formats the local search engine indexes."""
import io
import zipfile
from pathlib import Path
from typing import Union
from xml.etree import ElementTree

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".log", ".rst"}
RICH_SUFFIXES = {".docx", ".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | RICH_SUFFIXES

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# Refuse archives whose main XML part would inflate beyond this size (zip bombs).
MAX_DOCX_XML_BYTES = 64 * 1024 * 1024


class MissingDependencyError(RuntimeError):
    """An optional package is needed to read this format."""


def _docx_text(data: bytes) -> str:
    """Text of a .docx file, one Word paragraph per block (standard library only)."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.getinfo("word/document.xml").file_size > MAX_DOCX_XML_BYTES:
            raise ValueError("The .docx document is too large to index")
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for para in root.iter(f"{_W_NS}p"):
        parts = []
        for node in para.iter():
            if node.tag == f"{_W_NS}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W_NS}tab":
                parts.append("\t")
            elif node.tag in (f"{_W_NS}br", f"{_W_NS}cr"):
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise MissingDependencyError("Reading PDF files needs pypdf: pip install 'nanoprune[docs]'")
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)


def extract_text_from_bytes(data: bytes, name: str) -> str:
    """Decode a document given its raw bytes and file name."""
    suffix = Path(name).suffix.lower()
    if suffix == ".docx":
        return _docx_text(data)
    if suffix == ".pdf":
        return _pdf_text(data)
    if suffix in TEXT_SUFFIXES:
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: {name}")


def extract_text(path: Union[str, Path]) -> str:
    """Read a supported document from disk as plain text."""
    path = Path(path)
    return extract_text_from_bytes(path.read_bytes(), path.name)
