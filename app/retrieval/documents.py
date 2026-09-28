"""Markdown document loading for the local financial policy corpus."""

from pathlib import Path

from app.models.retrieval import Document
from app.retrieval._utils import slugify


class DocumentLoadError(ValueError):
    """Raised when a Markdown document cannot be loaded into the corpus."""


def load_markdown_documents(directory: str | Path) -> list[Document]:
    """Load every Markdown file in a directory in deterministic filename order."""
    document_directory = Path(directory)
    if not document_directory.is_dir():
        raise DocumentLoadError(f"Document directory does not exist: {document_directory}")

    documents = []
    for path in sorted(document_directory.glob("*.md")):
        documents.append(load_markdown_document(path))
    return documents


def load_markdown_document(path: str | Path) -> Document:
    """Load one Markdown document, requiring a non-empty level-one title."""
    document_path = Path(path)
    try:
        content = document_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DocumentLoadError(f"Unable to read document: {document_path}") from error

    title = _extract_title(content, document_path)
    return Document(
        document_id=slugify(document_path.stem),
        title=title,
        source=document_path.name,
        content=content,
    )


def _extract_title(content: str, path: Path) -> str:
    for line in content.splitlines():
        if line.startswith("# ") and line[2:].strip():
            return line[2:].strip()
    raise DocumentLoadError(f"Document requires a non-empty level-one heading: {path}")
