"""Deterministic, Markdown-aware document chunking."""

from collections.abc import Iterable

from app.models.retrieval import Document, DocumentChunk
from app.retrieval._utils import slugify

DEFAULT_MAX_CHARS = 1_200


def chunk_documents(
    documents: Iterable[Document], max_chars: int = DEFAULT_MAX_CHARS
) -> list[DocumentChunk]:
    """Chunk documents in input order using their Markdown section headings."""
    return [chunk for document in documents for chunk in chunk_document(document, max_chars)]


def chunk_document(document: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[DocumentChunk]:
    """Create stable chunks from level-two-and-deeper Markdown sections."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    sections = _extract_sections(document.content)
    chunks = []
    for section, text in sections:
        for part_number, fragment in enumerate(_split_text(text, max_chars), start=1):
            section_slug = slugify(section)
            suffix = "" if part_number == 1 else f"-{part_number}"
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}-{section_slug}{suffix}",
                    document_id=document.document_id,
                    title=document.title,
                    section=section,
                    source=document.source,
                    text=fragment,
                )
            )
    return chunks


def _extract_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_section = "Overview"
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("##") and line.lstrip("#").startswith(" "):
            if current_lines:
                sections.append((current_section, "\n".join(current_lines).strip()))
            current_section = line.lstrip("#").strip()
            current_lines = []
        elif line.startswith("# "):
            continue
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, "\n".join(current_lines).strip()))
    return [(section, text) for section, text in sections if text]


def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    fragments: list[str] = []
    current_words: list[str] = []
    current_length = 0
    for word in words:
        additional_length = len(word) + (1 if current_words else 0)
        if current_words and current_length + additional_length > max_chars:
            fragments.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += additional_length
    if current_words:
        fragments.append(" ".join(current_words))
    return fragments
