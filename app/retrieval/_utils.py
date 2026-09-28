"""Internal helpers shared by retrieval modules."""

import re


def slugify(value: str) -> str:
    """Convert a title or section heading into a stable citation identifier."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"
