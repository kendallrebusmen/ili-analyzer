"""The Nemi product brand: name, tagline, version label and the inline wordmark. Local only, no URLs."""

from __future__ import annotations

import ili

PRODUCT_NAME = "Nemi"
PRODUCT_TAGLINE = "ILI run comparison"


def product_label() -> str:
    """``Nemi <version>``, as shown on the reports and in the app."""
    return f"{PRODUCT_NAME} {ili.__version__}"


def wordmark_svg(size: int = 30) -> str:
    """Inline SVG wordmark: an ink ring, a teal arc and a teal centre dot (``--nemi-accent`` is set by the app CSS)."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 30 30" role="img" aria-label="{PRODUCT_NAME}">'
            '<circle cx="15" cy="15" r="11" fill="none" stroke="currentColor" stroke-width="3.2"/>'
            '<path d="M15 4 A11 11 0 0 1 26 15" fill="none" stroke="var(--nemi-accent)" stroke-width="3.2" '
            'stroke-linecap="round"/>'
            '<circle cx="15" cy="15" r="3" fill="var(--nemi-accent)"/></svg>')
