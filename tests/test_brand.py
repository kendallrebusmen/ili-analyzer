"""The Nemi brand: product label, local wordmark, and the app title."""

import re
from pathlib import Path

import ili
from ili import brand

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_product_label():
    assert brand.product_label() == f"Nemi {ili.__version__}"
    assert brand.PRODUCT_NAME == "Nemi" and brand.PRODUCT_TAGLINE == "ILI run comparison"


def test_wordmark_is_local_svg():
    svg = brand.wordmark_svg()
    assert 'role="img"' in svg and 'aria-label="Nemi"' in svg
    assert "http" not in svg and "xmlns" not in svg
    assert 'viewBox="0 0 30 30"' in svg and 'width="30"' in svg and "var(--nemi-accent)" in svg
    assert 'width="48" height="48"' in brand.wordmark_svg(48)


def test_app_title_is_nemi():
    match = re.search(r'^TITLE = "([^"]*)"', APP.read_text(encoding="utf-8"), re.M)
    assert match and match.group(1) == "Nemi"
