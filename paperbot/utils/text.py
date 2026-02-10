"""Text processing utilities for DOI extraction and title cleaning."""

import html
import re
from typing import Any, Optional

from bs4 import BeautifulSoup

# DOI regex pattern: 10.XXXX/... format
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def normalize_doi(doi: str) -> str:
    """Normalize DOI by removing URL prefixes and converting to lowercase."""
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("https://dx.doi.org/", "").replace("http://dx.doi.org/", "")
    return doi.strip().lower()


def extract_doi(entry: dict[str, Any]) -> Optional[str]:
    """Extract DOI from an RSS feed entry.

    Searches multiple fields in the entry for DOI patterns.

    Args:
        entry: Parsed RSS feed entry dictionary

    Returns:
        Normalized DOI string if found, None otherwise
    """
    # Check dedicated DOI fields
    for key in ["doi", "prism_doi", "dc_identifier", "id", "guid"]:
        val = entry.get(key)
        if isinstance(val, str):
            match = DOI_RE.search(val)
            if match:
                return normalize_doi(match.group(0))

    # Check common text fields
    for field in ["link", "summary", "title"]:
        val = entry.get(field)
        if isinstance(val, str):
            match = DOI_RE.search(val)
            if match:
                return normalize_doi(match.group(0))

    # Check content array (Atom feeds)
    content = entry.get("content")
    if isinstance(content, list):
        for item in content:
            val = item.get("value")
            if isinstance(val, str):
                match = DOI_RE.search(val)
                if match:
                    return normalize_doi(match.group(0))

    return None


def clean_title(text: str) -> str:
    """Clean title by removing MathML/HTML tags and normalizing whitespace.

    Handles messy titles from RSS feeds (e.g., ScienceDirect) that may contain
    embedded MathML, HTML tags, and HTML entities.

    Args:
        text: Raw title string

    Returns:
        Cleaned title string, or "(no title)" if empty
    """
    if not text or not isinstance(text, str):
        return text or "(no title)"

    # Remove <math ...>...</math> blocks (including attributes, multiline)
    text = re.sub(r"<math[\s>].*?</math>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove self-closing <math ... />
    text = re.sub(r"<math[\s\S]*?/>", " ", text, flags=re.IGNORECASE)
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Normalize whitespace
    text = " ".join(text.split()).strip()

    return text or "(no title)"


# ---------------------------------------------------------------------------
# LaTeX → plain-text conversion tables
# ---------------------------------------------------------------------------

_LATEX_GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ",
    r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π",
    r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ",
    r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ",
    r"\Psi": "Ψ", r"\Omega": "Ω",
}

_LATEX_SYMBOLS = {
    r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\sim": "~", r"\equiv": "≡", r"\propto": "∝",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\leftrightarrow": "↔",
    r"\Rightarrow": "⇒", r"\Leftarrow": "⇐",
    r"\langle": "⟨", r"\rangle": "⟩",
    r"\ldots": "…", r"\cdots": "⋯", r"\dots": "…",
    r"\circ": "°", r"\degree": "°",
    r"\AA": "Å", r"\angstrom": "Å",
}

# Pre-compiled regex for LaTeX commands (longest match first)
_LATEX_CMD_RE = re.compile(
    "|".join(
        re.escape(k)
        for k in sorted(
            list(_LATEX_GREEK) + list(_LATEX_SYMBOLS),
            key=len, reverse=True,
        )
    )
)

# Patterns applied in order
_LATEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    # \frac{a}{b} → a/b
    (re.compile(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}"), r"\1/\2"),
    # \sqrt{x} → √(x)
    (re.compile(r"\\sqrt\s*\{([^}]*)\}"), r"√(\1)"),
    # \text{...}, \mathrm{...}, \mathbf{...}, \textit{...}, \mathit{...}
    (re.compile(r"\\(?:text|mathrm|mathbf|mathit|textit|textrm|operatorname)\s*\{([^}]*)\}"), r"\1"),
    # \overline{X} → X̄  (approximate)
    (re.compile(r"\\overline\s*\{([^}]*)\}"), r"\1"),
    # \hat{x} → x
    (re.compile(r"\\hat\s*\{([^}]*)\}"), r"\1"),
    # \vec{x} → x
    (re.compile(r"\\vec\s*\{([^}]*)\}"), r"\1"),
    # \bar{x} → x
    (re.compile(r"\\bar\s*\{([^}]*)\}"), r"\1"),
    # ^{...} → superscript content
    (re.compile(r"\^\{([^}]*)\}"), r"^\1"),
    # _{...} → subscript content
    (re.compile(r"_\{([^}]*)\}"), r"_\1"),
    # Single-char super/sub: ^x → ^x, _x → _x (keep as-is)
]


def _latex_to_plain(text: str) -> str:
    """Best-effort conversion of inline LaTeX math to readable plain text."""
    # Strip math delimiters: $$...$$ and $...$
    text = re.sub(r"\$\$(.*?)\$\$", r" \1 ", text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", r" \1 ", text)
    # Also handle \( ... \) and \[ ... \]
    text = re.sub(r"\\\((.*?)\\\)", r" \1 ", text)
    text = re.sub(r"\\\[(.*?)\\\]", r" \1 ", text, flags=re.DOTALL)

    # Apply structural patterns (frac, sqrt, text wrappers, etc.)
    for pat, repl in _LATEX_PATTERNS:
        text = pat.sub(repl, text)

    # Replace Greek letters and symbols
    text = _LATEX_CMD_RE.sub(lambda m: {**_LATEX_GREEK, **_LATEX_SYMBOLS}[m.group()], text)

    # Clean up remaining backslash commands (e.g. \hspace, \quad, \,)
    text = re.sub(r"\\(?:hspace|quad|qquad|,|;|!|:)\b\s*(?:\{[^}]*\})?", " ", text)
    # Remove remaining unknown \commands but keep the next char
    text = re.sub(r"\\([a-zA-Z]+)\s*", r"\1 ", text)

    # Remove leftover braces
    text = text.replace("{", "").replace("}", "")
    # Normalise whitespace
    text = " ".join(text.split())
    return text


def _strip_mathml_to_text(text: str) -> str:
    """Remove MathML blocks (<math>, <mml:math>) so only plain text remains."""
    soup = BeautifulSoup(text, "html.parser")
    for math_tag in soup.find_all(["math", "mml:math"]):
        math_tag.decompose()
    body = soup.find("body")
    if body:
        return body.decode_contents()
    # Fragment without body (e.g. parser difference)
    return str(soup)


def clean_abstract(text: str) -> str:
    """Clean abstract text.

    1. Strip MathML blocks and replace with their text content (e.g. b=1/2⟨111⟩).
    2. Strip leading "Abstract" / "ABSTRACT" prefix (with optional colon/dash).
    3. Convert inline LaTeX math to readable plain-text Unicode.
    4. Normalise whitespace.
    """
    if not text:
        return text

    # Remove MathML tags
    text = _strip_mathml_to_text(text)

    # Strip leading "Abstract" prefix
    text = re.sub(r"^\s*abstract[\s.:;—–-]*", "", text, flags=re.IGNORECASE).strip()

    # Convert LaTeX to plain text
    text = _latex_to_plain(text)

    return text.strip()


def parse_published(entry: dict[str, Any]) -> Optional[str]:
    """Parse publication date from RSS entry.

    Args:
        entry: Parsed RSS feed entry dictionary

    Returns:
        ISO format date string (YYYY-MM-DD) if found, None otherwise
    """
    from dateutil import parser as dtparser

    for field in ["published", "updated"]:
        val = entry.get(field)
        if val:
            try:
                dt = dtparser.parse(val)
                return dt.date().isoformat()
            except Exception:
                pass

    return None
