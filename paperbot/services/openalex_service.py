"""OpenAlex API client for enriching paper metadata (authors, journal, abstract)."""

import httpx
import html
from typing import Any


OPENALEX_WORKS_URL = "https://api.openalex.org/works/https://doi.org/{doi}"
TIMEOUT = 10.0


async def get_paper_info(doi: str) -> dict[str, Any]:
    """Fetch paper metadata from OpenAlex by DOI.

    Args:
        doi: DOI string (with or without https://doi.org/ prefix).

    Returns:
        Dict with keys: authors (list of display_name), journal (str), abstract (str).
        On error: {"error": "message"}.
    """
    doi = doi.strip()
    if not doi:
        return {"error": "DOI is empty"}
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "", 1)
    url = OPENALEX_WORKS_URL.format(doi=doi)

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return {"error": f"OpenAlex returned {response.status_code}"}

            data = response.json()

            # 1. Authors
            authors = [
                auth.get("author", {}).get("display_name", "")
                for auth in data.get("authorships", [])
            ]
            authors = [a for a in authors if a]

            # 2. Journal name
            journal = (
                data.get("primary_location", {})
                .get("source", {})
                .get("display_name", "N/A")
            )

            # 3. Abstract: reconstruct from inverted index
            inv_index = data.get("abstract_inverted_index")
            abstract = "N/A"
            if inv_index:
                word_counts = {}
                for word, pos_list in inv_index.items():
                    for pos in pos_list:
                        word_counts[pos] = word
                abstract = " ".join(
                    word_counts[i] for i in sorted(word_counts.keys())
                )

            abstract = html.unescape(abstract)
            return {
                "authors": authors,
                "journal": journal,
                "abstract": abstract,
            }
    except httpx.TimeoutException:
        return {"error": "OpenAlex 요청 시간 초과"}
    except Exception as e:
        return {"error": str(e)}
