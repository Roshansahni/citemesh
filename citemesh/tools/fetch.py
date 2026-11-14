"""Polite page fetcher: obeys robots.txt, handles HTML + PDF, strips injection."""
import io
import logging
import re
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from ..utils import sanitize
from .web import UA

log = logging.getLogger("citemesh.tools")
_robots: dict[str, RobotFileParser] = {}


def allowed(url: str) -> bool:
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in _robots:
        rp = RobotFileParser()
        try:
            r = requests.get(base + "/robots.txt", headers={"User-Agent": UA}, timeout=6)
            if r.status_code in (401, 403):
                rp.parse(["User-agent: *", "Disallow: /"])
            elif r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                rp.parse([])
        except requests.RequestException:
            rp.parse([])
        _robots[base] = rp
    return _robots[base].can_fetch(UA, url)


def pdf_to_text(data: bytes, max_pages: int = 15) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((pg.extract_text() or "") for pg in reader.pages[:max_pages])


def fetch_text(url: str, max_chars: int = 20000) -> str:
    """Return cleaned text of a page/PDF, or '' if blocked/failed."""
    if not allowed(url):
        log.info("blocked by robots.txt: %s", url)
        return ""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = pdf_to_text(r.content)
        elif "html" in ctype or "text" in ctype:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
        else:
            return ""
    except Exception as e:
        log.warning("fetch failed %s: %s", url, e)
        return ""
    return sanitize(re.sub(r"\s+", " ", text))[:max_chars]
