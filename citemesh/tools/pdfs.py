"""Load the user's own PDFs from data/pdfs as sources."""
import logging
from pathlib import Path

from pypdf import PdfReader

from ..utils import sanitize

log = logging.getLogger("citemesh.tools")


def load_local_pdfs(folder: str, max_chars: int = 60000) -> list[dict]:
    out = []
    for path in sorted(Path(folder).glob("*.pdf")):
        try:
            text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        except Exception as e:  # corrupt/encrypted PDFs
            log.warning("cannot read %s: %s", path.name, e)
            continue
        text = sanitize(" ".join(text.split()))[:max_chars]
        if text:
            out.append({"title": path.stem, "url": path.resolve().as_uri(), "snippet": text[:300], "text": text, "kind": "pdf"})
    return out
