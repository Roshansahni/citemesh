"""Small helpers: sanitising untrusted text, normalising, slugs, interleaving."""
import re
from itertools import zip_longest

# Defence-in-depth against prompt injection in fetched pages. Real defences are
# (1) data delimiters in prompts, (2) quote verification, (3) agents have no
# side-effect tools. This regex just strips the most blatant attempts.
INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)"
    r"|disregard .{0,40}instructions|new instructions\s*:|system prompt"
    r"|you are now (a|an|the) )",
    re.I,
)


def sanitize(text: str) -> str:
    return INJECTION.sub("[removed]", text)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def slugify(s: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n] or "report"


def interleave(groups: list[list]) -> list:
    """Round-robin merge so one source type cannot crowd out the others."""
    sentinel = object()
    out = []
    for row in zip_longest(*groups, fillvalue=sentinel):
        out.extend(x for x in row if x is not sentinel)
    return out
