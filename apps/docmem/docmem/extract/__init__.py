"""Wikipedia wikitext extractors, by name: each turns a revision's wikitext into markdown and its links.

A wiki document's "extractor" names the one that made its passages' text; docmem.sources.read_span reads them back
with the same one.
"""

from . import wikiplain, wikivalues

RENDERERS = {
    wikiplain.EXTRACTOR: wikiplain.to_markdown_with_links,
    wikivalues.EXTRACTOR: wikivalues.to_markdown_with_links,
}


def to_markdown_with_links(extractor: str, title: str, wikitext: str):
    """(markdown, links) of a wikitext by the named extractor."""
    if extractor not in RENDERERS:
        raise ValueError(f"unknown wiki extractor {extractor!r} (one of {sorted(RENDERERS)})")
    return RENDERERS[extractor](title, wikitext)
