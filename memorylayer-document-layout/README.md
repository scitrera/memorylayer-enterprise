# MemoryLayer document layout

A dependency-free AGPL Python library for locating quotations and cited figures in MemoryLayer's
version-1 OCR layouts. It works on live pages and saved snapshots, with no server,
network requests, model calls, customer imports or raw source-file access.

```python
from memorylayer_document_layout import locate_quote

regions = locate_quote(quote, transcript, layout, image_sha256=expected_image_hash)
table = locate_quote(row_quote, transcript, layout, mode="table_row",
                     image_sha256=expected_image_hash)
figures = locate_quote(observed_quote, transcript, layout, mode="figure",
                       locator='Images below the "Sample artwork" heading',
                       image_sha256=expected_image_hash)
# Explicit page-local OCR marker; not the document's printed figure numbering:
figure = locate_quote("[figure 2]", transcript, layout, mode="figure",
                      image_sha256=expected_image_hash)
```

Results contain region_id, image_sha256, normalized bbox and origin="ocr".
`validate_layout(layout, transcript, image_sha256=...)` strips private/provider
metadata and rejects stale identities. Invalid rectangles remain unlocated.
Duplicate or ambiguous quotations do not acquire highlights. Text matching is
whitespace-normalized; table matching requires every cell in one complete,
ordered row. The table mode recognizes OCR currency dollars misread as LaTeX
math delimiters; digits, signs and percentages still have to agree. It returns
the existing whole-table box, never inferred row/cell rectangles.

Figure mode connects the OCR transcript's page-local `[figure N]` placeholders
with the recorded image boxes, including images whose OCR text is empty. It
requires the complete default rendered transcript to agree with layout reading
order, including omitted headers/page numbers. Custom placeholder/caption
rendering falls back to no match. Alternatively, a full unique heading or
caption named in the quote/locator selects the immediately following figure
group. A caption between figures requires an explicit below/beneath relationship.
These matches ignore case and whitespace only; they do not recognize semantic
paraphrases, figure contents, partial headings or inferred subregions. A group
returns every existing whole-figure box, not a single merged rectangle. Invalid
boxes, stale hashes, duplicate anchors and conflicting direction remain unlocated.
Explicit markers take precedence over contextual headings.

This is a display locator, not authorization or claim verification. Callers own
source permission checks, accepted citation selection, and review receipts.
The matcher does not alter source transcripts, layouts or verification records.

Install this directory directly (`pip install ./memorylayer-document-layout`),
or pin a Git commit using `#subdirectory=memorylayer-document-layout`. Version
0.2.0 is independently versioned; PyPI publication is not currently assumed.
The enterprise installer includes the package from the same checkout. Run its
synthetic tests with `python -m pytest memorylayer-document-layout/tests`.
