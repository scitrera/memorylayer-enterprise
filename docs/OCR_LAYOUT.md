# Durable OCR layout

New page transcriptions preserve the provider's raw completion and all parsed
regions, including text, headings, tables and figures. Both distributed
`document_transcribe` and monolithic ingestion use the same persistence helper.
This adds no model calls and requires no database migration.

`DocumentPage.metadata.ocr_layout` is a versioned optional object:

| Field | Meaning |
| --- | --- |
| `version` | `1` |
| `coordinate_space` | `normalized_0_1`, relative to the original rendered image, origin at top left |
| `image_sha256`, `image_width`, `image_height` | Exact input image bytes and pixel dimensions |
| `transcript_sha256` | Exact persisted UTF-8 rendered transcript, including optional figure captions |
| `model`, `provider`, `output_contract` | Winning transcription provider and parsing contract |
| `regions` | Ordered `{id, index, label, text, bbox}` records; bbox is `[x0,y0,x1,y1]` |
| `raw_ocr` | Optional `{storage_path, sha256}` for the unmodified UTF-8 model output |

Raw completions are stored beneath the document's `raw-ocr` directory with a
content hash in the filename. Re-transcribing a page replaces its metadata but
retains previous raw blobs. The path is an internal storage identifier, not a
browser URL; normal workspace authorization applies to reading document data.
The metadata contains region text but does not duplicate the full raw completion.

Region IDs depend on the source image hash, region order, text, label and box.
Boxes use the same orientation as the stored page image. Invalid, inverted,
zero-area or out-of-bounds boxes become `null`; they are never silently clamped
into apparently valid highlights. Raw output remains available for a future
parser/localization improvement. Ungrounded providers have no usable boxes.

A consumer must match image and transcript hashes, then associate the supporting
quotation with one or more regions. An OCR block may contain surrounding text;
it is not word-level localization or independent proof that a claim is correct.
Duplicate/ambiguous passages should retain page-level navigation.

Persistence decodes one existing base64 page and inspects its image header off
the event loop; it does not allocate a second full decoded raster. A raw-output
storage failure fails ingestion instead of silently losing provenance. Retrying
with unchanged bytes reuses the same raw path and region identities. Replacement
transcriptions remove stale layout and figure metadata before applying new data.

Pages ingested before this change remain readable without `ocr_layout`. Their
missing text/table geometry cannot be recovered from figure metadata alone;
reprocess/localize those pages explicitly if highlights are required. Merely
upgrading the service does not re-run OCR or change existing transcripts.
