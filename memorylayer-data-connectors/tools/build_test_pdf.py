#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Generate a minimal 1-page PDF for integration testing.

Produces a valid PDF without any external dependencies (no reportlab,
fpdf, etc.) — just raw PDF syntax.  The content is a fixed string that
can be searched for in RAG results after ingestion.

Usage:
    python tools/build_test_pdf.py [output_path]

Default output: tests/integration/fixtures/sample.pdf
"""
from __future__ import annotations

import os
import sys


def build_minimal_pdf(text: str = "MemoryLayer PDF integration test document. "
                      "This content should be searchable after ingestion.") -> bytes:
    """Build a minimal valid PDF containing *text* on a single page."""
    # PDF objects (1-indexed)
    # 1: Catalog
    # 2: Pages
    # 3: Page
    # 4: Font (Helvetica)
    # 5: Content stream

    stream_content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    stream_bytes = stream_content.encode("latin-1")

    objects: list[bytes] = []
    offsets: list[int] = []

    def add_obj(obj_num: int, content: str) -> None:
        data = f"{obj_num} 0 obj\n{content}\nendobj\n".encode("latin-1")
        objects.append(data)

    add_obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add_obj(3, "<< /Type /Page /Parent 2 0 R "
               "/MediaBox [0 0 612 792] "
               "/Contents 5 0 R "
               "/Resources << /Font << /F1 4 0 R >> >> >>")
    add_obj(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add_obj(5, f"<< /Length {len(stream_bytes)} >>\nstream\n".rstrip("\n"))

    # Build the final PDF bytes
    header = b"%PDF-1.4\n"
    body = b""
    for i, obj_data in enumerate(objects):
        offsets.append(len(header) + len(body))
        body += obj_data

    # Object 5 needs special handling for the stream
    # Re-build object 5 properly with stream
    objects_rebuilt: list[bytes] = []
    for i, obj_data in enumerate(objects):
        if i < 4:
            objects_rebuilt.append(obj_data)
        else:
            # Object 5: content stream
            obj5 = (f"5 0 obj\n<< /Length {len(stream_bytes)} >>\n"
                    f"stream\n").encode("latin-1")
            obj5 += stream_bytes
            obj5 += b"\nendstream\nendobj\n"
            objects_rebuilt.append(obj5)

    body = b""
    offsets = []
    for obj_data in objects_rebuilt:
        offsets.append(len(header) + len(body))
        body += obj_data

    # Cross-reference table
    xref_offset = len(header) + len(body)
    xref = f"xref\n0 {len(offsets) + 1}\n"
    xref += "0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"

    trailer = (f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\n"
               f"startxref\n{xref_offset}\n%%EOF\n")

    return header + body + xref.encode("latin-1") + trailer.encode("latin-1")


def main() -> None:
    default_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "integration", "fixtures", "sample.pdf"
    )
    output_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    output_path = os.path.abspath(output_path)

    pdf_bytes = build_minimal_pdf()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Generated test PDF: {output_path} ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    main()
