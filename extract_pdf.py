"""
Step 2: Extract raw text from the BNS Gazette PDF.

This script does NOT try to be clever yet — it just pulls text page by page
so you can SEE what the raw extraction looks like (headers, footers, line
breaks, column issues) before writing any cleanup or chunking logic.

Usage:
    python extract_pdf.py path/to/bns.pdf
"""

import sys
import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> list[str]:
    """Return a list of raw text strings, one per page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append(text)
    doc.close()
    return pages


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_pdf.py path/to/bns.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_text(pdf_path)

    print(f"Total pages extracted: {len(pages)}\n")
    print("=" * 60)
    print("PREVIEW: First 3 pages (raw, unprocessed)")
    print("=" * 60)

    for i, page_text in enumerate(pages[:3], start=1):
        print(f"\n--- PAGE {i} ---\n")
        print(page_text)
        print("-" * 40)

    # Save full raw text to a file so you can scroll through everything,
    # not just the preview above.
    out_path = "raw_extracted_text.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, page_text in enumerate(pages, start=1):
            f.write(f"\n--- PAGE {i} ---\n")
            f.write(page_text)

    print(f"\nFull raw text saved to: {out_path}")
    print("Open that file and check for:")
    print("  - Repeated headers/footers (e.g. 'THE GAZETTE OF INDIA')")
    print("  - Section numbers split across lines")
    print("  - Two-column layout scrambling (text out of order)")
    print("  - Page numbers or watermarks mixed into the body text")


if __name__ == "__main__":
    main()