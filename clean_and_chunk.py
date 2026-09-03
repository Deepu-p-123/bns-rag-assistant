"""
Cleans raw Gazette PDF text and splits it into one chunk per BNS Section.

Usage:
    from clean_and_chunk import clean_raw_text, chunk_by_section

    with open("raw_extracted_text.txt", encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean_raw_text(raw)
    chunks = chunk_by_section(cleaned)   # list of {"section_number", "text"}
"""

import re


def clean_raw_text(raw_text: str) -> str:
    """Strip Gazette noise: footers, page numbers, rules, garbled Hindi lines."""
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"_{5,}", stripped):
            continue
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if "THE GAZETTE OF INDIA EXTRAORDINARY" in stripped:
            continue
        if re.match(r"^Sec\.\s*\d+\]", stripped):
            continue
        if re.match(r"^\[Part", stripped):
            continue
        if re.match(r"^PART\s+[IVXLCDM]+", stripped):
            continue
        if re.match(r"^--- PAGE \d+ ---$", stripped):
            continue
        if stripped == "EXTRAORDINARY":
            continue

        # Drop lines that are mostly Devanagari script
        devanagari_chars = sum(1 for c in stripped if "\u0900" <= c <= "\u097F")
        if devanagari_chars > len(stripped) * 0.3:
            continue

        # Drop lines that are garbled Hindi rendered via a legacy Latin font
        # (common in Gazette PDFs): dominated by vowel-less "words".
        words = re.findall(r"[a-zA-Z]+", stripped)
        if words:
            vowelless = sum(1 for w in words if not re.search(r"[aeiouAEIOU]", w))
            if vowelless / len(words) > 0.6:
                continue

        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def chunk_by_section(cleaned_text: str) -> list[dict]:
    """Split cleaned text into chunks, one per top-level Section number.

    If the same section number is matched more than once (this can happen
    due to PDF page-break artifacts repeating a heading), the occurrences
    are merged into a single chunk rather than producing duplicate IDs.
    """
    pattern = re.compile(r"\n(\d{1,3})\.\s+(?=[A-Z(\u201c])")
    text = "\n" + cleaned_text

    matches = list(pattern.finditer(text))
    raw_chunks = []
    for i, match in enumerate(matches):
        section_num = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 5:
            continue  # skip false positives (e.g. stray numbering)
        raw_chunks.append({"section_number": section_num, "text": body})

    # Merge duplicates, preserving first-seen order
    merged: dict[str, list[str]] = {}
    order = []
    for c in raw_chunks:
        if c["section_number"] not in merged:
            merged[c["section_number"]] = []
            order.append(c["section_number"])
        merged[c["section_number"]].append(c["text"])

    duplicates = {num: bodies for num, bodies in merged.items() if len(bodies) > 1}
    if duplicates:
        print(f"Note: merged {len(duplicates)} duplicated section number(s): "
              f"{', '.join(duplicates.keys())}")

    chunks = []
    for section_num in order:
        combined_body = "\n\n".join(merged[section_num])
        chunks.append({
            "section_number": section_num,
            "text": f"Section {section_num}. {combined_body}",
        })
    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "raw_extracted_text.txt"
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean_raw_text(raw)
    chunks = chunk_by_section(cleaned)
    print(f"Extracted {len(chunks)} section chunks.")
    for c in chunks[:5]:
        print(f"\n[Section {c['section_number']}] {c['text'][:150]}...")