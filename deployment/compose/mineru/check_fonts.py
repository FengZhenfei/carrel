"""MinerU image build check: PDFium must paint distinct glyphs without embedded fonts."""

import pypdfium2 as pdfium


def glyph_pdf(font: str, ordering: str, cmap: str, char: str, *, serif: bool) -> bytes:
    draw = f"BT /F1 32 Tf 8 12 Td <{ord(char):04X}> Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 48 48] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(draw) + draw + b"\nendstream",
        f"<< /Type /Font /Subtype /Type0 /BaseFont /{font} /Encoding /{cmap} "
        f"/DescendantFonts [6 0 R] >>".encode(),
        f"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /{font} /FontDescriptor 7 0 R "
        f"/CIDSystemInfo << /Registry (Adobe) /Ordering ({ordering}) /Supplement 2 >> >>".encode(),
        f"<< /Type /FontDescriptor /FontName /{font} /Flags {6 if serif else 4} /FontBBox [-250 -250 1100 1100] "
        f"/ItalicAngle 0 /Ascent 880 /Descent -120 /CapHeight 700 /StemV 90 >>".encode(),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(b"%d 0 obj\n" % i + obj + b"\nendobj\n")
    xref = len(out)
    out.extend(b"xref\n0 8\n0000000000 65535 f \n")
    for offset in offsets:
        out.extend(b"%010d 00000 n \n" % offset)
    out.extend(b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref)
    return bytes(out)


if __name__ == "__main__":
    failures = []
    for font, ordering, cmap, serif, pair in (
        ("STSong-Light", "GB1", "UniGB-UCS2-H", True, "中文"),
        ("SimHei", "GB1", "UniGB-UCS2-H", False, "中文"),
        ("MSung-Light", "CNS1", "UniCNS-UCS2-H", True, "繁體"),
        ("HeiseiMin-W3", "Japan1", "UniJIS-UCS2-H", True, "あい"),
        # Arphic UKai can claim Japanese Gothic but leave these common kanji blank.
        ("MS-PGothic", "Japan1", "UniJIS-UCS2-H", False, "経済"),
        ("HYSMyeongJo-Medium", "Korea1", "UniKS-UCS2-H", True, "한글"),
        # Korean serif can work even when Droid Sans Fallback shadows Noto for Gothic.
        ("Dotum", "Korea1", "UniKS-UCS2-H", False, "서울"),
        ("MalgunGothic", "Korea1", "UniKS-UCS2-H", False, "감사"),
    ):
        masks = []
        for char in pair:
            with pdfium.PdfDocument(glyph_pdf(font, ordering, cmap, char, serif=serif)) as document:
                masks.append(document[0].render(grayscale=True).to_pil().tobytes())
        # Blank output and repeated missing-glyph boxes are both failures.
        if any(min(mask) >= 128 for mask in masks) or masks[0] == masks[1]:
            failures.append(f"{font} ({cmap}): {pair}")
    if failures:
        raise SystemExit("PDFium could not render distinct glyphs for " + "; ".join(failures))
    print("PDFium rendered unembedded Simplified Chinese, Traditional Chinese, Japanese and Korean glyphs.")
