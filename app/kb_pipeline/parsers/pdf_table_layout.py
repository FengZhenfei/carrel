"""Conservative table ownership repairs based on page layout, never topic vocabulary."""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .common import bbox, footnote_of, item_text, label_of, page_idx, grid_to_markdown


def attach_table_notes(items: list[dict], path=None) -> list[dict]:
    """Local form rows belong to the preceding table, not the next heading.

    Repeated key/value baselines or a matching ruled two-cell source row provide
    ownership. All source rectangles and text survive as explicit provenance.
    """
    counts = Counter(item_text(i).strip() for i in items if label_of(i) in {"text", "title"})
    doc, pages = None, {}
    if path:
        try:
            import pymupdf
            doc = pymupdf.open(path)
        except Exception:
            pass
    canon = lambda x: re.sub(r"\s+", "", unicodedata.normalize("NFKC", x or ""))

    def ruled_row(parts):
        if doc is None or not page_idx(parts[0]) or not all(bbox(x) for x in parts):
            return False
        page_no = page_idx(parts[0])
        if not 1 <= page_no <= len(doc):
            return False
        page = doc[page_no-1]
        from .pdf_evidence import _box
        try:
            if page_no not in pages:
                pages[page_no] = [(t.extract(), _box(page, pymupdf.Rect(t.bbox)))
                    for t in page.find_tables().tables if t.row_count == 1 and t.col_count == 2]
        except Exception:
            return False
        text = canon("".join(item_text(x) for x in parts))
        for rows, box in pages[page_no]:
            from .pdf_evidence import _span_present
            native = canon("".join(rows[0]))
            if not all(rows[0]) or len(native) != len(text) or not _span_present(native, text):
                continue
            if all(box[0]-5 <= bbox(x)[0] and bbox(x)[2] <= box[2]+5
                   and box[1]-5 <= bbox(x)[1] and bbox(x)[3] <= box[3]+5 for x in parts):
                return True
        return False

    def append_note(prev, parts, text):
        updated = dict(prev)
        updated["table_footnote"] = [s for s in (footnote_of(prev), text) if s]
        updated["_pdf_table_note_sources"] = list(prev.get("_pdf_table_note_sources") or []) + [
            {"page": page_idx(x), "bbox": bbox(x), "text": item_text(x)} for x in parts]
        return updated

    out, i = [], 0
    try:
        while i < len(items):
            key = items[i]
            value = items[i + 1] if i + 1 < len(items) else {}
            prev = out[-1] if out else {}
            a, b, table = bbox(key), bbox(value), bbox(prev)
            kt, vt = item_text(key).strip(), item_text(value).strip()
            local = ("table" in label_of(prev) and label_of(key) in {"text", "title"} and table and a
                     and page_idx(prev) == page_idx(key) and 0 <= a[1]-table[3] <= max(25, 2*(a[3]-a[1]))
                     and table[0]-8 <= a[0] and a[2] <= table[2]+8)
            same_row = (a and b and abs(a[1]-b[1]) <= max(a[3]-a[1],b[3]-b[1])*.35
                        and a[2] < b[0] and abs(a[3]-b[3]) <= max(a[3]-a[1],b[3]-b[1])*.35)
            if (local and label_of(value) in {"text", "title"} and same_row
                    and page_idx(key) == page_idx(value) and b[2] <= table[2]+8
                    and 0 < len(kt) <= 24 and 0 < len(vt) <= 160
                    and (counts[kt] >= 2 or kt.endswith((":", "：")) or ruled_row([key,value]))):
                out[-1] = append_note(prev, [key,value], kt.rstrip(":：") + ": " + vt)
                i += 2
            elif local and 0 < len(kt) <= 180 and ruled_row([key]):
                out[-1] = append_note(prev, [key], kt)
                i += 1
            else:
                out.append(key)
                i += 1
    finally:
        if doc is not None:
            doc.close()
    return out


def restore_ruled_continuation_columns(blocks, path):
    """Restore omitted empty columns only when native ruled cells prove the mapping.

    No cell text is replaced. Every OCR cell must match its native counterpart,
    in row order, and the only added positions are empty in the entire table.
    """
    from .table_repair import _pipe_grid
    from .pdf_evidence import _rect, normalize_ranges, _span_present
    try:
        import pymupdf
        doc = pymupdf.open(path)
    except Exception:
        return
    canon = lambda x: re.sub(r"\s+", "", normalize_ranges(unicodedata.normalize("NFKC", x or "")))
    with doc:
        content = [b for b in blocks if b.metadata.get("source_type") not in
                   {"header", "footer", "page_header", "page_footer", "page_number"}]
        pairs = []
        last = None
        for index, block in enumerate(content):
            if block.block_type != "table":
                continue
            if last is not None:
                prior = content[last]
                between = content[last+1:index]
                if block.bbox and all(b.bbox and
                        min(b.bbox[2], block.bbox[2]) <= max(b.bbox[0], block.bbox[0]) for b in between):
                    pairs.append((prior, block))
            last = index
        for prev, block in pairs:
            if (prev.block_type != "table" or block.block_type != "table" or not prev.page_idx
                    or block.page_idx != prev.page_idx + 1 or not prev.bbox or not block.bbox
                    or block.caption or block.title or prev.bbox[3] < 700 or block.bbox[1] > 300
                    or max(abs(prev.bbox[k]-block.bbox[k]) for k in (0, 2)) > 40):
                continue
            a, old = _pipe_grid(prev.table_markdown or prev.text), _pipe_grid(block.table_markdown or block.text)
            if not a or not old:
                continue
            if len(a) == 1 and len(a[0]) == len(old[0]) and prev.bbox[3] >= 850 and block.bbox[1] <= 150:
                # A header stranded at the page bottom may continue with textual
                # rows. Require matching internal ruled columns on both pages.
                geometries = []
                for part, grid in ((prev, a), (block, old)):
                    page = doc[part.page_idx-1]
                    region = _rect(page, part.bbox)
                    found = []
                    try:
                        for t in page.find_tables().tables:
                            box = pymupdf.Rect(t.bbox)
                            native = t.extract()
                            if ((box & region).get_area()/max(box.get_area(),region.get_area(),1) >= .85
                                    and t.col_count == len(grid[0]) and native
                                    and all(len(canon(x)) == len(canon(y)) and _span_present(canon(x),canon(y))
                                            for x,y in zip(native[0],grid[0]))
                                    and all(t.rows[0].cells)):
                                found.append([cell[k]/page.rect.width for cell in t.rows[0].cells for k in (0,2)])
                    except Exception:
                        pass
                    geometries.append(found[0] if len(found)==1 else [])
                if all(geometries) and max(abs(x-y) for x,y in zip(*geometries)) <= .005:
                    block.metadata["native_continuation_columns"] = True
                    text = block.table_markdown or block.text
                    notes = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("|")]
                    block.text = block.table_markdown = "\n".join([grid_to_markdown([a[0]] + old)] + notes)
                    from ..chunking.packing import source_regions
                    block.metadata["source_regions"] = source_regions(block) + [{
                        "page": prev.page_idx, "bbox": prev.bbox, "kind": "table_header", "text": " | ".join(a[0])}]
                    block.metadata["inherited_table_header"] = {"page": prev.page_idx, "block_id": prev.block_id}
                    block.metadata["page_start"] = prev.page_idx
            if not 0 < len(a[0])-len(old[0]) <= 2:
                continue
            page = doc[block.page_idx-1]
            rect = _rect(page, block.bbox)
            try:
                candidates = page.find_tables().tables
            except Exception:
                continue
            for table in candidates:
                region = pymupdf.Rect(table.bbox)
                if (region & rect).get_area() / max(region.get_area(), rect.get_area(), 1) < .85:
                    continue
                native = table.extract()
                if len(native) != len(old) or table.col_count != len(a[0]):
                    continue
                empty = [j for j in range(table.col_count) if all(not canon(row[j]) for row in native)]
                from itertools import combinations
                matches = []
                for omitted in combinations(empty, table.col_count-len(old[0])):
                    if all([canon(c) for c in row] == [canon(c) for j,c in enumerate(nr) if j not in omitted]
                           for row,nr in zip(old,native)):
                        matches.append(omitted)
                if len(matches) != 1:
                    continue
                omitted = matches[0]
                rows = []
                for row in old:
                    source = iter(row)
                    rows.append(["" if j in omitted else next(source) for j in range(table.col_count)])
                notes = [l for l in (block.table_markdown or block.text).splitlines() if l.strip() and not l.strip().startswith("|")]
                block.text = block.table_markdown = "\n".join([grid_to_markdown(rows)] + notes)
                block.metadata["restored_empty_columns"] = list(omitted)
                break
