"""Recover undersampled PDF figures from source geometry, without model decisions."""
from __future__ import annotations

import math
from pathlib import Path

from ..models import ParsedBlock
from ..vision.images import DEFAULT_MAX_PIXELS, image_size


def restore_pdf_images(blocks: list[ParsedBlock], path: Path, cache_dir: Path,
                       max_pixels: int | None = DEFAULT_MAX_PIXELS) -> int:
    """Render only small figures for which the PDF has more source detail.

    Rendering at the embedded image's density also preserves masks, clipping,
    rotation and overlaid arrows/text. Ambiguous matches never select an xref.
    Pixel and edge limits apply before rasterization, as well as on the wire.
    """
    try:
        import pymupdf
        doc = pymupdf.open(path)
    except (ImportError, OSError, RuntimeError):
        return 0
    budget = min(int(max_pixels or DEFAULT_MAX_PIXELS), DEFAULT_MAX_PIXELS)
    if budget <= 0:
        doc.close()
        return 0
    changed, pages = 0, {}
    with doc:
        for b in blocks:
            if b.block_type not in {"image", "chart"} or not b.visual_ref or not b.page_idx or not b.bbox:
                continue
            old_size = image_size(Path(b.visual_ref))
            if not all(old_size) or max(old_size) >= 384:
                continue
            if not 1 <= b.page_idx <= len(doc) or len(b.bbox) != 4:
                continue
            if not all(math.isfinite(v) and 0 <= v <= 1000 for v in b.bbox):
                continue
            try:
                page = doc[b.page_idx - 1]
                if page.rotation:
                    continue  # Parser coordinates for rotated pages require a separate transform.
                box = pymupdf.Rect(b.bbox[0]*page.rect.width/1000, b.bbox[1]*page.rect.height/1000,
                                   b.bbox[2]*page.rect.width/1000, b.bbox[3]*page.rect.height/1000)
                if box.is_empty or min(box.width, box.height) < 2:
                    continue
                # Refuse a mismatched crop rather than attaching neighbouring content.
                if abs((box.width/box.height)/(old_size[0]/old_size[1])-1) > .15:
                    continue
                if b.page_idx not in pages:
                    pages[b.page_idx] = (page.get_image_info(), page.get_drawings())
                infos, drawings = pages[b.page_idx]
                matches = []
                for info in infos:
                    r = pymupdf.Rect(info['bbox'])
                    overlap = (r & box).get_area()
                    if overlap / max(r.get_area(), 1) >= .85 and overlap / box.get_area() >= .85:
                        a, bb, c, d, _, _ = info['transform']
                        density = min(info['width']/max(math.hypot(a, bb), 1),
                                      info['height']/max(math.hypot(c, d), 1))
                        matches.append(density)
                if len(matches) == 1:
                    scale = matches[0]
                elif not matches and (page.get_text('text', clip=box).strip() or
                        any((pymupdf.Rect(d['rect']) & box).get_area() > 0 for d in drawings)):
                    scale = 300 / 72  # Vector content has no bitmap resolution to recover.
                else:
                    continue
                # Pixmap coordinates round both clip edges outwards (up to two pixels).
                scale = min(scale, max(0, math.sqrt(budget / box.get_area()) - 2/min(box.width, box.height)),
                            4094 / max(box.width, box.height))
                # Upgrade thumbnails only when the source can provide a large,
                # substantially denser view. Small icons and already legible
                # figures do not incur another model reading.
                if (max(box.width, box.height)*scale < 1024 or
                        min(box.width*scale/old_size[0], box.height*scale/old_size[1]) < 3):
                    continue
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=box, alpha=False)
                target = cache_dir / 'evidence' / 'images' / (Path(b.block_id).name + '.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                pix.save(target)
                b.metadata['visual_resolution'] = {'source': 'pdf-region', 'original_size': list(old_size),
                                                    'size': [pix.width, pix.height]}
                b.visual_ref = str(target)
                changed += 1
            except (ValueError, OSError, RuntimeError, KeyError):
                continue  # Keep the existing picture if the source region cannot be read.
    return changed
