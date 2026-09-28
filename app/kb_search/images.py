"""Original image and crop for image evidence (Q25 / Q26). visual_ref is a relative key into the parse
cache (settings.cache_dir / visual_ref); an image chunk from a PDF prefers the original bitmap embedded
in the PDF (the copy in the parse cache is a re-encoded JPEG of clearly lower quality), falls back to a
high-resolution render of the page bbox when no embedded image matches, and only then to the cached
image. Cropping is a deterministic cut along the box the caller supplies; the DGX side does no
recognition of any kind."""
from __future__ import annotations

import hashlib
import io
import re
import threading
from pathlib import Path
from typing import Any

from PIL import Image

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA_CACHE: dict[tuple[str, int, int], str] = {}
_SHA_LOCK = threading.Lock()

RENDER_DPI = 220
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


def locate(q: Any, collection: str, point_id: str) -> dict[str, Any]:
    """Fetch an image chunk's payload by point_id; a non-image chunk or a missing point raises KeyError."""
    recs = q.retrieve(collection_name=collection, ids=[str(point_id)], with_payload=True, with_vectors=False)
    if not recs:
        raise KeyError(f"point {point_id} not found in {collection}")
    payload = dict(recs[0].payload or {})
    if payload.get("is_active") is False:
        raise KeyError(f"point {point_id} is no longer active")
    if not payload.get("visual_ref"):
        raise KeyError(f"point {point_id} has no image")
    payload["point_id"] = str(point_id)
    return payload


def _cached_file(settings: Any, payload: dict[str, Any]) -> Path:
    """visual_ref is a relative key under the parse cache directory; once resolved to an absolute path
    it must still lie inside the cache directory, and escaping references such as absolute paths or
    `..` are all treated as "no image"."""
    base = Path(settings.cache_dir).resolve()
    ref = str(payload.get("visual_ref") or "")
    target = (base / ref).resolve() if ref else base
    if not ref or target == base or base not in target.parents:
        raise KeyError(f"visual_ref {ref!r} is outside the parse cache")
    return target


def _file_sha256(path: Path) -> str:
    """sha256 of a mirror file, cached by (path, size, mtime) so the same file is not read twice."""
    st = path.stat()
    key = (str(path), int(st.st_size), int(st.st_mtime_ns))
    with _SHA_LOCK:
        cached = _SHA_CACHE.get(key)
    if cached:
        return cached
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    with _SHA_LOCK:
        if len(_SHA_CACHE) > 2048:
            _SHA_CACHE.clear()
        _SHA_CACHE[key] = digest
    return digest


def _pdf_path(settings: Any, payload: dict[str, Any]) -> Path | None:
    """The PDF in the mirror: used only when it can be proven to be the very version the chunk came
    from (Codex R4). The chunk's content_version is the file's sha256 anyway ("mtime:...:size:..." when
    it could not be computed), and the mirror file is checked against it; if it does not match, or the
    chunk carries no version, fall back to the image in the parse cache."""
    src = str(payload.get("source_path") or "")
    if not src.lower().endswith(".pdf"):
        return None
    p = Path(src)
    if not p.is_absolute():
        p = Path(settings.mirror_root) / src
    if not p.exists():
        return None
    # The file must still sit inside its (admitted) knowledge base directory: a mirror entry or the file
    # itself swapped for a symbolic link is not followed (security review F04 follow-up)
    from kb_pipeline.discovery import directory_admitted
    from kb_pipeline.localfs.scanner import inside_boundary

    parts = Path(src).parts if not Path(src).is_absolute() else ()
    top = parts[0] if len(parts) > 1 else ""            # "<knowledge base directory>/<relative path>"
    if top:
        if not directory_admitted(Path(settings.mirror_root), top)[0]:
            return None
        if not inside_boundary(p, (Path(settings.mirror_root) / top).resolve()):
            return None
    elif not inside_boundary(p, Path(settings.mirror_root).resolve()):
        return None
    version = str(payload.get("content_version") or "").strip().lower()
    if not version:
        return None
    try:
        st = p.stat()
        if _SHA_RE.match(version):
            return p if _file_sha256(p) == version else None
        m = re.match(r"^mtime:(\d+):size:(\d+)$", version)
        if m:
            return p if int(m.group(1)) == int(st.st_mtime_ns) and int(m.group(2)) == st.st_size else None
    except OSError:
        return None
    return None


def _as_png(image_bytes: bytes) -> tuple[bytes, int, int]:
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    buf = io.BytesIO()
    (img if img.mode in ("RGB", "RGBA", "L") else img.convert("RGB")).save(buf, format="PNG")
    return buf.getvalue(), img.size[0], img.size[1]


def _rect_iou(a: Any, b: Any) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    union = a.get_area() + b.get_area() - inter.get_area()
    return float(inter.get_area() / union) if union > 0 else 0.0


def _embedded_match(doc: Any, page_no: int, want_w: int, want_h: int, bbox_rect: Any | None) -> dict[str, Any] | None:
    """Which of the bitmaps embedded on the page is this chunk's image: with a bbox, match by where the
    image sits on the page (highest IoU and >= 0.3, Codex S04: two images of the same aspect ratio on
    one page cannot be told apart by pixel count); only without a bbox fall back to the rule "same
    aspect ratio (+-2%) and no smaller than the cached image". The image in the parse cache is a
    downscaled re-encoding of the embedded one (real machine: embedded 1358x1890, cached 918x1262), so
    equal dimensions are not reliable."""
    page = doc[page_no]
    want = want_w / max(1, want_h)
    items = page.get_images(full=True)
    if bbox_rect is not None:
        best, best_iou = None, 0.0
        for item in items:
            try:
                rects = page.get_image_rects(item[0])
            except Exception:
                rects = []
            for r in rects:
                iou = _rect_iou(r, bbox_rect)
                if iou > best_iou:
                    best, best_iou = item, iou
        if best is None or best_iou < 0.3:
            return None
        try:
            return doc.extract_image(best[0])
        except Exception:
            return None
    best, best_gap = None, None
    for item in items:
        w, h = int(item[2] or 0), int(item[3] or 0)
        if w <= 0 or h <= 0 or w < want_w * 0.9 or h < want_h * 0.9:
            continue
        if abs(w / h - want) > max(0.02, want * 0.02):
            continue
        gap = abs(w * h - want_w * want_h)          # the cached image is a downscaled copy of the embedded one: the closest size (and no smaller) is the best match, not the largest
        if best_gap is None or gap < best_gap:
            try:
                info = doc.extract_image(item[0])
            except Exception:
                continue
            best, best_gap = info, gap
    return best


def _bbox_rect(page: Any, bbox: list[float], cached_size: tuple[int, int] | None) -> Any | None:
    """Map the payload's bbox to page coordinates. MinerU's content_list gives page coordinates
    normalised to 0-1000 (verified on the real machine: on a 960x540 landscape page the bbox y reaches
    972), but PDF points have been seen as well; both interpretations are computed, and with a cached
    image the one whose aspect ratio is closer to it wins, otherwise the normalised one is taken."""
    import pymupdf

    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    pw, ph = float(page.rect.width), float(page.rect.height)
    cands: list[tuple[str, Any]] = []
    if max(x2, y2) <= 1000.0:
        cands.append(("permille", pymupdf.Rect(x1 / 1000 * pw, y1 / 1000 * ph, x2 / 1000 * pw, y2 / 1000 * ph)))
    if x2 <= pw * 1.02 and y2 <= ph * 1.02:
        cands.append(("points", pymupdf.Rect(x1, y1, x2, y2)))
    if not cands:
        scale = pw / max(x2, 1.0)
        cands.append(("scaled", pymupdf.Rect(x1 * scale, y1 * scale, x2 * scale, y2 * scale)))
    rects = [(name, r & page.rect) for name, r in cands]
    rects = [(name, r) for name, r in rects if not r.is_empty and r.width >= 4 and r.height >= 4]
    if not rects:
        return None
    if cached_size and cached_size[1] > 0 and len(rects) > 1:
        want = cached_size[0] / cached_size[1]
        rects.sort(key=lambda nr: abs(nr[1].width / max(nr[1].height, 1e-6) - want))
    return rects[0][1]


def original_image(settings: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Returns {bytes, mime, source, width, height}. source: pdf-embedded / pdf-render / cache."""
    cached = _cached_file(settings, payload)
    cached_size: tuple[int, int] | None = None
    if cached.exists():
        try:
            with Image.open(cached) as im:
                cached_size = im.size
        except Exception:
            cached_size = None
    pdf = _pdf_path(settings, payload)
    page_idx = payload.get("page_idx")
    if pdf is not None and page_idx is not None:
        try:
            import pymupdf

            doc = pymupdf.open(str(pdf))
            try:
                # The page number in the payload is 1-based (parsers.common.page_idx adds 1 to MinerU's
                # 0-based index); look at this page only, never guess across pages
                page_no = max(0, int(page_idx) - 1)
                if page_no < doc.page_count:
                    page = doc[page_no]
                    rect = _bbox_rect(page, list(payload.get("bbox") or []), cached_size)
                    if cached_size:
                        info = _embedded_match(doc, page_no, cached_size[0], cached_size[1], rect)
                        if info and info.get("image"):
                            ext = str(info.get("ext") or "").lower()
                            if ext in ("png", "jpg", "jpeg"):
                                w, h = int(info["width"]), int(info["height"])
                                return {"bytes": info["image"], "mime": "image/png" if ext == "png" else "image/jpeg",
                                        "source": "pdf-embedded", "width": w, "height": h, "page": page_no + 1}
                            data, w, h = _as_png(info["image"])
                            return {"bytes": data, "mime": "image/png", "source": "pdf-embedded", "width": w, "height": h, "page": page_no + 1}
                    if rect is not None:
                        pix = page.get_pixmap(clip=rect, dpi=RENDER_DPI, alpha=False)
                        return {"bytes": pix.tobytes("png"), "mime": "image/png", "source": "pdf-render", "width": pix.width, "height": pix.height, "page": page_no + 1}
            finally:
                doc.close()
        except Exception:
            pass
    if not cached.exists():
        raise KeyError(f"image file missing: {payload.get('visual_ref')}")
    data = cached.read_bytes()
    mime = MIME.get(cached.suffix.lower(), "application/octet-stream")
    w, h = cached_size or (0, 0)
    return {"bytes": data, "mime": mime, "source": "cache", "width": w, "height": h}


def resolve_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Two notations for the box: all values within 0-1 are fractions, otherwise values within 0-1000
    are per-mille (the grounding convention); pixel coordinates are not accepted (the image the caller
    saw may have been downscaled, so pixels would not line up). Returns a pixel box (already clamped to
    the image)."""
    if not bbox or len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    vals = [float(v) for v in bbox]
    if any(v < 0 for v in vals) or max(vals) > 1000.0:
        raise ValueError("bbox values must be fractions (0–1) or per-mille (0–1000)")
    if all(v <= 1.0 for v in vals):
        x1, y1, x2, y2 = vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height
    else:
        x1, y1, x2, y2 = vals[0] / 1000 * width, vals[1] / 1000 * height, vals[2] / 1000 * width, vals[3] / 1000 * height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box = (max(0, int(x1)), max(0, int(y1)), min(width, int(round(x2))), min(height, int(round(y2))))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError("bbox is empty after clamping to the image")
    return box


def crop_image(full: dict[str, Any], bbox: list[float], *, pad: int = 16) -> dict[str, Any]:
    img = Image.open(io.BytesIO(full["bytes"]))
    img.load()
    w, h = img.size
    x1, y1, x2, y2 = resolve_bbox(bbox, w, h)
    pad = max(0, int(pad))
    box = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
    out = img.crop(box)
    buf = io.BytesIO()
    (out if out.mode in ("RGB", "RGBA", "L") else out.convert("RGB")).save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "mime": "image/png", "source": full.get("source"), "width": out.size[0], "height": out.size[1],
            "box": list(box), "full_width": w, "full_height": h}
