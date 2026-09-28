from __future__ import annotations

import base64
import hashlib
import math
from io import BytesIO
from pathlib import Path


# Raster formats both local vision models accept. They are served by vLLM, which
# decodes images with Pillow, so the practical boundary is "what Pillow opens
# and what we are willing to put on the wire". Vector formats (svg) and HEIC
# are deliberately out: Pillow needs extra plugins for them.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Formats that can go on the wire untouched. Anything else is re-encoded.
_PASSTHROUGH_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

# Matches the --mm-processor-kwargs max_pixels both vLLM containers run with
# (3686400 = 1920x1920). The server would downscale anyway; doing it here keeps
# request bodies small and makes the bytes a cached caption/vector was computed
# from reproducible.
DEFAULT_MAX_PIXELS = 3_686_400


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# An image Pillow cannot even identify may still be worth showing the model,
# but only within reason -- an unbounded read_bytes() defeated the
# decompression-bomb guard with a single oversized file.
MAX_UNIDENTIFIED_PASSTHROUGH_BYTES = 20 * 1024 * 1024

_EXIF_ORIENTATION_TAG = 0x0112


def load_image_for_model(path: Path, max_pixels: int | None = DEFAULT_MAX_PIXELS) -> tuple[bytes, str]:
    """Return (bytes, mime) ready to be base64-embedded in a model request.

    PNG/JPEG/WebP within the pixel budget and without an EXIF rotation are
    sent byte-for-byte (so MinerU crops reach the VLM exactly as before).
    Everything else -- BMP/GIF/TIFF, oversized images, rotated JPEGs, 16-bit
    scans -- is decoded, normalised (EXIF transpose, 8-bit, RGB(A)),
    downscaled to the budget and re-encoded (JPEG stays JPEG, the rest
    becomes PNG). Decompression bombs and truncated files raise: callers
    pre-filter with image_size() and skip such blocks.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    budget = int(max_pixels) if max_pixels else 0
    try:
        image = Image.open(path)
    except UnidentifiedImageError:
        raw = path.read_bytes()
        if len(raw) > MAX_UNIDENTIFIED_PASSTHROUGH_BYTES:
            raise RuntimeError(
                f"unidentified image too large to pass through: {path.name} ({len(raw)} bytes)"
            )
        return raw, _mime_from_suffix(path)
    with image:
        fmt = (image.format or "").upper()
        pixels = int(image.width) * int(image.height)
        try:
            orientation = int(image.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
        except Exception:
            orientation = 1
        if fmt in _PASSTHROUGH_MIME and (budget <= 0 or pixels <= budget) and orientation == 1:
            return path.read_bytes(), _PASSTHROUGH_MIME[fmt]

        out_format = "JPEG" if fmt == "JPEG" else "PNG"
        work = image
        if work.mode in {"I", "I;16", "I;16B", "I;16L", "I;16N"}:
            # 16-bit grayscale clips through mode I on convert("RGB") and
            # renders as a blank white frame; scale to 8-bit first.
            work = work.point(lambda value: value / 256).convert("L")
        if orientation != 1:
            # A portrait phone photo left un-transposed is captioned (and
            # embedded) sideways; the re-encode drops the EXIF tag, so the
            # rotation must be applied to the pixels.
            transposed = ImageOps.exif_transpose(work)
            if transposed is not None:
                work = transposed
        has_alpha = work.mode in {"RGBA", "LA", "PA"} or (work.mode == "P" and "transparency" in work.info)
        if out_format == "JPEG":
            converted = work.convert("RGB")
        else:
            converted = work.convert("RGBA" if has_alpha else "RGB")
        width, height = int(converted.width), int(converted.height)
        pixels = width * height
        if budget > 0 and pixels > budget:
            ratio = math.sqrt(budget / pixels)
            target = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            converted = converted.resize(target, resampling)
        buffer = BytesIO()
        if out_format == "JPEG":
            converted.save(buffer, format="JPEG", quality=90, optimize=True)
            return buffer.getvalue(), "image/jpeg"
        converted.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"


def _mime_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".bmp":
        return "image/bmp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    return "image/png"


def image_data_url(path: Path, max_pixels: int | None = DEFAULT_MAX_PIXELS) -> str:
    data, mime = load_image_for_model(path, max_pixels)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
