from __future__ import annotations

import json
import math
import os
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ..models import ParsedBlock
from .mineru_pdf import MinerUServiceError
from .mineru_pptx import aggregate_mineru_slide_blocks, mineru_pptx_blocks
from ..vision.vlm import compose_prompt
from .visual_blocks import enrich_blocks_with_vlm, ensure_visual_blocks_have_text, tidy_visual_blocks


def parse_pptx_enhanced(
    *,
    mineru_url: str,
    vlm_base_url: str,
    vlm_api_key: str,
    vlm_model_id: str,
    vlm_concurrency: int,
    path: Path,
    cache_dir: Path,
    timeout: int = 43200,
    vlm_options: dict | None = None,
    caption_cache_root: Path | None = None,
    vlm_prompt: str | None = None,
    progress_cb=None,
) -> list[ParsedBlock]:
    structure_path = prepare_pptx_for_structure_parsers(path, cache_dir)
    print(f"[parser] pptx mineru start file={path.name}", flush=True)
    structured = mineru_pptx_blocks(
        mineru_url=mineru_url,
        path=structure_path,
        cache_dir=cache_dir,
        timeout=timeout,
    )
    print(f"[parser] pptx mineru done blocks={len(structured)} file={path.name}", flush=True)
    blocks = aggregate_mineru_slide_blocks(structured, path)
    if not blocks:
        raise MinerUServiceError("MinerU returned no usable PPTX blocks")

    image_blocks = [
        block
        for block in blocks
        if block.block_type in {"image", "chart"} and block.visual_ref
    ]
    print(f"[parser] pptx vlm candidates={len(image_blocks)} file={path.name}", flush=True)
    enrich_blocks_with_vlm(
        image_blocks,
        base_url=vlm_base_url,
        api_key=vlm_api_key,
        model_id=vlm_model_id,
        cache_dir=cache_dir,
        concurrency=vlm_concurrency,
        prompt=compose_prompt("figure", vlm_prompt),
        filter_decorative=True,
        vlm_options=vlm_options,
        caption_cache_root=caption_cache_root,
        progress_cb=progress_cb,
    )
    tidy_visual_blocks(image_blocks)
    ensure_visual_blocks_have_text(image_blocks, doc_name=path.name)
    return blocks


def prepare_pptx_for_structure_parsers(path: Path, cache_dir: Path) -> Path:
    max_pixels = max_structure_image_pixels()
    if max_pixels <= 0:
        return path

    out_dir = cache_dir / "pptx_structure_input"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_path = out_dir / f"{path.stem}.structure-safe.pptx"
    marker = out_dir / "structure-safe.json"
    if marker.exists():
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data.get("source") == str(path) and data.get("changed") and safe_path.exists():
                return safe_path
            if data.get("source") == str(path) and not data.get("changed"):
                return path
        except Exception:
            pass

    try:
        from PIL import Image
    except Exception as exc:
        print(f"[parser] pptx structure image-sanitize skipped missing Pillow file={path.name} error={exc!r}", flush=True)
        return path

    changed: list[dict[str, object]] = []
    tmp_path = safe_path.with_suffix(".tmp.pptx")
    old_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with ZipFile(path, "r") as zin, ZipFile(tmp_path, "w", ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                new_data = data
                if info.filename.startswith("ppt/media/") and _looks_like_raster_image(data):
                    # One undecodable or exotic-mode picture must not abort
                    # the whole PPTX before MinerU even runs; ship the
                    # original bytes for that image instead.
                    try:
                        new_data, change = _downsample_pptx_image(
                            Image=Image,
                            filename=info.filename,
                            data=data,
                            max_pixels=max_pixels,
                        )
                    except Exception as exc:
                        print(
                            f"[parser] pptx structure image-sanitize kept original "
                            f"image={info.filename} error={exc!r}",
                            flush=True,
                        )
                        new_data, change = data, None
                    if change:
                        changed.append(change)
                zout.writestr(info, new_data)
    finally:
        Image.MAX_IMAGE_PIXELS = old_max_pixels

    if changed:
        tmp_path.replace(safe_path)
        marker.write_text(
            json.dumps(
                {"source": str(path), "changed": True, "max_pixels": max_pixels, "images": changed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[parser] pptx structure image-sanitize changed={len(changed)} "
            f"file={path.name} safe_file={safe_path}",
            flush=True,
        )
        return safe_path

    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass
    marker.write_text(
        json.dumps({"source": str(path), "changed": False, "max_pixels": max_pixels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _looks_like_raster_image(data: bytes) -> bool:
    return data.startswith(
        (
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff",
            b"GIF87a",
            b"GIF89a",
            b"BM",
            b"II*\x00",
            b"MM\x00*",
        )
    )


def _downsample_pptx_image(*, Image, filename: str, data: bytes, max_pixels: int) -> tuple[bytes, dict[str, object] | None]:
    with Image.open(BytesIO(data)) as image:
        original_width, original_height = int(image.width), int(image.height)
        original_pixels = original_width * original_height
        if original_pixels <= max_pixels:
            return data, None

        ratio = math.sqrt(max_pixels / original_pixels)
        target_size = (max(1, int(original_width * ratio)), max(1, int(original_height * ratio)))
        resized = image.copy()
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        resized.thumbnail(target_size, resampling)

        image_format = "JPEG" if filename.lower().endswith((".jpg", ".jpeg")) else "PNG"
        if image_format == "JPEG" and resized.mode not in {"RGB", "L"}:
            resized = resized.convert("RGB")
        elif image_format == "PNG" and resized.mode not in {"RGB", "RGBA", "L", "LA"}:
            # CMYK (and friends) cannot be written as PNG; P keeps alpha.
            has_alpha = "A" in resized.mode or (resized.mode == "P" and "transparency" in resized.info)
            resized = resized.convert("RGBA" if has_alpha else "RGB")

        out = BytesIO()
        save_kwargs: dict[str, object] = {"optimize": True}
        if image_format == "JPEG":
            save_kwargs["quality"] = 90
        resized.save(out, format=image_format, **save_kwargs)
        new_data = out.getvalue()
        return new_data, {
            "filename": filename,
            "original_size": [original_width, original_height],
            "new_size": [int(resized.width), int(resized.height)],
            "original_pixels": original_pixels,
            "new_pixels": int(resized.width) * int(resized.height),
            "original_bytes": len(data),
            "new_bytes": len(new_data),
        }


def max_structure_image_pixels() -> int:
    try:
        return max(0, int(os.getenv("PPTX_STRUCTURE_MAX_IMAGE_PIXELS", "40000000")))
    except Exception:
        return 40_000_000
