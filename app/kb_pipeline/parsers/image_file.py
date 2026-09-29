from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..models import ParsedBlock
from ..utils import infer_doc_type
from ..vision.images import IMAGE_SUFFIXES
from ..vision.vlm import compose_prompt
from .errors import NonRetryableParseError
from .visual_blocks import enrich_blocks_with_vlm


IMAGE_FILE_PROFILE = "image-vlm-v1"
# Skip reasons that mean the picture itself cannot be sent to the model: retrying changes nothing. A missing
# API key or a cached copy that has gone missing are environment problems and are not listed here
PERMANENT_SKIP_REASONS = {"unreadable_image", "degenerate_aspect"}


def parse_image_file(
    *,
    path: Path,
    cache_dir: Path,
    vlm_base_url: str,
    vlm_api_key: str,
    vlm_model_id: str,
    vlm_concurrency: int,
    vlm_options: dict[str, Any] | None = None,
    caption_cache_root: Path | None = None,
    vlm_prompt: str | None = None,
    progress_cb=None,
) -> list[ParsedBlock]:
    """A standalone image file is one visual block.

    The picture is copied into the parse cache so visual_ref stays under the
    cache root like every MinerU crop (the mirror copy may move or vanish while
    the point lives on through its retention window). The block then takes the
    same road as an embedded figure: VLM description first, visual vector
    second (parse_job), text vector from the description.
    """
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise NonRetryableParseError(f"not an image file: {path.name}")
    from PIL import Image

    with path.open("rb") as probe:        # cannot open / read (permissions, disk): an environment problem, retried as usual; the checks below test the picture itself
        probe.read(1)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = int(image.width), int(image.height)
            image_format = (image.format or "").upper()
            mode = image.mode
    except Exception as exc:
        raise NonRetryableParseError(f"unreadable image file {path.name}: {exc!r}") from exc

    asset_dir = cache_dir / "image"
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset = asset_dir / path.name
    if not asset.exists() or asset.stat().st_size != path.stat().st_size:
        shutil.copy2(path, asset)

    block = ParsedBlock(
        parser="image-file",
        parser_profile=IMAGE_FILE_PROFILE,
        doc_type=infer_doc_type(path.name),
        block_type="image",
        text="",
        block_id="image-0001",
        # The stem is the only text the file carries; it guarantees a chunk
        # (and therefore a point with the visual vector) even when the VLM has
        # nothing to say about the picture.
        title=path.stem,
        visual_ref=str(asset),
        metadata={
            "image_width": width,
            "image_height": height,
            "image_format": image_format,
            "image_mode": mode,
            "image_bytes": int(path.stat().st_size),
        },
    )
    print(f"[parser] image file={path.name} size={width}x{height} format={image_format}", flush=True)
    enrich_blocks_with_vlm(
        [block],
        base_url=vlm_base_url,
        api_key=vlm_api_key,
        model_id=vlm_model_id,
        cache_dir=cache_dir,
        concurrency=vlm_concurrency,
        prompt=compose_prompt("image_file", vlm_prompt, filename=path.name),
        # A file the user put in the knowledge base on its own is never
        # "decorative", whatever its pixel size.
        filter_decorative=False,
        vlm_options=vlm_options,
        caption_cache_root=caption_cache_root,
        progress_cb=progress_cb,
    )
    # For an embedded figure a failed caption costs one block's description;
    # for a standalone image it is the whole document, so fail the job and let
    # the worker's retry schedule try again instead of indexing a blank. A
    # skip is the same outcome for a standalone file: no description, no
    # vector. When the picture itself is the reason (unreadable bytes that
    # slipped past verify, an aspect ratio the model rejects) a retry cannot
    # change anything, so the job fails for good instead of backing off.
    status = str(block.metadata.get("vlm_status") or "")
    if status == "failed":
        raise RuntimeError(f"VLM caption failed for image file {path.name}: {block.metadata.get('vlm_error')}")
    if status == "skipped":
        reason = block.metadata.get("vlm_skip_reason")
        error = NonRetryableParseError if reason in PERMANENT_SKIP_REASONS else RuntimeError
        raise error(f"VLM caption skipped for image file {path.name}: {reason}")
    return [block]
