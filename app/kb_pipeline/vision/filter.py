from __future__ import annotations

VLM_MAX_ASPECT = 100


def degenerate_image(width: int | None, height: int | None) -> str | None:
    """Input compatibility guard for vision processors with bounded aspect ratios.

    This says nothing about the image's meaning. Small images remain eligible;
    dimensions and filenames cannot establish that an image is decorative.
    """
    if width is None or height is None:
        return None
    if max(width, height) / max(1, min(width, height)) > VLM_MAX_ASPECT:
        return "degenerate_aspect"
    return None

