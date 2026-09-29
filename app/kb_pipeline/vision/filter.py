from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecorativeDecision:
    decorative_skip: bool
    decorative_reason: str | None = None


VLM_MAX_ASPECT = 100      # the Qwen-VL processor rejects images whose aspect ratio at native size exceeds 200; anything over 100 can only be a divider, a border strip or a header rule


def degenerate_image(width: int | None, height: int | None) -> str | None:
    """An image the vision model cannot accept at all: things like a 797x2 rule (aspect ratio 398) that
    MinerU cropped out of a docx / pdf. Returns the reason, or None if it is fine. Small images do not
    count: the processor upscales them to its minimum size, and a 16x16 icon is described as usual;
    only the aspect ratio is a hard limit. On 2026-09-11 a docx in kb_005 retried as a whole because of
    this: VLM 400 -> "description failed 1/1" -> not indexed -> back again 20 minutes later."""
    if width is None or height is None:
        return None
    if max(width, height) / max(1, min(width, height)) > VLM_MAX_ASPECT:
        return "degenerate_aspect"
    return None


def conservative_decorative_filter(
    *,
    width: int | None,
    height: int | None,
    object_name: str = "",
) -> DecorativeDecision:
    """Decorative images that can be settled before calling the model: very small icons, or files whose name
    says they are a background / divider. The rest (logos, watermarks, stamps) is judged by the model in its
    description (the decorative field of the vlm result)."""
    name = object_name.lower()
    if width is not None and height is not None:
        if width < 80 and height < 80:
            return DecorativeDecision(True, "tiny_icon")
    if any(token in name for token in ("background", "divider", "separator")):
        return DecorativeDecision(True, "decorative_name")
    return DecorativeDecision(False, None)

