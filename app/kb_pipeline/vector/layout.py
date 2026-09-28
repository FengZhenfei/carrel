from __future__ import annotations

from dataclasses import dataclass


# Named vectors of a knowledge-base collection. Qdrant fixes the set of named
# vectors at collection creation (a name cannot be added later -- verified on
# 1.18: "Not existing vector name error"), so both are always declared even
# while the visual path is switched off; a point simply omits the vector it
# does not have.
TEXT_VECTOR = "text"
VISUAL_VECTOR = "visual"


@dataclass(frozen=True)
class VectorLayout:
    text_size: int
    visual_size: int

    def sizes(self) -> dict[str, int]:
        return {TEXT_VECTOR: int(self.text_size), VISUAL_VECTOR: int(self.visual_size)}

    def describe(self) -> str:
        return ",".join(f"{name}={size}" for name, size in self.sizes().items())
