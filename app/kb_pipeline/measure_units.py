"""Canonical spelling of measurement units (split into its own module in the 2026-09-09 final review F04: the
parse side's image-text correction and the graph side's reconciliation / concepts / trends share one copy).

Spelling normalization only, no conversion; conversion applies only to whitelisted dimensions, see
graph/reconcile. A leading M / m carries physical meaning (mega / milli), so the canonical key keeps its
case: MW and mW, MHz and mHz are different units, and downstream must not casefold them again.
"""
from __future__ import annotations

import unicodedata
from typing import Any

# Unit normalization: alias → canonical spelling (spelling only, no conversion; conversion applies only to
# whitelisted dimensions, see reconcile)
_UNIT_ALIASES = {
    "ma": "mA", "毫安": "mA", "milliamp": "mA", "milliamps": "mA", "ua": "µA", "μa": "µA", "µa": "µA", "微安": "µA",
    "a": "A", "安": "A", "mv": "mV", "毫伏": "mV", "v": "V", "伏": "V", "伏特": "V", "kv": "kV",
    "ns": "ns", "纳秒": "ns", "ps": "ps", "us": "µs", "μs": "µs", "µs": "µs", "微秒": "µs", "ms": "ms", "毫秒": "ms", "s": "s", "秒": "s",
    "mhz": "MHz", "兆赫": "MHz", "兆赫兹": "MHz", "khz": "kHz", "ghz": "GHz", "hz": "Hz",
    "°c": "°C", "℃": "°C", "摄氏度": "°C", "c": "°C", "°f": "°F", "℉": "°F",
    "mw": "mW", "毫瓦": "mW", "w": "W", "瓦": "W", "ohm": "Ω", "ohms": "Ω", "ω": "Ω", "欧姆": "Ω", "kω": "kΩ", "kohm": "kΩ",
    "pf": "pF", "nf": "nF", "uf": "µF", "μf": "µF", "µf": "µF",
    "mm": "mm", "毫米": "mm", "cm": "cm", "厘米": "cm", "m": "m", "米": "m", "kg": "kg", "千克": "kg", "公斤": "kg", "g": "g", "克": "g",
    "mmhg": "mmHg", "mmol/l": "mmol/L", "umol/l": "µmol/L", "μmol/l": "µmol/L", "µmol/l": "µmol/L", "mg/dl": "mg/dL", "mg/l": "mg/L",
    "g/l": "g/L", "u/l": "U/L", "iu/l": "IU/L", "miu/ml": "mIU/mL", "ng/ml": "ng/mL", "pg/ml": "pg/mL", "10^9/l": "10^9/L",
    "×10^9/l": "10^9/L", "x10^9/l": "10^9/L", "×10~9/l": "10^9/L", "10^12/l": "10^12/L", "×10^12/l": "10^12/L", "×10~12/l": "10^12/L",
    "fl": "fL", "pg": "pg", "%": "%", "百分比": "%", "次/分": "bpm", "bpm": "bpm", "次/min": "bpm", "元": "CNY", "人民币": "CNY", "usd": "USD", "$": "USD",
}


_CANONICAL_UNITS = set(_UNIT_ALIASES.values())
_CANONICAL_UNITS_FOLDED = {u.casefold() for u in _CANONICAL_UNITS}


def canonical_unit(unit: Any) -> str:
    """Canonical spelling of a unit: case, Chinese / English aliases and OCR spellings like ×10~9/L are
    normalized; unrecognized input is returned as is. A leading M / m carries physical meaning (mega /
    milli): when the original starts with M and the alias would turn it into m (or the other way round),
    and what remains after dropping that first letter is itself a unit (MW → W, mHz → Hz), that is not a
    spelling difference and is kept as is; only when the remainder is not a unit (MMOL/L, MMHG) is it an
    all-caps writing habit, normalized as usual (Codex review N06)."""
    text = unicodedata.normalize("NFKC", str(unit or "")).strip()
    if not text:
        return ""
    compact = text.replace(" ", "")
    if compact in _CANONICAL_UNITS:
        return compact
    hit = _UNIT_ALIASES.get(compact.casefold())
    if hit is None:
        return text
    if (compact[:1] in ("M", "m") and hit[:1] in ("M", "m") and compact[:1] != hit[:1]
            and compact[1:].casefold() in _CANONICAL_UNITS_FOLDED):
        return text
    return hit


def unit_key(unit: Any) -> str:
    """Key of a unit for grouping / comparison: canonical spelling with whitespace removed, case preserved
    (final review F04). Empty unit → empty string."""
    return canonical_unit(unit).replace(" ", "")
