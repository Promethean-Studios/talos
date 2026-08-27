"""Core data types shared across the Talos data pipeline.

A :data:`Record` is the atomic unit that flows through every stage of the
pipeline. It is a plain ``dict`` with at least a ``text`` field plus arbitrary
*metadata* keys (``url``, ``source``, ``lang``, ``quality``, ...). Keeping it
a ``dict`` (rather than a dataclass) keeps it trivially JSON/Parquet
serialisable and lets processors attach derived fields (e.g. ``num_tokens``)
without changing the type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, TypedDict

#: A processed record: ``text`` plus optional metadata.
Record = Dict[str, Any]


class TextRecord(TypedDict, total=False):
    """Typed view of the minimal contract every reader must produce."""

    text: str
    url: Optional[str]
    source: str
    lang: Optional[str]
    quality: Optional[float]
    num_tokens: Optional[int]


#: Required key that every :data:`Record` must carry.
REQUIRED_FIELD = "text"


def validate_record(record: Record) -> None:
    """Raise :class:`ValueError` if ``record`` is not a usable document.

    Catches malformed records early so a bad source row fails loudly at the
    reader boundary instead of silently corrupting downstream counts.
    """
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict, got {type(record).__name__}")
    text = record.get(REQUIRED_FIELD)
    if not isinstance(text, str):
        raise ValueError(
            f"record['{REQUIRED_FIELD}'] must be a str, got {type(text).__name__}"
        )


@dataclass
class PipelineStats:
    """Running counters for a pipeline run (input/dropped/kept per stage)."""

    total_input: int = 0
    total_kept: int = 0
    total_dropped: int = 0
    #: dropped[stage_name] -> number of records dropped by that stage
    dropped_by_stage: Dict[str, int] = field(default_factory=dict)

    def record_input(self) -> None:
        self.total_input += 1

    def record_kept(self) -> None:
        self.total_kept += 1

    def record_dropped(self, stage: str) -> None:
        self.total_dropped += 1
        self.dropped_by_stage[stage] = self.dropped_by_stage.get(stage, 0) + 1

    def summary(self) -> Dict[str, Any]:
        return {
            "total_input": self.total_input,
            "total_kept": self.total_kept,
            "total_dropped": self.total_dropped,
            "dropped_by_stage": dict(self.dropped_by_stage),
        }


def merge_sources_counter(
    counter: Dict[str, int], records: Iterable[Record]
) -> Dict[str, int]:
    """Increment a per-``source`` counter for each record's source field."""
    out = dict(counter)
    for rec in records:
        src = rec.get("source", "unknown")
        out[src] = out.get(src, 0) + 1
    return out
