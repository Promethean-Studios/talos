"""Talos data pipeline (Phase 3): streaming readers, processors, mixer, writers.

Pure stdlib + numpy (optional extras: pyarrow, datasets, langdetect — all
guarded lazy imports). See ``data/README.md`` for the architecture and how to
configure a mixed multilingual+code+math corpus.
"""
from __future__ import annotations

from data.dedup import ExactDedupFilter, MinHashNearDupFilter, minhash_signature
from data.langid import (
    HeuristicLanguageIdentifier,
    LanguageIdentifier,
    make_language_identifier,
)
from data.mixer import WeightedMixer
from data.processors import (
    BlacklistRegexFilter,
    CodeFenceFilter,
    FieldRemap,
    LanguageFilter,
    LengthFilter,
    Processor,
    ProcessorChain,
    QualityHeuristicFilter,
    RegexFilter,
    TokenCounter,
    URLBlacklistFilter,
    processor_from_config,
)
from data.readers import (
    DatasetReader,
    HuggingFaceReader,
    JSONLReader,
    ParquetReader,
    TextReader,
    reader_from_config,
)
from data.types import PipelineStats, Record
from data.writers import ShardedWriter

__all__ = [
    "BlacklistRegexFilter",
    "CodeFenceFilter",
    "DatasetReader",
    "ExactDedupFilter",
    "FieldRemap",
    "HeuristicLanguageIdentifier",
    "HuggingFaceReader",
    "JSONLReader",
    "LanguageFilter",
    "LanguageIdentifier",
    "LengthFilter",
    "MinHashNearDupFilter",
    "ParquetReader",
    "PipelineStats",
    "Processor",
    "ProcessorChain",
    "QualityHeuristicFilter",
    "Record",
    "RegexFilter",
    "ShardedWriter",
    "TextReader",
    "TokenCounter",
    "URLBlacklistFilter",
    "WeightedMixer",
    "make_language_identifier",
    "minhash_signature",
    "processor_from_config",
    "reader_from_config",
]
