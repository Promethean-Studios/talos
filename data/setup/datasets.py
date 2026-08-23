"""Registry of legally-usable public datasets that Talos's downloader knows.

This is *configuration + metadata only* — no data is vendored into the repo.
Each dataset entry documents: where it lives, its license, how to fetch it, and
how to prepare it into the JSONL/Parquet formats the pipeline reads.

Nothing here is fetched at build/test time. Use ``python -m data.setup.download
--dataset NAME --out DIR [--dry-run]`` to fetch and prepare on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetSpec:
    """A downloadable, legally-usable public dataset."""

    name: str
    license: str
    license_url: Optional[str]
    source: str  # e.g. "hf" (HuggingFace), "http" (direct), "torrent"
    hf_id: Optional[str] = None
    subset: Optional[str] = None
    urls: List[str] = field(default_factory=list)
    split: str = "train"
    text_field: str = "text"
    notes: str = ""
    #: for huggingface datasets: which fields to keep as metadata
    keep_fields: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"[{self.name}] source={self.source} license={self.license}\n"
            f"  {self.notes}"
        )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="pile",
        license="MIT (per subset; Pile is a mix — see Pile paper for per-source "
        "licenses)",
        license_url="https://pile.eleuther.ai/",
        source="hf",
        hf_id="EleutherAI/the_pile",
        subset="all",
        split="train",
        text_field="text",
        notes=(
            "The Pile: 22 subsets (e.g. Pile-CC, OpenWebText2, Books3, "
            "ArXiv, PubMed, GitHub, StackExchange). weights = per-subset "
            "'pile_subset' field. Download legally via HF; note per-subset "
            "licenses in the Pile paper before use."
        ),
    ),
    DatasetSpec(
        name="redpajama",
        license="RedPajama data license (Apache-2.0 code, per-source data "
        "licenses)",
        license_url="https://github.com/togethercomputer/RedPajama-Data",
        source="hf",
        hf_id="togethercomputer/RedPajama-Data-1T",
        split="train",
        text_field="text",
        notes=(
            "RedPajama v1: CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia, "
            "StackExchange subsets available as split files."
        ),
    ),
    DatasetSpec(
        name="fineweb",
        license="FineWeb is released under the ODC-By license",
        license_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb",
        source="hf",
        hf_id="HuggingFaceFW/fineweb",
        subset="sample-10BT",
        split="train",
        text_field="text",
        keep_fields=["url", "dump", "language", "quality"],
        notes="FineWeb: a high-quality web-text corpus. Use sample-10BT for a "
        "lightweight start, or CC-MAIN-2024-10 for the full ~15T tokens.",
    ),
    DatasetSpec(
        name="starcoder",
        license="Acceptable Use + per-language licenses (StarCoder Data "
        "Agreement)",
        license_url="https://huggingface.co/datasets/bigcode/starcoderdata",
        source="hf",
        hf_id="bigcode/starcoderdata",
        subset="python",
        split="train",
        text_field="content",
        notes="StarCoder data: per-language subsets (python, javascript, java, "
        "cpp, ...). A large code corpus; sample one language subset for dev.",
    ),
    DatasetSpec(
        name="codealpaca",
        license="CodeAlpaca-20k is Apache-2.0 (generated from Self-Instruct)",
        license_url="https://github.com/sahil280114/codealpaca",
        source="hf",
        hf_id="sahil280114/CodeAlpaca-20k",
        split="train",
        text_field="text",
        notes="Instruction-tuning (code) sample: 20k instruction/input/output "
        "examples. Ready-to-train on; good for a small mixed dev corpus.",
    ),
    DatasetSpec(
        name="numina_math",
        license="Apache-2.0 (NuminaMath-CoT sample)",
        license_url="https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
        source="hf",
        hf_id="AI-MO/NuminaMath-CoT",
        split="train",
        text_field="problem",
        keep_fields=["solution", "problem"],
        notes="NuminaMath: competition math problems + CoT solutions. The full "
        "set is large; a documented legal sample is enough for a dev corpus.",
    ),
    DatasetSpec(
        name="wiki_multilingual",
        license="Wikipedia: CC BY-SA 3.0 / GFDL",
        license_url="https://en.wikipedia.org/wiki/Wikipedia:Copyrights",
        source="hf",
        hf_id="wikipedia",
        subset="en",
        split="train",
        text_field="text",
        notes="Wikipedia dumps per language (subset='en' | 'fr' | 'de' | 'pl' "
        "| 'ru' | 'ja' | ...). Good multilingual natural-language source.",
    ),
    DatasetSpec(
        name="c4_sample",
        license="C4 is released under ODC-By",
        license_url="https://www.tensorflow.org/datasets/catalog/c4",
        source="hf",
        hf_id="allenai/c4",
        subset="en",
        split="train",
        text_field="text",
        notes="C4 (Colossal Clean Crawled Corpus). Streaming supported; sample "
        "a tiny slice for dev, stream the rest for training.",
    ),
]

_REGISTRY = {d.name: d for d in DATASETS}


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset by name, raising KeyError with a helpful message."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; known: {sorted(_REGISTRY)}"
        ) from None


def known_datasets() -> List[str]:
    return sorted(_REGISTRY)
