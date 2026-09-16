"""Deterministic synthetic corpus used by tokenizer format-compatibility tests.

This module is deliberately free of tokenizer imports so it can be consumed by
both the legacy (pre-memory-fix) trainer — used to produce the committed
``tests/data/tokenizer_vocab512_legacy.json`` fixture — and the current trainer
in the tests. The corpus is small, deterministic and self-contained (no
copyrighted data): multilingual prose, code and mathematics.
"""

EN_WORDS = (
    "the quick brown fox jumps over lazy dog model attention transformer neural "
    "network tokenizer byte merge training inference scaling efficient reasoning "
    "capability research language open source foundation recursive descent"
).split()

CODE_LINES = [
    "def foo(x):",
    "    return x + 1",
    "class Tokenizer:",
    "        self._rank = 0",
    "import torch as th",
    "for i in range(10):",
    "    print(i)",
    "if __name__ == '__main__':",
]

FIXTURE_CORPUS = [
    "The quick brown fox jumps over the lazy dog while the model trains.",
    "Attention is all you need but byte-level BPE is also quite useful.",
    "你好世界 語言模型 深度学习 神经网络",
    "العربية اللغة العربية موديل",
    "हिन्दी मॉडल नेटवर्क",
    "Русский кириллица модель",
    "def foo(x):\n    return x + 1\n\nif __name__ == '__main__':\n    print('hi')",
    "math ∑ f(x) = ∫ x² dx ≈ 42 · ¾ ∞ ≤ ≥ ± √ 🚀",
    "NUL\x00byte\x00and\x00more",
    "the quick brown fox jumps over lazy dog model attention transformer neural",
    "network tokenizer byte merge training inference scaling efficient reasoning",
    "capability research language open source foundation recursive descent",
    "".join(chr(97 + (i % 26)) for i in range(4000)),  # long repetitive text
]