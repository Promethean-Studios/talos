"""Tests for the DDM disk-tiered KV cache (``experiments/ddm_kv_cache.py``).

Covers the design's required test list (``ddm_experiment_design.md`` §6.3):

(a) raw fp32 round-trip of one and multiple completed blocks — bitwise exact;
(b) tiered-vs-resident logits within ``1e-4`` through prefill + several
    decode steps (full attention ⇒ every decode recalls the whole prefix);
(c) length / ``reset()`` / ``close()`` cleanup, including unlink-on-failure;
(d) ledger byte arithmetic (exact per-block payload, cumulative read/write
    totals, persistent/transient accounting);
(e) temporary cold-store files are removed.

All tests are deterministic and CPU-fast (tiny geometry, no training).
"""
from __future__ import annotations

import os
from typing import Tuple

import pytest
import torch

from configs.presets import tiny_config
from experiments.ddm_kv_cache import DiskTieredKVCache, HEADER_BYTES
from inference.generate import decode_step, prefill
from model import TalosGPT
from model.utils import set_seed

LOGIT_ATOL = 1e-4  # established fp32 guardrail (tests/test_inference.py)

# One layer's completed 64-token K+V block payload in fp32 bytes:
# 64 tokens x 2 (K,V) x 2 KV heads x 16 head_dim x 4 bytes = 16,384.
BLOCK_TOKENS = 64
BLOCK_PAYLOAD_BYTES = BLOCK_TOKENS * 2 * 2 * 16 * 4
GEOMETRY = dict(num_layers=2, num_kv_heads=2, head_dim=16, max_seq_len=512)


def _kv(t: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (batch=1, kv_heads=2, T, head_dim=16) fp32 K/V pair."""
    g = torch.Generator().manual_seed(seed)
    shape = (1, 2, t, 16)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


def _cache(**kwargs) -> DiskTieredKVCache:
    params = dict(GEOMETRY, batch_size=1, block_size=BLOCK_TOKENS)
    params.update(kwargs)
    return DiskTieredKVCache(**params)


@pytest.fixture()
def tiny_model() -> TalosGPT:
    set_seed(0)
    model = TalosGPT(tiny_config().derive())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# (a) raw block round-trip, bitwise exact
# ---------------------------------------------------------------------------
def test_single_block_round_trip_exact() -> None:
    cache = _cache()
    try:
        k, v = _kv(BLOCK_TOKENS, seed=1)
        rk, rv = cache.update(0, k, v, 0)  # prefill path: caller's tensors
        assert torch.equal(rk, k) and torch.equal(rv, v)
        # Completed block is offloaded: cold payload exact, hot tail empty.
        assert cache.file_payload_bytes() == BLOCK_PAYLOAD_BYTES
        assert cache.persistent_kv_bytes() == 0
        assert cache.ledger.blocks_written == 1
        assert cache.ledger.blocks_read == 0  # prefill must not read back
        # Raw file read-back is bit-preserving fp32.
        k2, v2 = cache._read_block(0, 0)
        assert torch.equal(k2, k) and torch.equal(v2, v)
        assert k2.dtype == torch.float32 and k2.shape == (1, 2, BLOCK_TOKENS, 16)
    finally:
        cache.close()


def test_multi_block_round_trip_exact() -> None:
    cache = _cache()
    try:
        blocks = [_kv(BLOCK_TOKENS, seed=10 + i) for i in range(3)]
        prefix_k = torch.cat([b[0] for b in blocks], dim=2)
        prefix_v = torch.cat([b[1] for b in blocks], dim=2)
        got: Tuple[torch.Tensor, torch.Tensor] | None = None
        for i, (k, v) in enumerate(blocks):
            got = cache.update(0, k, v, i * BLOCK_TOKENS)
        assert got is not None
        # Third update (decode-style append) reconstructs the whole prefix.
        assert torch.equal(got[0], prefix_k) and torch.equal(got[1], prefix_v)
        # Every stored block is bitwise identical to its source.
        for i, (k, v) in enumerate(blocks):
            k2, v2 = cache._read_block(0, i)
            assert torch.equal(k2, k) and torch.equal(v2, v)
        # Second layer stores into its own record region independently.
        k1, v1 = _kv(BLOCK_TOKENS, seed=99)
        cache.update(1, k1, v1, 0)
        assert torch.equal(cache._read_block(1, 0)[0], k1)
        assert torch.equal(cache._read_block(0, 0)[0], blocks[0][0])  # unchanged
        assert cache.ledger.blocks_written == 4
        assert cache.ledger.file_payload_current_bytes == 4 * BLOCK_PAYLOAD_BYTES
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# (b) tiered-vs-resident logit equivalence through prefill + decode
# ---------------------------------------------------------------------------
def test_tiered_logits_match_resident(tiny_model: TalosGPT) -> None:
    set_seed(1)
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 130), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        # Resident control via the canonical path.
        logits_r, cache_r = prefill(tiny_model, prompt)
        tok_r = logits_r[:, -1:, :].argmax(dim=-1)
        res_r = [decode_step(tiny_model, cache_r, tok_r, cache_r.length)]
        for _ in range(6):
            tok_r = res_r[-1][:, -1:, :].argmax(dim=-1)
            res_r.append(decode_step(tiny_model, cache_r, tok_r, cache_r.length))

        # Tiered cache through the same public prefill(cache=...) path.
        tier = DiskTieredKVCache.for_model(tiny_model)
        try:
            logits_t, cache_t = prefill(tiny_model, prompt, cache=tier)
            assert cache_t is tier and tier.length == 130
            assert float((logits_r - logits_t).abs().max()) <= LOGIT_ATOL
            tok_t = logits_t[:, -1:, :].argmax(dim=-1)
            for step_r in res_r:
                out_t = decode_step(tiny_model, cache_t, tok_t, cache_t.length)
                assert float((step_r - out_t).abs().max()) <= LOGIT_ATOL
                tok_t = out_t[:, -1:, :].argmax(dim=-1)
            # Full attention ⇒ every decode step recalled the cold prefix.
            assert tier.ledger.blocks_read > 0
            assert tier.ledger.bytes_read_total > 0
        finally:
            tier.close()


# ---------------------------------------------------------------------------
# (c) length / reset / close cleanup, unlink-on-failure
# ---------------------------------------------------------------------------
def test_length_tracking_and_reset() -> None:
    cache = _cache()
    try:
        assert cache.length == 0 and cache.last_len() == 0
        k, v = _kv(70, seed=3)  # 70 tokens => 1 block offloaded, 6-token tail
        cache.update(0, k, v, 0)
        assert cache.length == 70
        k1, v1 = _kv(1, seed=4)
        cache.update(1, k1, v1, 0)
        assert cache.length == 70 and cache.last_len() == 70
        assert cache.persistent_kv_bytes() == (6 + 1) * 2 * 2 * 16 * 4
        payload_before = cache.file_payload_bytes()
        assert payload_before == BLOCK_PAYLOAD_BYTES
        cache.reset()
        assert cache.length == 0
        assert cache.persistent_kv_bytes() == 0
        assert cache.file_payload_bytes() == 0
        assert cache.ledger.blocks_written == 0 and cache.ledger.blocks_read == 0
        assert os.path.getsize(cache.path) == HEADER_BYTES  # cold payload dropped
        # Still usable after reset.
        k2, v2 = _kv(BLOCK_TOKENS, seed=5)
        out = cache.update(0, k2, v2, 0)
        assert torch.equal(out[0], k2) and cache.length == BLOCK_TOKENS
    finally:
        cache.close()


def test_close_idempotent_and_removes_temp_file() -> None:
    cache = _cache()
    path = cache.path
    assert os.path.exists(path)
    k, v = _kv(BLOCK_TOKENS, seed=6)
    cache.update(0, k, v, 0)
    cache.close()
    assert not os.path.exists(path)  # unlink-on-close even with data
    cache.close()  # idempotent, never raises
    with pytest.raises(RuntimeError):
        cache.update(0, k, v, 0)


def test_close_unlinks_after_failed_update(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache()
    path = cache.path
    k, v = _kv(100, seed=7)  # completes a block => hits os.pwrite mid-update

    def _boom(*args: object, **kwargs: object) -> int:
        raise OSError("simulated write failure")

    monkeypatch.setattr(os, "pwrite", _boom)
    with pytest.raises(OSError):
        cache.update(0, k, v, 0)
    monkeypatch.undo()
    cache.close()  # must still unlink (failed-transaction cleanup)
    assert not os.path.exists(path)


def test_header_validation_rejects_mismatch() -> None:
    cache = _cache()
    try:
        with pytest.raises(ValueError, match="run-id mismatch"):
            _cache(path=cache.path, run_id="f" * 32)
        with pytest.raises(ValueError, match="geometry mismatch"):
            _cache(num_kv_heads=4, path=cache.path, run_id=cache.run_id)
        with pytest.raises(OSError):
            _cache(path=cache.path + ".missing", run_id=cache.run_id)
        # Same geometry + run id attaches fine (do this last: close() unlinks).
        twin = _cache(path=cache.path, run_id=cache.run_id)
        twin.close()
    finally:
        cache.close()


def test_update_rejects_bad_protocol() -> None:
    cache = _cache()
    try:
        k, v = _kv(4, seed=8)
        cache.update(0, k, v, 0)
        # Non-sequential append (hole/gap) is outside the experiment contract.
        with pytest.raises(ValueError, match="sequential"):
            cache.update(0, * _kv(4, seed=9), start_pos=10)
        # Batch mismatch (KVCache parity).
        with pytest.raises(ValueError, match="batch"):
            kb, vb = torch.randn(2, 2, 4, 16), torch.randn(2, 2, 4, 16)
            cache.update(0, kb, vb, 4)
        # dtype / device / geometry mismatches.
        with pytest.raises(ValueError, match="dtype"):
            cache.update(0, k.double(), v, 4)
        with pytest.raises(ValueError, match="kv heads"):
            cache.update(0, torch.randn(1, 4, 4, 16), torch.randn(1, 4, 4, 16), 4)
        # Overflowing max_seq_len (4 + 509 = 513 > 512).
        with pytest.raises(ValueError, match="max_seq_len"):
            cache.update(0, *_kv(509, seed=10), start_pos=4)
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# (d) ledger byte arithmetic + per-move / per-step events
# ---------------------------------------------------------------------------
def test_ledger_byte_arithmetic() -> None:
    cache = _cache()
    try:
        for layer in range(2):
            for i in range(3):  # 3 blocks per layer = 192 tokens
                k, v = _kv(BLOCK_TOKENS, seed=20 + 10 * layer + i)
                cache.update(layer, k, v, i * BLOCK_TOKENS)
        led = cache.ledger
        assert led.blocks_written == 6
        assert led.bytes_written_total == 6 * BLOCK_PAYLOAD_BYTES
        assert led.file_payload_current_bytes == 6 * BLOCK_PAYLOAD_BYTES
        assert led.file_payload_peak_bytes == 6 * BLOCK_PAYLOAD_BYTES
        assert [e["block_index"] for e in led.write_events if e["layer"] == 0] == [0, 1, 2]
        evt = led.write_events[0]
        assert evt["payload_bytes"] == BLOCK_PAYLOAD_BYTES
        assert evt["block_tokens"] == BLOCK_TOKENS
        for field in ("write_materialize_ms", "write_pwrite_ms", "write_sync_ms", "write_total_ms"):
            assert field in evt and evt[field] >= 0.0
        # Decode-style appends recall their cold prefix: update #2 read 2 blocks
        # and #3 read 3, on each layer (5 per layer, 10 total so far).
        assert led.blocks_read == 10
        assert led.bytes_read_total == 10 * BLOCK_PAYLOAD_BYTES
        # A decode-style append on layer 0 recalls all 3 cold blocks.
        k, v = _kv(1, seed=40)
        rk, rv = cache.update(0, k, v, 192)
        assert rk.shape == (1, 2, 193, 16) and rv.shape == (1, 2, 193, 16)
        assert led.blocks_read == 13
        assert led.bytes_read_total == 13 * BLOCK_PAYLOAD_BYTES
        step = led.step_events[-1]
        assert step["blocks_read"] == 3 and step["bytes_read"] == 3 * BLOCK_PAYLOAD_BYTES
        assert step["tier_prepare_ms"] >= 0.0
        for field in ("read_pread_ms", "read_reconstruct_ms", "read_total_ms"):
            assert field in led.read_events[0] and led.read_events[0][field] >= 0.0
        # Persistent hot tail is now: layer 0 = 1 token, layer 1 = 0 tokens.
        assert cache.persistent_kv_bytes() == 1 * 2 * 2 * 16 * 4
        inv = cache.memory_inventory()
        assert inv["persistent_kv_ram_peak_bytes"] >= inv["persistent_kv_ram_current_bytes"]
        assert inv["transient_kv_staging_peak_bytes"] == (193 * 2) * 2 * 16 * 4
        assert inv["bytes_read_total"] == 13 * BLOCK_PAYLOAD_BYTES
        assert inv["bytes_written_total"] == 6 * BLOCK_PAYLOAD_BYTES
        # Transient staging must dwarf the persistent tail (it is not a saving).
        assert inv["transient_kv_staging_peak_bytes"] > inv["persistent_kv_ram_current_bytes"]
    finally:
        cache.close()


def test_block_payload_matches_design_constant() -> None:
    """Design §3A: one layer/block payload is exactly 16,384 bytes."""
    cache = _cache()
    try:
        assert cache._record_bytes == 16_384
        k, v = _kv(BLOCK_TOKENS, seed=50)
        cache.update(0, k, v, 0)
        assert os.path.getsize(cache.path) == HEADER_BYTES + BLOCK_PAYLOAD_BYTES
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# (e) temp files are removed
# ---------------------------------------------------------------------------
def test_temp_files_are_private_and_removed() -> None:
    paths = []
    try:
        caches = [_cache() for _ in range(2)]
        paths = [c.path for c in caches]
        assert len(set(paths)) == 2  # per-run private files
        assert all(p.startswith(os.path.join(os.path.dirname(paths[0]), "talos-ddm-kv-")) for p in paths)
        for c in caches:
            c.close()
        assert all(not os.path.exists(p) for p in paths)
    finally:
        for p in paths:
            if os.path.exists(p):  # pragma: no cover - only on failure
                os.unlink(p)
