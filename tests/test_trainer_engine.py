"""T4 training-engine pass regression tests (P1-P5).

CPU-run, memory-lean: SDPA equivalence, attention-backend flag routing, AMP
integration (scaler/clip/bad-step, CPU-safe), checkpoint atomicity + corrupt
rejection + fallback, sidecar field presence, and the expected-state-dict
shape pin used by resume validation.
"""
from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from data.packed import PackedTokenDataset
from model.attention import (
    PlainAttentionBackend,
    SDPAAttentionBackend,
    build_attention_backend,
)
from scripts.train_oasst1 import (
    RUN_METADATA_FILENAME,
    CheckpointStager,
    TrainingHistory,
    build_preset_model,
    expected_state_dict_shapes,
    make_grad_scaler,
    save_checkpoint,
    train_run,
    validate_resume_checkpoint,
)
from tests.test_trainer_stability import _args, _write_synthetic_jsonl  # reuse


# ---------------------------------------------------------------------------
# P1a — SDPA equivalence (CPU, tight tolerance)
# ---------------------------------------------------------------------------
def _qkv(batch=2, seq=32, heads=4, kv=4, d=16, seed=7):
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, kv, seq, d)
    v = torch.randn(batch, kv, seq, d)
    return q, k, v


def test_sdpa_matches_plain_causal_tight():
    q, k, v = _qkv()
    scale = q.shape[-1] ** -0.5
    ref = PlainAttentionBackend()(q, k, v, causal=True, scale=scale)
    out = SDPAAttentionBackend()(q, k, v, causal=True, scale=scale)
    assert torch.allclose(out, ref, atol=2e-6, rtol=2e-6)


def test_sdpa_matches_plain_windowed():
    q, k, v = _qkv(seq=16)
    scale = q.shape[-1] ** -0.5
    ref = PlainAttentionBackend()(q, k, v, window_size=5, causal=True, scale=scale)
    out = SDPAAttentionBackend()(q, k, v, window_size=5, causal=True, scale=scale)
    assert torch.allclose(out, ref, atol=2e-6, rtol=2e-6)


def test_sdpa_decode_matches_plain_suffix():
    # is_causal with nq != ns uses a PREFIX convention in torch; the SDPA
    # backend must build the suffix-convention mask for decode (single query
    # attending every cached key) — a 1-token decode over 8 keys == the last
    # row of an 8-token prefill.
    q = torch.randn(2, 4, 1, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    scale = 16 ** -0.5
    ref = PlainAttentionBackend()(q, k, v, causal=True, scale=scale)
    out = SDPAAttentionBackend()(q, k, v, causal=True, scale=scale)
    assert torch.allclose(out, ref, atol=2e-6, rtol=2e-6)


def test_model_logits_plain_vs_sdpa_close():
    torch.manual_seed(0)
    m1 = build_preset_model("tiny", attention_backend="plain").eval()
    m2 = build_preset_model("tiny", attention_backend="sdpa").eval()
    m2.load_state_dict(m1.state_dict())
    ids = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        l1, _ = m1(ids)
        l2, _ = m2(ids)
    assert torch.allclose(l1, l2, atol=2e-5, rtol=2e-5)


def test_attention_backend_flag_routing():
    assert isinstance(
        build_preset_model("tiny", attention_backend="plain").backend,
        PlainAttentionBackend,
    )
    assert isinstance(
        build_preset_model("tiny", attention_backend="sdpa").backend,
        SDPAAttentionBackend,
    )
    with pytest.raises(ValueError):
        build_preset_model("tiny", attention_backend="bogus")


# ---------------------------------------------------------------------------
# P1b/P3 — AMP integration (CPU-safe logic)
# ---------------------------------------------------------------------------
def test_amp_none_and_fp16_cpu_produce_identical_grads(tmp_path):
    """amp='none' is the old path; amp='fp16' on CPU (scaler disabled) must
    produce the same gradients — the autocast+scaler plumbing is a no-op."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    outs = []
    for amp in ("none", "fp16"):
        out = str(tmp_path / f"out_{amp}")
        metrics = train_run(_args(
            tmp_path, out, data=src,
            max_steps_per_epoch=2, val_max_steps=1, seq=16,
            attention_backend="sdpa", amp=amp, log_every_steps=1,
        ))
        ckpt = torch.load(os.path.join(out, "step-2.pt"),
                          map_location="cpu", weights_only=False)
        outs.append(ckpt["model_state_dict"]["layers.0.attention.q_proj.weight"])
    assert torch.allclose(outs[0], outs[1], atol=2e-3, rtol=2e-3), (
        "amp fp16 on CPU (autocast casts matmuls to half; scaler no-op) must "
        "be numerically close to amp none"
    )


def test_amp_fp16_with_plain_backend_rejected(tmp_path):
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    with pytest.raises(ValueError, match="requires the SDPA"):
        train_run(_args(
            tmp_path, str(tmp_path / "out"), data=src,
            attention_backend="plain", amp="fp16", max_steps_per_epoch=1,
            val_max_steps=1,
        ))


def test_scaler_overflow_counts_toward_bad_steps(tmp_path, monkeypatch):
    """A GradScaler overflow (found_inf) must surface through unscale_ into
    the always-on gradient guard: the step is skipped, the event recorded,
    and consecutive-bad-step counting drives the abort — P1b/P3."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out = str(tmp_path / "out")

    class OverflowScaler:
        def __init__(self, *a, **k):
            self.n_calls = 0

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss * 8.0

        def unscale_(self, opt):
            # Simulate a scaling overflow by poisoning the grads.
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad[() if p.grad.dim() == 0 else 0] = float("inf")

        def step(self, opt):
            pass

        def update(self):
            pass

    monkeypatch.setattr("scripts.train_oasst1.make_grad_scaler",
                        lambda amp, device: OverflowScaler() if amp == "fp16"
                        else make_grad_scaler(amp, device))
    metrics = train_run(_args(
        tmp_path, out, data=src, max_steps_per_epoch=1, val_max_steps=1,
        seq=16, attention_backend="sdpa", amp="fp16",
        max_consecutive_bad_steps=2,
    ))
    assert metrics["total_bad_steps"] >= 1
    assert metrics["consecutive_bad_steps"] == 1
    assert metrics["bad_steps"][0]["tensor"].startswith("grad:")


# ---------------------------------------------------------------------------
# P1e — packed fast path (cached mmap/pinned/prefetch) preserves the stream
# ---------------------------------------------------------------------------
def _packed_mini(tmp_path, seq=8, batch=2):
    import numpy as np
    from tests.fixture_corpus import FIXTURE_CORPUS
    from scripts.prepare_corpus import prepare_corpus
    from scripts.train_oasst1 import train_tokenizer_for_run
    from data.readers import JSONLReader
    from tokenizer.tokenizer import ByteLevelBPETokenizer

    src = str(tmp_path / "docs.jsonl")
    with open(src, "w", encoding="utf-8") as fh:
        for doc in FIXTURE_CORPUS:
            fh.write(json.dumps({"text": doc}) + "\n")
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer_for_run(src, tok_path, preset="tiny", num_merges=60)
    packed_dir = str(tmp_path / "packed")
    prepare_corpus(
        JSONLReader(src), ByteLevelBPETokenizer.from_file(tok_path),
        out_dir=packed_dir, target_tokens=400, val_tokens=60,
        tokenizer_path=tok_path, seq_len=seq, dtype="int32",
        val_skip_docs=2, stream_buffer_docs=4, rows_per_shard=8,
        dataset_meta={"dataset": "synthetic"}, args_echo={"smoke": True},
    )
    manifest = json.load(open(os.path.join(packed_dir, "manifest.json")))
    from data.packed import packed_phase_shard_paths
    return packed_dir, tok_path, packed_phase_shard_paths(
        manifest, packed_dir, "train"
    ), manifest


def test_packed_fast_path_stream_equals_classic(tmp_path):
    packed_dir, tok_path, paths, manifest = _packed_mini(tmp_path)
    seq = manifest["seq_len"]
    kw = dict(seq_len=seq, batch_size=2, expected_dtype="int32",
              max_id=1024, drop_last=True)
    classic = list(PackedTokenDataset(paths, **kw))
    fast = list(PackedTokenDataset(paths, cache_mmaps=True, pin_memory=True,
                                   prefetch=2, **kw))
    assert len(fast) == len(classic)
    for a, b in zip(classic, fast):
        # fast path yields int64 pooled buffers; classic yields int32 —
        # values must be identical.
        assert torch.equal(a.long(), b.long()), (
            "fast-path token stream must equal classic"
        )


def test_packed_train_e2e_with_prefetch(tmp_path):
    packed_dir, tok_path, _, _ = _packed_mini(tmp_path)
    out = str(tmp_path / "out")
    metrics = train_run(_args(
        tmp_path, out, data=None, packed_dir=packed_dir,
        tokenizer_json=tok_path, epochs=1, seq=None, batch=2,
        max_steps_per_epoch=4, val_max_steps=1,
        attention_backend="sdpa", prefetch=2, pin_memory=True,
        log_every_steps=2,
    ))
    assert metrics["engine"]["prefetch"] == 2
    assert metrics["engine"]["pin_memory"] is True
    assert metrics["interval_logs"], "interval logs must be present"
    assert metrics["performance"]["run_wide_tok_s"] is not None


# ---------------------------------------------------------------------------
# P5 — checkpoint atomicity, corrupt rejection + fallback, staging
# ---------------------------------------------------------------------------
def test_checkpoint_shape_mismatch_rejected_with_fallback(tmp_path):
    """A checkpoint whose state-dict tensor shape does not match the config
    must be rejected by validate_resume_checkpoint (tensor-head check) and
    the directory scanner must fall back to the previous valid checkpoint."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out = str(tmp_path / "out")
    metrics = train_run(_args(
        tmp_path, out, data=src, max_steps_per_epoch=1, val_max_steps=1,
        seq=16, log_every_steps=1,
    ))
    good = torch.load(os.path.join(out, "step-1.pt"),
                      map_location="cpu", weights_only=False)
    # Corrupt the tensor head of a NEWER checkpoint: different shape.
    bad = copy.deepcopy(good)
    bad["step"] = 2
    bad["model_state_dict"]["embed_tokens.weight"] = torch.zeros(
        512, 64
    )
    torch.save(bad, os.path.join(out, "step-2.pt"))
    with pytest.raises(ValueError, match="has shape"):
        validate_resume_checkpoint(bad, "tiny")
    # Directory resume picks step-1 (valid) and skips step-2 with a warning.
    resumed = train_run(_args(
        tmp_path, out + "-resume", data=src,
        resume=out, max_steps_per_epoch=1, val_max_steps=1, seq=16,
        epochs=2,  # resumed ckpt is already at epoch 1 — continue to epoch 2
    ))
    assert resumed["resumed_from"].endswith("step-1.pt")


def test_checkpoint_staging_copies_to_out_dir(tmp_path):
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out = str(tmp_path / "out")
    staging = str(tmp_path / "staging")
    train_run(_args(
        tmp_path, out, data=src, max_steps_per_epoch=1, val_max_steps=1,
        seq=16, ckpt_staging_dir=staging, log_every_steps=1,
    ))
    assert os.path.isfile(os.path.join(staging, "step-1.pt"))
    assert os.path.isfile(os.path.join(out, "step-1.pt"))


def test_expected_state_dict_shapes_pinned_against_real_model():
    from configs.presets import tiny_config
    model = build_preset_model("tiny", attention_backend="plain")
    actual = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    expected = {
        k: tuple(v) for k, v in expected_state_dict_shapes(
            tiny_config().derive()
        ).items()
    }
    assert actual == expected


# ---------------------------------------------------------------------------
# P4 — sidecar field presence
# ---------------------------------------------------------------------------
def test_run_metadata_records_engine_fields(tmp_path):
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out = str(tmp_path / "out")
    train_run(_args(
        tmp_path, out, data=src, max_steps_per_epoch=2, val_max_steps=1,
        seq=16, attention_backend="sdpa", amp="none", log_every_steps=1,
        ckpt_staging_dir=str(tmp_path / "staging"),
    ))
    sidecar = json.load(open(os.path.join(out, RUN_METADATA_FILENAME)))
    cfg = sidecar["training_config"]
    for key in ("attention_backend", "amp", "fused_optim", "pin_memory",
                "prefetch", "compile", "log_every_steps", "ckpt_staging_dir",
                "ckpt_fsync"):
        assert key in cfg, f"sidecar missing training_config.{key}"
    metrics_sidecar = json.load(open(os.path.join(out, "metrics.json")))
    assert metrics_sidecar["engine"]["attention_backend"] == "sdpa"
    assert "interval_logs" in metrics_sidecar
    assert "performance" in metrics_sidecar
    assert "checkpoint_save_total_s" in metrics_sidecar


def test_checkpoint_stager_drain_noop_without_staging(tmp_path):
    st = CheckpointStager(None, str(tmp_path))
    assert st.drain(timeout=5) is True