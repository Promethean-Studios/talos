"""Disk-tiered KV cache for the DDM memory-tier hypothesis experiment.

Implements the **primary experimental variant** of the DDM memory-tier design
(``ddm_experiment_design.md``): a KV cache whose *hot tier* (CPU RAM) holds only
the current unfinished block of tokens, while every **completed 64-token K/V
block is offloaded to one run-private temporary file** (a raw, fixed-layout
record file — not ``torch.save``) and recalled on every decode ``update``,
because the ``tiny`` preset uses full attention and therefore needs the whole
prefix for every decode step.

This is a *hypothesis-experiment* artifact, not a production cache: the point
is to measure, with exact byte accounting and per-block timings, whether a
lower file tier can lower persistent application-level KV RAM at an acceptable
throughput cost on the 254,272-parameter tiny prototype. It deliberately
implements no caching policy beyond the prespecified eviction/recall rules.

Protocol subset (duck-typed by ``model.TalosGPT`` / ``inference.generate``):

* ``length`` (property) / ``last_len()`` — cached-token count (all layers).
* ``update(layer, key, value, start_pos)`` — write one layer's new K/V and get
  back the full prefix ``(batch, num_kv_heads, cur_len, head_dim)``.
* ``reset()`` — drop all sequence state (and the cold payload) for reuse.
* ``close()`` — close the backing file and unlink it (idempotent; safe after
  failed updates).

Memory accounting (design §3D) is exposed by :class:`CacheLedger` and
:meth:`DiskTieredKVCache.memory_inventory` as *exact* logical byte counts:

* ``persistent`` — K/V retained in RAM between layer calls (the hot tail only).
* ``transient``  — K/V reconstructed by the cache solely to execute the current
  attention operation (never counted as a saving; never retained).
* ``cold file``  — payload bytes currently in the file tier, plus cumulative
  bytes read/written and total lower-tier data-motion time.
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = ["CacheLedger", "DiskTieredKVCache", "HEADER_BYTES", "MAGIC"]

# ---------------------------------------------------------------------------
# Raw fixed-layout record file
# ---------------------------------------------------------------------------
# Header (little-endian, zero-padded to HEADER_BYTES):
#   magic | version | dtype_code | num_layers | num_kv_heads | head_dim |
#   block_size | batch | max_blocks_per_layer | run_id (32 ASCII chars)
_MAGIC = b"TLDDMKV1"
_VERSION = 1
_DTYPE_FP32 = 32  # only fp32 payloads are supported (design §2: fp32 bytes only)
_HEADER_STRUCT = struct.Struct("<8sIIIIIIII32s")
HEADER_BYTES = 128
_RUN_ID_LEN = 32


class CacheLedger:
    """Event + byte ledger for one tiered cache instance.

    Collects per-block write/read events with exact payload byte counts and
    ``time.perf_counter()`` timings, cumulative totals, and the persistent /
    transient memory peaks required by the experiment design (§3A, §3D).
    All byte counts are unrounded integers; all times are milliseconds.
    """

    def __init__(self, fdatasync_enabled: bool) -> None:
        self.fdatasync_enabled = bool(fdatasync_enabled)
        self.write_events: List[Dict[str, Any]] = []
        self.read_events: List[Dict[str, Any]] = []
        self.step_events: List[Dict[str, Any]] = []
        self.bytes_written_total = 0
        self.bytes_read_total = 0
        self.move_time_total_ms = 0.0
        self.blocks_written = 0
        self.blocks_read = 0
        self.file_payload_current_bytes = 0
        self.file_payload_peak_bytes = 0
        self.persistent_peak_bytes = 0
        self.transient_peak_bytes = 0
        # Advisory page-eviction bookkeeping (os.posix_fadvise DONTNEED).
        self.fadvise: Dict[str, Any] = {"supported": False, "requested": 0, "failed": 0}
        self.fdatasync: Dict[str, Any] = {"calls": 0, "failed": 0}
        self._persistent_current_bytes = 0

    # -- current-state helpers ---------------------------------------------------
    def note_persistent(self, nbytes: int) -> None:
        """Record the persistent hot-K/V bytes retained between layer calls."""
        self._persistent_current_bytes = int(nbytes)
        self.persistent_peak_bytes = max(self.persistent_peak_bytes, int(nbytes))

    def note_transient(self, nbytes: int) -> None:
        """Record a cache-reconstructed staging size (peak-tracked)."""
        self.transient_peak_bytes = max(self.transient_peak_bytes, int(nbytes))

    def persistent_kv_bytes(self) -> int:
        """K/V bytes currently retained in RAM between layer calls."""
        return self._persistent_current_bytes

    def file_payload_bytes(self) -> int:
        """K/V payload bytes currently stored in the cold file tier."""
        return self.file_payload_current_bytes

    def clear(self) -> None:
        """Reset all events/counters/peaks (used by ``reset()``)."""
        self.write_events.clear()
        self.read_events.clear()
        self.step_events.clear()
        self.bytes_written_total = 0
        self.bytes_read_total = 0
        self.move_time_total_ms = 0.0
        self.blocks_written = 0
        self.blocks_read = 0
        self.file_payload_current_bytes = 0
        self.file_payload_peak_bytes = 0
        self.persistent_peak_bytes = 0
        self.transient_peak_bytes = 0
        self.fadvise["requested"] = 0
        self.fadvise["failed"] = 0
        self.fdatasync["calls"] = 0
        self.fdatasync["failed"] = 0
        self._persistent_current_bytes = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes_written_total": self.bytes_written_total,
            "bytes_read_total": self.bytes_read_total,
            "move_time_total_ms": round(self.move_time_total_ms, 4),
            "blocks_written": self.blocks_written,
            "blocks_read": self.blocks_read,
            "file_payload_current_bytes": self.file_payload_current_bytes,
            "file_payload_peak_bytes": self.file_payload_peak_bytes,
            "persistent_kv_ram_current_bytes": self._persistent_current_bytes,
            "persistent_kv_ram_peak_bytes": self.persistent_peak_bytes,
            "transient_kv_staging_peak_bytes": self.transient_peak_bytes,
            "cold_page_eviction": dict(self.fadvise),
            "fdatasync": {"enabled": self.fdatasync_enabled, **self.fdatasync},
            "num_write_events": len(self.write_events),
            "num_read_events": len(self.read_events),
            "num_step_events": len(self.step_events),
        }


class DiskTieredKVCache:
    """KV cache with a RAM hot tail and a file cold tier for completed blocks.

    Layout: one private temp file holds a fixed metadata header (magic,
    geometry, dtype, run id — used to reject mismatched attachments) followed
    by fixed-size records at offset ``HEADER + (layer * max_blocks + index) *
    record_bytes``. Each record is the raw fp32 bytes of K then V for one
    completed block: ``(batch, num_kv_heads, block_tokens, head_dim)`` each.

    Semantics mirror :class:`model.cache.KVCache` for the sequential-append
    protocol used by prefill + decode (arbitrary in-place rewrites are outside
    the experiment's contract and raise ``ValueError``). The global ``length``
    is the ``max`` over per-layer written lengths, matching the all-layer cache.

    Args:
        num_layers / num_kv_heads / head_dim / max_seq_len / batch_size:
            cache geometry (must match the model exactly).
        dtype: payload dtype; only ``torch.float32`` is supported (raw fp32
            bytes-only layout per design §2).
        device: must be ``cpu`` (or ``None``).
        block_size: tokens per offloaded block (64 in the experiment).
        sync: ``fdatasync`` each completed block before the advisory
            page-eviction request (design §2 file-I/O policy).
        path: attach to an existing cold-store file instead of creating a
            private temp file (used by tests to exercise header validation).
        run_id: expected run id when attaching to an existing file.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        block_size: int = 64,
        sync: bool = True,
        path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if dtype is not torch.float32:
            raise ValueError(
                "DiskTieredKVCache stores raw fp32 payload bytes only "
                f"(got dtype={dtype}); the experiment design requires fp32"
            )
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError("DiskTieredKVCache supports device 'cpu' only")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.sync = bool(sync)
        self._dtype = dtype
        self._device = torch.device("cpu")
        self._max_blocks = int(math.ceil(self.max_seq_len / self.block_size))
        # One record = K + V for one layer's block: B*H*T*D fp32 bytes each.
        self._record_bytes = (
            self.batch_size * self.num_kv_heads * self.block_size * self.head_dim * 4 * 2
        )
        self._layer_len: List[int] = [0] * self.num_layers
        self._tail_k: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._tail_v: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.ledger = CacheLedger(fdatasync_enabled=self.sync)
        self._closed = False
        self._fadvise_available = hasattr(os, "posix_fadvise") and hasattr(
            os, "POSIX_FADV_DONTNEED"
        )
        self.ledger.fadvise["supported"] = self._fadvise_available

        self.run_id = run_id if run_id is not None else uuid.uuid4().hex
        if len(self.run_id) != _RUN_ID_LEN:
            raise ValueError(f"run_id must be {_RUN_ID_LEN} ASCII chars")
        if path is None:
            fd, self.path = tempfile.mkstemp(prefix="talos-ddm-kv-", suffix=".bin")
            self._write_header(fd)
        else:
            self.path = str(path)
            fd = os.open(self.path, os.O_RDWR)
            try:
                self._validate_header(fd)
            except Exception:
                os.close(fd)
                raise
        self._fd = fd

    # -- construction helpers -----------------------------------------------------
    @classmethod
    def for_model(
        cls,
        model: Any,
        batch_size: int = 1,
        block_size: int = 64,
        sync: bool = True,
        path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> "DiskTieredKVCache":
        """Build a tiered cache shaped for ``model`` (its config + dtype)."""
        cfg = model.config
        return cls(
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.max_seq_len,
            batch_size=batch_size,
            dtype=next(model.parameters()).dtype,
            device=torch.device("cpu"),
            block_size=block_size,
            sync=sync,
            path=path,
            run_id=run_id,
        )

    def _header_bytes(self) -> bytes:
        blob = _HEADER_STRUCT.pack(
            _MAGIC,
            _VERSION,
            _DTYPE_FP32,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.batch_size,
            self._max_blocks,
            self.run_id.encode("ascii"),
        )
        return blob.ljust(HEADER_BYTES, b"\0")

    def _write_header(self, fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, self._header_bytes())

    def _validate_header(self, fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        header = os.read(fd, HEADER_BYTES)
        if len(header) < HEADER_BYTES:
            raise ValueError(
                f"cold-store file {self.path!r} is not a valid Talos DDM tier file"
            )
        (
            magic,
            version,
            dtype_code,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            batch,
            _max_blocks,
            run_id,
        ) = _HEADER_STRUCT.unpack(header[: _HEADER_STRUCT.size])
        if magic != _MAGIC:
            raise ValueError(f"cold-store file {self.path!r} has wrong magic {magic!r}")
        expected = (
            _VERSION,
            _DTYPE_FP32,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.batch_size,
        )
        got = (version, dtype_code, num_layers, num_kv_heads, head_dim, block_size, batch)
        if got != expected:
            raise ValueError(
                "cold-store file geometry mismatch: file="
                f"{got} (version, dtype_code, layers, kv_heads, head_dim, "
                f"block_size, batch), requested={expected}"
            )
        stored_run = run_id.decode("ascii", errors="replace")
        if stored_run != self.run_id:
            raise ValueError(
                f"cold-store file run-id mismatch: file={stored_run!r}, "
                f"requested={self.run_id!r}"
            )

    # -- cache protocol -------------------------------------------------------------
    @property
    def length(self) -> int:
        """Number of cached tokens (max over layers; 0 when empty)."""
        return max(self._layer_len) if self._layer_len else 0

    def last_len(self) -> int:
        """Alias for :attr:`length` (used by the model for decode positions)."""
        return self.length

    def update(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write one layer's new K/V and return the full prefix for attention.

        Prefill (``start_pos == 0`` on an empty layer) returns the caller's
        just-computed tensors directly (the whole prefix is already in hand;
        completed blocks are still offloaded). A decode append recalls every
        completed cold block for that layer, appends the hot tail, and returns
        a contiguous **transient** prefix ``(batch, num_kv_heads, cur_len,
        head_dim)``. The global length becomes ``max(existing, start_pos + T)``
        (per-layer appends; consistent with the all-layer ``KVCache``).
        """
        if self._closed:
            raise RuntimeError("DiskTieredKVCache is closed")
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(f"layer {layer} out of range [0, {self.num_layers})")
        self._validate_kv(key, "key")
        self._validate_kv(value, "value")
        _batch, _kv, t, _dim = key.shape
        if t < 1:
            raise ValueError(f"update() requires at least one token, got T={t}")
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")
        layer_len = self._layer_len[layer]
        is_prefill = start_pos == 0 and layer_len == 0
        if not is_prefill and start_pos != layer_len:
            raise ValueError(
                "DiskTieredKVCache supports sequential appends only: "
                f"layer {layer} holds {layer_len} tokens, got start_pos={start_pos}"
            )
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"sequence end {end} exceeds max_seq_len {self.max_seq_len}; "
                "call cache.reset() or increase max_seq_len"
            )

        # Hot tier: append the new tokens, offloading completed blocks.
        self._append_tail(layer, key, value)

        if is_prefill:
            # Return the caller's just-computed in-memory tensors directly;
            # do not read back data that is already present (design §6).
            returned = (key, value)
        else:
            returned = self._reconstruct(layer, end)

        self._layer_len[layer] = end
        self.ledger.note_persistent(self._persistent_bytes())
        return returned

    def _validate_kv(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.dtype is not self._dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} != cache dtype {self._dtype}")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on cpu, got {tensor.device}")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4D (B,H,T,D), got {tuple(tensor.shape)}")
        if tensor.shape[0] != self.batch_size:
            raise ValueError(f"cache holds batch {self.batch_size}, got batch {tensor.shape[0]}")
        _b, kv, _t, dim = tensor.shape
        if kv != self.num_kv_heads:
            raise ValueError(f"expected {self.num_kv_heads} kv heads, got {kv}")
        if dim != self.head_dim:
            raise ValueError(f"expected head_dim {self.head_dim}, got {dim}")

    # -- hot tier -------------------------------------------------------------------
    def _append_tail(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        tk, tv = self._tail_k[layer], self._tail_v[layer]
        self._tail_k[layer] = key if tk is None else torch.cat((tk, key), dim=2)
        self._tail_v[layer] = value if tv is None else torch.cat((tv, value), dim=2)
        self._drain_tail(layer)

    def _drain_tail(self, layer: int) -> None:
        """Offload complete leading blocks of the layer's tail to the file tier."""
        while True:
            tk, tv = self._tail_k[layer], self._tail_v[layer]
            if tk is None or int(tk.shape[2]) < self.block_size:
                return
            block_index = sum(1 for e in self.ledger.write_events if e["layer"] == layer)
            self._write_block(
                layer,
                block_index,
                tk[:, :, : self.block_size, :],
                tv[:, :, : self.block_size, :],
            )
            rest_k = tk[:, :, self.block_size :, :]
            rest_v = tv[:, :, self.block_size :, :]
            if int(rest_k.shape[2]) == 0:
                self._tail_k[layer] = None
                self._tail_v[layer] = None
            else:
                # Keep the retained hot buffer contiguous so its nbytes are the
                # honest persistent-KV accounting (small: < block_size tokens).
                self._tail_k[layer] = rest_k.contiguous()
                self._tail_v[layer] = rest_v.contiguous()

    def _persistent_bytes(self) -> int:
        """Exact K/V bytes this cache retains in RAM right now (all layers)."""
        total = 0
        for lk, lv in zip(self._tail_k, self._tail_v):
            if lk is not None:
                total += lk.numel() * lk.element_size()
            if lv is not None:
                total += lv.numel() * lv.element_size()
        return total

    # -- cold tier I/O ----------------------------------------------------------------
    def _block_offset(self, layer: int, block_index: int) -> int:
        return HEADER_BYTES + (layer * self._max_blocks + block_index) * self._record_bytes

    def _write_block(
        self, layer: int, block_index: int, block_k: torch.Tensor, block_v: torch.Tensor
    ) -> None:
        """Serialize one completed block to raw fp32 bytes and commit it.

        Records separate materialization/copy, ``pwrite``, ``fdatasync`` (if
        enabled) and total timings, then requests advisory page eviction with
        ``os.posix_fadvise(..., POSIX_FADV_DONTNEED)`` where supported.
        """
        t0 = time.perf_counter()
        data = (
            block_k.detach().cpu().contiguous().numpy().tobytes()
            + block_v.detach().cpu().contiguous().numpy().tobytes()
        )
        t1 = time.perf_counter()
        offset = self._block_offset(layer, block_index)
        # Loop: os.pwrite may legitimately write fewer bytes than requested.
        remaining = memoryview(data)
        file_offset = offset
        while remaining:
            written = os.pwrite(self._fd, remaining, file_offset)
            if written <= 0:  # pragma: no cover - would mean a full device
                raise OSError(f"pwrite returned {written} at offset {file_offset}")
            remaining = remaining[written:]
            file_offset += written
        t2 = time.perf_counter()
        sync_ms = 0.0
        if self.sync:
            t2b = time.perf_counter()
            try:
                os.fdatasync(self._fd)
                self.ledger.fdatasync["calls"] += 1
            except OSError:
                self.ledger.fdatasync["failed"] += 1
            sync_ms = (time.perf_counter() - t2b) * 1000.0
        t3 = time.perf_counter()
        if self._fadvise_available:
            try:
                os.posix_fadvise(self._fd, offset, len(data), os.POSIX_FADV_DONTNEED)
                self.ledger.fadvise["requested"] += 1
            except OSError:
                self.ledger.fadvise["failed"] += 1
        t4 = time.perf_counter()
        total_ms = (t4 - t0) * 1000.0
        self.ledger.write_events.append(
            {
                "block_tokens": self.block_size,
                "layer": int(layer),
                "block_index": int(block_index),
                "payload_bytes": len(data),
                "write_materialize_ms": round((t1 - t0) * 1000.0, 6),
                "write_pwrite_ms": round((t2 - t1) * 1000.0, 6),
                "write_sync_ms": round(sync_ms, 6),
                "write_total_ms": round(total_ms, 6),
            }
        )
        self.ledger.bytes_written_total += len(data)
        self.ledger.blocks_written += 1
        self.ledger.move_time_total_ms += total_ms
        self.ledger.file_payload_current_bytes += len(data)
        self.ledger.file_payload_peak_bytes = max(
            self.ledger.file_payload_peak_bytes, self.ledger.file_payload_current_bytes
        )

    def _read_block(self, layer: int, block_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read one block's raw fp32 record and rebuild its K/V tensors."""
        nbytes = self._record_bytes
        offset = self._block_offset(layer, block_index)
        t0 = time.perf_counter()
        data = os.pread(self._fd, nbytes, offset)
        if len(data) != nbytes:
            raise OSError(
                f"cold-store short read at offset {offset}: got {len(data)} of {nbytes} bytes"
            )
        t1 = time.perf_counter()
        half = nbytes // 2
        shape = (self.batch_size, self.num_kv_heads, self.block_size, self.head_dim)
        # bytearray => writable buffer => torch can share it without a warning.
        k = torch.frombuffer(bytearray(data[:half]), dtype=torch.float32).view(shape)
        v = torch.frombuffer(bytearray(data[half:]), dtype=torch.float32).view(shape)
        t2 = time.perf_counter()
        pread_ms = (t1 - t0) * 1000.0
        reconstruct_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0
        self.ledger.read_events.append(
            {
                "block_tokens": self.block_size,
                "layer": int(layer),
                "block_index": int(block_index),
                "payload_bytes": nbytes,
                "read_pread_ms": round(pread_ms, 6),
                "read_reconstruct_ms": round(reconstruct_ms, 6),
                "read_total_ms": round(total_ms, 6),
            }
        )
        self.ledger.bytes_read_total += nbytes
        self.ledger.blocks_read += 1
        self.ledger.move_time_total_ms += total_ms
        return k, v

    def _reconstruct(self, layer: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recall all cold blocks + hot tail into one contiguous transient prefix."""
        n_cold = end // self.block_size  # completed blocks below the tail
        t0 = time.perf_counter()
        cold_ks: List[torch.Tensor] = []
        cold_vs: List[torch.Tensor] = []
        for block_index in range(n_cold):
            k, v = self._read_block(layer, block_index)
            cold_ks.append(k)
            cold_vs.append(v)
        tk, tv = self._tail_k[layer], self._tail_v[layer]
        if cold_ks:
            cold_k = torch.cat(cold_ks, dim=2)
            cold_v = torch.cat(cold_vs, dim=2)
            k = cold_k if tk is None else torch.cat((cold_k, tk), dim=2)
            v = cold_v if tv is None else torch.cat((cold_v, tv), dim=2)
        else:
            # No cold blocks yet: the tail *is* the prefix (a view of the
            # persistent hot buffer — no extra staging allocation).
            k = tk
            v = tv
        assemble_ms = (time.perf_counter() - t0) * 1000.0
        cur_len = int(k.shape[2])
        if cur_len != end:  # pragma: no cover - internal invariant
            raise AssertionError(f"reconstructed prefix length {cur_len} != expected {end}")
        if k is not None and cold_ks:
            # Cache-created staging for the current attention op only; never
            # retained (design §6: transient, not a memory saving).
            self.ledger.note_transient((k.numel() + v.numel()) * k.element_size())
        self.ledger.step_events.append(
            {
                "layer": int(layer),
                "start_pos": end - 1,
                "cur_len": end,
                "blocks_read": n_cold,
                "bytes_read": n_cold * self._record_bytes,
                "tier_prepare_ms": round(assemble_ms, 6),
                "persistent_after_bytes": self._persistent_bytes(),
            }
        )
        return k, v

    # -- lifecycle --------------------------------------------------------------------
    def reset(self) -> None:
        """Drop all sequence state, the cold payload, and ledger history."""
        if self._closed:
            raise RuntimeError("DiskTieredKVCache is closed")
        self._layer_len = [0] * self.num_layers
        self._tail_k = [None] * self.num_layers
        self._tail_v = [None] * self.num_layers
        os.ftruncate(self._fd, HEADER_BYTES)
        self.ledger.clear()

    def close(self) -> None:
        """Close the backing file and unlink it. Idempotent; never raises.

        Unlinks even after failed updates or short/corrupt reads so a failed
        transaction cannot leave temporary data behind (design §6).
        """
        if self._closed:
            return
        self._closed = True
        fd, path = self._fd, self.path
        self._fd = -1
        if fd is not None and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # -- inspection -------------------------------------------------------------------
    def file_payload_bytes(self) -> int:
        """K/V payload bytes currently stored in the cold file tier."""
        return self.ledger.file_payload_current_bytes

    def persistent_kv_bytes(self) -> int:
        """K/V bytes currently retained in RAM between layer calls."""
        return self.ledger.persistent_kv_bytes()

    def memory_inventory(self) -> Dict[str, int]:
        """Exact logical byte counts (design §3D quantities 2–4)."""
        led = self.ledger
        return {
            "persistent_kv_ram_current_bytes": led.persistent_kv_bytes(),
            "persistent_kv_ram_peak_bytes": led.persistent_peak_bytes,
            "transient_kv_staging_peak_bytes": led.transient_peak_bytes,
            "cold_file_payload_current_bytes": led.file_payload_current_bytes,
            "cold_file_payload_peak_bytes": led.file_payload_peak_bytes,
            "bytes_read_total": led.bytes_read_total,
            "bytes_written_total": led.bytes_written_total,
            "move_time_total_ms": round(led.move_time_total_ms, 4),
        }

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"len={self.length}"
        return (
            f"DiskTieredKVCache(layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, block={self.block_size}, {state})"
        )
