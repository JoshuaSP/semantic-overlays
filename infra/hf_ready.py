"""Serialize the model download: ONE process fetches, every other rank blocks.

ROOT CAUSE of the recurring HF cache flake (RESULTS.md logs 5 failures over two
days, and it recurred on the training volume on 2026-08-11):

    mp.spawn starts N ranks in one container, all sharing the SAME
    goggles-hf-cache volume, and every rank calls
    AutoModelForCausalLM.from_pretrained() at the same instant. huggingface_hub
    is not safe against N concurrent writers materialising the same snapshot:
    the ranks race on the blob/symlink layout and can leave an INCOMPLETE shard
    set behind. The corruption is silent at write time and only surfaces later,
    on some subsequent run, as

        OSError: Qwen/Qwen3.5-9B does not appear to have a file named
                 model.safetensors-00002-of-00004.safetensors

    which reads like a missing upstream file but is really our own race.

FIX (two parts, both required):

  1. Exactly one process downloads. Rank 0 calls snapshot_download(); all other
     ranks wait on a barrier until it returns. No concurrent writers, ever.
  2. Every rank then loads with local_files_only=True. After the barrier the
     snapshot is complete, so nothing needs the network — and if anything IS
     missing, the run fails loudly at load time instead of silently racing to
     re-fetch and corrupting the cache again for the next job.

Two barrier flavours, because our jobs differ:
  - torch.distributed is already initialised (training)  -> dist.barrier()
  - plain mp.spawn with no process group (eval)          -> filesystem sentinel,
    which is sufficient because mp.spawn ranks share one container filesystem.
"""

import os
import time

import config

SENTINEL = "/tmp/.hf_snapshot_ready"
TIMEOUT_S = 40 * 60


def ensure_model(rank: int, ddp: bool = False, timeout_s: int = TIMEOUT_S) -> str:
    """Download on rank 0 only; other ranks block. Returns the snapshot path."""
    from huggingface_hub import snapshot_download

    if rank == 0:
        path = snapshot_download(config.MODEL_ID, max_workers=8)
        # Publish readiness for the no-process-group case. Written last, so its
        # existence implies the download returned.
        with open(SENTINEL, "w") as f:
            f.write(path)
        _barrier(rank, ddp)
        return path

    if ddp:
        _barrier(rank, ddp)
        from huggingface_hub import snapshot_download as sd
        return sd(config.MODEL_ID, local_files_only=True)

    t0 = time.time()
    while not os.path.exists(SENTINEL):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(
                f"rank {rank}: timed out after {timeout_s}s waiting for rank 0 to "
                f"finish downloading {config.MODEL_ID}")
        time.sleep(2)
    return open(SENTINEL).read().strip()


def _barrier(rank: int, ddp: bool) -> None:
    if not ddp:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier(device_ids=[rank])
