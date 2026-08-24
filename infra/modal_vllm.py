"""Modal app serving Qwen3.5-9B via vLLM's OpenAI-compatible API.

Deploy:   modal deploy infra/modal_vllm.py
Endpoint: printed on deploy (https://<workspace>--inference-goggles-vllm-serve.modal.run)
Auth:     Authorization: Bearer <config.VLLM_API_KEY>

Shared between the baseline eval harness and the training side's teacher
rollouts. Model id / GPU / vLLM version all come from infra/config.py.
"""

import subprocess

import modal

import config

# CUDA devel base (nvcc present, needed for flashinfer's JIT) — same image
# pattern as the other sandbox Modal apps (see ../vlmplay/serve.py).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        f"vllm=={config.VLLM_VERSION}",
        "huggingface_hub[hf_xet]",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    # Bake the RESOLVED model id into the container env: config.py re-imports
    # inside the container, and a deploy-time GOGGLES_MODEL override would
    # otherwise vanish there — the container would silently serve the default
    # model under this app's name. Must precede add_local_* (build-step rule).
    .env({"GOGGLES_MODEL": config.MODEL_ID})
    .add_local_python_source("config")
)

import os as _os
# Dedicated cache by default: the serve container must never share a write
# path with training (concurrent snapshot_download corrupts the volume).
hf_cache = modal.Volume.from_name(
    _os.environ.get("GOGGLES_HF_CACHE", "goggles-hf-serve"), create_if_missing=True)
vllm_cache = modal.Volume.from_name("goggles-vllm-cache", create_if_missing=True)

app = modal.App(config.APP_NAME)


@app.function(
    image=image,
    gpu=config.GPU,
    timeout=60 * 60,
    scaledown_window=10 * 60,
    max_containers=int(_os.environ.get("GOGGLES_VLLM_CONTAINERS", "1")),
    # gated models (Llama replication track) need the HF token
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=config.VLLM_PORT, startup_timeout=20 * 60)
def serve():
    import os
    from huggingface_hub import snapshot_download
    # Fill any snapshot gaps ONLINE first (a no-op when complete — but offline
    # mode's completeness check demands even .gitattributes/LICENSE/README,
    # which vLLM's own downloader skips), THEN go cache-only so scale-out
    # containers never race hub downloads or flake on hub revalidation.
    snapshot_download(config.MODEL_ID)
    os.environ["HF_HUB_OFFLINE"] = "1"
    cmd = [
        "vllm",
        "serve",
        config.MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(config.VLLM_PORT),
        "--max-model-len", str(config.MAX_MODEL_LEN),
        "--max-logprobs", str(config.MAX_LOGPROBS),
        "--api-key", config.VLLM_API_KEY,
    ]
    if "qwen" in config.MODEL_ID.lower():
        # think-token parser exists only in Qwen tokenizers; on other
        # families this flag crashes the server at startup
        cmd += ["--reasoning-parser", "qwen3"]
    subprocess.Popen(cmd)
