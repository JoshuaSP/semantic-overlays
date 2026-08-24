"""vLLM general plugin: request-scoped goggle masks inside the V1 EngineCore.

Why a plugin: AsyncLLM always runs EngineCore in a spawned subprocess
(vllm 0.21.0's EngineCoreClient.make_client raises NotImplementedError for
asyncio without multiprocessing), so a monkeypatch made in the API process
never reaches the model. This package is pip-installed in the serving image
and registered under the `vllm.general_plugins` entry-point group, which vLLM
loads in EVERY process — including the spawned EngineCore (core.py calls
load_general_plugins() in EngineCore.__init__) — so the patch below runs in
the process that actually owns the GPU model runner.

Mask transport: a request carries
    SamplingParams.extra_args = {"goggles": {"adapter": <name>,
                                             "bits": [int, ...]}}
which rides the msgspec RPC into the worker and is visible there on
CachedRequestState.sampling_params. bits[i] is the quality bitmask for
absolute prompt position i; positions >= len(bits) (decode tokens) get 0 —
goggles are prefill-only. Chunked prefill and preemption-resume are handled
by construction because we index bits by the runner's own absolute positions.

Adapter weights load lazily in the worker on the first goggled batch, from
the JSON spec in $GOGGLES_ADAPTER_SPEC:
    {name: {"path": <ckpt>, "n_layers": int, "d_model": int}}
The checkpoint decides the module: an "arch" meta key -> OverlapGoggles
(bitmask semantics); otherwise the binary injection-track Goggles (bit 0
only). Unknown adapter names, a missing spec, and the phrase arch (needs
frozen base-model phrase embeddings this plugin does not compute) all raise —
a goggled request must never silently run ungoggled.
"""

import json
import os
import sys

_state = {
    "patched": False,
    "hooked": False,
    "slots": [],       # adapter modules; index = slot
    "slot_of": {},     # adapter name -> slot
    "resolved": {},    # req_id -> (slot, np bits) | None
    "current": {"aidx": None, "bits": None},   # flat per-token, set each step
}


class _BinaryShim:
    """Injection-track Goggles (one adapter per layer, boolean mask) behind
    the OverlapGoggles channel API: n_qualities=1, bit 0 = marked."""

    def __init__(self, g):
        self.g = g
        self.n_qualities = 1

    def _channel(self, layer_idx, q, h):
        return self.g.adapters[layer_idx](h)


def _load_adapter(name, device):
    if name in _state["slot_of"]:
        return _state["slot_of"][name]
    import torch
    raw = os.environ.get("GOGGLES_ADAPTER_SPEC")
    if not raw:
        raise RuntimeError("goggled request arrived but GOGGLES_ADAPTER_SPEC "
                           "is unset in this process")
    spec = json.loads(raw)
    if name not in spec:
        raise KeyError(f"unknown goggle adapter {name!r}; spec has "
                       f"{sorted(spec)}")
    s = spec[name]
    libdir = os.environ.get("GOGGLES_LIB_DIR", "/root")
    if libdir not in sys.path:
        sys.path.insert(0, libdir)
    ck = torch.load(s["path"], map_location="cpu", weights_only=True)
    sd = ck["goggles"] if "goggles" in ck else ck
    if "arch" in ck:
        from goggles_lib import OverlapGoggles
        mod = OverlapGoggles(s["n_layers"], s["d_model"], ck["n_qualities"],
                             ck["arch"], d_hidden=ck["d_hidden"],
                             d_code=ck["d_code"])
        mod.load_state_dict(sd)
        mod.to(device)
        if ck["arch"] == "phrase":
            # The phrase arm conditions on frozen embeddings of the quality
            # phrases, held as a NON-PERSISTENT buffer — absent from the
            # checkpoint by design. They are deterministic, so they are
            # precomputed on the training stack (infra/xfer_phrase.py) and
            # shipped beside the weights; the serving image cannot load the
            # HF model to recompute them. Missing file = hard failure, never
            # a silent forward with a zero identity.
            pp = s.get("phrases")
            if not pp or not os.path.exists(pp):
                raise RuntimeError(
                    f"phrase arm {name!r} needs precomputed phrase embeddings; "
                    f"{pp!r} missing (run infra/xfer_phrase.py::phrases)")
            E = torch.load(pp, map_location="cpu", weights_only=True)["phrases"]
            if tuple(E.shape) != (ck["n_qualities"], s["d_model"]):
                raise RuntimeError(f"phrase embeddings {tuple(E.shape)} != "
                                   f"({ck['n_qualities']}, {s['d_model']})")
            mod.set_phrases(E.to(device))
            print(f"[goggles-plugin] phrase identities set from {pp}",
                  flush=True)
    else:
        from goggles_lib import Goggles
        dh = sd["adapters.0.w_gate.weight"].shape[0]
        g = Goggles(s["n_layers"], s["d_model"], d_hidden=dh)
        g.load_state_dict(sd)
        g.to(device)
        mod = _BinaryShim(g)
    slot = len(_state["slots"])
    _state["slots"].append(mod)
    _state["slot_of"][name] = slot
    print(f"[goggles-plugin] loaded adapter {name!r} from {s['path']} "
          f"(slot {slot}, pid {os.getpid()})", flush=True)
    return slot


def _hook_for(layer_idx):
    import torch

    def pre_hook(module, args, kwargs):
        cur = _state["current"]
        aidx, bits = cur["aidx"], cur["bits"]
        if aidx is None:
            return None
        # vLLM splits the stream into (hidden_states, residual) with fused
        # add-norm; the true stream entering a layer is their sum (or
        # hidden_states alone at layer 0). Compute the adapter input from the
        # sum, add the delta to hidden_states — identical math to training.
        # Arg passing differs by architecture: Qwen's vLLM layers are called
        # with keywords, Llama's positionally (positions, hidden_states,
        # residual). Read from whichever carries them, and write back the same
        # way — a KeyError here means the adapter never fires (2026-08-24).
        hs_in_kwargs = "hidden_states" in kwargs
        if hs_in_kwargs:
            hs = kwargs["hidden_states"]
            res = kwargs.get("residual", None)
        else:
            if len(args) < 2:
                raise RuntimeError(
                    f"layer {layer_idx}: cannot locate hidden_states "
                    f"(kwargs={list(kwargs)}, {len(args)} positional args)")
            hs = args[1]
            res = args[2] if len(args) > 2 else kwargs.get("residual", None)
        n = min(hs.shape[0], aidx.shape[0])
        a, b = aidx[:n], bits[:n]
        m = a >= 0
        if not bool(m.any()):
            return None
        hs = hs.clone()
        h_full = hs[:n][m] + (res[:n][m] if res is not None else 0)
        h_full = h_full.to(torch.float32)
        mb, ma = b[m], a[m]
        delta = torch.zeros_like(h_full)
        for slot, gog in enumerate(_state["slots"]):
            s = ma == slot
            if not bool(s.any()):
                continue
            sb, sh = mb[s], h_full[s]
            d = torch.zeros_like(sh)
            for q in range(gog.n_qualities):
                qsel = ((sb >> q) & 1) == 1
                if bool(qsel.any()):
                    d[qsel] = d[qsel] + gog._channel(layer_idx, q, sh[qsel])
            delta[s] = d
        hs[:n][m] = hs[:n][m] + delta.to(hs.dtype)
        if hs_in_kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hs
        else:
            args = list(args)
            args[1] = hs
            args = tuple(args)
        return args, kwargs

    return pre_hook


def _ensure_hooks(runner):
    if _state["hooked"]:
        return
    # structural discovery: the wrapper class varies (ForCausalLM vs
    # ForConditionalGeneration), decoder layers don't.
    # Index source varies by architecture: Qwen's vLLM layers expose
    # .layer_idx, Llama's do not (they carry a `prefix` path instead), so fall
    # back to the trailing integer of the module's dotted name
    # ("model.layers.15" -> 15). Without the fallback the hook count is zero
    # and the adapter silently never fires (2026-08-24).
    found = []
    for name, m in runner.model.named_modules():
        if not m.__class__.__name__.endswith("DecoderLayer"):
            continue
        idx = getattr(m, "layer_idx", None)
        if idx is None:
            tail = name.rsplit(".", 1)[-1]
            idx = int(tail) if tail.isdigit() else None
        if idx is None:
            raise RuntimeError(
                f"decoder layer {name!r} has no resolvable index; refusing to "
                "install hooks in an unknown order")
        found.append((int(idx), m))
    if not found:
        raise RuntimeError("no decoder layers found on runner.model")
    for idx, layer in sorted(found, key=lambda t: t[0]):
        layer.register_forward_pre_hook(_hook_for(idx), with_kwargs=True)
    layers = found
    _state["hooked"] = True
    print(f"[goggles-plugin] hooks installed on {len(layers)} layers "
          f"(pid {os.getpid()})", flush=True)


def _install_patch():
    if _state["patched"]:
        return
    import numpy as np
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig = GPUModelRunner._prepare_inputs
    resolved = _state["resolved"]
    current = _state["current"]

    def patched(self, scheduler_output, num_scheduled_tokens, *a, **kw):
        out = orig(self, scheduler_output, num_scheduled_tokens, *a, **kw)
        _ensure_hooks(self)
        for rid in scheduler_output.finished_req_ids:
            resolved.pop(rid, None)
        num_reqs = self.input_batch.num_reqs
        nst = np.asarray(num_scheduled_tokens)
        req_indices = np.repeat(np.arange(num_reqs), nst)
        cu = np.cumsum(nst)
        qpos = np.arange(cu[-1]) - np.repeat(cu - nst, nst)
        positions = self.input_batch.num_computed_tokens_cpu[req_indices] + qpos
        total = int(cu[-1])
        aidx = np.full(total, -1, np.int8)
        bits = np.zeros(total, np.int64)
        any_goggled = False
        for r in range(num_reqs):
            rid = self.input_batch.req_ids[r]
            if rid not in resolved:
                sp = self.requests[rid].sampling_params
                g = ((sp.extra_args or {}).get("goggles")
                     if sp is not None else None)
                if g is None:
                    resolved[rid] = None
                else:
                    slot = _load_adapter(g["adapter"], self.device)
                    resolved[rid] = (slot, np.asarray(g["bits"], np.int64))
            ent = resolved[rid]
            if ent is None:
                continue
            slot, rbits = ent
            sel = req_indices == r
            pos = positions[sel]
            valid = pos < len(rbits)
            rb = np.zeros(int(sel.sum()), np.int64)
            rb[valid] = rbits[pos[valid]]
            if rb.any():
                bits[sel] = rb
                aidx[sel] = np.where(rb != 0, slot, -1)
                any_goggled = True
        if any_goggled:
            dev = self.device
            current["aidx"] = torch.from_numpy(aidx).to(dev, non_blocking=True)
            current["bits"] = torch.from_numpy(bits).to(dev, non_blocking=True)
        else:
            current["aidx"] = None
            current["bits"] = None
        return out

    GPUModelRunner._prepare_inputs = patched
    _state["patched"] = True
    print(f"[goggles-plugin] runner patch installed (pid {os.getpid()})",
          flush=True)


def register():
    """Entry point for the vllm.general_plugins group — called in every vLLM
    process, including the spawned EngineCore that owns the model. Patching
    the class is cheap; adapters load lazily and only in the process where a
    goggled batch actually reaches a GPUModelRunner."""
    _install_patch()
    print(f"[goggles-plugin] registered (pid {os.getpid()})", flush=True)
