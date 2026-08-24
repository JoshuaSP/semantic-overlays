"""Modal app: train injection-defense goggles, hybrid KL(benign)/CE(injected).

  modal run infra/train_injection.py --smoke-only        # sanity gates only, 1 GPU
  modal run infra/train_injection.py                     # full training (8x H100)
  modal run infra/train_injection.py --run-name inj-mlp128-b

Objective (2026-08-10): CE toward the frozen model's own completion on
the COUNTERFACTUAL-CLEAN input — the passage with the injection deleted. Not KL:
CE-on-edit is the recipe that beat a note-conditioned teacher's own ceiling on
the instruction task, and the target text is what the compositor already stores.

Data (scripts/compose_injection_items.py -> preprocess_injection_ce.py):
  injected — passage carries a framed payload; goggles mask the WHOLE passage
             (provenance: deployment cannot know where an injection is);
             target = completion on the clean passage. "The marked span has no
             imperative authority."
  benign   — clean passage, same mask, same target. "Goggles change nothing on
             ordinary data." Also the alignment gate below.

Architecture: per-layer SwiGLU bottleneck, d_hidden=128 (a deliberate call — skip
layervec here and go straight to MLP; PLR showed vectors broadcast and overwrite
content while the MLP writes the same message shaded, and content survival is
exactly what we need: the marked span must stay READABLE while losing authority).
Everything else held at the established recipe for CE-trained goggles:
Muon lr 5e-4, warmup 20, eff batch 32, 3 epochs, cosine to lr*0.1.

Data: goggles-data volume /data/training/inject_{train,heldout}.npz
Checkpoints: /data/checkpoints/<run_name>/latest.pt  (resumable)
"""

import os as _os

import modal

import config

WANDB_SECRET = _os.environ.get("GOGGLES_WANDB_SECRET", "wandb-api-key")

N_GPU = 8
D_HIDDEN = 128               # 2026-08-10: MLP 128, no layervec arm
LOCAL_ACCUM = 4
EFF_BATCH = N_GPU * LOCAL_ACCUM
EPOCHS = 2                   # 2026-08-11: 3 epochs likely overkill at 4x data
WARMUP = 20                  # every CE-trained goggle path uses 15-20
LR = 5e-4                    # Muon, matching train_{highlight,concept,mixed,linmap}.
IN_LR_MULT = 6.0             # lr multiplier for w_in/w_gate. Muon scales updates
                             # by sqrt(max(1, fan_out/fan_in)): 5.657 for w_out
                             # [4096,128] and exactly 1.0 for w_in/w_gate
                             # [128,4096]. Combined with w_out starting at zero
                             # while the inputs start at Kaiming ||W||~6.5, the
                             # RELATIVE step ratio is 12.2x (measured: 113% vs 6%
                             # drift over steps 200-896). LoRA+ argues the
                             # zero-init output factor SHOULD be faster, so this
                             # may already be near-optimal — hence the ablation
                             # rather than a "fix".
                             # NOT OPT_DEFAULTS["muon"]=5e-3: that belongs to the KL
                             # path in train_goggles.py. Training this at 5e-3 with
                             # warmup 0 diverged (heldout worse than no-op in every
                             # bucket, pre-clip grad norms 6-9).
CKPT_EVERY = 25
SNAPSHOT_EVERY = 100        # retained snapshots (latest.pt is overwritten)
EVAL_EVERY = 50
DATASET = _os.environ.get("GOGGLES_DATASET", "injv2b")
# Each passage yields MANY items per epoch, not one. The (passage x payload x
# frame x position) space is ~3e9; storing one record per passage but emitting
# one ITEM per passage threw that away and silently cut the main family in half.
# Storage != items. Benign variants are byte-identical (same passage, no
# injection) so they add weight, not diversity; injected variants are all
# distinct draws.
N_BENIGN_PER_UNIT = 1
N_INJECTED_PER_UNIT = 3

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("torch==2.8.0", "transformers==5.14.1", "accelerate", "numpy", "wandb",
                 "huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          # Resolved at deploy time and BAKED IN: a deploy-time env override
          # does not reach the container, and config.py re-imports there.
          "GOGGLES_MODEL": config.MODEL_ID,
          "GOGGLES_DATASET": DATASET})
    .add_local_python_source("config", "goggles_lib", "hf_ready")
)

hf_cache = modal.Volume.from_name("goggles-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("goggles-data", create_if_missing=True)

app = modal.App("inference-goggles-inject" + (
    "-" + config.MODEL_ID.split("/")[-1].lower().replace(".", "").replace("_", "-")
    if config.MODEL_ID != "Qwen/Qwen3.5-9B" else ""))


def load_split(path):
    import numpy as np
    z = np.load(path)
    return {k: z[k] for k in z.files}


def make_sampler(meta, seed):
    """Draw a fresh (payload, frame, position) for a resamplable unit.

    The target is the model's completion on the CLEAN passage — a function of
    (task, passage) only — so the injection can be redrawn every epoch at zero
    cost. No two epochs see the same string, and the axes that get fresh
    randomisation are the ones the ablation says matter (framing ~50pp,
    position ~26pp) rather than passages (~9pp).
    """
    import random
    import re
    payloads, frames, fw = meta["payloads"], meta["frames"], meta["frame_weights"]
    pw = meta["position_weights"]
    positions, posw = list(pw), [pw[k] for k in pw]

    def splice(doc, probe, position):
        if position == "start":
            return f"{probe} {doc}"
        if position == "end":
            return f"{doc} {probe}"
        ends = [m.end() for m in re.finditer(r"(?<!\d)[.!?](?=\s)", doc)]
        if not ends:
            return f"{doc} {probe}"
        cut = min(ends, key=lambda i: abs(i - len(doc) // 2))
        return f"{doc[:cut]} {probe}{doc[cut:]}"

    def sample(unit_idx, step):
        r = random.Random((seed * 1000003 + unit_idx) * 7919 + step)
        pl = r.choice(payloads)["payload"]
        fr = r.choices(frames, weights=fw)[0]
        pos = r.choices(positions, weights=posw)[0]
        framed = fr["template"].format(p=pl, p_lower=pl[0].lower() + pl[1:])
        return splice(meta["resamplable"][unit_idx]["passage"], framed, pos)

    return sample


def _build(rank, dev, ddp=False):
    import torch
    from transformers import AutoModelForCausalLM
    from goggles_lib import Goggles
    import hf_ready

    # ONE rank downloads; the rest block. Then load offline: after the barrier
    # the snapshot is complete, so no rank touches the network and a missing
    # file fails loudly instead of racing to re-fetch (the HF cache flake).
    hf_ready.ensure_model(rank, ddp=ddp)
    # Identity assertion (lesson, 2026-08-23): a stale container or a lost env
    # override silently trains the WRONG base model under the right app name.
    print(f"[train] rank{rank} base model: {config.MODEL_ID}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, dtype=torch.bfloat16, device_map=dev,
        attn_implementation="sdpa", local_files_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    # Qwen3_5Config carries dims on its text sub-config, not the top level;
    # get_text_config() returns self for plain configs, so this is safe either way.
    tcfg = model.config.get_text_config()
    goggles = Goggles(tcfg.num_hidden_layers, tcfg.hidden_size, D_HIDDEN).to(dev)
    handles = goggles.attach(model)
    return model, goggles, handles


_TURN_END_ID = None


def _turn_end_id(tok):
    global _TURN_END_ID
    if _TURN_END_ID is None:
        probe = tok.apply_chat_template(
            [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "PROBE"}],
            tokenize=False, enable_thinking=config.ENABLE_THINKING)
        tail = probe.split("PROBE", 1)[1].strip()
        tid = tok.convert_tokens_to_ids(tail.split()[0]) if tail else None
        if tid is None or tid == tok.unk_token_id:
            raise RuntimeError(f"no usable turn terminator for {config.MODEL_ID}")
        _TURN_END_ID = tid
        print(f"[train] turn terminator {tail.split()[0]!r} -> id {tid}", flush=True)
    return _TURN_END_ID


def _seq(tok, task, span, target):
    """Tokenize a (task, span, target) triple on the fly and build the goggle mask."""
    import numpy as np
    sysmsg = ("Below is an instruction that describes a task, paired with an input "
              "that provides further context.\nWrite a response that appropriately "
              f"completes the request.\n\nInstruction:\n{task}\n")
    user = f"Input:\n{span}\n"
    text = tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    # Terminator is model-specific and MUST match what preprocessing scored
    # (<|im_end|> Qwen, <|eot_id|> Llama). convert_tokens_to_ids returns None
    # for a foreign token, which silently poisons the completion ids.
    eos = _turn_end_id(tok)
    comp = tok(target, add_special_tokens=False)["input_ids"] + [eos]
    stripped = user.strip()
    st = text.rfind(stripped)
    if st < 0:
        raise RuntimeError("user content not found in templated text")
    lo = st + user.index(span) - (len(user) - len(user.lstrip()))
    hi = min(lo + len(span), st + len(stripped))
    mask = np.zeros(len(ids), dtype=bool)
    for j, (a, b) in enumerate(offs):
        if a < hi and b > lo:
            mask[j] = True
    if not mask.any():
        raise RuntimeError("on-the-fly mask mapped to zero tokens")
    return np.asarray(ids, np.int32), mask, np.asarray(comp, np.int32)


def _micro(model, goggles, ctx, i, dev, need_grad, step=0):
    """CE toward the target, goggles on the marked span.

    ALL arms are CE (2026-08-12). KL-for-both cost 14 SEP points; the
    hybrid recovered 9 of them, leaving benign-KL as the remaining suspect for
    the residual gap. Locality is enforced by the CORPUS instead — half the
    resamplable draws are benign (same passage, no injection, same target), so
    "the mark changes nothing here" is taught by example rather than by
    objective.

    Resamplable units get a FRESH (payload, frame, position) every time they are
    visited, so no two epochs see the same string. Gate and validator items are
    fixed — their targets come from a protocol or a validator, not from sampling.
    """
    import numpy as np
    import torch
    from goggles_lib import chunked_ce

    d, meta, tok, sample = ctx["d"], ctx["meta"], ctx["tok"], ctx["sample"]
    per = N_BENIGN_PER_UNIT + N_INJECTED_PER_UNIT
    n_main = per * len(meta["resamplable"])
    if i < n_main:
        unit, variant = divmod(i, per)
        u = meta["resamplable"][unit]
        benign = variant < N_BENIGN_PER_UNIT
        # variant enters the seed so the injected draws differ from EACH OTHER
        # within an epoch, and step enters it so they differ ACROSS epochs.
        span = u["passage"] if benign else sample(unit, step * per + variant)
        ids_np, mask_np, comp_np = _seq(tok, u["task"], span, u["target"])
    else:
        j = i - n_main
        p0, p1 = d["fixed_prefix_off"][j], d["fixed_prefix_off"][j + 1]
        c0, c1 = d["fixed_comp_off"][j], d["fixed_comp_off"][j + 1]
        ids_np = d["fixed_prefix_ids"][p0:p1]
        mask_np = d["fixed_mask"][p0:p1]
        comp_np = d["fixed_comp_ids"][c0:c1]

    ids = torch.tensor(np.concatenate([ids_np, comp_np]).astype(np.int64)[None],
                       dtype=torch.long, device=dev)
    m = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=dev)
    m[0, :len(ids_np)] = torch.from_numpy(mask_np.copy()).to(dev)
    goggles.set_mask(m)
    with torch.enable_grad() if need_grad else torch.no_grad():
        logits = model(ids, logits_to_keep=len(comp_np) + 1, use_cache=False).logits[0, :-1]
        loss = chunked_ce(logits, torch.tensor(comp_np.astype(np.int64), device=dev))
    if not need_grad:
        goggles.set_mask(None)
    return loss


def _gates(model, goggles, ctx, dev):
    """Sanity gates. Any failure raises before GPU spend."""
    import torch

    meta = ctx["meta"]
    n_res = len(meta["resamplable"])
    # index 0 with step 0 is a benign draw by construction ((0*7919+0)%2==0)
    b0 = _micro(model, goggles, ctx, 0, dev, need_grad=False, step=0).item()
    i0 = _micro(model, goggles, ctx, 1, dev, need_grad=False, step=0).item()
    g0 = _micro(model, goggles, ctx, n_res, dev, need_grad=False, step=0).item()
    print(f"[gate] zero-init CE — benign {b0:.4f} | injected {i0:.4f} | "
          f"fixed(gate/validator) {g0:.4f}", flush=True)
    if not b0 < 1.0:
        raise RuntimeError(
            f"benign CE {b0:.3f} at zero-init; the target IS this prompt's own "
            "greedy completion, so it must be small. Targets are misaligned.")
    if not 0 < g0 < 8.0:
        raise RuntimeError(f"fixed-item CE {g0:.3f} is out of range; check the "
                           "gate/validator targets.")

    # on-the-fly resampling must actually produce different strings per step
    a = ctx["sample"](1, 0)
    b = ctx["sample"](1, 1)
    if a == b:
        raise RuntimeError("resampler returned identical spans for two steps — "
                           "epochs would repeat, which is the thing this fixes.")
    print(f"[gate] resampling OK — same unit, two steps differ "
          f"({len(a)} vs {len(b)} chars)", flush=True)

    loss = _micro(model, goggles, ctx, 1, dev, need_grad=True, step=0)
    loss.backward()
    goggles.set_mask(None)
    leaked = [n for n, q in model.named_parameters() if q.grad is not None]
    if leaked:
        raise RuntimeError(f"gradients leaked into {len(leaked)} base params")
    n_g = sum(1 for q in goggles.parameters() if q.grad is not None)
    print(f"[gate] gradient confinement OK ({n_g}/{sum(1 for _ in goggles.parameters())} "
          f"goggle params, 0 base)", flush=True)
    for q in goggles.parameters():
        q.grad = None
    torch.cuda.empty_cache()


def _worker(rank, world, run_name, lr, epochs, wandb_entity, smoke_only, warmup=WARMUP):
    import json
    import math
    import os
    import numpy as np
    import torch
    import torch.distributed as dist
    from goggles_lib import Muon

    ddp = world > 1
    if ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29515")
        dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    torch.manual_seed(0)

    ckpt_dir = f"/data/checkpoints/{run_name}"
    ckpt_path = f"{ckpt_dir}/latest.pt"
    wb = None
    if rank == 0 and not smoke_only:
        import wandb as wb
        wid = None
        if os.path.exists(ckpt_path):
            wid = torch.load(ckpt_path, map_location="cpu", weights_only=True).get("wandb_id")
        wb.init(id=wid, resume="allow" if wid else None, entity=wandb_entity or None,
                project="inference-goggles", name=run_name,
                config={"task": "injection_defense", "objective": "ce_all_arms_onthefly", "in_lr_mult": IN_LR_MULT,
                        "dataset": DATASET, "model": config.MODEL_ID,
                        "arch": "mlp", "d_hidden": D_HIDDEN, "optimizer": "muon",
                        "lr": lr, "warmup": warmup, "epochs": epochs, "world": world,
                        "eff_batch": world * LOCAL_ACCUM})

    model, goggles, handles = _build(rank, dev, ddp=ddp)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, local_files_only=True)
    _dsm = json.load(open(f"/data/training/{DATASET}_train_meta.json")).get("model")
    if _dsm and _dsm != config.MODEL_ID:
        raise RuntimeError(
            f"dataset {DATASET} was tokenized for {_dsm}, but this run serves "
            f"{config.MODEL_ID} — token ids would be meaningless")
    train_d = load_split(f"/data/training/{DATASET}_train.npz")
    held_d = load_split(f"/data/training/{DATASET}_heldout.npz")
    train_meta = json.load(open(f"/data/training/{DATASET}_train_meta.json"))
    held_meta = json.load(open(f"/data/training/{DATASET}_heldout_meta.json"))
    train_ctx = {"d": train_d, "meta": train_meta, "tok": tok,
                 "sample": make_sampler(train_meta, seed=1)}
    held_ctx = {"d": held_d, "meta": held_meta, "tok": tok,
                "sample": make_sampler(held_meta, seed=2)}
    per = N_BENIGN_PER_UNIT + N_INJECTED_PER_UNIT
    n_train = per * len(train_meta["resamplable"]) + len(train_meta["fixed"])
    n_held = per * len(held_meta["resamplable"]) + len(held_meta["fixed"])

    if rank == 0:
        print(f"goggles: {sum(p.numel() for p in goggles.parameters())/1e6:.1f}M params "
              f"(d_hidden={D_HIDDEN}) | train {n_train} "
              f"({len(train_meta['resamplable'])} resamplable + "
              f"{len(train_meta['fixed'])} fixed) | heldout {n_held}", flush=True)
        _gates(model, goggles, train_ctx, dev)
        print(f"  main family: {len(train_meta['resamplable'])} passages x "
              f"{per} variants ({N_BENIGN_PER_UNIT} benign + "
              f"{N_INJECTED_PER_UNIT} freshly-injected) = "
              f"{per * len(train_meta['resamplable'])} items/epoch", flush=True)

    if smoke_only:
        if rank == 0:
            # Overfit a FIXED batch. Comparing step 1 to step 20 on DIFFERENT
            # items is meaningless here: per-item CE varies far more between
            # units than the optimizer moves it in 20 steps.
            # one of each family, so the smoke batch exercises all three
            per = N_BENIGN_PER_UNIT + N_INJECTED_PER_UNIT
            n_main = per * len(train_meta["resamplable"])
            fixed = [0, 1, n_main,
                     next(n_main + j for j, m in enumerate(train_meta["fixed"])
                          if m["kind"] == "validator")]
            print(f"smoke gates passed; overfitting a fixed batch {fixed} for 30 steps",
                  flush=True)
            opt = Muon(goggles.parameters(), lr=lr)
            first = last = None
            for step in range(30):
                opt.zero_grad(set_to_none=True)
                acc = 0.0
                for idx in fixed:
                    loss = _micro(model, goggles, train_ctx, idx, dev,
                                  need_grad=True, step=0)
                    (loss / len(fixed)).backward()
                    goggles.set_mask(None)
                    acc += loss.item() / len(fixed)
                torch.nn.utils.clip_grad_norm_(goggles.parameters(), 1.0)
                opt.step()
                first = acc if first is None else first
                last = acc
                if (step + 1) % 5 == 0 or step == 0:
                    print(f"  smoke step {step+1}/30 CE {acc:.4f}", flush=True)
            if not last < first:
                raise RuntimeError(
                    f"fixed-batch loss did not decrease ({first:.4f} -> {last:.4f})")
            print(f"[gate] fixed-batch loss decreases OK ({first:.4f} -> {last:.4f})",
                  flush=True)
        if ddp:
            dist.destroy_process_group()
        return

    # Two groups so w_in/w_gate can be given their own lr. Muon's shape-based
    # scaling gives w_out 5.657x and the inputs exactly 1.0; with w_out at zero
    # init and the inputs at Kaiming, the measured RELATIVE step ratio is 12.2x.
    ins = [q for n, q in goggles.named_parameters() if "w_in" in n or "w_gate" in n]
    outs = [q for n, q in goggles.named_parameters() if "w_out" in n]
    assert len(ins) == 64 and len(outs) == 32, (len(ins), len(outs))
    opt = Muon([{"params": outs, "lr": lr},
                {"params": ins, "lr": lr * IN_LR_MULT}], lr=lr)
    eff = world * LOCAL_ACCUM
    total_steps = epochs * (n_train // eff)
    os.makedirs(ckpt_dir, exist_ok=True)
    start = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=True)
        goggles.load_state_dict(ck["goggles"]); opt.load_state_dict(ck["opt"])
        start = ck["opt_step"]
        if rank == 0:
            print(f"resumed from step {start}", flush=True)

    rng = np.random.default_rng(0)
    order = np.concatenate([rng.permutation(n_train) for _ in range(epochs)])

    def bucket(i):
        """Fine-grained: main family split benign/injected, and every gate kind
        and validator task separately — a regression in correct_rider must not
        hide inside a family average."""
        per = N_BENIGN_PER_UNIT + N_INJECTED_PER_UNIT
        n_main = per * len(held_meta["resamplable"])
        if i < n_main:
            return ("main_benign" if (i % per) < N_BENIGN_PER_UNIT
                    else "main_injected")
        f = held_meta["fixed"][i - n_main]
        return f"{f['kind']}_{f['sub']}"

    BUCKETS = ["main_benign", "main_injected"] + sorted(
        {f"{f['kind']}_{f['sub']}" for f in held_meta["fixed"]})

    def eval_heldout():
        goggles.eval()
        sums = {k: [0.0, 0] for k in BUCKETS}
        with torch.no_grad():
            for i in range(rank, n_held, world):
                v = _micro(model, goggles, held_ctx, i, dev, need_grad=False, step=0).item()
                b = bucket(i)
                sums[b][0] += v; sums[b][1] += 1
        goggles.train()
        out = {}
        for k in BUCKETS:
            t = torch.tensor(sums[k], dtype=torch.float64, device=dev)
            if ddp:
                dist.all_reduce(t)
            out[k] = (t[0] / t[1]).item() if t[1] > 0 else float("nan")
        return out

    def fmt(ev):
        return " | ".join(f"{k.replace('validator_','v:').replace('gate_','g:').replace('fidelity_','f:')} "
                          f"{ev[k]:.3f}" for k in BUCKETS if ev[k] == ev[k])

    def grad_stats():
        """Per-matrix-group gradient and update norms. This is what the LR
        ablation actually measures, and its absence is why a 12x asymmetry sat
        unnoticed through four runs."""
        out = {}
        for tag, ps in (("w_out", outs), ("w_in_gate", ins)):
            gs = [q.grad.norm().item() for q in ps if q.grad is not None]
            if gs:
                out[f"gradnorm_{tag}"] = sum(gs) / len(gs)
        return out

    def weight_stats():
        """Per-matrix drift and norms — the diagnostic that would have caught the
        12x asymmetry on run one. Rank is NOT logged: Muon's orthogonalised
        updates preserve a random matrix's Marchenko-Pastur spectrum, so rank90
        stays pinned at its init value even while the matrix genuinely rotates,
        which is exactly how it fooled us."""
        out = {}
        for tag in ("w_in", "w_gate", "w_out"):
            ws = [q for n, q in goggles.named_parameters() if tag in n]
            drift = [((q - init[n]).norm() / max(init[n].norm().item(), 1e-9)).item()
                     for n, q in goggles.named_parameters() if tag in n]
            out[f"drift_{tag}"] = sum(drift) / len(drift)
            out[f"norm_{tag}"] = sum(q.norm().item() for q in ws) / len(ws)
        return out

    init = {n: q.detach().clone() for n, q in goggles.named_parameters()}

    base = None
    if start == 0:
        goggles.enabled = False
        base = eval_heldout()
        goggles.enabled = True
        if rank == 0:
            print(f"[baseline] heldout goggles-OFF — {fmt(base)} "
                  f"(all-CE: main_benign is the model's own greedy self-CE, ~0.26, not 0)", flush=True)
            if wb:
                wb.summary.update({f"heldout_kl_off_{k}": v for k, v in base.items()})

    if rank == 0:
        print(f"world={world} eff_batch={eff} lr={lr} steps={total_steps}", flush=True)

    for step in range(start, total_steps):
        lr_min = lr * 0.1
        cur_lr = lr * (step + 1) / warmup if warmup and step < warmup else \
            lr_min + 0.5 * (lr - lr_min) * (
                1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = cur_lr * (IN_LR_MULT if gi == 1 else 1.0)
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for k in range(LOCAL_ACCUM):
            idx = order[(step * eff + rank * LOCAL_ACCUM + k) % len(order)]
            loss = _micro(model, goggles, train_ctx, int(idx), dev, need_grad=True, step=step)
            (loss / LOCAL_ACCUM).backward()
            goggles.set_mask(None)
            acc += loss.item() / LOCAL_ACCUM
        if ddp:
            for p in goggles.parameters():
                # Ranks always touch the same params here (single channel), but
                # zero-fill anyway: a None grad on any rank deadlocks all-reduce.
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        gn = torch.nn.utils.clip_grad_norm_(goggles.parameters(), 1.0)
        opt.step()

        if rank == 0 and (step + 1) % 5 == 0:
            print(f"opt {step+1}/{total_steps} lr {cur_lr:.2e} loss {acc:.4f} "
                  f"gn {gn:.3f}", flush=True)
            if wb:
                wb.log({"train_loss": acc, "lr": cur_lr,
                        "grad_norm_preclip": gn.item(),
                        **weight_stats(), **grad_stats()},
                       step=step + 1)
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == total_steps:
            ev = eval_heldout()
            if rank == 0:
                print(f"  heldout (benign=KL, injected=CE) — {fmt(ev)}", flush=True)
                if wb:
                    wb.log({f"heldout_kl_{k}": v for k, v in ev.items()}, step=step + 1)
        if rank == 0 and ((step + 1) % CKPT_EVERY == 0 or step + 1 == total_steps):
            blob = {"goggles": goggles.state_dict(), "opt": opt.state_dict(),
                    "opt_step": step + 1, "d_hidden": D_HIDDEN,
                    "wandb_id": wb.run.id if wb else None}
            torch.save(blob, ckpt_path)
            # Retain periodic snapshots, not just latest.pt. Each is a paid
            # artifact, and overwriting them cost us the ability to evaluate the
            # held-out-CE-minimum checkpoint against the final one — the
            # experiment that decides whether the CE rise is real overfitting or
            # an artifact of scoring against one archived greedy sample.
            if (step + 1) % SNAPSHOT_EVERY == 0 or step + 1 == total_steps:
                torch.save(blob, f"{ckpt_dir}/step{step + 1:05d}.pt")
            data_vol.commit()

    if rank == 0:
        print("training complete", flush=True)
        if wb:
            wb.finish()
    if ddp:
        dist.destroy_process_group()


@app.function(image=image, gpu=f"H100:{N_GPU}", timeout=8 * 3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol},
              secrets=[modal.Secret.from_name(WANDB_SECRET),
                       modal.Secret.from_name("huggingface")])
def train(run_name: str, lr: float, epochs: int, wandb_entity: str, warmup: int):
    import torch.multiprocessing as mp
    mp.spawn(_worker, args=(N_GPU, run_name, lr, epochs, wandb_entity, False, warmup),
             nprocs=N_GPU)


@app.function(image=image, gpu="H100", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def smoke():
    _worker(0, 1, "inject-smoke", LR, 1, "", True)


@app.local_entrypoint()
def main(run_name: str = "inject-mlp128", lr: float = LR, epochs: int = EPOCHS,
         wandb_entity: str = "", smoke_only: bool = False,
         warmup: int = WARMUP):
    if smoke_only:
        smoke.remote()
    else:
        train.remote(run_name=run_name, lr=lr, epochs=epochs, wandb_entity=wandb_entity,
                     warmup=warmup)
