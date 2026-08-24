"""Build training tensors with ON-THE-FLY injection resampling.

Run with the tokenizer venv:
  .venv-tok/bin/python scripts/preprocess_injection_v2.py

WHY THIS EXISTS. v1 froze one arbitrary (payload, frame, position) draw per
passage into the npz and replayed those exact tuples every epoch. The
combinatorial space is ~8k passages x 243 payloads x 56 frames x 3 positions
(~3e9); we sampled 8,000 points from it and then showed the model the same 8,000
again. Worse, scaling 2k -> 8k units multiplied PASSAGES only — the pools stayed
fixed, so each payload recurred ~25x instead of ~8x and 32% of items repeated a
(payload, frame) combination. Our own ablation says passages are the weakest
axis (~9pp) and framing the strongest (~50pp), so we quadrupled the thing that
matters least.

THE KEY PROPERTY THAT MAKES RESAMPLING FREE: the training target does not depend
on the injection. For an injected item the target is the model's completion on
the CLEAN passage, a function of (task, passage) only; the teacher prefix is the
clean rendering. So payload/frame/position can be redrawn at train time with no
new API calls — we already have every clean completion we need.

So this stores the RAW pieces (passage, task, clean target) plus the pools, and
the training loop splices and tokenizes per item per epoch. No two epochs see
the same string.

Output: data/training/injv2_{train,heldout}.npz
  Ragged arrays of TOKENIZED CLEAN prefixes + targets (shared across all draws),
  and a JSON sidecar with the raw text needed to re-splice on the fly.

The gate and validator families are FIXED items (their targets are read off a
protocol or a validator, not sampled), so they are stored pre-rendered and are
not resampled. Only the main family varies per epoch.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "infra"))
import config  # noqa: E402

COMPOSED = REPO / "data" / "injectgen" / "composed"
import os as _os
_GM = _os.environ.get("GOGGLES_MODEL", "Qwen/Qwen3.5-9B")
_MODEL_TAG = "" if _GM == "Qwen/Qwen3.5-9B" else "-" + _GM.split("/")[-1].lower()
SHARED = COMPOSED  # model-agnostic inputs (frames.json, raw corpora)
if _MODEL_TAG:
    COMPOSED = COMPOSED.parent / ("composed" + _MODEL_TAG)
    print(f"[replica] corpus: {COMPOSED}", flush=True)
OUT_DIR = REPO / "data" / "training"
HELDOUT_FRAC = 0.1

tok = AutoTokenizer.from_pretrained(config.MODEL_ID)


def _turn_end() -> str:
    """Assistant-turn terminator, read off the chat template.

    Model-specific (<|im_end|> on Qwen, <|eot_id|> on Llama). Hardcoding it
    is silently catastrophic on another family: convert_tokens_to_ids returns
    the UNKNOWN-token id, and training would score that as the stop token.
    """
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": "PROBE"}],
        tokenize=False, enable_thinking=config.ENABLE_THINKING)
    tail = probe.split("PROBE", 1)[1].strip()
    if not tail:
        raise RuntimeError("chat template emits no assistant-turn terminator")
    return tail.split()[0]


TURN_END = _turn_end()
EOS = tok.convert_tokens_to_ids(TURN_END)
if EOS is None or EOS == tok.unk_token_id:
    raise RuntimeError(f"terminator {TURN_END!r} is not a real token id")
print(f"[preprocess] turn terminator {TURN_END!r} -> id {EOS}", flush=True)
TEMPLATE_USER = "Input:\n{}\n"


def templated(messages, target):
    """Prefix ids + offsets + completion ids, with the stop token scored."""
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=config.ENABLE_THINKING)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    comp = tok(target, add_special_tokens=False)["input_ids"] + [EOS]
    full = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": target}],
        tokenize=False, enable_thinking=config.ENABLE_THINKING)
    if not full.startswith(text + target + TURN_END):
        raise RuntimeError("template drift")
    return text, enc["input_ids"], enc["offset_mapping"], comp


def span_mask(prefix_text, user_content, char_span, ids, offsets, expected):
    stripped = user_content.strip()
    start = prefix_text.rfind(stripped)
    if start < 0:
        raise RuntimeError("user content not found in templated text")
    end = start + len(stripped)
    lead = len(user_content) - len(user_content.lstrip())
    lo = start + char_span[0] - lead
    hi = min(start + char_span[1] - lead, end)
    got = prefix_text[lo:hi]
    if got != expected and got != expected.rstrip():
        raise RuntimeError(f"mask span mismatch: {got[:60]!r} vs {expected[:60]!r}")
    m = np.zeros(len(ids), dtype=bool)
    for i, (a, b) in enumerate(offsets):
        if a < hi and b > lo:
            m[i] = True
    if not m.any():
        raise RuntimeError("mask mapped to zero tokens")
    return m


def main():
    # --name writes a DIFFERENT npz so earlier datasets stay reproducible.
    # "injv2" is what every arm through 6x/per=1 trained on: gate 1200,
    # validator 1500, no fidelity family. Do not overwrite it.
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--name", default="injv2")
    _name = _ap.parse_args().name
    main_items = [json.loads(l) for l in open(COMPOSED / "items.jsonl") if l.strip()]
    gate = [json.loads(l) for l in open(COMPOSED / "gate_items.jsonl") if l.strip()]
    val = [json.loads(l) for l in open(COMPOSED / "validator_items.jsonl") if l.strip()]
    # Fidelity family: reproduce the marked span verbatim. Added because the
    # mix had nothing that ever rewarded READING the span accurately -- every
    # other family rewards not obeying it -- and exact verbatim copy measured
    # 80.8% goggled vs 99.2% ungoggled. Fixed items: the target is the span
    # itself, so there is nothing to resample.
    fid_path = COMPOSED / "fidelity_items.jsonl"
    fid = ([json.loads(l) for l in open(fid_path) if l.strip()]
           if fid_path.exists() else [])

    # One RESAMPLABLE unit per (passage, task): keep the benign twin's rendering
    # (the clean prefix, which is also the teacher) and the shared clean target.
    units = {}
    for it in main_items:
        if it["kind"] != "benign":
            continue
        units[it["unit_id"]] = {
            "unit_id": it["unit_id"], "title": it["title"], "task": it["task"],
            "passage": it["masked_text"], "target": it["target"],
        }
    print(f"{len(units)} resamplable units | {len(gate)} gate | {len(val)} validator "
          f"| {len(fid)} fidelity")

    payloads = json.loads((COMPOSED / "payloads_screened.json").read_text())
    frames = [f for f in json.loads((SHARED / "frames.json").read_text())
              if not f.get("reference_only")]
    rates = json.loads((COMPOSED / "frame_ranking_metrics.json").read_text())["frames"]
    fw = [rates[f["id"]]["rate"] + 0.15 for f in frames]

    titles = sorted({u["title"] for u in units.values()})
    rng = np.random.default_rng(0)
    held = set(rng.permutation(titles)[: max(1, int(len(titles) * HELDOUT_FRAC))])

    splits = {"train": {"units": [], "fixed": []}, "heldout": {"units": [], "fixed": []}}
    for u in units.values():
        splits["heldout" if u["title"] in held else "train"]["units"].append(u)
    # STRATIFIED heldout for the fixed families: sample per (kind, sub-kind) so
    # every cell has a readable count without growing the heldout set. A uniform
    # 10% draw left `noise` at 8 items, too few to read a per-kind metric from.
    import collections
    r = random.Random(0)
    cells = collections.defaultdict(list)
    for it in gate + val + fid:
        cells[(it["kind"], it.get("gate_kind") or it.get("task_name"))].append(it)
    for key, group in sorted(cells.items()):
        r.shuffle(group)
        n_h = max(20, int(len(group) * HELDOUT_FRAC))
        splits["heldout"]["fixed"].extend(group[:n_h])
        splits["train"]["fixed"].extend(group[n_h:])
    r.shuffle(splits["train"]["fixed"]); r.shuffle(splits["heldout"]["fixed"])
    print("heldout per cell:", {f"{k[0]}/{k[1]}": max(20, int(len(v)*HELDOUT_FRAC))
                                for k, v in sorted(cells.items())})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, d in splits.items():
        # --- resamplable units: tokenize the CLEAN prefix + shared target once ---
        pids, poff, cids, coff, meta = [], [0], [], [0], []
        for u in d["units"]:
            msgs = [{"role": "system", "content": config_system(u["task"])},
                    {"role": "user", "content": TEMPLATE_USER.format(u["passage"])}]
            text, ids, offs, comp = templated(msgs, u["target"])
            m = span_mask(text, msgs[1]["content"], [7, 7 + len(u["passage"])],
                          ids, offs, u["passage"])
            pids.append(np.asarray(ids, np.int32)); cids.append(np.asarray(comp, np.int32))
            poff.append(poff[-1] + len(ids)); coff.append(coff[-1] + len(comp))
            meta.append({"unit_id": u["unit_id"], "title": u["title"],
                         "kind": "resamplable", "task": u["task"],
                         "passage": u["passage"], "target": u["target"],
                         "mask_tokens": int(m.sum())})
        # --- fixed items (gate + validator): pre-rendered, never resampled ---
        fpids, fpoff, fcids, fcoff, fmask, fmeta = [], [0], [], [0], [], []
        for it in d["fixed"]:
            text, ids, offs, comp = templated(it["messages"], it["target"])
            m = span_mask(text, it["messages"][1]["content"], it["mask_char_span"],
                          ids, offs, it["masked_text"])
            fpids.append(np.asarray(ids, np.int32)); fcids.append(np.asarray(comp, np.int32))
            fmask.append(m)
            fpoff.append(fpoff[-1] + len(ids)); fcoff.append(fcoff[-1] + len(comp))
            fmeta.append({"item_id": it["item_id"], "kind": it["kind"],
                          "sub": it.get("gate_kind") or it.get("task_name"),
                          "mask_tokens": int(m.sum())})
        np.savez_compressed(
            OUT_DIR / f"{_name}_{split}.npz",
            clean_prefix_ids=np.concatenate(pids) if pids else np.zeros(0, np.int32),
            clean_prefix_off=np.asarray(poff, np.int64),
            clean_comp_ids=np.concatenate(cids) if cids else np.zeros(0, np.int32),
            clean_comp_off=np.asarray(coff, np.int64),
            fixed_prefix_ids=np.concatenate(fpids), fixed_prefix_off=np.asarray(fpoff, np.int64),
            fixed_mask=np.concatenate(fmask),
            fixed_comp_ids=np.concatenate(fcids), fixed_comp_off=np.asarray(fcoff, np.int64),
        )
        (OUT_DIR / f"{_name}_{split}_meta.json").write_text(json.dumps({
            # stamped so training can refuse a dataset tokenized for another
            # model: token ids from a different tokenizer are meaningless
            "model": config.MODEL_ID, "turn_end": TURN_END,
            "resamplable": meta, "fixed": fmeta,
            "payloads": payloads, "frames": frames, "frame_weights": fw,
            "position_weights": {"end": 0.6, "middle": 0.25, "start": 0.15},
        }))
        print(f"{split}: {len(meta)} resamplable + {len(fmeta)} fixed "
              f"({sum(len(c) for c in cids) + sum(len(c) for c in fcids):,} target tokens)")


def config_system(task):
    return ("Below is an instruction that describes a task, paired with an input "
            "that provides further context.\nWrite a response that appropriately "
            f"completes the request.\n\nInstruction:\n{task}\n")


if __name__ == "__main__":
    main()
