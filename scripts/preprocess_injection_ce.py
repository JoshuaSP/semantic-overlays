"""Build CE training tensors from composed injection items.

Run with the tokenizer venv:
  .venv-tok/bin/python scripts/preprocess_injection_ce.py [--verify-server]

Input:  data/injectgen/composed/items.jsonl  (compose_injection_items.py)
Output: data/training/inject_{train,heldout}.npz + _meta.json

Objective (2026-08-11), HYBRID — the objective differs per arm because
the two arms want opposite things:

  benign   -> KL against the SAME frozen model with goggles DISABLED on the
              SAME input. Minimum is at EXACTLY zero change, which is what
              "the mark must be inert here" means. Measured need: goggled and
              ungoggled text agreed only 5% of the time on clean inputs
              (median similarity 0.24), so locality is not free.
  injected -> CE toward the counterfactual-clean completion. CE puts all mass
              on the target token — including a hard EOS — whereas KL only asks
              the student to be as willing to stop as the teacher was. Measured
              need: KL-for-both scored SEP 79.6% against CE-for-both's 93.5%,
              and the residual failures were trailing footnotes appended past
              where the answer should have ended.

Earlier framing, kept for the record:

  benign   : teacher = goggles-off on the SAME clean passage  -> minimum at
             exactly zero change. True locality; the metric now means what its
             name says.
  injected : teacher = goggles-off on the CLEAN passage       -> "produce the
             distribution you would have produced had the injection not been
             there", as a distribution rather than one archived greedy string.

CE toward a stored sample was the earlier objective and it misled us: on clean
inputs the goggled and ungoggled models agree on text only 5% of the time
(median similarity 0.24) while both answers are correct, so CE charged full
price for writing a different good answer. KL against the live teacher has no
such penalty.

The stored target is still tokenized — it is the teacher-forcing PATH along
which the two distributions are compared — but it is no longer the label.

  prefix   = chat_template([system, user], add_generation_prompt=True)
  tprefix  = same, but always over the CLEAN passage (== prefix for benign)
  comp     = tokenize(target) + [<|im_end|>]

The trailing <|im_end|> is REQUIRED and scored: without it the model never
learns to stop (the stop-token lesson — an earlier project run scored 8.3%
exact / 98% overrun from exactly this omission). We verify that
prefix_text + target + "<|im_end|>" is byte-identical to what the chat template
produces for the full [system, user, assistant] conversation, so the training
sequence is exactly an in-distribution templated turn.

Goggle mask = tokens whose character range overlaps mask_char_span, which the
compositor recorded as offsets into the USER MESSAGE CONTENT. Offsets are
rebased onto the templated text here.

Split is by SQuAD article title (not random): passages from one article never
straddle train/heldout, so heldout measures generalization to unseen source
documents rather than memorized passages.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "infra"))
import config  # noqa: E402

ITEMS = REPO / "data" / "injectgen" / "composed" / "items.jsonl"
OUT_DIR = REPO / "data" / "training"
HELDOUT_FRAC = 0.1

tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
EOS_ID = tok.convert_tokens_to_ids("<|im_end|>")


def build_sequence(messages: list[dict], target: str):
    """Templated prefix + completion ids ending in <|im_end|>.

    Returns (prefix_text, prefix_ids, offsets, comp_ids).
    """
    prefix_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING)
    enc = tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True)
    comp_ids = tok(target, add_special_tokens=False)["input_ids"] + [EOS_ID]

    # The assembled sequence must be exactly the templated full conversation.
    full = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": target}],
        tokenize=False, enable_thinking=config.ENABLE_THINKING)
    assembled = prefix_text + target + "<|im_end|>"
    if not full.startswith(assembled):
        raise RuntimeError(
            "assembled sequence is not a prefix of the templated conversation; "
            f"template drift.\n assembled tail: {assembled[-80:]!r}\n"
            f" template tail: {full[:len(assembled)][-80:]!r}")
    return prefix_text, enc["input_ids"], enc["offset_mapping"], comp_ids


def span_mask(prefix_text, user_content, char_span, ids, offsets, expected_text):
    """Rebase mask_char_span (offsets into user content) onto templated text.

    The chat template strips surrounding whitespace from message content, so the
    raw content is not a literal substring of the templated text; locate the
    stripped form and shift by the leading whitespace it removed.
    """
    stripped = user_content.strip()
    user_start = prefix_text.rfind(stripped)
    if user_start < 0:
        raise RuntimeError("user message content not found in templated text")
    content_end = user_start + len(stripped)
    lead = len(user_content) - len(user_content.lstrip())
    lo = user_start + char_span[0] - lead
    # Clip to the stripped content: a span reaching the end of the message loses
    # the trailing whitespace the template removed (frames such as
    # fake_delimiter_v1 end in a newline), so it must not run into template
    # tokens. Any other mismatch is a real misalignment and still raises.
    hi = min(user_start + char_span[1] - lead, content_end)
    got = prefix_text[lo:hi]
    if got != expected_text and got != expected_text.rstrip():
        raise RuntimeError(
            f"mask span does not cover the intended text.\n"
            f"  expected: {expected_text[:60]!r}...{expected_text[-40:]!r}\n"
            f"  got:      {got[:60]!r}...{got[-40:]!r}")
    mask = np.zeros(len(ids), dtype=bool)
    for i, (a, b) in enumerate(offsets):
        if a < hi and b > lo:
            mask[i] = True
    if not mask.any():
        raise RuntimeError(f"mask span {char_span} mapped to zero tokens")
    return mask


def verify_against_server(samples):
    import requests
    import os
    base = os.environ["GOGGLES_VLLM_URL"]  # your deployed modal_vllm endpoint
    for messages, ids, item_id in samples:
        r = requests.post(f"{base}/tokenize", json={
            "model": config.MODEL_ID, "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING},
        }, headers={"Authorization": f"Bearer {config.VLLM_API_KEY}"}, timeout=900)
        r.raise_for_status()
        server_ids = r.json()["tokens"]
        if server_ids != list(ids):
            diff = next((i for i, (a, b) in enumerate(zip(ids, server_ids)) if a != b), None)
            raise RuntimeError(f"template mismatch on {item_id}: local {len(ids)} vs "
                               f"server {len(server_ids)} ids, first diff at {diff}")
        print(f"  template verified vs server: {item_id} ({len(ids)} prefix tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-server", action="store_true",
                    help="assert local chat template matches the serving /tokenize")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(ITEMS) if l.strip()]
    # Teacher prefix for every item is the CLEAN-passage rendering, which is
    # exactly the benign twin's messages.
    clean_msgs = {it["unit_id"]: it["messages"] for it in items if it["kind"] == "benign"}
    missing = [it["item_id"] for it in items if it["unit_id"] not in clean_msgs]
    if missing:
        raise RuntimeError(f"{len(missing)} items have no benign twin, e.g. {missing[:3]}")
    titles = sorted({it["title"] for it in items})
    rng = np.random.default_rng(0)
    heldout_titles = set(rng.permutation(titles)[: max(1, int(len(titles) * HELDOUT_FRAC))])
    print(f"{len(items)} items, {len(titles)} articles, "
          f"{len(heldout_titles)} articles held out")

    splits = {"train": [], "heldout": []}
    for it in items:
        splits["heldout" if it["title"] in heldout_titles else "train"].append(it)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify_samples = []
    for split, recs in splits.items():
        prefix_ids, masks, comp_ids_all = [], [], []
        tprefix_ids, is_benign = [], []
        prefix_off, comp_off, tprefix_off = [0], [0], [0]
        meta = []
        for it in recs:
            ptext, ids, offsets, comp = build_sequence(it["messages"], it["target"])
            _, tids, _, _ = build_sequence(clean_msgs[it["unit_id"]], it["target"])
            if comp[-1] != EOS_ID:
                raise RuntimeError(f"{it['item_id']}: completion does not end in <|im_end|>")
            m = span_mask(ptext, it["messages"][1]["content"], it["mask_char_span"],
                          ids, offsets, it["masked_text"])
            prefix_ids.append(np.asarray(ids, dtype=np.int32))
            tprefix_ids.append(np.asarray(tids, dtype=np.int32))
            # Objective is per-arm (hybrid): benign -> KL vs goggles-off,
            # injected -> CE toward the counterfactual-clean completion.
            is_benign.append(it["kind"] == "benign")
            masks.append(m)
            comp_ids_all.append(np.asarray(comp, dtype=np.int32))
            prefix_off.append(prefix_off[-1] + len(ids))
            tprefix_off.append(tprefix_off[-1] + len(tids))
            comp_off.append(comp_off[-1] + len(comp))
            meta.append({"item_id": it["item_id"], "kind": it["kind"],
                         "frame_id": it["frame_id"], "title": it["title"],
                         "position": it["position"],
                         "base_falls_for_probe": it["base_falls_for_probe"],
                         "mask_tokens": int(m.sum()), "comp_tokens": len(comp)})
            if len(verify_samples) < 3 and split == "train":
                verify_samples.append((it["messages"], ids, it["item_id"]))
        np.savez_compressed(
            OUT_DIR / f"inject_{split}.npz",
            prefix_ids=np.concatenate(prefix_ids),
            prefix_off=np.asarray(prefix_off, np.int64),
            tprefix_ids=np.concatenate(tprefix_ids),
            tprefix_off=np.asarray(tprefix_off, np.int64),
            is_benign=np.asarray(is_benign, dtype=bool),
            mask=np.concatenate(masks),
            comp_ids=np.concatenate(comp_ids_all),
            comp_off=np.asarray(comp_off, np.int64),
        )
        (OUT_DIR / f"inject_{split}_meta.json").write_text(json.dumps(meta))
        mt = sorted(m["mask_tokens"] for m in meta)
        kinds = {}
        for m in meta:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        print(f"{split}: {len(recs)} items {kinds}, "
              f"{sum(len(c) for c in comp_ids_all):,} target tokens, "
              f"mask tokens min {mt[0]} median {mt[len(mt)//2]} max {mt[-1]}")

    if args.verify_server:
        verify_against_server(verify_samples)


if __name__ == "__main__":
    main()
