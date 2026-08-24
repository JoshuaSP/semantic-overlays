"""Batched generation with prefill-only goggle masks (greedy by default).

Every eval harness here generated ONE sequence at a time — prefill, then a
Python loop decoding a single token per forward. An H100 holds 32+ concurrent
sequences at nearly the same per-step latency, so that costs roughly an order of
magnitude of throughput on every eval we run.

The reason it was written that way is the mask, and the mask is also why
batching needs care:

  - Prefixes have different lengths, so a batch must be LEFT-padded: the
    generation point is then the last column for every row, and the KV cache
    lines up.
  - Left padding means position_ids can NOT be inferred from arange - they must
    come from the attention mask, or every padded row is fed wrong positions and
    silently produces garbage that still looks like fluent text.
  - The goggle mask is per-row and must be shifted by each row's pad amount.
    Get this wrong and the adapter fires on the wrong tokens; nothing crashes,
    the numbers are just quietly meaningless.

Goggles are a PREFILL-ONLY intervention, so the mask is set for the prefill
forward and cleared before the decode loop — same contract as the unbatched
path, and the hook still refuses a mask whose length does not match.

Verified against the unbatched implementation on stored results; see
scripts/verify_batched_gen.py.

SAMPLING (added 2026-08-17 for the ASIDE temp-0.7 x 3-seed SEP protocol) is
OPT-IN and off by default. temperature=0.0 keeps the original argmax path
untouched -- no RNG is created, seeded, or consumed -- so every pre-existing
greedy invocation is byte-identical to before. temperature>0 REQUIRES an
explicit seed: an unseeded sampling run is unreproducible, and this harness
writes results that end up in a paper.

Two properties of seeded batched sampling worth knowing before quoting numbers:

  - Draws come from one per-call torch.Generator over the whole batch, so a
    given row's tokens depend on the batch it landed in. Resuming a crashed run
    (different done-set -> different batching and rank sharding) therefore does
    NOT reproduce the completions the uninterrupted run would have produced.
    Statistically equivalent, bitwise different. Seed + rank fixes the stream,
    not the row-to-draw assignment.
  - No top-k/top-p truncation is applied unless asked for. "temperature 0.7"
    in the protocol is read literally as pure temperature sampling; HF's
    generate() would instead silently fold in the model's generation_config
    defaults (Qwen ships top_p/top_k), which is a different distribution.
"""


def generate_batched(model, goggles, tok, rows, dev, max_new_tokens,
                     batch_size=16, eos_id=None, progress=None, on_batch=None,
                     temperature=0.0, seed=None, top_p=1.0, top_k=0,
                     return_meta=False):
    """rows: list of (prefix_ids, goggle_mask) numpy pairs. Returns completions.

    prefix_ids : 1-D int array, the templated prompt
    goggle_mask: 1-D bool array, same length, True where the adapter applies

    on_batch(start, comps) fires after EACH chunk with that chunk's completions
    and its offset into `rows`, so the caller can persist them immediately.

    Pass it. Without it nothing reaches disk until every row is generated: the
    first batched full-SEP run sat 15% in (~27k generations) with a zero-byte
    output file, so a crash would have discarded all of it. The cross-run resume
    logic was fine -- it just had nothing to resume from.

    temperature : 0.0 (default) = greedy argmax, the original code path.
                  >0 = sample from softmax(logits/temperature); needs `seed`.
    seed        : int, required when temperature>0. Used verbatim -- callers
                  running multiple ranks must offset it themselves so ranks do
                  not share a stream.
    top_p/top_k : optional truncation, default OFF (1.0 / 0) = untruncated.
    return_meta : also report per-row {finish_reason, completion_tokens}, so
                  callers can measure the 1024-cap truncation rate instead of
                  hardcoding finish_reason="stop". Changes the return type and
                  adds a third argument to on_batch; default False keeps the
                  original contract.
    """
    import numpy as np
    import torch

    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    sampling = temperature > 0
    if sampling and seed is None:
        raise ValueError(
            "temperature>0 requires an explicit seed: unseeded sampling is "
            "unreproducible and these results are reported in a paper")
    gen_rng = None
    if sampling:
        gen_rng = torch.Generator(device=dev)
        gen_rng.manual_seed(int(seed))

    def _next_token(logits):
        """logits: [B, vocab] at the current position -> [B] next token ids."""
        if not sampling:
            return logits.argmax(-1)          # untouched original path
        lg = logits.float() / temperature
        if top_k:
            kth = lg.topk(top_k, dim=-1).values[:, -1:]
            lg = lg.masked_fill(lg < kth, float("-inf"))
        if top_p < 1.0:
            srt, idx = lg.sort(dim=-1, descending=True)
            p = srt.softmax(-1)
            # keep the top-1 always: (cumsum - p) is 0 for the first column.
            keep_sorted = (p.cumsum(-1) - p) < top_p
            keep = torch.zeros_like(keep_sorted).scatter(1, idx, keep_sorted)
            lg = lg.masked_fill(~keep, float("-inf"))
        return torch.multinomial(lg.softmax(-1), 1, generator=gen_rng).squeeze(1)

    if eos_id is None:
        from goggles_lib import turn_end_id
        eos_id = turn_end_id(tok)
    pad_id = eos_id  # padding is masked out; the id itself never matters
    out = [None] * len(rows)

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        B = len(chunk)
        L = max(len(ids) for ids, _ in chunk)

        ids = torch.full((B, L), pad_id, dtype=torch.long, device=dev)
        att = torch.zeros((B, L), dtype=torch.long, device=dev)
        gog = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for r, (p, m) in enumerate(chunk):
            n = len(p)
            ids[r, L - n:] = torch.tensor(p.astype(np.int64), device=dev)
            att[r, L - n:] = 1
            gog[r, L - n:] = torch.from_numpy(m.copy()).to(dev)   # shifted by the pad

        # position_ids from the attention mask, NOT arange: with left padding the
        # first real token of a short row sits at column L-n, and arange would
        # tell the model it is at position L-n instead of 0.
        pos = (att.cumsum(-1) - 1).clamp(min=0)

        goggles.set_mask(gog)
        with torch.no_grad():
            # logits_to_keep=1: only the last position's logits are ever read,
            # but the default materialises [B, L, vocab]. With vocab=248320 that
            # is 27 GB at B=256 -- on its own enough to OOM an 80 GB H100 before
            # any KV cache. Keeping one position makes it 0.13 GB.
            o = model(ids, attention_mask=att, position_ids=pos, use_cache=True,
                      logits_to_keep=1)
        goggles.set_mask(None)          # prefill-only: never applies during decode

        past = o.past_key_values
        nxt = _next_token(o.logits[:, -1])                    # [B]
        gen = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        cur_pos = pos[:, -1]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                done |= (nxt == eos_id)
                if bool(done.all()):
                    break
                for r in range(B):
                    if not bool(done[r]):
                        gen[r].append(int(nxt[r]))
                att = torch.cat([att, (~done).long().unsqueeze(1)], dim=1)
                cur_pos = cur_pos + 1
                o = model(nxt.unsqueeze(1), attention_mask=att,
                          position_ids=cur_pos.unsqueeze(1),
                          past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = _next_token(o.logits[:, -1])

        chunk_out = [tok.decode(gen[r], skip_special_tokens=True) for r in range(B)]
        # A row that never emitted eos ran the loop out: its completion is
        # TRUNCATED at the cap, and a witness past the cutoff is invisible to
        # SEP scoring. Previously every record was stamped finish_reason="stop",
        # so the ~3.8% truncation artifact had to be re-derived by hand.
        metas = [{"finish_reason": "length" if len(gen[r]) >= max_new_tokens else "stop",
                  "completion_tokens": len(gen[r])} for r in range(B)]
        for r in range(B):
            out[start + r] = (chunk_out[r], metas[r]) if return_meta else chunk_out[r]
        if on_batch:
            on_batch(start, chunk_out, metas) if return_meta else on_batch(start, chunk_out)
        if progress:
            progress(min(start + B, len(rows)), len(rows))
    return out
