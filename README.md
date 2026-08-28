# Semantic Overlays — injection-defense reproduction

Minimal code to reproduce the prompt-injection results of **Semantic
Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and
Steering Vectors** ([arXiv:2608.23873](https://arxiv.org/abs/2608.23873)):
corpus construction, adapter training on two frozen base models
(Qwen3.5-9B and Llama-3.1-8B-Instruct), serving, and the
SEP / TensorTrust / PIArena evaluations.

**Related releases**

| | |
|---|---|
| paper | [arXiv:2608.23873](https://arxiv.org/abs/2608.23873) |
| interactive demo | [semantic-overlays.vercel.app](https://semantic-overlays.vercel.app) |
| training corpus | [semantic-overlays-injection](https://huggingface.co/datasets/joshuapenman/semantic-overlays-injection) |
| trained adapters | [semantic-overlays-adapters](https://huggingface.co/joshuapenman/semantic-overlays-adapters) |
| Quadrat-IPI harness | [quadrat-model-eval](https://github.com/JoshuaSP/quadrat-model-eval) |

The companion dataset (composed training corpus, frame bank, payload
screening records, and frame rankings for both base models) is on
Hugging Face as `semantic-overlays-injection`. Place its contents at
`data/injectgen/` relative to this repo's root.

## Layout

```
infra/            Modal apps: serving, training, goggled evaluation
infra/goggles_plugin/   vLLM plugin: applies overlays at marked prefill positions
evals/            frozen-model baseline harnesses (SEP, TensorTrust, PIArena)
scripts/          tokenizing, judging, scoring (run locally)
scripts/corpus/   how the released corpus was built -- not needed to reproduce
web/              the interactive demo (Next.js; see web/README.md)
```

Everything GPU-bound runs on [Modal](https://modal.com) (H100s); `scripts/`
runs locally. Set `GOGGLES_VLLM_API_KEY` to a bearer token of your choice
before deploying. To target Llama instead of the default Qwen3.5-9B, export
`GOGGLES_MODEL=meta-llama/Llama-3.1-8B-Instruct` -- every app, output path, and
volume namespace derives from it (`infra/config.py`).

## Reproducing the paper

Four steps. You do not need to build a corpus: the released one is on Hugging
Face, already screened and ranked against both base models.

1. **Get the corpus.** The dataset is laid out by base model; the tokenizer
   reads one model's files from `composed/`, so pick a model as you unpack.

   ```bash
   huggingface-cli download joshuapenman/semantic-overlays-injection \
     --repo-type dataset --local-dir data/hf-corpus

   mkdir -p data/injectgen/composed data/injectgen/short_span
   cp data/hf-corpus/qwen3.5-9b/*.json*   data/injectgen/composed/
   cp -r data/hf-corpus/shared/*          data/injectgen/composed/
   cp data/hf-corpus/qwen3.5-9b/short_span/* data/injectgen/short_span/
   ```

   For Llama, substitute `llama-3.1-8b-instruct/` and drop the `short_span`
   line and the `--short-span` flag below: that overlay was trained before the
   short-span family existed.

2. **Tokenize it.** This writes the `.npz` the trainer reads, stamped with the
   model id so a dataset tokenized for one base model cannot silently be used
   with another.

   ```bash
   python scripts/preprocess_injection_v2.py --name injv2b-ss --short-span
   modal volume put goggles-data data/training/injv2b-ss_train.npz /training/
   modal volume put goggles-data data/training/injv2b-ss_heldout.npz /training/
   modal volume put goggles-data data/training/injv2b-ss_train_meta.json /training/
   modal volume put goggles-data data/training/injv2b-ss_heldout_meta.json /training/
   ```

3. **Train** (about 5 h on 8xH100 for the reported checkpoint):

   ```bash
   modal run --detach infra/train_injv2b_ss.py --run-name injv2b-ss-6x-per4
   ```

   Or skip training entirely: every checkpoint the paper reports is released as
   `semantic-overlays-adapters` on Hugging Face.

4. **Evaluate.** `--arm off` reproduces the frozen rows with the same harness,
   so both arms of every table come from one command each.

   ```bash
   modal run infra/eval_sep_goggled.py       --ckpt injv2b-ss-6x-per4 --arm on --full
   modal run infra/eval_tensortrust_goggled.py --ckpt injv2b-ss-6x-per4 --arm on --benchmark hijacking
   modal run infra/eval_tensortrust_goggled.py --ckpt injv2b-ss-6x-per4 --arm on --benchmark extraction
   modal run infra/eval_piarena_goggled.py   --ckpt injv2b-ss-6x-per4 --arm on
   modal run infra/eval_fidelity.py          --ckpt injv2b-ss-6x-per4 --arm on --n-items 500
   ```

## Building a corpus from scratch (optional)

Only needed to derive a corpus for a **new base model**. Two of these steps
measure against the model itself, so they cannot be inherited: payload
screening keeps only payloads the frozen model answers standalone (witness
metrics are unsound otherwise), and frame ranking measures each frame's
standalone injection rate, which sets sampling shares. For the two models in
the paper, both records ship with the released corpus.

```bash
modal deploy infra/modal_vllm.py                    # serve the frozen model
uv run scripts/corpus/screen_payloads.py --base-url <endpoint>/v1
uv run scripts/corpus/rank_frames.py     --base-url <endpoint>/v1
uv run scripts/corpus/compose_injection_items.py --base-url <endpoint>/v1
uv run scripts/corpus/gen_gate_items.py      --n-units 4800 --unique
uv run scripts/corpus/gen_validator_items.py --n-units 6000
uv run scripts/corpus/gen_fidelity_items.py  --n 1920
uv run scripts/corpus/compose_short_span_items.py --url <endpoint>/generate
```

`--unique` on the gate family is not optional: without it, 4,800 requested
items yield about 4,166 distinct (span, target) pairs, and the duplicates are
not extra signal.

   two judged PIArena families (two-step, evidence-grounded);
   `scripts/score_sep_boundary.py` applies the corrected SEP grading
   rule described in the paper; `scripts/judge_sep.py` is the SEP
   compliance judge. `infra/eval_tensortrust_aside165.py` replicates
   ASIDE's 165-row TensorTrust harness for the published comparison.
8. **Serve with overlays** (interactive):
   `modal deploy infra/goggled_vllm.py` — vLLM itself is
   unmodified (stock 0.21.0 from PyPI) --- `infra/goggles_plugin/` is a
   pip package registered under vLLM's `general_plugins` entry point,
   which applies the trained adapters at marked prefill positions;
   unmarked requests reproduce the frozen model exactly.

## Notes for reproduction

- The Llama checkpoint was trained on an earlier data mix than the Qwen
  one, so its numbers are a lower bound on the recipe rather than a
  matched comparison.
- Eval outputs checkpoint per item and resume; killing and relaunching
  any eval is safe.
- All decoding is greedy at temperature 0 unless a script says
  otherwise; seeded-sampling variants used for the reproducibility
  check are flags on the SEP eval (`--temperature 0.7 --seed N`).
