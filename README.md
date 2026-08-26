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
infra/    Modal apps: serving, training, goggled evaluation
infra/goggles_plugin/  vLLM plugin: applies overlays at marked prefill positions
evals/    frozen-model baseline harnesses (SEP, TensorTrust, PIArena)
scripts/  corpus construction, judging, and scoring (run locally)
web/      the interactive demo (Next.js; see web/README.md)
```

Trained adapter checkpoints for every overlay set the demo serves —
the do-not-execute overlay on both base models, the twelve visual
marks, the four asserted languages, and the twelve carried
instructions — are released as `semantic-overlays-adapters` on
Hugging Face. With those, the demo and every evaluation run without
any training.

Everything GPU-bound runs on [Modal](https://modal.com) (H100s); the
scripts/ directory runs locally against a deployed endpoint. Set
`GOGGLES_VLLM_API_KEY` to a bearer token of your choice before
deploying. To target Llama instead of the default Qwen3.5-9B, export
`GOGGLES_MODEL=meta-llama/Llama-3.1-8B-Instruct` — every app, output
path, and volume namespace derives from it (`infra/config.py`).

## Pipeline

1. **Serve the frozen model** (baselines + corpus derivation need it):
   `modal deploy infra/modal_vllm.py`
2. **Screen payloads** against the frozen model (keeps only payloads it
   answers standalone; witness metrics are unsound otherwise):
   `uv run scripts/screen_payloads.py --base-url <endpoint>/v1`
3. **Rank frames** (per-model standalone injection rate per frame;
   sampling shares derive from this):
   `uv run scripts/rank_frames.py --base-url <endpoint>/v1`
4. **Compose and tokenize the corpus**:
   `uv run scripts/compose_injection_items.py`, then
   `uv run scripts/preprocess_injection_ce.py` (Qwen) or
   `uv run scripts/preprocess_injection_v2.py` (Llama). The assistant
   turn's terminator is read from the chat template — it is
   model-specific, and hardcoding it silently breaks both training and
   generation stopping on the other model.
5. **Train** (≈4 h on 8×H100 for the reported checkpoints):
   `modal run infra/train_injv2b_6x_per4.py::train` (Qwen) or
   `GOGGLES_MODEL=... modal run infra/train_llama_inj.py::train`
6. **Evaluate the marked arm** (`--arm off` reproduces the frozen rows
   with the same harness):
   `modal run infra/eval_sep_goggled.py --ckpt <name> --arm on --n-items 1000`
   `modal run infra/eval_tensortrust_goggled.py --ckpt <name> --arm on --benchmark hijacking`
   (and `extraction`), `modal run infra/eval_piarena_goggled.py --ckpt <name> --arm on`,
   `modal run infra/eval_fidelity.py --ckpt <name>` for the copy-rate.
7. **Baselines and scoring**: `evals/{sep,tensortrust,piarena}.py` run
   the frozen/prompt-defense arms; `scripts/judge_piarena.py` scores the
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

- Corpus derivation is per base model by design: payload screening and
  frame ranking (steps 2–3) must be re-run against any new base model.
  Between the paper's two models a quarter of the kept payload sets are
  disjoint and the frame rankings correlate at only 0.49.
- Eval outputs checkpoint per item and resume; killing and relaunching
  any eval is safe.
- All decoding is greedy at temperature 0 unless a script says
  otherwise; seeded-sampling variants used for the reproducibility
  check are flags on the SEP eval (`--temperature 0.7 --seed N`).
