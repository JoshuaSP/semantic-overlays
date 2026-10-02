# Injection (`inject` / do-not-execute) corpus: generation prompts and derivation

Reconstructed 2026-10-01 in response to reviewers of the FLMSec workshop paper
(`paper/submissions/2026-08-flmsec/40_Semantic_Overlays_Mitigatin.pdf`), who said the
prompts that generated our data are not published.

Every prompt, model output and code excerpt in this directory was copied programmatically
from the Claude Code transcripts, the repo, or git objects by a staging script.
Nothing was retyped. `SHA256SUMS` covers every file. This README describes the files and
does not reproduce any frame template. Frame text lives only in
`release_data/shared/frames.json` and in the copied authoring outputs.

`generation_params.json` lists the model, agent id, session and UTC start and end of every authoring call.

Corpus script paths point to this repository's `scripts/corpus/`. Other paths (`data/...`, session transcripts) refer to the private development repository; `scripts/build_injection_items.py` (abandoned approach) and the two Quadrat judges are not included here, but their prompts are copied below.

Transcript references are `session:line_index`, where line_index is the 0-based line of
the session JSONL under `~/.claude/projects/-Users-joshuapenman-sandbox-inference-goggles/`.
The main session is `ee22aab6-3eed-426a-870a-0026c5bd22b6` (`ee22aab6` below). The
short-span session is `d28ac381-bd2e-4b12-8cf1-0648c6ff7df0`.

## What in the corpus is model-authored

The corpus has no per-item model authoring. Passages are SQuAD v2 contexts and trivia
payloads are TriviaQA `rc.nocontext` questions, both fetched by script. Every target is
the frozen base model's own greedy completion on the clean passage. Several fixed text
banks were still written by an LLM, either in one authoring pass or inline in code:

| component | where it lives | authored by | how |
|---|---|---|---|
| 50 attack frames | `release_data/shared/frames.json` | 6 Opus 5 subagents, ee22aab6, 2026-08-11 04:39–04:51 UTC | prompts in `frames/authoring_calls/` |
| 6 seed frames (`insistent_v1/v2`, `authority_v1/v2`, `bare`, `conversational_v1`) | same file | main-session model (claude-fable-5), ee22aab6:799, 2026-08-11 01:31 UTC | written straight into `frames.json` during the design dialogue. There was no dedicated prompt. The context is in `frames/main_session/01_design_and_seed_frames.jsonl`. |
| `sep_verbatim` | same file, `reference_only` | SEP benchmark's own prefix | used as the ranking reference bar and never composed into training |
| directive ("hijack") bank, 12 entries | `scripts/corpus/compose_injection_items.py` `DIRECTIVES` (lines in `payloads/directive_bank_location.json`) | claude-fable-5, inline in the Write at ee22aab6:802 | unchanged since then (checked) |
| transform task bank, 20 entries | same file, `TRANSFORM_TASKS` | claude-opus-5, ee22aab6:1168, then pruned to selective tasks only at ee22aab6:1276 after Joshua approved at :1267 | `tasks/` |
| gate, validator and fidelity pools (protocols, riders, attack payloads, filler, exfiltration and fake-system strings, copy instructions) | `scripts/corpus/gen_gate_items.py`, `gen_validator_items.py`, `gen_fidelity_items.py`. Symbol line ranges are in `fixed_families/code_locations.json`. | claude-opus-5, ee22aab6:3717 / :3845 / :5841 (2026-08-12 to 13) | inline in code. Gate injections also draw on `frames.json`. Joshua's design instructions are in `fixed_families/gate_family_dialogue.jsonl`. |
| short-span topic tasks | `data/injectgen/composed/topic_tasks.json` (copied to `fixed_families/`; the authoring transcript is withheld because it contained a credential) | claude-opus-5, d28ac381:9993 (one task replaced at :10014), 2026-08-27 | Bash heredoc. Added after submission. |

### Frames (the 56 composable templates)

**Authoring calls** (`frames/authoring_calls/NN_*.{prompt.txt,final_output.txt,meta.json,tool_calls.jsonl}`).
Each was a `general-purpose` subagent spawned with `model: opus` and observed as
`claude-opus-5`, in session ee22aab6 on 2026-08-11:

| # | description | styles | frames returned |
|---|---|---|---|
| 01 | Author prototype attack frames | one each of insistent, authority, fake_delimiter, persona_swap, task_complete, politeness, urgency, metadata_spoof, reasoning_bait, nested_quote | 10 (`*_v1`, `insistent_v3`, `authority_v3`) |
| 02 | Author frames: insistent authority | insistent, authority | 8 (`*_10`–`*_13`) |
| 03 | Author frames: delimiter metadata | fake_delimiter, metadata_spoof | 8 |
| 04 | Author frames: persona task_complete | persona_swap, task_complete | 8 |
| 05 | Author frames: politeness urgency | politeness, urgency | 8 |
| 06 | Author frames: reasoning_bait nested_quote | reasoning_bait, nested_quote | 8 |

The style list is the ten families in call 01's prompt. Adding `bare` and
`conversational` from the seed set gives the paper's twelve styles. Every prompt bans
SEP's verbatim prefix and requires exactly one `{p}`/`{p_lower}` placeholder.

**Human checkpoint.** Joshua chose "prototype first" (ee22aab6:939, options at :914).
The 10 prototype frames were shown to him verbatim (:1031). He questioned one caveat
(:1035), the model reworded one frame (:1038), and he approved the fan-out (:1048). See
`frames/main_session/02_*`, `03_*`, `04_*`.

**Merge, and no filtering.** All 50 authored frames plus the 7 seed entries went into
`frames.json`, giving 57 entries, of which 56 are composable. The validator
(`frames/orchestrator_code/merge_frames.py`, plus the patch for its own `{p_lower}` bug)
rejected only duplicate ids/templates, wrong placeholder counts, render failures and the
reserved SEP string. No frame was dropped on quality or measured strength. The "56" in
the paper is simply everything authored, minus `sep_verbatim`.

**Mapping.** `frames/frame_provenance.json` maps every entry to its source:

- 48 are byte-identical to the subagent's output.
- `metadata_spoof_11` differs only in literal braces doubled for `str.format`
  (`orchestrator_code/normalize_delimiter_metadata_batches.sh`).
- `task_complete_v1` was **reworded by the orchestrating model** (Opus 5) to remove a
  reference to a "summary" task. The original is in `01_prototype_attack_frames.final_output.txt`;
  the reworded one is in `orchestrator_code/write_prototype_batch_and_merge.sh`.
- The 6 seeds are byte-identical to the ee22aab6:799 Write.

The orchestrator re-typed the subagent outputs into `frame_batches/*.json` via heredocs.
The comparison above shows that transcription was lossless apart from the two changes
listed.

**Ranking and weighting.** Frames are weighted, not selected.

- `scripts/corpus/rank_frames.py` (ee22aab6:1072) measures each frame's standalone injection
  rate on held-out passages. It was re-run twice after two measurement bugs were found:
  QA tasks were replaced by transform tasks, and payloads were screened (:1144–:1264).
- Joshua chose "keep the full spread, weighted toward the 40%+ frames" (:1267).
- Composition-time weights come from `STYLE_TARGETS` × within-style rate in
  `compose_injection_items.py:load_frame_weights`. The per-epoch re-draw that training
  actually sees uses rate + 0.15 in `scripts/preprocess_injection_v2.py:191`.
- Per-model rankings are copied to `frames/ranking/`. The full per-request data is in
  `release_data/<model>/frame_ranking.jsonl`.

### Payloads

- **TriviaQA:** 500 rows were fetched from the HF datasets-server
  (`payloads/triviaqa_fetch_command.jsonl`, ee22aab6:788, aliases of 5+ characters kept).
  They were then screened by `scripts/corpus/screen_payloads.py`, which keeps the ones the
  frozen model answers correctly standalone (231/494 for Qwen, 263 for Llama).
- **Directive bank:** this is the paper's "programmatic bank of format, language, and
  behavior hijacks". It is a fixed list of 12 strings written by the main-session model
  in code. No prompt and no per-item model call was involved.

### Targets

Targets are the frozen model's greedy completion on the clean passage
(`compose_injection_items.py:complete`). Any target with `finish_reason != "stop"` is
rejected and logged. No model edits targets for this corpus.

## The earlier "task-document pairs" approach (2026-08-04): abandoned

`abandoned_task_doc_pairs/` holds `data/injectgen/metaprompt.md` and the 3 Opus 5
subagent calls with their prompts and outputs:

- "Generate injection training items" wrote `batch_prototype.json`.
- "Generate task-document pairs for ablation" wrote `ablation_docs.json`.
- "Generate revised validation batch" wrote `batch_v2.json`.

These produced the 26-item `data/injectgen/items.jsonl` via `scripts/build_injection_items.py`.
On 2026-08-11 the design pivoted to composition from shelf datasets (ee22aab6:760–:776,
`pivot_dialogue.jsonl`). Nothing reads those files except `build_injection_items.py` and
`ablate_probe_style.py`, and no item from them is in `release_data` or any training
tensor.

It influenced the corpus in two indirect ways:

1. The ablation over `ablation_docs.json` produced the "~23% bare vs ~80%
   insistent/authority" measurement. That figure is quoted in authoring prompts 01, 02
   and 05, and the other three state that the frame drives injection success. It
   motivated the frame library.
2. The seed frames' style names, and the "ADMIN NOTICE" label, mirror metaprompt.md's
   framing mix.

## Eval-side judges (not datagen)

`eval_judges/` holds files for `judge_sep`, `judge_piarena`, `judge_quadrat` and
`judge_quadrat_taxonomy`:

- the script at HEAD and at the submission-era commit `74e6101` (2026-08-24), where that
  commit has it;
- each prompt constant extracted verbatim by AST (`prompt__<NAME>__L<a>-<b>.py.txt`);
- the default models in `judge_models.json`.

All four call temperature 0 through Prime Inference's OpenAI-compatible endpoint, and
`MODEL` can be overridden by `JUDGE_MODEL`. The two Quadrat judges can instead route to
OpenRouter when `JUDGE_BACKEND=openrouter`. Prompt text for SEP and PIArena is unchanged
since 74e6101; only the default model and PIArena's `--arm` default changed.
`judge_copy`, `judge_probe`, `judge_behav_or` and `judge_decline_leaks` are not
injection evals and are excluded.

## Gaps & discrepancies

1. **The workshop PDF misdescribes the injection targets.** Its §2.3 lists "injection
   resistance" among the overlays whose targets come from "having a stronger model write
   or edit the frozen model's own completion". In fact the injection targets are the
   frozen model's unedited clean-passage completions. The accurate description
   (`appendix_corpus.tex`) is `\notworkshop` and was added after submission (commit
   74e6101 message). So the submitted PDF contains neither the "no per-item synthetic
   data" text nor the 56/twelve-styles text.
2. **"Programmatic" undersells authorship.** The hijack bank, the task bank, the
   gate/validator/fidelity pools and 6 frames are LLM-written strings embedded in code.
   They contain no per-item synthetic data, but they are model-authored. The 50 other
   frames came from the Opus subagent prompts here.
3. **`ai_usage.tex`** (ICLR build only; absent from the workshop PDF) has three problems:
   - It names "Claude Opus and Claude Sonnet" for synthetic data. The injection frames
     were Opus 5, but the seed frames and directive bank were Claude Fable 5, which it
     does not mention as a data author.
   - It says "targets were generated by Claude … every target passed mechanical
     validation". That does not apply to this corpus, whose targets come from the frozen
     model.
   - It says "Experimental design was done by the human authors alone". The transcripts
     show the assistant proposing the frame-library composition, the selective-task
     restriction and the weighting scheme, with Joshua approving or choosing among the
     options (ee22aab6:760–:776, :914/:939, :1246/:1267).
4. **The judge model for the workshop numbers is unverified.** At 74e6101 (submission
   era) the default judge for SEP and PIArena was `openai/gpt-4.1-nano`. On 2026-08-27 it
   became `anthropic/claude-haiku-4.5`, with a comment claiming "every published number
   on this project uses Haiku". The stored verdict files
   (`data/baselines/sep/judged.jsonl`, dated 08-13; `piarena/judged.jsonl`, rewritten
   10-01) carry no judge-model field. The transcripts show no `JUDGE_MODEL` override for
   SEP or PIArena, only Haiku for Quadrat (post-submission). So the model behind the
   PDF's SEP judge audit (67 of 314 hits) cannot be confirmed from records.
5. **The fixed families were regenerated after submission.** On 2026-08-27 the Qwen
   files were regenerated, and the originals are kept in
   `data/injectgen/composed/_pre_regen_backup/`:

   | family | before | after (released) |
   |---|---|---|
   | gate | 1,200 | 4,800 |
   | validator | 1,500 | 6,000 |
   | fidelity | 1,181 | 1,920 |

   Short spans were also added on 08-27. Which sizes the workshop-paper adapter trained
   on is not established here, and needs checking against the run config.
6. **Two frames were changed after authoring:** `task_complete_v1` (reworded) and
   `metadata_spoof_11` (brace escaping). See the frames section.
7. **The seed frames have no prompt.** They were written in the main conversation, so
   "the prompt" is the surrounding dialogue, copied as JSONL.
8. **Thinking blocks are omitted** from the transcript excerpts. They are empty or signed
   in these logs.
