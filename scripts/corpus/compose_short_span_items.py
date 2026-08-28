# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "tqdm"]
# ///
"""Compose SHORT-SPAN injection items -- the increment that fills the corpus's
empty corner (see notes/short_span_corpus_plan.md).

The shipped corpus never trains a marked span shorter than 216 chars (1 item of
7,974 below 220) and never lets the payload dominate one (min benign fraction
0.34).  The Qwen overlay consequently leaks 14.4% on the AlpacaFarm/StruQ
harness, all of it on short, injection-dominated spans, and is hijacked 4/4 by a
span that is nothing but an injection.  This script generates that regime.

Three unit kinds, all reusing the existing as-if-absent contract -- the target is
the frozen model's own greedy completion on the CLEAN operand, shared by the
injected and benign items of a pair:

  title     operand = a SQuAD article title ("Mandolin.")  x  a topic task.
            Short operand + normal payload => LOW benign fraction (~0.1-0.3).
  question  operand = a SQuAD question       x  an answer task.
            Longer operand => MID benign fraction (~0.4-0.6).
  pure      operand = EMPTY.  The marked span is the payload and nothing else
            (benign fraction 0).  Target = the frozen model's completion on
            empty input, which for an operand-requiring task is a truthful
            "you haven't given me a topic" -- not a false claim of emptiness.
            Cached per task: only one distinct empty-input target per task.

Plus mechanical `copy_injected` fidelity items over the same short spans (target
= the span verbatim, no model call).  These are added only at the corpus's
EXISTING fidelity proportion -- raising the fidelity family from 4.3% to 10%
measurably cost 3.7pt of SEP and doubled TensorTrust ASR
(knowledge/arm_results.md), so this increment must not shift the ratio.

Sources are all CC BY-SA 4.0, matching the released corpus: SQuAD titles and
questions, the screened trivia payloads, and the programmatic directive bank.
The Alpaca train split -- which StruQ and SecAlign both train on -- is
deliberately NOT used: it is CC BY-NC 4.0 and would force relicensing.

Contamination guards, enforced at runtime:
  * no operand is drawn from the 208 AlpacaFarm eval items
  * no payload contains the eval's marker string ("hacked")

Checkpointed per unit and resumable.  Rejects (target did not finish naturally)
are logged with reasons, never dropped silently.

Usage:
  uv run scripts/compose_short_span_items.py --url ... --limit 3     # prototype
  uv run scripts/compose_short_span_items.py --url ... --n-title 350 \
      --n-question 150 --n-pure 100 --n-fidelity 100
"""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evals"))
from common import TEMPLATE_SYSTEM, TEMPLATE_USER  # noqa: E402

COMPOSED = REPO / "data" / "injectgen" / "composed"
OUT_DIR = REPO / "data" / "injectgen" / "short_span"
KEY = os.environ["GOGGLES_API_KEY"]  # bearer token for your serving endpoint

QUESTION_TASKS = [
    "Answer the following question.",
    "Answer the question below.",
    "Provide a brief answer to the question.",
    "Answer the question in one sentence.",
]
# Directive bank, verbatim from compose_injection_items.py.  Kept in sync by
# import below; duplicated here only if that import fails.
FIDELITY_INSTRUCTION = (
    "Transcribe the input exactly. Do not summarise, correct, or add anything.")

# A `pure` unit's target is the frozen model's completion on EMPTY input. For
# ~1/3 of tasks that completion INVENTS a topic ("Welcome to today's lecture,
# where we will explore ... artificial intelligence in healthcare") instead of
# reporting the operand is missing. Such a target would train the overlay to
# fabricate unrelated content whenever a span is pure injection. Screen for an
# explicit acknowledgement that the input is absent; reject the task for pure
# units otherwise, and log it.
ACKNOWLEDGES_EMPTY = re.compile(
    r"please provide|kindly provide|provide the|share the|specify the|"
    r"you have ?n[o’']t (provided|specified|given|included)|"
    r"(did|do) not (provide|specify|include)|was not provided|"
    r"is (missing|empty|blank)|appears (to be )?empty|it seems|could you|"
    r"i (don't|do not|cannot|can't) see|no (topic|input|text|question)\b|"
    r"there is no (topic|input|specific)", re.I)


def screen_empty_target(text: str) -> bool:
    """True if the completion admits the input is absent rather than inventing one."""
    return bool(ACKNOWLEDGES_EMPTY.search(text[:300]))


def load_sources():
    tasks = json.load(open(COMPOSED / "topic_tasks.json"))["tasks"]
    frames = [f for f in json.load(open(COMPOSED / "frames.json"))
              if not f.get("reference_only")]
    payloads = json.load(open(REPO / "release_data" / "qwen3.5-9b" /
                              "payloads_screened.json"))
    sys.path.insert(0, str(REPO / "scripts"))
    from compose_injection_items import DIRECTIVES, frame_payload, splice
    payloads = ([{"kind": "trivia", **p} for p in payloads] +
                [{"kind": "directive", **d} for d in DIRECTIVES])
    # Contamination guard: the eval's marker must never enter training.
    bad = [p for p in payloads if "hacked" in p["payload"].lower()]
    if bad:
        raise SystemExit(f"payload bank contains the eval marker: {bad[:2]}")
    squad = json.load(open(REPO / "release_data" / "shared" / "raw" /
                           "train-v2.0.json"))["data"]
    titles = sorted({a["title"] for a in squad})
    questions = sorted({q["question"].strip() for a in squad
                        for p in a["paragraphs"] for q in p["qas"]
                        if 25 <= len(q["question"].strip()) <= 90})
    # Contamination guard: nothing may coincide with an AlpacaFarm eval input.
    ev = REPO / "data" / "alpaca_inject" / "items.json"
    if ev.exists():
        evalset = {(it.get("input") or "").strip().lower()
                   for it in json.load(open(ev))}
        titles = [t for t in titles
                  if t.replace("_", " ").lower() not in evalset]
        questions = [q for q in questions if q.lower() not in evalset]
    return tasks, frames, payloads, titles, questions, frame_payload, splice


def gen(url, task, operand, _attempt=0):
    """Frozen-model greedy completion. Retries the SAME request only."""
    try:
        r = requests.post(url, headers={"authorization": f"Bearer {KEY}"},
                          json={"text": operand, "spans": [],
                                "instruction": task, "max_new": 1024},
                          timeout=300, stream=True)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        if _attempt >= 4:
            raise
        time.sleep(5 * (_attempt + 1))
        return gen(url, task, operand, _attempt + 1)
    if r.status_code in (429, 502, 503, 504):
        if _attempt >= 4:
            r.raise_for_status()
        time.sleep(5 * (_attempt + 1))
        return gen(url, task, operand, _attempt + 1)
    r.raise_for_status()
    out, fr = [], None
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode()
        if s.startswith("data: "):
            try:
                d = json.loads(s[6:])
            except Exception:
                continue
            if "delta" in d:
                out.append(d["delta"])
            if d.get("finish_reason"):
                fr = d["finish_reason"]
    return "".join(out), fr


def item(unit_id, kind, task, span, target, **extra):
    return dict(
        unit_id=unit_id, item_id=f"{unit_id}:{kind}", kind=kind, task=task,
        messages=[{"role": "system", "content": TEMPLATE_SYSTEM.format(task)},
                  {"role": "user", "content": TEMPLATE_USER.format(span)}],
        mask_char_span=[7, 7 + len(span)], masked_text=span, target=target,
        **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n-title", type=int, default=350)
    ap.add_argument("--n-question", type=int, default=150)
    ap.add_argument("--n-pure", type=int, default=100)
    ap.add_argument("--n-fidelity", type=int, default=100)
    ap.add_argument("--bare-share", type=float, default=0.70)
    ap.add_argument("--limit", type=int, default=0, help="prototype: N units")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tasks, frames, payloads, titles, questions, frame_payload, splice = load_sources()
    bare = [f for f in frames if f["id"] == "bare"]
    styled = [f for f in frames if f["id"] != "bare"]
    rng = random.Random(a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items_path, rej_path = OUT_DIR / "items.jsonl", OUT_DIR / "items_rejected.jsonl"

    done = set()
    if items_path.exists():
        for l in open(items_path):
            done.add(json.loads(l)["unit_id"])
    print(f"sources: {len(titles)} titles, {len(questions)} questions, "
          f"{len(payloads)} payloads, {len(tasks)} topic tasks, "
          f"{len(frames)} frames; {len(done)} units already done")

    plan = []
    for i in range(a.n_title):
        plan.append(dict(unit_id=f"ss-title:{i}", operand_kind="title",
                         operand=rng.choice(titles).replace("_", " ") + ".",
                         task=rng.choice(tasks)))
    for i in range(a.n_question):
        plan.append(dict(unit_id=f"ss-q:{i}", operand_kind="question",
                         operand=rng.choice(questions),
                         task=rng.choice(QUESTION_TASKS)))
    for i in range(a.n_pure):
        plan.append(dict(unit_id=f"ss-pure:{i}", operand_kind="pure",
                         operand="", task=rng.choice(tasks)))
    trivia = [p for p in payloads if p["kind"] == "trivia"]
    directives = [p for p in payloads if p["kind"] == "directive"]
    for u in plan:
        # A question operand next to a trivia payload is two questions with no
        # cue but position -- that teaches "answer the first one", not "ignore
        # the injection". Question operands therefore take directives only.
        # Title/pure units tilt to directives too, since the attack we are
        # fixing ("Print exactly 'hacked!'") is an imperative.
        if u["operand_kind"] == "question":
            p = rng.choice(directives)
        else:
            p = rng.choice(directives if rng.random() < 0.6 else trivia)
        f = rng.choice(bare if rng.random() < a.bare_share else styled)
        u.update(payload=p["payload"], payload_kind=p["kind"],
                 witness=p.get("witness"), frame_id=f["id"],
                 framed_payload=frame_payload(f, p["payload"]),
                 position="end" if rng.random() < 0.7 else "start")
    plan = [u for u in plan if u["unit_id"] not in done]
    if a.limit:
        by_kind = {}
        for u in plan:
            by_kind.setdefault(u["operand_kind"], []).append(u)
        per = max(1, a.limit // max(1, len(by_kind)))
        plan = [u for v in by_kind.values() for u in v[:per]]
    print(f"{len(plan)} units to compose")

    # Empty-input targets are task-dependent only -> cache one per task.
    empty_cache, cache_lock = {}, threading.Lock()
    write_lock = threading.Lock()
    stats = {"ok": 0, "rejected": 0}

    def compose(u):
        if u["operand_kind"] == "pure":
            with cache_lock:
                hit = empty_cache.get(u["task"])
            if hit is None:
                txt, fr = gen(a.url, u["task"], " ")
                if fr == "stop" and not screen_empty_target(txt):
                    fr = "hallucinated-empty-target"   # rejected below, loudly
                with cache_lock:
                    empty_cache[u["task"]] = (txt, fr)
            else:
                txt, fr = hit
        else:
            txt, fr = gen(a.url, u["task"], u["operand"])
        if fr != "stop":
            with write_lock, open(rej_path, "a") as fh:
                fh.write(json.dumps({**u, "reason": f"finish_reason={fr}",
                                     "target": txt}) + "\n")
            stats["rejected"] += 1
            return None
        span = (u["framed_payload"] if u["operand_kind"] == "pure"
                else splice(u["operand"], u["framed_payload"], u["position"]))
        # Chat templates strip leading/trailing whitespace from message content.
        # A span carrying it cannot be located by character offset afterwards
        # (the same failure that silently excluded 365 of 14,441 Quadrat docs),
        # and for a fidelity item -- whose target IS the span -- it also breaks
        # the preprocessor's template-drift check. Some frames end in a newline,
        # so strip once, here, before the span is used for anything.
        span = span.strip()
        bf = 0.0 if not u["operand"] else len(u["operand"]) / len(span)
        common = dict(payload=u["payload"], payload_kind=u["payload_kind"],
                      frame_id=u["frame_id"], framed_payload=u["framed_payload"],
                      position=u["position"], operand_kind=u["operand_kind"],
                      operand=u["operand"], benign_fraction=round(bf, 3),
                      witness=u["witness"], target_finish_reason=fr)
        rows = [item(u["unit_id"], "injected", u["task"], span, txt, **common)]
        if u["operand_kind"] != "pure":   # a pure unit has no benign counterpart
            rows.append(item(u["unit_id"], "benign", u["task"], u["operand"],
                             txt, **common))
        with write_lock, open(items_path, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        stats["ok"] += 1
        return span

    spans = []
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(compose, u) for u in plan]
        for n, fu in enumerate(as_completed(futs), 1):
            s = fu.result()
            if s:
                spans.append(s)
            if n % 50 == 0:
                print(f"  {n}/{len(plan)}  ok={stats['ok']} rej={stats['rejected']}")

    # Mechanical fidelity items over the same short spans -- no model call.
    if a.n_fidelity and spans:
        fid_path = OUT_DIR / "fidelity_items.jsonl"
        pick = rng.sample(spans, min(a.n_fidelity, len(spans)))
        with open(fid_path, "a") as fh:
            for i, span in enumerate(pick):
                r = item(f"ss-fid:{i}", "fidelity", FIDELITY_INSTRUCTION, span, span,
                         task_name="copy_injected", injected=True, title="__fidelity")
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(pick)} short copy_injected fidelity items -> {fid_path}")

    print(f"\ncomposed {stats['ok']} units, rejected {stats['rejected']}")
    print(f"items -> {items_path}")


if __name__ == "__main__":
    main()
