"use client";

import { useState } from "react";
import OutputPanes from "./OutputPanes";
import { useGpu } from "./GpuStatus";
import { useGenerate } from "@/lib/useGenerate";
import { hueOf, LANG_DECK, LANGUAGES } from "@/lib/instructions";
import { Armed } from "./StampEditor";
import { Span } from "@/lib/marks";

// Python-colored glasses (§4), structured: fixed "Snippet N:" titles
// delineate the snippets (so questions can refer to them BY NUMBER), every
// snippet body is editable AND clickable, and the question is editable.
// Interaction rule: with a language armed, clicking a snippet stamps it;
// with nothing armed, clicking a snippet (or the question) edits it.

type Snip = { body: string; channel: string | null };

function assemble(snippets: Snip[], question: string) {
  const text =
    snippets.map((s, i) => `Snippet ${i + 1}:\n${s.body}`).join("\n\n") +
    "\n\n" +
    question;
  const spans: Span[] = [];
  let pos = 0;
  snippets.forEach((s, i) => {
    const head = `Snippet ${i + 1}:\n`;
    pos += head.length;
    if (s.channel)
      spans.push({ adapter: "plr", channel: s.channel, lo: pos, hi: pos + s.body.length });
    pos += s.body.length + 2; // "\n\n"
  });
  return { text, spans };
}

export default function LanguagesSection() {
  const [idx, setIdx] = useState(0);
  const [snippets, setSnippets] = useState<Snip[]>(
    LANG_DECK[0].snippets.map((s) => ({ ...s }))
  );
  const [question, setQuestion] = useState(LANG_DECK[0].question);
  const [armed, setArmed] = useState<Armed>(null);
  const [editing, setEditing] = useState<number | "question" | null>(null);
  const { notifyWake } = useGpu();
  const { on, off, run, reset, busy } = useGenerate(notifyWake);

  function load(i: number) {
    const n = LANG_DECK.length;
    const k = ((i % n) + n) % n;
    setIdx(k);
    setSnippets(LANG_DECK[k].snippets.map((s) => ({ ...s })));
    setQuestion(LANG_DECK[k].question);
    setEditing(null);
    reset();
  }

  // arming a language closes any open editor (the marks-editor rule)
  function arm(next: Armed) {
    setEditing(null);
    setArmed(next);
  }

  function clickSnippet(i: number) {
    if (armed === null) {
      setEditing(i);
      return;
    }
    // multi4 training always has exactly ONE marked snippet — multiple marks
    // are out of distribution and visibly degrade, so stamping snippet i
    // clears any mark elsewhere.
    setSnippets(
      snippets.map((s, k) => {
        if (k !== i) return { ...s, channel: null };
        if (armed === "erase" || s.channel === (armed as { channel: string }).channel)
          return { ...s, channel: null };
        return { ...s, channel: (armed as { channel: string }).channel };
      })
    );
    setArmed(null); // one stamp per pickup: drop the tool after use
  }

  const { text, spans } = assemble(snippets, question);

  return (
    <section className="card" id="languages">
      <div className="card-head">
        <h2>Python-colored gels</h2>
        <p className="blurb">
          The overlay changes how the model understands what language the
          snippet is in. Arm a language and
          click a snippet to stamp it; with nothing armed, click any snippet
          (or the question) to edit it. Questions can name snippets by number.
        </p>
      </div>

      <div className="deck">
        <button className="ghost" onClick={() => load(idx - 1)}>‹</button>
        <span className="deck-name">
          {LANG_DECK[idx].name}
          <span className="deck-count">{idx + 1}/{LANG_DECK.length}</span>
        </span>
        <button className="ghost" onClick={() => load(idx + 1)}>›</button>
      </div>

      <div className="palette">
        <span className="palette-label">asserted language</span>
        <div className="palette-row">
          {LANGUAGES.map((l) => (
            <button
              key={l.channel}
              className={
                "chip" +
                (armed !== null && armed !== "erase" && armed.channel === l.channel
                  ? " armed"
                  : "")
              }
              style={{ borderColor: l.hue }}
              onClick={() =>
                arm(
                  armed !== null && armed !== "erase" && armed.channel === l.channel
                    ? null
                    : { channel: l.channel, label: l.label }
                )
              }
            >
              {l.label}
            </button>
          ))}
          <button
            className={"chip" + (armed === "erase" ? " armed" : "")}
            onClick={() => arm(armed === "erase" ? null : "erase")}
          >
            ✕ clear
          </button>
        </div>
      </div>

      <div className="snippets">
        {snippets.map((s, i) => (
          <div className="snippet" key={i}>
            <div className="snippet-title">Snippet {i + 1}:</div>
            {editing === i ? (
              <textarea
                className="custom-text code"
                value={s.body}
                autoFocus
                rows={Math.max(3, s.body.split("\n").length + 1)}
                onBlur={() => setEditing(null)}
                onChange={(e) =>
                  setSnippets(
                    snippets.map((x, k) => (k === i ? { ...x, body: e.target.value } : x))
                  )
                }
              />
            ) : (
              <div className="snippet-box">
                <pre
                  className={"snippet-body" + (armed ? " armable" : " editable")}
                  style={
                    s.channel
                      ? {
                          background: `color-mix(in srgb, ${hueOf(s.channel)} 14%, transparent)`,
                          borderLeft: `3px solid ${hueOf(s.channel)}`,
                        }
                      : undefined
                  }
                  onClick={() => clickSnippet(i)}
                >
                  {s.body}
                </pre>
                {s.channel && (
                  <span className="stamp-tag" style={{ background: hueOf(s.channel) }}>
                    {s.channel}
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
        <div className="snippet-divider" />
        {editing === "question" ? (
          <textarea
            className="custom-text"
            value={question}
            autoFocus
            rows={2}
            onBlur={() => setEditing(null)}
            onChange={(e) => setQuestion(e.target.value)}
          />
        ) : (
          <div
            className="question-line editable"
            onClick={() => armed === null && setEditing("question")}
          >
            {question}
          </div>
        )}
      </div>

      <div className="run-row">
        <button
          className="generate"
          disabled={busy}
          onClick={() => run({ text, spans, maxNew: 500 }, false)}
        >
          {busy ? "generating…" : "Generate"}
        </button>
        <button
          className="generate secondary"
          disabled={busy}
          onClick={() => run({ text, spans, maxNew: 500 }, true)}
        >
          Compare overlays on / off
        </button>
        {spans.length > 0 ? (
          <span className="goggle-state on">
            🖍️ marked {spans.map((s) => s.channel.replace("written in ", "")).join(", ")}
          </span>
        ) : (
          <span className="goggle-state">no overlays — surface language wins</span>
        )}
      </div>

      <OutputPanes on={on} off={off} />
    </section>
  );
}
