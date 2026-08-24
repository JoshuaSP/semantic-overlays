"use client";

import { useState } from "react";
import MarkingEditor from "./MarkingEditor";
import OutputPanes from "./OutputPanes";
import { useGpu } from "./GpuStatus";
import { useGenerate } from "@/lib/useGenerate";
import { MARKS_DECK } from "@/lib/sections";
import { Span } from "@/lib/marks";

export default function MarksSection() {
  const [idx, setIdx] = useState(0);
  const [text, setText] = useState(MARKS_DECK[0].text);
  const [spans, setSpans] = useState<Span[]>(MARKS_DECK[0].spans);
  const { notifyWake } = useGpu();
  const { on, off, run, reset, busy } = useGenerate(notifyWake);

  function load(i: number) {
    const n = MARKS_DECK.length;
    const k = ((i % n) + n) % n;
    setIdx(k);
    setText(MARKS_DECK[k].text);
    setSpans(MARKS_DECK[k].spans);
    reset();
  }

  return (
    <section className="card" id="marks">
      <div className="card-head">
        <h2>Invisible highlighters: marks &amp; overlapped marks</h2>
        <p className="blurb">
          The text the model sees is just the text that you see below. All the
          data about the marks comes through the overlays. There are no hidden
          control characters or tokens.
        </p>
      </div>

      <div className="deck">
        <button className="ghost" onClick={() => load(idx - 1)}>‹</button>
        <span className="deck-name">
          {MARKS_DECK[idx].name}
          <span className="deck-count">
            {idx + 1}/{MARKS_DECK.length}
          </span>
        </span>
        <button className="ghost" onClick={() => load(idx + 1)}>›</button>
      </div>

      <MarkingEditor
        text={text}
        spans={spans}
        palette="qover"
        onText={setText}
        onSpans={setSpans}
      />

      <div className="run-row">
        <button
          className="generate"
          disabled={busy}
          onClick={() => run({ text, spans, maxNew: 300 }, false)}
        >
          {busy ? "generating…" : "Generate"}
        </button>
        {spans.length > 0 ? (
          <span className="goggle-state on">
            🖍️ {spans.length} overlay{spans.length > 1 ? "s" : ""} on
          </span>
        ) : (
          <span className="goggle-state">no overlays — plain model</span>
        )}
      </div>

      <OutputPanes on={on} off={off} />
    </section>
  );
}
