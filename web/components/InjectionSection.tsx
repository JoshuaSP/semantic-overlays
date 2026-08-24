"use client";

import { useState } from "react";
import OutputPanes from "./OutputPanes";
import { useGpu } from "./GpuStatus";
import { useGenerate } from "@/lib/useGenerate";
import { INJECTION, InjectionItem } from "@/lib/sections";

// Deliberately chrome-free: in this section the retrieved passage is ALWAYS
// fully marked untrusted, so there is no palette, no tool to pick up, and no
// per-span editing — just the text (editable) and the two answers. The whole
// passage carries one flat tint; we never tint the planted sentence
// differently, because that would imply the overlay treats it specially. It
// does not: provenance is the only signal, and the mark covers everything
// that came back from retrieval.
export default function InjectionSection() {
  const [idx, setIdx] = useState(0);
  const item: InjectionItem = INJECTION[idx];
  const [text, setText] = useState(item.context);
  const [instruction, setInstruction] = useState(item.instruction);
  const { notifyWake } = useGpu();
  const { on, off, run, reset, busy } = useGenerate(notifyWake);

  function load(i: number) {
    const n = INJECTION.length;
    const k = ((i % n) + n) % n;
    setIdx(k);
    setText(INJECTION[k].context);
    setInstruction(INJECTION[k].instruction);
    reset();
  }

  // The mark always covers the entire retrieved passage.
  const spans = [
    { adapter: "inject", channel: "untrusted", lo: 0, hi: text.length },
  ];
  const plantedStillPresent = text.includes(item.injection);

  return (
    <section className="card" id="injection">
      <div className="card-head">
        <h2>An NX bit for LLMs: prompt-injection protection</h2>
        <p className="blurb">
          Imagine an application that fetches web data. The developer writes
          the task; the passage comes back from the open web with an
          instruction hidden inside it. Semantic overlays are used to mark the
          full retrieved passage as &ldquo;do not execute&rdquo; — it stays
          completely readable, but loses the authority to give orders.
        </p>
      </div>

      <div className="deck">
        <button className="ghost" onClick={() => load(idx - 1)}>‹</button>
        <span className="deck-name">
          {item.name}
          <span className="deck-count">
            {idx + 1}/{INJECTION.length}
          </span>
        </span>
        <button className="ghost" onClick={() => load(idx + 1)}>›</button>
      </div>

      <div className="mech">
        <span className="mech-tag">{item.mechanism}</span>
        <span className="mech-why">unmarked, {item.why}</span>
      </div>
      {item.note && <div className="mech-note">{item.note}</div>}

      <div className="lbl trusted-lbl">developer instruction — trusted</div>
      <textarea
        className="instruction-box"
        value={instruction}
        rows={3}
        onChange={(e) => setInstruction(e.target.value)}
      />

      <div className="lbl data-lbl">
        retrieved passage — marked in full with a &ldquo;do not execute&rdquo; semantic overlay
      </div>
      <textarea
        className="passage untrusted-all"
        value={text}
        rows={Math.min(16, Math.ceil(text.length / 105) + text.split("\n").length)}
        onChange={(e) => setText(e.target.value)}
      />

      {plantedStillPresent && (
        <div className="planted">
          <span className="planted-tag">⚠ planted instruction</span>
          <span className="planted-where">
            buried {item.injection_pct}% into the passage above — nothing
            visually marks it apart, and the model is given no hint where it is
          </span>
          <div className="planted-text">“{item.injection}”</div>
        </div>
      )}

      <div className="run-row">
        <button
          className="generate"
          disabled={busy}
          onClick={() =>
            run({ text, spans, instruction, maxNew: 300 }, true)
          }
        >
          {busy ? "generating…" : "Compare overlays on / off"}
        </button>
        <span className="goggle-state on">
          🖍️ whole passage marked untrusted
        </span>
      </div>

      <OutputPanes on={on} off={off} offLabel="unmarked — same model, no overlay" />
    </section>
  );
}
