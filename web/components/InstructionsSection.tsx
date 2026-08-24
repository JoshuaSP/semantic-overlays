"use client";

import { useState } from "react";
import OutputPanes from "./OutputPanes";
import { useGpu } from "./GpuStatus";
import { useGenerate } from "@/lib/useGenerate";
import {
  BehavSpan,
  INSTR_DECK,
  INSTRUCTIONS,
} from "@/lib/instructions";
import { Armed, ClickSpanText, DragMarkText } from "./StampEditor";
import { remapSpans, Span } from "@/lib/marks";

// Secret instructions (§5). Deck prompts are heldout multi-request items
// whose request spans are pre-segmented and CLICKABLE: arm an instruction,
// click a request, the stamp lands. A free-typed prompt has no segmentation,
// so the editor switches to marks-style drag selection — same palette.
export default function InstructionsSection() {
  const [idx, setIdx] = useState(0);
  const [custom, setCustom] = useState(false);
  const [editingText, setEditingText] = useState(false);
  const [text, setText] = useState(INSTR_DECK[0].text);
  const [spans, setSpans] = useState<BehavSpan[]>([]);
  const [armed, setArmed] = useState<Armed>(null);
  const { notifyWake } = useGpu();
  const { on, off, run, reset, busy } = useGenerate(notifyWake);

  function load(i: number) {
    const n = INSTR_DECK.length;
    const k = ((i % n) + n) % n;
    setIdx(k);
    setCustom(false);
    setEditingText(false);
    setText(INSTR_DECK[k].text);
    setSpans([]);
    reset();
  }

  function startCustom() {
    setCustom(true);
    setEditingText(true);
    setText("");
    setSpans([]);
    reset();
  }

  // Arming an instruction while typing commits the text and switches to
  // stamping mode — same interaction rule as the marks editor.
  function arm(next: Armed) {
    if (custom && editingText && text.trim().length > 0) setEditingText(false);
    setArmed(next);
  }

  return (
    <section className="card" id="instructions">
      <div className="card-head">
        <h2>Secret instructions</h2>
        <p className="blurb">
          Several requests, one prompt. Stamp a single request with an
          instruction no token states — only that answer transforms; the
          others are answered normally. Arm an instruction, then click a
          request to stamp it.
        </p>
      </div>

      <div className="deck">
        <button className="ghost" onClick={() => load(idx - 1)}>‹</button>
        <span className="deck-name">
          {custom ? "your own prompt" : INSTR_DECK[idx].name}
          {!custom && (
            <span className="deck-count">{idx + 1}/{INSTR_DECK.length}</span>
          )}
        </span>
        <button className="ghost" onClick={() => load(idx + 1)}>›</button>
        <button
          className={"ghost custom-toggle" + (custom ? " active" : "")}
          onClick={() => (custom ? load(idx) : startCustom())}
        >
          {custom ? "back to the deck" : "write your own"}
        </button>
      </div>

      <div className="palette">
        <span className="palette-label">instructions</span>
        <div className="palette-row">
          {INSTRUCTIONS.map((ins) => (
            <button
              key={ins.channel}
              className={
                "chip" +
                (armed !== null && armed !== "erase" && armed.channel === ins.channel
                  ? " armed"
                  : "")
              }
              style={{ borderColor: ins.hue }}
              onClick={() =>
                arm(
                  armed !== null && armed !== "erase" && armed.channel === ins.channel
                    ? null
                    : { channel: ins.channel, label: ins.label }
                )
              }
            >
              {ins.label}
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

      {custom ? (
        editingText ? (
          <textarea
            className="custom-text"
            placeholder="Type a prompt with a few separate requests in it — then arm an instruction to start stamping."
            value={text}
            onChange={(e) => {
              const next = e.target.value;
              // marks follow the edit live; only a span whose own words
              // were changed is dropped
              setSpans(
                remapSpans(text, next, spans as unknown as Span[]) as unknown as typeof spans
              );
              setText(next);
            }}
            rows={6}
            onBlur={() => {
              if (text.trim().length > 0) setEditingText(false);
            }}
          />
        ) : (
          <>
            <div className="edit-row">
              <button className="ghost" onClick={() => setEditingText(true)}>
                edit text
              </button>
            </div>
            <DragMarkText text={text} spans={spans} armed={armed} onSpans={setSpans} />
          </>
        )
      ) : (
        <ClickSpanText
          text={text}
          regions={INSTR_DECK[idx].spans}
          spans={spans}
          armed={armed}
          onSpans={setSpans}
        />
      )}

      <div className="run-row">
        <button
          className="generate"
          disabled={busy || text.trim().length === 0}
          onClick={() => run({ text, spans, maxNew: 1800 }, false)}
        >
          {busy ? "generating…" : "Generate"}
        </button>
        <button
          className="generate secondary"
          disabled={busy || text.trim().length === 0}
          onClick={() => run({ text, spans, maxNew: 1800 }, true)}
        >
          Compare overlays on / off
        </button>
        {spans.length > 0 ? (
          <span className="goggle-state on">
            🖍️ {spans.length} stamped request{spans.length > 1 ? "s" : ""}
          </span>
        ) : (
          <span className="goggle-state">
            {armed ? "now click a request to stamp it" : "no stamps — plain model"}
          </span>
        )}
      </div>

      <OutputPanes on={on} off={off} />
    </section>
  );
}
