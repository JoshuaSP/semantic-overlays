"use client";

import { useEffect, useRef, useState } from "react";
import {
  applyMark,
  Chan,
  clearRange,
  COLORS,
  MARK_TYPES,
  MarkType,
  parseQover,
  qoverChannel,
  remapSpans,
  Seg,
  segment,
  Span,
  wordSnap,
} from "@/lib/marks";

type Props = {
  text: string;
  spans: Span[];
  palette: "qover" | "inject";
  onText: (t: string) => void;
  onSpans: (s: Span[]) => void;
};

type Tool = { kind: "mark"; chan: Chan } | { kind: "erase" };

// Char offset in the box's rendered text under the pointer. The box renders
// the text verbatim (pre-wrap, no injected glyphs), so Range.toString()
// lengths ARE text offsets.
function offsetFromPoint(
  box: HTMLElement,
  x: number,
  y: number
): number | null {
  let node: Node | null = null;
  let off = 0;
  const d = document as Document & {
    caretPositionFromPoint?: (
      x: number,
      y: number
    ) => { offsetNode: Node; offset: number } | null;
    caretRangeFromPoint?: (x: number, y: number) => Range | null;
  };
  if (d.caretPositionFromPoint) {
    const p = d.caretPositionFromPoint(x, y);
    if (p) {
      node = p.offsetNode;
      off = p.offset;
    }
  } else if (d.caretRangeFromPoint) {
    const r = d.caretRangeFromPoint(x, y);
    if (r) {
      node = r.startContainer;
      off = r.startOffset;
    }
  }
  if (!node || !box.contains(node)) return null;
  const pre = document.createRange();
  pre.selectNodeContents(box);
  pre.setEnd(node, off);
  return pre.toString().length;
}

function segClasses(seg: Seg): string {
  const cls: string[] = [];
  for (const m of seg.marks) {
    const q = parseQover(m.span.channel);
    if (q) {
      if (q.t === "highlighted") cls.push(`hl-${q.c}`);
      if (q.t === "underlined") cls.push(`ul-${q.c}`);
      if (q.t === "circled") {
        cls.push(`ci-${q.c}`);
        if (m.start) cls.push("ci-start");
        if (m.end) cls.push("ci-end");
      }
    } else if (m.span.channel === "untrusted") {
      cls.push("untrusted");
    }
  }
  return cls.join(" ");
}

const TYPE_LABEL: Record<MarkType, string> = {
  highlighted: "highlight",
  underlined: "underline",
  circled: "circle",
};

function sameTool(a: Tool | null, b: Tool): boolean {
  if (!a) return false;
  if (a.kind !== b.kind) return false;
  if (a.kind === "mark" && b.kind === "mark")
    return (
      a.chan.adapter === b.chan.adapter && a.chan.channel === b.chan.channel
    );
  return true;
}

export default function MarkingEditor({
  text,
  spans,
  palette,
  onText,
  onSpans,
}: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(text);
  const [tool, setTool] = useState<Tool | null>(null);
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);

  // Esc puts the overlay down.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setTool(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function toggleTool(t: Tool) {
    // Picking up an overlay while editing commits the draft and leaves edit
    // mode — the natural next act after typing is marking, not "done".
    if (editing) {
      setEditing(false);
      if (draft !== text) {
        onText(draft);
        onSpans(remapSpans(text, draft, spans)); // marks follow the edit
      }
    }
    setTool((cur) => (sameTool(cur, t) ? null : t));
  }

  function onBoxMouseDown(e: React.MouseEvent) {
    if (!tool || editing || e.button !== 0) return;
    const box = boxRef.current;
    if (!box) return;
    const start = offsetFromPoint(box, e.clientX, e.clientY);
    if (start == null) return;
    e.preventDefault();
    // tool/text/spans are captured here; they can't change mid-drag.
    const armed = tool;
    const cur = { a: start, b: start };
    setDrag({ ...cur });
    const move = (ev: MouseEvent) => {
      const off = offsetFromPoint(box, ev.clientX, ev.clientY);
      if (off != null) {
        cur.b = off;
        setDrag({ ...cur });
      }
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      setDrag(null);
      const snap = wordSnap(text, cur.a, cur.b);
      if (!snap) return;
      onSpans(
        armed.kind === "erase"
          ? clearRange(text, spans, snap[0], snap[1])
          : applyMark(text, spans, armed.chan, snap[0], snap[1])
      );
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  // Live preview: render the spans as they WILL be after release — including
  // the same-mark-type eviction — so painting over an old mark shows the
  // overwrite in real time.
  let shown = spans;
  if (drag && tool) {
    const snap = wordSnap(text, drag.a, drag.b);
    if (snap) {
      shown =
        tool.kind === "erase"
          ? clearRange(text, spans, snap[0], snap[1])
          : applyMark(text, spans, tool.chan, snap[0], snap[1]);
    }
  }

  const toolState = !tool
    ? palette === "qover"
      ? "pick up an overlay, then drag across words"
      : "pick up the untrusted marker, then drag across words"
    : tool.kind === "erase"
      ? "eraser in hand — drag across words · Esc to put down"
      : `${tool.chan.channel} in hand — drag across words · Esc to put down`;

  return (
    <div className="editor">
      <div className="toolbar">
        {palette === "qover" ? (
          <div className="palette">
            {MARK_TYPES.map((t) => (
              <div className="palette-row" key={t}>
                <span className="palette-label">{TYPE_LABEL[t]}</span>
                {COLORS.map((c) => {
                  const chan = { adapter: "qover", channel: qoverChannel(t, c) };
                  const armed = sameTool(tool, { kind: "mark", chan });
                  return (
                    <button
                      key={c}
                      className={`chip chip-${t} chip-${c}${armed ? " armed" : ""}`}
                      title={chan.channel}
                      onClick={() => toggleTool({ kind: "mark", chan })}
                    >
                      Aa
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        ) : (
          <button
            className={`chip chip-untrusted${
              sameTool(tool, {
                kind: "mark",
                chan: { adapter: "inject", channel: "untrusted" },
              })
                ? " armed"
                : ""
            }`}
            onClick={() =>
              toggleTool({
                kind: "mark",
                chan: { adapter: "inject", channel: "untrusted" },
              })
            }
          >
            untrusted marker
          </button>
        )}
        <div className="toolbar-side">
          {palette === "inject" && (
            <button
              className="ghost"
              onClick={() =>
                onSpans(
                  applyMark(
                    text,
                    spans,
                    { adapter: "inject", channel: "untrusted" },
                    0,
                    text.length
                  )
                )
              }
            >
              mark all untrusted
            </button>
          )}
          <button
            className={`ghost${sameTool(tool, { kind: "erase" }) ? " armed" : ""}`}
            onClick={() => toggleTool({ kind: "erase" })}
          >
            eraser
          </button>
          <button className="ghost" onClick={() => onSpans([])}>
            clear all
          </button>
          {editing ? (
            <button
              className="ghost primary"
              onClick={() => {
                setEditing(false);
                if (draft !== text) {
                  onText(draft);
                  onSpans(remapSpans(text, draft, spans)); // marks follow the edit
                }
              }}
            >
              done editing
            </button>
          ) : (
            <button
              className="ghost"
              onClick={() => {
                setDraft(text);
                setTool(null);
                setEditing(true);
              }}
            >
              edit text
            </button>
          )}
        </div>
      </div>

      <div className="toolstate">{toolState}</div>

      {editing ? (
        <textarea
          className="markbox markbox-edit"
          value={draft}
          rows={Math.max(6, draft.split("\n").length + 1)}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => {
            // clicking anywhere outside stores the edit, so Generate right
            // after typing picks up the new text
            setEditing(false);
            if (draft !== text) {
              onText(draft);
              onSpans(remapSpans(text, draft, spans));
            }
          }}
        />
      ) : (
        <div
          className={`markbox${tool ? " tool-armed" : ""}`}
          ref={boxRef}
          onMouseDown={onBoxMouseDown}
        >
          {segment(text, shown).map((seg) => (
            <span key={`${seg.lo}-${seg.hi}`} className={segClasses(seg)}>
              {seg.text}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
