"use client";

import { useRef, useState } from "react";
import { hueOf, BehavSpan } from "@/lib/instructions";
import { wordSnap } from "@/lib/marks";

// Two span-selection modes over one stamped-text renderer:
//   click mode — the deck's pre-segmented spans are clickable units; the
//                armed instruction stamps a whole span (click again clears).
//   drag mode  — free-typed target text has no pre-segmented spans, so
//                selection is marks-style press-drag-release, word-snapped.
// Either way the output is the same BehavSpan[] the backend takes.

export type Armed = { channel: string; label: string } | "erase" | null;

function offsetFromPoint(box: HTMLElement, x: number, y: number): number | null {
  let node: Node | null = null;
  let off = 0;
  const d = document as Document & {
    caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null;
    caretRangeFromPoint?: (x: number, y: number) => Range | null;
  };
  if (d.caretPositionFromPoint) {
    const p = d.caretPositionFromPoint(x, y);
    if (p) { node = p.offsetNode; off = p.offset; }
  } else if (d.caretRangeFromPoint) {
    const r = d.caretRangeFromPoint(x, y);
    if (r) { node = r.startContainer; off = r.startOffset; }
  }
  if (!node || !box.contains(node)) return null;
  const pre = document.createRange();
  pre.selectNodeContents(box);
  pre.setEnd(node, off);
  return pre.toString().length;
}

function stampStyle(channel: string): React.CSSProperties {
  const hue = hueOf(channel);
  return {
    background: `color-mix(in srgb, ${hue} 22%, transparent)`,
    borderBottom: `2px solid ${hue}`,
    borderRadius: "3px",
  };
}

// ---------------------------------------------------------------------------

export function ClickSpanText({
  text,
  regions,
  spans,
  armed,
  onSpans,
  mono = false,
}: {
  text: string;
  regions: { lo: number; hi: number }[];
  spans: BehavSpan[];
  armed: Armed;
  onSpans: (s: BehavSpan[]) => void;
  mono?: boolean;
}) {
  function clickRegion(r: { lo: number; hi: number }) {
    if (!armed) return;
    const cur = spans.find((s) => s.lo === r.lo && s.hi === r.hi);
    const rest = spans.filter((s) => !(s.lo === r.lo && s.hi === r.hi));
    if (armed === "erase" || (cur && cur.channel === armed.channel)) {
      onSpans(rest);
      return;
    }
    onSpans(
      [...rest, { adapter: "behav" as const, channel: armed.channel, ...r }].sort(
        (a, b) => a.lo - b.lo
      )
    );
  }

  const parts: React.ReactNode[] = [];
  let pos = 0;
  const sorted = [...regions].sort((a, b) => a.lo - b.lo);
  for (const r of sorted) {
    if (r.lo > pos) parts.push(<span key={`t${pos}`}>{text.slice(pos, r.lo)}</span>);
    const stamped = spans.find((s) => s.lo === r.lo && s.hi === r.hi);
    parts.push(
      <span
        key={`r${r.lo}`}
        className={"ispan" + (stamped ? " stamped" : "") + (armed ? " armable" : "")}
        style={stamped ? stampStyle(stamped.channel) : undefined}
        onClick={() => clickRegion(r)}
      >
        {text.slice(r.lo, r.hi)}
        {stamped && (
          <span className="stamp-tag" style={{ background: hueOf(stamped.channel) }}>
            {stamped.channel}
          </span>
        )}
      </span>
    );
    pos = r.hi;
  }
  if (pos < text.length) parts.push(<span key={`t${pos}`}>{text.slice(pos)}</span>);
  return <div className={"stamp-text" + (mono ? " code" : "")}>{parts}</div>;
}

// ---------------------------------------------------------------------------

export function DragMarkText({
  text,
  spans,
  armed,
  onSpans,
}: {
  text: string;
  spans: BehavSpan[];
  armed: Armed;
  onSpans: (s: BehavSpan[]) => void;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [anchor, setAnchor] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  function commit(a: number, b: number) {
    const snapped = wordSnap(text, a, b);
    if (!snapped) return;
    const [lo, hi] = snapped;
    // one instruction per token: painting evicts every overlapped piece
    const rest = spans.flatMap((s) => {
      if (s.hi <= lo || s.lo >= hi) return [s];
      const keep: BehavSpan[] = [];
      if (s.lo < lo) keep.push({ ...s, hi: lo });
      if (s.hi > hi) keep.push({ ...s, lo: hi });
      return keep.filter((p) => text.slice(p.lo, p.hi).trim().length > 0);
    });
    if (armed && armed !== "erase")
      rest.push({ adapter: "behav", channel: armed.channel, lo, hi });
    onSpans(rest.sort((x, y) => x.lo - y.lo));
  }

  const preview =
    anchor !== null && hover !== null ? wordSnap(text, anchor, hover) : null;

  // flat segmentation at every span/preview boundary
  const cuts = new Set<number>([0, text.length]);
  for (const s of spans) { cuts.add(s.lo); cuts.add(s.hi); }
  if (preview) { cuts.add(preview[0]); cuts.add(preview[1]); }
  const pts = [...cuts].sort((a, b) => a - b);

  return (
    <div
      ref={boxRef}
      className={"stamp-text dragable" + (armed ? " armable" : "")}
      onPointerDown={(e) => {
        if (!armed) return;
        const o = offsetFromPoint(boxRef.current!, e.clientX, e.clientY);
        if (o !== null) { setAnchor(o); setHover(o); }
      }}
      onPointerMove={(e) => {
        if (anchor === null) return;
        const o = offsetFromPoint(boxRef.current!, e.clientX, e.clientY);
        if (o !== null) setHover(o);
      }}
      onPointerUp={(e) => {
        if (anchor === null) return;
        const o = offsetFromPoint(boxRef.current!, e.clientX, e.clientY);
        if (o !== null) commit(anchor, o);
        setAnchor(null); setHover(null);
      }}
      onPointerLeave={() => { setAnchor(null); setHover(null); }}
    >
      {pts.slice(0, -1).map((lo, i) => {
        const hi = pts[i + 1];
        if (hi <= lo) return null;
        const s = spans.find((x) => x.lo <= lo && x.hi >= hi);
        const inPreview = preview && lo >= preview[0] && hi <= preview[1];
        return (
          <span
            key={lo}
            className={inPreview ? "drag-preview" : undefined}
            style={s ? stampStyle(s.channel) : undefined}
          >
            {text.slice(lo, hi)}
            {s && s.lo === lo && (
              <span className="stamp-tag" style={{ background: hueOf(s.channel) }}>
                {s.channel}
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
