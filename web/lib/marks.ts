// Span model shared by the editor, decks, and the backend API.
// A span is char offsets [lo, hi) into the *user text* plus the adapter
// channel it activates — exactly the JSON the backend /generate takes.

export const MARK_TYPES = ["highlighted", "underlined", "circled"] as const;
export const COLORS = ["red", "blue", "green", "yellow"] as const;
export type MarkType = (typeof MARK_TYPES)[number];
export type Color = (typeof COLORS)[number];

export type Chan = { adapter: string; channel: string };
export type Span = Chan & { lo: number; hi: number };

export function qoverChannel(t: MarkType, c: Color): string {
  return `${t} in ${c}`;
}

export function parseQover(
  channel: string
): { t: MarkType; c: Color } | null {
  const m = channel.match(/^(\w+) in (\w+)$/);
  if (!m) return null;
  const t = m[1] as MarkType;
  const c = m[2] as Color;
  if (!MARK_TYPES.includes(t) || !COLORS.includes(c)) return null;
  return { t, c };
}

// Exclusivity group: the model was trained with <=1 color per mark type per
// token, so applying "underlined in blue" must evict any other underline
// color from the painted range. Binary adapters (inject) are one group.
export function groupKey(s: Chan): string {
  const p = parseQover(s.channel);
  return p ? `${s.adapter}:${p.t}` : s.adapter;
}

// Shrink [lo, hi) to exclude boundary whitespace. Critical for tokenizer
// alignment: BPE attaches the space between words to the FOLLOWING word's
// token, so a span piece that ends on that space still claims the next
// word's token — and two same-type spans abutting at a word boundary would
// collide on it (the backend 400s loudly on exactly this).
function tighten(
  text: string,
  lo: number,
  hi: number,
  s: Span
): Span | null {
  while (lo < hi && /\s/.test(text[lo])) lo++;
  while (hi > lo && /\s/.test(text[hi - 1])) hi--;
  if (lo >= hi) return null;
  return { ...s, lo, hi };
}

// Cut [lo, hi) out of a span, returning the tightened surviving pieces.
function cut(text: string, s: Span, lo: number, hi: number): Span[] {
  if (s.hi <= lo || s.lo >= hi) return [s];
  const out: Span[] = [];
  if (s.lo < lo) {
    const p = tighten(text, s.lo, lo, s);
    if (p) out.push(p);
  }
  if (s.hi > hi) {
    const p = tighten(text, hi, s.hi, s);
    if (p) out.push(p);
  }
  return out;
}

export function applyMark(
  text: string,
  spans: Span[],
  mark: Chan,
  lo: number,
  hi: number
): Span[] {
  const painted = tighten(text, lo, hi, { ...mark, lo, hi });
  if (!painted) return spans;
  const g = groupKey(mark);
  const out = spans.flatMap((s) =>
    groupKey(s) === g ? cut(text, s, painted.lo, painted.hi) : [s]
  );
  out.push(painted);
  return out.sort((a, b) => a.lo - b.lo || a.hi - b.hi);
}

export function clearRange(
  text: string,
  spans: Span[],
  lo: number,
  hi: number
): Span[] {
  return spans.flatMap((s) => cut(text, s, lo, hi));
}

// Snap a raw drag range to word boundaries: the emplaced span runs from the
// start of the first word the drag touches to the end of the last. A drag
// (or click) that touches no word returns null.
export function wordSnap(
  text: string,
  a: number,
  b: number
): [number, number] | null {
  const lo0 = Math.max(0, Math.min(a, b));
  const hi0 = Math.min(text.length, Math.max(a, b));
  let lo: number | null = null;
  let hi: number | null = null;
  const re = /\S+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    const ws = m.index;
    const we = ws + m[0].length;
    if (lo === null && we > lo0) lo = ws;
    if (ws <= hi0) hi = we;
  }
  if (lo === null || hi === null || lo >= hi) return null;
  return [lo, hi];
}

export type SegMark = { span: Span; start: boolean; end: boolean };
export type Seg = { lo: number; hi: number; text: string; marks: SegMark[] };

// Flat decomposition of the text at every span boundary; each segment lists
// the spans covering it (with start/end flags so circles can round only
// their outer corners). Overlaps of different mark types compose freely.
export function segment(text: string, spans: Span[]): Seg[] {
  const cuts = new Set<number>([0, text.length]);
  for (const s of spans) {
    cuts.add(Math.max(0, Math.min(text.length, s.lo)));
    cuts.add(Math.max(0, Math.min(text.length, s.hi)));
  }
  const pts = [...cuts].sort((a, b) => a - b);
  const segs: Seg[] = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const lo = pts[i];
    const hi = pts[i + 1];
    if (hi <= lo) continue;
    const marks = spans
      .filter((s) => s.lo < hi && s.hi > lo)
      .map((span) => ({ span, start: span.lo === lo, end: span.hi === hi }));
    segs.push({ lo, hi, text: text.slice(lo, hi), marks });
  }
  return segs;
}

// Carry spans across a text edit instead of wiping them. Each span is
// re-located by its own snippet: unique match -> remapped; ambiguous match
// disambiguated by preceding context; no match (the marked words themselves
// were edited) -> that span alone is dropped. Editing an unrelated sentence
// never costs you your marks.
export function remapSpans(
  oldText: string,
  newText: string,
  spans: Span[]
): Span[] {
  if (oldText === newText) return spans;
  const out: Span[] = [];
  for (const s of spans) {
    const snippet = oldText.slice(s.lo, s.hi);
    if (!snippet.trim()) continue;
    const first = newText.indexOf(snippet);
    if (first < 0) continue;
    let lo = first;
    if (newText.indexOf(snippet, first + 1) >= 0) {
      // ambiguous: anchor on up to 24 chars of preceding context
      const ctx = oldText.slice(Math.max(0, s.lo - 24), s.lo);
      const hit = ctx ? newText.indexOf(ctx + snippet) : -1;
      if (hit >= 0) lo = hit + ctx.length;
    }
    out.push({ ...s, lo, hi: lo + snippet.length });
  }
  // a remap can land two spans of one exclusivity group on the same range;
  // keep the first and drop exact-duplicate ranges within a group
  const seen = new Set<string>();
  return out
    .sort((a, b) => a.lo - b.lo || a.hi - b.hi)
    .filter((s) => {
      const k = `${groupKey(s)}:${s.lo}:${s.hi}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
}
