// Secret-instructions + Python-colored-glasses channel model.
// Channel strings must match the backend ADAPTERS["behav"].channels exactly;
// bit order is the training bit order (transforms 0-7, languages 8-11).

import instrDeckRaw from "./instructions-deck.json";
import langDeckRaw from "./languages-deck.json";

export type Instruction = {
  channel: string; // backend channel name
  label: string; // chip label
  hue: string; // css color for stamps
};

// Best-compliance four of the eight trained transforms (decline 100%,
// nested bullets 100%, caps 89%, German 81%); the rest stay trained and
// servable, just not offered in the demo palette.
export const INSTRUCTIONS: Instruction[] = [
  { channel: "decline", label: "Decline (safety filter)", hue: "#dc322f" },
  { channel: "hypernested", label: "Nested bullets", hue: "#859900" },
  { channel: "caps", label: "ALL CAPS", hue: "#b58900" },
  { channel: "german", label: "German", hue: "#6c71c4" },
];

// Served by ADAPTERS["plr"] (rosetta-v3d-phrase128); channel strings must
// match that registry entry exactly.
export const LANGUAGES: Instruction[] = [
  { channel: "Python", label: "Python", hue: "#268bd2" },
  { channel: "JavaScript", label: "JavaScript", hue: "#6c71c4" },
  { channel: "Ruby", label: "Ruby", hue: "#2aa198" },
  { channel: "C", label: "C", hue: "#b58900" },
];

export function hueOf(channel: string): string {
  const all = [...INSTRUCTIONS, ...LANGUAGES];
  return all.find((i) => i.channel === channel)?.hue ?? "#93a1a1";
}

export type ClickSpan = { lo: number; hi: number; channel?: string };
export type InstrItem = {
  name: string;
  item_id: string;
  text: string;
  spans: { lo: number; hi: number }[];
};
export type LangItem = {
  name: string;
  example_id: string;
  qtype: string;
  snippets: { body: string; channel: string | null }[];
  question: string;
};

export const INSTR_DECK = instrDeckRaw as InstrItem[];
export const LANG_DECK = langDeckRaw as LangItem[];

// Instructions are ONLY the trained channels above. Free text refers to the
// TARGET passage: a user-typed prompt has no pre-segmented request spans, so
// the editor switches from click-to-stamp to marks-style drag selection —
// same instruction palette, same span JSON either way.
export type BehavSpan = {
  adapter: "behav" | "plr";
  lo: number;
  hi: number;
  channel: string;
};
