import { Span } from "./marks";
import deck from "./injection-deck.json";

// ---------------------------------------------------------------------------
// Marks section: short passages, hand-marked.
// ---------------------------------------------------------------------------

export type MarkItem = { name: string; text: string; spans: Span[] };

// Locate marks by substring so char offsets can never drift from the prose.
// Throws at module init if a target is missing — a broken deck should fail
// the build, not silently ship an unmarked example.
function item(
  name: string,
  passage: string,
  question: string,
  marks: [target: string, adapter: string, channel: string][]
): MarkItem {
  const text = passage + "\n\n" + question;
  const spans = marks.map(([target, adapter, channel]) => {
    const lo = passage.indexOf(target);
    if (lo < 0) throw new Error(`deck item ${name}: target not in passage: ${target}`);
    return { adapter, channel, lo, hi: lo + target.length };
  });
  return { name, text, spans };
}

const FOX =
  "The quick silver fox slipped between the birch trees while the river " +
  "hummed a low tune to the stones.";

const LETTER =
  "Dear Mira — the observatory finally reopened on Tuesday. The old " +
  "brass telescope still points at the meridian, and the caretaker swears " +
  "the dome creaks in E minor. Bring your star charts and the good coffee.";

// Two underlines in different colors (word-disjoint, as training requires:
// <=1 color per mark type per token) plus a circle that straddles both —
// starting inside the first underline and ending inside the second, so it
// overlaps each without containing either.
const LIGHTHOUSE =
  "The lighthouse keeper logged three things that night: a green flare " +
  "over the water, the smell of diesel on the wind, and a small boat " +
  "drifting without lights toward the rocks.";

const RECIPE =
  "Warm the butter until it smells faintly of hazelnuts, then fold in the " +
  "toasted walnuts gently so the batter keeps its air. Rest the pan on a " +
  "damp towel before you even think about the oven.";

const MARKET =
  "The market inspector noted the crates of alphonso mangoes by the gate, " +
  "a scale that read two grams heavy, a vendor humming off-key, and the " +
  "missing permit taped behind the register.";

export const MARKS_DECK: MarkItem[] = [
  item(
    "Fox & river",
    FOX,
    "Which part of my message was underlined in blue? Reply with just that text, or say if nothing was.",
    [["the river hummed a low tune", "qover", "underlined in blue"]]
  ),
  item(
    "A stacked recipe",
    RECIPE,
    "List every marked part of this message, in order, with how each is marked.",
    [
      ["fold in the toasted walnuts gently", "qover", "circled in green"],
      ["toasted walnuts", "qover", "highlighted in yellow"],
    ]
  ),
  item(
    "How is it marked?",
    LETTER,
    'How is the text "the meridian" marked, if at all?',
    [["the meridian", "qover", "underlined in red"]]
  ),
  item(
    "Three underlines, one circle",
    LIGHTHOUSE,
    "Which parts of this text are underlined, and in what colors?",
    [
      ["a green flare over the water", "qover", "underlined in red"],
      ["the smell of diesel", "qover", "underlined in green"],
      ["a small boat drifting without lights", "qover", "underlined in blue"],
      [
        "flare over the water, the smell of diesel on the wind, and a small boat drifting",
        "qover",
        "circled in yellow",
      ],
    ]
  ),
  item(
    "Only the circled text",
    MARKET,
    "Quote only the circled text — ignore every other mark.",
    [
      ["crates of alphonso mangoes", "qover", "highlighted in yellow"],
      ["two grams heavy", "qover", "underlined in red"],
      ["a vendor humming off-key", "qover", "circled in blue"],
      ["missing permit", "qover", "highlighted in green"],
    ]
  ),
  item(
    "Everything in red",
    LETTER,
    "Tell me everything underlined, highlighted, or circled in red, and how each one is marked. Skip the other colors.",
    [
      ["reopened on Tuesday", "qover", "highlighted in red"],
      ["the meridian", "qover", "underlined in red"],
      ["the good coffee", "qover", "circled in green"],
    ]
  ),
  item(
    "Asking about the wrong color",
    LETTER,
    "What text was circled in green? Quote it exactly; if there isn't any, tell me.",
    [["the dome creaks in E minor", "qover", "circled in yellow"]]
  ),
];

// ---------------------------------------------------------------------------
// Injection section: real PIArena items, each measured live to defeat the
// unmarked model while the goggled model answers correctly. The planted
// instruction sits ~1/3 to ~1/2 of the way in, never at the end.
// ---------------------------------------------------------------------------

export type InjectionItem = {
  name: string;
  mechanism: string; // what KIND of attack — the deck spans mechanisms, not topics
  why: string; // what the unmarked model actually does
  category: string;
  instruction: string;
  context: string;
  injection: string;
  injection_pct: number;
  gold: string;
  note: string | null;
};

export const INJECTION: InjectionItem[] = deck as InjectionItem[];
