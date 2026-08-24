import InjectionSection from "@/components/InjectionSection";
import InstructionsSection from "@/components/InstructionsSection";
import LanguagesSection from "@/components/LanguagesSection";
import MarksSection from "@/components/MarksSection";
import { GpuBadge, GpuProvider } from "@/components/GpuStatus";

export default function Home() {
  return (
    <GpuProvider>
      <main>
        <header className="topbar">
          <div className="brand">
            <span className="brand-goggles">🖍️</span> Semantic Overlays
          </div>
          <nav>
            <a href="#injection">The NX bit</a>
            <a href="#marks">Invisible highlighters</a>
            <a href="#languages">Python-colored gels</a>
            <a href="#instructions">Secret instructions</a>
          </nav>
          <GpuBadge />
        </header>

        <div className="intro">
          <h1>
            Introducing <em>Semantic Overlays.</em>
          </h1>
          <p>
            Everything a language model sees is tokens, and everything between
            the special tokens is just text. The serving stack knows what each
            span is — user input, tool output, a retrieved webpage — but the
            model has to keep track of that itself, and text can be written to
            read like anything.
          </p>
          <p>
            Semantic Overlays add a second channel: small trained adapters on a
            frozen model that fire only at marked token positions, annotating a
            span in the residual stream itself. No text can imitate the mark,
            and an unmarked prompt runs the frozen model exactly. Everything
            below runs live.
          </p>
        </div>

        <InjectionSection />
        <MarksSection />
        <LanguagesSection />
        <InstructionsSection />

        <footer>
          Qwen3.5-9B, frozen; per-position adapters distilled per channel. One
          GPU, continuous batching, request-scoped overlays.
        </footer>
      </main>
    </GpuProvider>
  );
}
