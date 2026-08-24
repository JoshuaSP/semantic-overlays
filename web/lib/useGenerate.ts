"use client";

import { useCallback, useRef, useState } from "react";
import { Span } from "./marks";

export type Pane = { text: string; running: boolean; secs: number | null; truncated?: boolean };
export const EMPTY_PANE: Pane = { text: "", running: false, secs: null };

export type GenArgs = {
  text: string;
  spans: Span[];
  instruction?: string;
  maxNew?: number;
};

/** Streaming generate against the proxy. Returns two panes (goggled / plain)
 *  so a caller can run one or both; both fire concurrently and the engine
 *  batches them, so a comparison costs about one generation of wall clock. */
export function useGenerate(onWake?: () => void) {
  const [on, setOn] = useState<Pane>(EMPTY_PANE);
  const [off, setOff] = useState<Pane>(EMPTY_PANE);
  const abortRef = useRef<AbortController | null>(null);

  const stream = useCallback(
    async (
      args: GenArgs,
      useSpans: boolean,
      set: (f: (p: Pane) => Pane) => void,
      signal: AbortSignal
    ) => {
      const t0 = performance.now();
      set(() => ({ text: "", running: true, secs: null }));
      try {
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            text: args.text,
            spans: useSpans ? args.spans : [],
            instruction: args.instruction,
            max_new: args.maxNew ?? 2048,
          }),
          signal,
        });
        if (!res.ok || !res.body) {
          const detail = await res.text();
          set(() => ({
            text: `⚠ ${res.status}: ${detail}`,
            running: false,
            secs: null,
          }));
          return;
        }
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const events = buf.split("\n\n");
          buf = events.pop() ?? "";
          for (const ev of events) {
            const line = ev.trim();
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6);
            if (payload === "[DONE]") continue;
            const { delta, finish_reason } = JSON.parse(payload);
            if (delta !== undefined)
              set((p) => ({ ...p, text: p.text + delta }));
            if (finish_reason === "length")
              set((p) => ({ ...p, truncated: true }));
          }
        }
        set((p) => ({
          ...p,
          running: false,
          secs: (performance.now() - t0) / 1000,
        }));
      } catch (e) {
        if (!signal.aborted)
          set(() => ({ text: `⚠ ${String(e)}`, running: false, secs: null }));
        else set((p) => ({ ...p, running: false }));
      }
    },
    []
  );

  const run = useCallback(
    async (args: GenArgs, compare: boolean) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      onWake?.();
      setOff(EMPTY_PANE);
      await Promise.all([
        stream(args, true, setOn, ctrl.signal),
        compare
          ? stream(args, false, setOff, ctrl.signal)
          : Promise.resolve(),
      ]);
    },
    [stream, onWake]
  );

  const reset = useCallback(() => {
    setOn(EMPTY_PANE);
    setOff(EMPTY_PANE);
  }, []);

  return { on, off, run, reset, busy: on.running || off.running };
}
