"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

type State = "checking" | "warm" | "booting" | "cold";
type Ctx = { state: State; notifyWake: () => void; since: number | null };

const GpuCtx = createContext<Ctx>({
  state: "checking",
  notifyWake: () => {},
  since: null,
});
export const useGpu = () => useContext(GpuCtx);

// Polls the CPU-only heartbeat endpoint (see /api/status): a poll can
// neither keep the GPU warm nor boot it, so fast honest polling is free.
// Only real requests (generates) reset the backend's 15-min idle clock.
const POLL_WARM_MS = 15_000;
const POLL_BOOT_MS = 2_000; // near-live flip: boots finish faster than they used to

export function GpuProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<State>("checking");
  const [since, setSince] = useState<number | null>(null);
  const booting = useRef(false);
  const warm = useRef(false);

  const probe = useCallback(async () => {
    try {
      const r = await fetch("/api/status", { cache: "no-store" });
      const j = await r.json();
      if (j.warm) {
        booting.current = false;
        warm.current = true;
        setSince(null);
        setState("warm");
        return true;
      }
    } catch {
      /* fall through to not-warm */
    }
    warm.current = false;
    // Not warm: "booting" only if a wake was actually requested, else "cold".
    // We deliberately do NOT auto-wake on page load — a boot costs GPU
    // minutes, so the user (or a Generate click) has to ask for it.
    setState(booting.current ? "booting" : "cold");
    return false;
  }, []);

  const notifyWake = useCallback(() => {
    if (booting.current || warm.current) return;
    booting.current = true;
    setSince(Date.now());
    setState("booting");
    // Fire-and-forget long request that actually holds the boot open — the
    // short status probe can be abandoned before the container starts.
    fetch("/api/status?wake=1", { cache: "no-store" }).catch(() => {});
  }, []);

  useEffect(() => {
    let live = true;
    let t: ReturnType<typeof setTimeout>;
    const loop = async () => {
      const isWarm = await probe();
      if (!live) return;
      t = setTimeout(loop, isWarm ? POLL_WARM_MS : POLL_BOOT_MS);
    };
    loop();
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [probe]);

  return (
    <GpuCtx.Provider value={{ state, notifyWake: notifyWake, since }}>
      {children}
    </GpuCtx.Provider>
  );
}

export function GpuBadge() {
  const { state: liveState, notifyWake, since: liveSince } = useGpu();
  const [, tick] = useState(0);
  // Styling override: ?gpu=cold|booting|warm forces the badge state so the
  // cold/booting layouts can be inspected without a real boot.
  const [forced, setForced] = useState<State | null>(null);
  const [forcedSince] = useState(() => Date.now() - 9_000);
  useEffect(() => {
    const v = new URLSearchParams(window.location.search).get("gpu");
    if (v === "cold" || v === "booting" || v === "warm") setForced(v);
  }, []);
  const state = forced ?? liveState;
  const since = forced === "booting" ? forcedSince : liveSince;

  useEffect(() => {
    if (state !== "booting") return;
    const i = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(i);
  }, [state]);

  if (state === "warm")
    return (
      <span className="badge warm">
        <i className="dot" /> GPU warm
      </span>
    );
  if (state === "booting") {
    return (
      <span className="badge booting">
        <i className="dot pulse" /> GPU booting
      </span>
    );
  }
  if (state === "cold")
    return (
      <button className="badge cold" onClick={notifyWake}>
        <i className="dot" /> GPU cold
      </button>
    );
  return (
    <span className="badge">
      <i className="dot" /> checking…
    </span>
  );
}
