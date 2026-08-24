"use client";

import { Pane } from "@/lib/useGenerate";

function Body({ p }: { p: Pane }) {
  return (
    <div className="output">
      {p.text || <span className="cursor">▋</span>}
      {p.running && p.text && <span className="cursor">▋</span>}
      {p.truncated && !p.running && (
        <div className="trunc-note">⚠ cut off at the token limit</div>
      )}
    </div>
  );
}

export default function OutputPanes({
  on,
  off,
  offLabel = "unmarked (plain model)",
}: {
  on: Pane;
  off: Pane;
  offLabel?: string;
}) {
  const showOff = off.text !== "" || off.running;
  const showOn = on.text !== "" || on.running;
  if (!showOn && !showOff) return null;
  return (
    <div className={showOff ? "panes two" : "panes"}>
      <div className="pane">
        <div className="pane-head on">
          🖍️ overlays on
          {on.secs !== null && (
            <span className="pane-secs">{on.secs.toFixed(1)}s</span>
          )}
        </div>
        <Body p={on} />
      </div>
      {showOff && (
        <div className="pane">
          <div className="pane-head offhead">
            {offLabel}
            {off.secs !== null && (
              <span className="pane-secs">{off.secs.toFixed(1)}s</span>
            )}
          </div>
          <Body p={off} />
        </div>
      )}
    </div>
  );
}
