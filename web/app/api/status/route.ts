import { NextRequest } from "next/server";

export const maxDuration = 300;  // hobby-plan ceiling; wake client retries past this

// Two modes:
//   GET /api/status         short probe -> {warm}. Never blocks the UI.
//   GET /api/status?wake=1  long request that actually holds a cold start
//                           open until the container answers.
// The short probe alone is not enough to boot the backend reliably: it is
// abandoned after a few seconds, well before a ~10 min cold start finishes.
export async function GET(req: NextRequest) {
  const url = process.env.GOGGLES_BACKEND_URL;
  const key = process.env.GOGGLES_API_KEY;
  if (!url || !key) {
    return Response.json({ warm: false, error: "env not configured" });
  }
  const wake = req.nextUrl.searchParams.get("wake") === "1";
  // Non-wake probes go to the CPU-only heartbeat endpoint when configured:
  // it reads a Dict the GPU container stamps, so a probe can neither keep
  // the GPU warm nor boot it. Only ?wake=1 touches the GPU app itself.
  const statusUrl = process.env.GOGGLES_STATUS_URL;
  try {
    if (!wake && statusUrl) {
      const r = await fetch(statusUrl, {
        signal: AbortSignal.timeout(5_000),
        cache: "no-store",
      });
      const j = await r.json();
      return Response.json({ warm: !!j.warm });
    }
    const r = await fetch(`${url}/sections`, {
      headers: { authorization: `Bearer ${key}` },
      signal: AbortSignal.timeout(wake ? 780_000 : 3_000),
      cache: "no-store",
    });
    return Response.json({ warm: r.ok });
  } catch {
    return Response.json({ warm: false });
  }
}
