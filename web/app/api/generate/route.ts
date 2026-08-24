import { NextRequest } from "next/server";

// Streamed generations run 30-70s+ (plus cold start); without this the
// hobby default would cut streams mid-answer.
export const maxDuration = 300;

// Streaming proxy to the Modal goggled-vLLM backend. The bearer key lives
// server-side only; the browser never sees it. The upstream SSE body is
// passed through untouched.

const WINDOW_MS = 60_000;
const MAX_PER_WINDOW = 12;
const hits = new Map<string, number[]>();

function rateLimited(ip: string): boolean {
  const now = Date.now();
  const arr = (hits.get(ip) ?? []).filter((t) => now - t < WINDOW_MS);
  if (arr.length >= MAX_PER_WINDOW) return true;
  arr.push(now);
  hits.set(ip, arr);
  return false;
}

export async function POST(req: NextRequest) {
  const url = process.env.GOGGLES_BACKEND_URL;
  const key = process.env.GOGGLES_API_KEY;
  if (!url || !key) {
    return new Response("backend env not configured", { status: 500 });
  }
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0] ?? "local";
  if (rateLimited(ip)) {
    return new Response("rate limited; try again in a minute", { status: 429 });
  }

  const body = await req.json();
  if (typeof body.text !== "string" || !body.text.trim()) {
    return new Response("empty text", { status: 400 });
  }
  const upstream = await fetch(`${url}/generate`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${key}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      text: body.text,
      spans: Array.isArray(body.spans) ? body.spans : [],
      // RAG framing: the task rides in the system prompt. Dropping this
      // silently reads as "the model ignored the question", so it must be
      // forwarded whenever the client sends it.
      ...(body.instruction ? { instruction: body.instruction } : {}),
      // Transform answers median ~1,400 tokens; the old 300-default/1024-clamp
      // silently truncated most multi-request answers (found 2026-08-19).
      max_new: Math.min(Number(body.max_new) || 2048, 4096),
    }),
  });
  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return new Response(`backend ${upstream.status}: ${detail}`, {
      status: upstream.status,
    });
  }
  return new Response(upstream.body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
    },
  });
}
