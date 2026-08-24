# Semantic Overlays — web demo

The interactive demo: mark spans of a prompt with trained overlays
(non-executable, visual marks, asserted programming language, carried
instructions) and compare the frozen model's behavior with and without
the mark. All model calls go through a thin Next.js API proxy to a
deployed overlay-serving endpoint; the browser never sees the backend
key.

## Run it yourself

1. **Serve the model with overlays** (repo root):
   download the released adapter checkpoints into the Modal volume, set
   a bearer token, and deploy the goggled endpoint —
   ```
   modal volume put goggles-data <local>/checkpoints /checkpoints
   export GOGGLES_VLLM_API_KEY=<your-token>
   modal deploy infra/goggled_vllm.py
   ```
   The deploy prints the serve-api and health URLs.
2. **Configure the app**: copy `.env.example` to `.env.local` and fill
   in
   ```
   GOGGLES_BACKEND_URL=<serve-api URL>
   GOGGLES_STATUS_URL=<health URL>
   GOGGLES_API_KEY=<your-token>
   ```
3. **Run**: `pnpm install && pnpm dev` (or deploy to any Next.js host
   with the same three environment variables).

The decks under `lib/*.json` are the demo's curated examples; edit them
freely. GPU containers scale to zero when idle; the status endpoint
lets the UI show cold-start state honestly.
