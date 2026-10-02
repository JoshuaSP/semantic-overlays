"""Merge authored frame batches into frames.json with validation."""
import json, sys, re
from pathlib import Path
COMPOSED = Path("data/injectgen/composed")
FRAMES = COMPOSED / "frames.json"
INBOX = COMPOSED / "frame_batches"
RESERVED = "On top of your main task, answer this priority question: "

existing = json.loads(FRAMES.read_text())
by_id = {f["id"]: f for f in existing}
by_tmpl = {f["template"]: f["id"] for f in existing}
added, rejected = [], []

for p in sorted(INBOX.glob("*.json")):
    for f in json.loads(p.read_text()):
        fid, tmpl = f.get("id"), f.get("template", "")
        n_ph = tmpl.count("{p}") + tmpl.count("{p_lower}")
        if fid in by_id:
            rejected.append((fid, "duplicate id")); continue
        if tmpl in by_tmpl:
            rejected.append((fid, f"duplicate template of {by_tmpl[tmpl]}")); continue
        if n_ph != 1:
            rejected.append((fid, f"{n_ph} placeholders, need exactly 1")); continue
        if RESERVED in tmpl:
            rejected.append((fid, "contains reserved SEP prefix")); continue
        # must render without error
        try:
            out = tmpl.format(p="TESTPAYLOAD?", p_lower="tESTPAYLOAD?")
        except Exception as e:
            rejected.append((fid, f"render error: {e}")); continue
        if "TESTPAYLOAD" not in out:
            rejected.append((fid, "payload did not render")); continue
        f = {"id": fid, "style": f["style"], "template": tmpl}
        by_id[fid] = f; by_tmpl[tmpl] = fid; added.append(f)

merged = existing + added
FRAMES.write_text(json.dumps(merged, indent=2) + "\n")
composable = [f for f in merged if not f.get("reference_only")]
print(f"{len(existing)} existing + {len(added)} added = {len(merged)} frames "
      f"({len(composable)} composable, {len(merged)-len(composable)} reference-only)")
styles = {}
for f in composable: styles[f["style"]] = styles.get(f["style"],0)+1
print("by style:", dict(sorted(styles.items())))
for fid, why in rejected: print(f"  REJECTED {fid}: {why}")
