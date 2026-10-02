python3 - <<'PYEOF'
import re
from pathlib import Path
p = Path("/private/tmp/claude-501/-Users-joshuapenman-sandbox-inference-goggles/ee22aab6-3eed-426a-870a-0026c5bd22b6/scratchpad/merge_frames.py")
s = p.read_text()
s = s.replace(
 '            out = tmpl.format(p="TESTPAYLOAD?", p_lower="tESTPAYLOAD?")',
 '            # first char must be insensitive to {p_lower} lowercasing it\n'
 '            out = tmpl.format(p="ZZTESTPAYLOAD?", p_lower="zZTESTPAYLOAD?")')
p.write_text(s)
print("validator fixed")
PYEOF
python3 /private/tmp/claude-501/-Users-joshuapenman-sandbox-inference-goggles/ee22aab6-3eed-426a-870a-0026c5bd22b6/scratchpad/merge_frames.py
