"""Goggle modules, layer hooks, and the top-K+tail KL loss. Torch-side (Modal)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoggleAdapter(nn.Module):
    """SwiGLU bottleneck on an RMS-normed input: W_out(SiLU(W_gate n(h)) * W_in n(h)).

    W_out zero-init (exact no-op at start). The input RMSNorm is essential:
    the bilinear SwiGLU is quadratic in its input, and residual streams carry
    massive outlier activations (~1e4) — unnormed, deltas compound through
    depth and overflow bf16.
    """

    def __init__(self, d_model, d_hidden=32):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_in = nn.Linear(d_model, d_hidden, bias=False)
        self.w_out = nn.Linear(d_hidden, d_model, bias=False)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, h):
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.w_out(F.silu(self.w_gate(hn)) * self.w_in(hn))


class Goggles(nn.Module):
    """One adapter per decoder layer, applied to the layer INPUT at masked positions.

    Usage:
        goggles = Goggles(n_layers, d_model).to(device)   # fp32 params
        handles = goggles.attach(model)                    # pre-forward hooks
        goggles.set_mask(mask_bool_1xS)                    # None disables (decode)
        ... forward/backward ...
        goggles.detach(handles)
    """

    def __init__(self, n_layers, d_model, d_hidden=32):
        super().__init__()
        self.adapters = nn.ModuleList(GoggleAdapter(d_model, d_hidden) for _ in range(n_layers))
        self._mask = None
        self.enabled = True

    def set_mask(self, mask):  # bool tensor [B, S] or None
        self._mask = mask

    def _make_hook(self, idx):
        def hook(module, args, kwargs):
            if not self.enabled or self._mask is None:
                return None
            if args:
                h, rest = args[0], args[1:]
                from_kwargs = False
            else:
                h, rest = kwargs["hidden_states"], ()
                from_kwargs = True
            if h.shape[1] != self._mask.shape[1]:
                raise RuntimeError(
                    f"goggle mask length {self._mask.shape[1]} != seq length {h.shape[1]}; "
                    "set_mask(None) during decode steps")
            m = self._mask.to(h.device)
            delta = self.adapters[idx](h[m].to(torch.float32)).to(h.dtype)
            h = h.clone()
            h[m] = h[m] + delta
            if from_kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = h
                return args, kwargs
            return (h, *rest), kwargs
        return hook

    def attach(self, hf_model):
        layers = hf_model.model.layers
        if len(layers) != len(self.adapters):
            raise RuntimeError(f"{len(layers)} model layers vs {len(self.adapters)} adapters")
        return [layer.register_forward_pre_hook(self._make_hook(i), with_kwargs=True)
                for i, layer in enumerate(layers)]

    @staticmethod
    def detach(handles):
        for h in handles:
            h.remove()


class QualityGoggles(nn.Module):
    """Quality-readout ablation: one channel per QUALITY (mark x color), applied
    at marked positions at every decoder-layer input. Three arms:

      arch="vec"      one vector V[q] in R^d, the SAME vector added at every
                      layer (re-added each layer input, so the stream
                      accumulates it through depth)
      arch="layervec" V[l, q] — a different vector per layer
      arch="mlp"      per (layer, quality) SwiGLU bottleneck (input RMSNorm —
                      required, see GoggleAdapter — w_out zero-init),
                      position-dependent delta

    All arms zero-init -> exact no-op at start. Mask = int tensor [B, S]:
    -1 = unmarked, q = quality index. Interface matches Goggles."""

    def __init__(self, n_layers, d_model, n_qualities, arch, d_hidden=8, layer_ids=None):
        super().__init__()
        if arch not in ("vec", "layervec", "mlp"):
            raise ValueError(f"arch must be vec|layervec|mlp, got {arch!r}")
        self.arch = arch
        # layer_ids: which decoder layers get the channel (None = all). The
        # slot dimension of v/adapters indexes INTO layer_ids, not raw layers.
        self.layer_ids = list(range(n_layers)) if layer_ids is None else list(layer_ids)
        L = len(self.layer_ids)
        if arch == "vec":
            self.v = nn.Parameter(torch.zeros(n_qualities, d_model))
        elif arch == "layervec":
            self.v = nn.Parameter(torch.zeros(L, n_qualities, d_model))
        else:
            self.adapters = nn.ModuleList(
                nn.ModuleList(GoggleAdapter(d_model, d_hidden) for _ in range(n_qualities))
                for _ in range(L))
        self.n_layers = n_layers
        self._assign = None
        self.enabled = True

    def set_mask(self, assign):  # int tensor [B, S] (-1 off / q = quality) or None
        if assign is not None and assign.dtype == torch.bool:
            raise TypeError("QualityGoggles.set_mask takes quality indices (-1 = off)")
        self._assign = assign

    def _make_hook(self, idx):
        def hook(module, args, kwargs):
            if not self.enabled or self._assign is None:
                return None
            if args:
                h, rest = args[0], args[1:]
                from_kwargs = False
            else:
                h, rest = kwargs["hidden_states"], ()
                from_kwargs = True
            if h.shape[1] != self._assign.shape[1]:
                raise RuntimeError(
                    f"goggle mask length {self._assign.shape[1]} != seq length {h.shape[1]}; "
                    "set_mask(None) during decode steps")
            assign = self._assign.to(h.device)
            m = assign >= 0
            if not bool(m.any()):
                return None
            qs = assign[m]                                        # [N]
            if self.arch == "vec":
                delta = self.v[qs]
            elif self.arch == "layervec":
                delta = self.v[idx][qs]
            else:
                h_m = h[m].to(torch.float32)
                delta = torch.zeros_like(h_m)
                for q in qs.unique().tolist():
                    sel = qs == q
                    delta[sel] = self.adapters[idx][q](h_m[sel])
            h = h.clone()
            h[m] = h[m] + delta.to(h.dtype)
            if from_kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = h
                return args, kwargs
            return (h, *rest), kwargs
        return hook

    def attach(self, hf_model):
        layers = hf_model.model.layers
        if len(layers) != self.n_layers:
            raise RuntimeError(f"{len(layers)} model layers vs {self.n_layers} goggle layers")
        return [layers[l].register_forward_pre_hook(self._make_hook(slot), with_kwargs=True)
                for slot, l in enumerate(self.layer_ids)]

    detach = Goggles.detach


class CodeAdapter(nn.Module):
    """SwiGLU bottleneck on [n(h); e] where e is a SMALL learnable code
    (d_code, not d_model). The dictionary arm's shared machinery: instruction/
    quality identity enters only as (w_gate·e, w_in·e) — a chosen low-dim
    handle, placed by SGD rather than inherited from frozen phrase geometry.
    No norm on e: it is a free parameter and learns its own scale.
    w_out zero-init -> exact no-op regardless of e."""

    def __init__(self, d_model, d_code, d_hidden):
        super().__init__()
        self.w_gate = nn.Linear(d_model + d_code, d_hidden, bias=False)
        self.w_in = nn.Linear(d_model + d_code, d_hidden, bias=False)
        self.w_out = nn.Linear(d_hidden, d_model, bias=False)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, h, e):  # h [N, d_model], e [N, d_code]
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        x = torch.cat([hn, e], -1)
        return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))


class OverlapGoggles(nn.Module):
    """Multi-hot marks: one position may carry SEVERAL qualities at once
    (upstream constraint: <=1 color per mark type -> <=3 marks/token).

    Mask = int tensor [B, S] BITMASK: bit q set = quality q active, 0 = off.
    The delta at a position is the SUM of the active qualities' channel
    outputs, each computed from the same pre-edit state — parallel-in-layer
    composition, order-invariant by construction; marks interact only through
    depth (A's layer-8 write changes what B's layer-9 goggle reads).

    arms:
      layervec  V[l, q, d] summed over active bits        (composition control)
      mlp       per (layer, quality) SwiGLU-d_hidden on n(h)   — arm A, lookup
      dict      ONE shared per-layer SwiGLU on [n(h); e], e = learnable
                per-layer d_code code per quality (random init)  — arm B
      phrase    shared per-layer SwiGLU on [n(h); n(e)], e = FROZEN pooled
                phrase embedding [n_q, d_model] via set_phrases   — arm C

    All arms zero-init -> exact no-op at start. Interface matches
    QualityGoggles (set_mask / enabled / attach / detach)."""

    def __init__(self, n_layers, d_model, n_qualities, arch, d_hidden=128,
                 d_code=128, layer_ids=None):
        super().__init__()
        if arch not in ("layervec", "mlp", "dict", "phrase"):
            raise ValueError(f"arch must be layervec|mlp|dict|phrase, got {arch!r}")
        self.arch = arch
        self.layer_ids = list(range(n_layers)) if layer_ids is None else list(layer_ids)
        L = len(self.layer_ids)
        self.n_qualities = n_qualities
        if arch == "layervec":
            self.v = nn.Parameter(torch.zeros(L, n_qualities, d_model))
        elif arch == "mlp":
            self.adapters = nn.ModuleList(
                nn.ModuleList(GoggleAdapter(d_model, d_hidden) for _ in range(n_qualities))
                for _ in range(L))
        elif arch == "dict":
            self.shared = nn.ModuleList(
                CodeAdapter(d_model, d_code, d_hidden) for _ in range(L))
            g = torch.Generator().manual_seed(1234)
            self.codes = nn.Parameter(
                torch.randn(L, n_qualities, d_code, generator=g) / d_code ** 0.5)
        else:  # phrase
            self.shared = nn.ModuleList(
                ConditionedGoggleAdapter(d_model, d_hidden) for _ in range(L))
            self.register_buffer("phrases", torch.zeros(n_qualities, d_model),
                                 persistent=False)
            self._phrases_set = False
        self.n_layers = n_layers
        self._mhot = None
        self.enabled = True

    def set_phrases(self, e):  # [n_qualities, d_model] frozen embeddings
        if e.shape != self.phrases.shape:
            raise ValueError(f"phrases {tuple(e.shape)} != {tuple(self.phrases.shape)}")
        self.phrases.copy_(e.to(self.phrases.device, torch.float32))
        self._phrases_set = True

    def set_mask(self, mhot):  # int tensor [B, S] bitmask, or None
        if mhot is not None:
            if mhot.dtype == torch.bool:
                raise TypeError("OverlapGoggles.set_mask takes a quality BITMASK")
            if int(mhot.max()) >= (1 << self.n_qualities):
                raise ValueError(f"bitmask value {int(mhot.max())} out of range "
                                 f"for {self.n_qualities} qualities")
        self._mhot = mhot

    def _channel(self, idx, q, h_q):
        if self.arch == "layervec":
            return self.v[idx, q].expand_as(h_q)
        if self.arch == "mlp":
            return self.adapters[idx][q](h_q)
        if self.arch == "dict":
            return self.shared[idx](h_q, self.codes[idx, q].expand(len(h_q), -1))
        if not self._phrases_set:
            raise RuntimeError("phrase arm: call set_phrases before forward")
        return self.shared[idx](h_q, self.phrases[q].expand(len(h_q), -1))

    def _make_hook(self, idx):
        def hook(module, args, kwargs):
            if not self.enabled or self._mhot is None:
                return None
            if args:
                h, rest = args[0], args[1:]
                from_kwargs = False
            else:
                h, rest = kwargs["hidden_states"], ()
                from_kwargs = True
            if h.shape[1] != self._mhot.shape[1]:
                raise RuntimeError(
                    f"goggle mask length {self._mhot.shape[1]} != seq length "
                    f"{h.shape[1]}; set_mask(None) during decode steps")
            mhot = self._mhot.to(h.device)
            m = mhot > 0
            if not bool(m.any()):
                return None
            bits = mhot[m]                                    # [N] int
            h_m = h[m].to(torch.float32)
            delta = torch.zeros_like(h_m)
            for q in range(self.n_qualities):
                sel = ((bits >> q) & 1) == 1
                if bool(sel.any()):
                    delta[sel] = delta[sel] + self._channel(idx, q, h_m[sel])
            h = h.clone()
            h[m] = h[m] + delta.to(h.dtype)
            if from_kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = h
                return args, kwargs
            return (h, *rest), kwargs
        return hook

    attach = QualityGoggles.attach
    detach = Goggles.detach


class ConceptPooler(nn.Module):
    """Pool a concept phrase's final hidden states into one vector.

    mode="attn": learned-query attention. A Q/K layer with a single learned
      query collapses algebraically to one vector (q^T W_k h_j = (W_k^T q).h_j),
      so the pooler is just w in R^d: scores = softmax(h w / sqrt(d)). Zero-init
      w -> exactly uniform mean-pool at start.
    mode="last": the final token's state. In a causal LM this has attended to
      the whole phrase, and measured on our bank it separates minimal pairs
      about 2x better than mean-pooling (spanish/french relative distance
      0.10 -> 0.21) because mean-pooling dilutes the one discriminative token.
    """

    def __init__(self, d_model, mode="attn"):
        super().__init__()
        self.mode = mode
        self.w = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, hs):  # [T, d] -> [d]
        if self.mode == "last":
            return hs[-1]
        s = torch.softmax(hs @ self.w[0] / hs.shape[-1] ** 0.5, dim=0)
        return (s[:, None] * hs).sum(0)


class ConditionedGoggleAdapter(nn.Module):
    """SwiGLU bottleneck on [n(h) ; n(e)] — concept embedding e concatenated
    per position, doubling the input dim. e gets its own RMSNorm for the same
    reason h does: final hidden states carry outliers, and SwiGLU is bilinear.
    w_out zero-init -> exact no-op at start regardless of e."""

    def __init__(self, d_model, d_hidden=32, bias=False):
        super().__init__()
        # bias matters once the concept embeddings are centered: uncentered, the
        # shared mean direction supplies a de-facto constant term (W_e.c), and
        # centering deletes it. w_out stays bias-free so zero-init is an exact no-op.
        self.w_gate = nn.Linear(2 * d_model, d_hidden, bias=bias)
        self.w_in = nn.Linear(2 * d_model, d_hidden, bias=bias)
        self.w_out = nn.Linear(d_hidden, d_model, bias=False)
        nn.init.zeros_(self.w_out.weight)
        if bias:
            nn.init.zeros_(self.w_gate.bias); nn.init.zeros_(self.w_in.bias)

    def forward(self, h, e):  # h [N, d], e [N, d] (per-position concept embedding)
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        en = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)
        x = torch.cat([hn, en], -1)
        return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))


class ConditionedGoggles(nn.Module):
    """Concept-conditioned variant of Goggles with PER-POSITION concepts, so one
    sequence can carry several differently-marked spans at once.

    Usage:
        goggles.set_concepts([hs_a, hs_b])   # each [T_k, d] fp32, frozen-model states
        goggles.set_mask(assign)             # int tensor [B, S]: -1 = off, k = concept k
        ... forward/backward ...
        goggles.set_mask(None)               # decode steps

    The shared pooler runs inside every hook (per active concept) so its
    gradient flows through all layers and gradient-checkpoint recompute stays
    correct — K<=3 concepts x <=8 tokens, cost is nil."""

    def __init__(self, n_layers, d_model, d_hidden=32, pool_mode="attn", bias=False):
        super().__init__()
        self.adapters = nn.ModuleList(
            ConditionedGoggleAdapter(d_model, d_hidden, bias=bias) for _ in range(n_layers))
        self.pooler = ConceptPooler(d_model, mode=pool_mode)
        self._assign = None
        self._concept_hs = None
        self.enabled = True

    def set_mask(self, assign):  # int tensor [B, S] (-1 off / k = concept idx) or None
        if assign is not None and assign.dtype == torch.bool:
            raise TypeError("ConditionedGoggles.set_mask takes concept indices "
                            "(-1 = off), not a bool mask")
        self._assign = assign

    def set_concepts(self, hs_list):  # list of fp32 [T_k, d] tensors, or None
        self._concept_hs = hs_list

    def _make_hook(self, idx):
        def hook(module, args, kwargs):
            if not self.enabled or self._assign is None:
                return None
            if args:
                h, rest = args[0], args[1:]
                from_kwargs = False
            else:
                h, rest = kwargs["hidden_states"], ()
                from_kwargs = True
            if h.shape[1] != self._assign.shape[1]:
                raise RuntimeError(
                    f"goggle mask length {self._assign.shape[1]} != seq length {h.shape[1]}; "
                    "set_mask(None) during decode steps")
            assign = self._assign.to(h.device)
            m = assign >= 0
            k = int(assign.max().item()) + 1
            if self._concept_hs is None or k > len(self._concept_hs):
                raise RuntimeError(f"mask references concept {k - 1} but "
                                   f"{0 if self._concept_hs is None else len(self._concept_hs)} "
                                   "concepts are set; call set_concepts first")
            es = torch.stack([self.pooler(hs.to(h.device)) for hs in self._concept_hs[:k]])
            e_per = es[assign[m]]  # [N, d] — each position gets its own concept's embedding
            delta = self.adapters[idx](h[m].to(torch.float32), e_per).to(h.dtype)
            h = h.clone()
            h[m] = h[m] + delta
            if from_kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = h
                return args, kwargs
            return (h, *rest), kwargs
        return hook

    def attach(self, hf_model):
        layers = hf_model.model.layers
        if len(layers) != len(self.adapters):
            raise RuntimeError(f"{len(layers)} model layers vs {len(self.adapters)} adapters")
        return [layer.register_forward_pre_hook(self._make_hook(i), with_kwargs=True)
                for i, layer in enumerate(layers)]

    detach = Goggles.detach


def rope_partial(x, cos, sin):
    """Qwen3.5 partial rotary: rotate the first cos.shape[-1] dims of x, pass the rest.

    x [N, H, hd], cos/sin [N, rd] (the per-position rows of the model's own
    position_embeddings, which already include attention_scaling)."""
    rd = cos.shape[-1]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    c, s = cos[:, None, :], sin[:, None, :]
    x1, x2 = x_rot[..., : rd // 2], x_rot[..., rd // 2:]
    rotated = torch.cat((-x2, x1), -1)
    return torch.cat([x_rot * c + rotated * s, x_pass], -1)


def qwen35_phrase_kv(attn, h, cos, sin):
    """K/V for a phrase exactly as Qwen3_5Attention computes them: k_proj ->
    per-head k_norm -> partial RoPE at the phrase's own positions (0..m-1).

    h [1, m, d] post-input_layernorm states, cos/sin [1, m, rd].
    Returns (K [m, n_kv, hd], V [m, n_kv, hd]) in h.dtype."""
    hd = attn.head_dim
    k = attn.k_norm(attn.k_proj(h).view(h.shape[0], h.shape[1], -1, hd))[0]
    v = attn.v_proj(h).view(h.shape[0], h.shape[1], -1, hd)[0]
    return rope_partial(k, cos[0], sin[0]), v


class SideAttnGoggles(nn.Module):
    """Out-of-band instruction channel via SIDE ATTENTION at the full-attention
    layers (Qwen3.5 is hybrid: only every 4th layer has softmax attention).

    Each goggled position runs a SECOND attention computation with its own
    softmax: its real (RoPE'd, q-normed, per-head) query against the frozen
    K/V of the instruction phrase, computed by the frozen model at the SAME
    layer with the phrase at positions 0..m-1 — i.e. the instruction is
    represented exactly as if it sat at the top of the document, but it is
    never in the token stream. The result is gated per head and added to the
    layer's attention output at that position only.

    Nothing instruction-specific is learned: instruction identity rides on the
    phrase's own KV (no low-dimensional bottleneck to collapse — the fix for
    the rank-2 control-code failure). Learned, shared across all instructions:
      - per-layer LoRA on o_proj (B zero-init: opens at the model's own O)
      - per-layer gate MLP d_model -> d_gate -> n_heads, final layer zero-init
        -> exact no-op at start; per-head so each head's contribution is
        individually controllable
    gate_override (float) bypasses the gate MLP with a fixed uniform scale —
    the zero-training probe: with LoRA at zero this is the pure frozen model
    reading the phrase through its own attention machinery.

    Interface matches ConditionedGoggles (set_concepts / set_mask / enabled /
    attach / detach) so the teacher-forced KL harness drops in. set_concepts
    takes a list of (K, V) pairs shaped [L_full, m, n_kv, hd]."""

    def __init__(self, layer_ids, d_model, n_heads, head_dim, lora_rank=32, d_gate=16):
        super().__init__()
        self.layer_ids = list(layer_ids)
        self.n_heads, self.head_dim = n_heads, head_dim
        d_attn = n_heads * head_dim
        L = len(self.layer_ids)
        self.lora_a = nn.ModuleList(nn.Linear(d_attn, lora_rank, bias=False) for _ in range(L))
        self.lora_b = nn.ModuleList(nn.Linear(lora_rank, d_model, bias=False) for _ in range(L))
        self.gate = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_gate), nn.SiLU(), nn.Linear(d_gate, n_heads))
            for _ in range(L))
        for i in range(L):
            nn.init.zeros_(self.lora_b[i].weight)
            nn.init.zeros_(self.gate[i][2].weight)
            nn.init.zeros_(self.gate[i][2].bias)
        self._assign = None
        self._concepts = None
        self.gate_override = None
        self.enabled = True

    def set_mask(self, assign):  # int tensor [B, S] (-1 off / k = concept idx) or None
        if assign is not None and assign.dtype == torch.bool:
            raise TypeError("SideAttnGoggles.set_mask takes concept indices (-1 = off)")
        self._assign = assign

    def set_concepts(self, kv_list):  # list of (K [L,m,n_kv,hd], V [L,m,n_kv,hd])
        if kv_list is not None and any(kv is None for kv in kv_list):
            raise RuntimeError("set_concepts got a None slot — an example references an "
                               "instruction with no KV bank entry (dropped from the grid)")
        self._concepts = kv_list

    def _make_hook(self, slot):
        def hook(module, args, kwargs, output):
            if not self.enabled or self._assign is None:
                return None
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            if h.shape[1] != self._assign.shape[1]:
                raise RuntimeError(
                    f"goggle mask length {self._assign.shape[1]} != seq length {h.shape[1]}; "
                    "set_mask(None) during decode steps")
            assign = self._assign.to(h.device)
            m = assign >= 0
            if not bool(m.any()):
                return None
            if self._concepts is None or int(assign.max()) >= len(self._concepts):
                raise RuntimeError(f"mask references concept {int(assign.max())} but "
                                   f"{0 if self._concepts is None else len(self._concepts)} "
                                   "concepts are set; call set_concepts first")
            cos, sin = kwargs["position_embeddings"]
            B = h.shape[0]
            if cos.shape[0] == 1 and B > 1:
                cos, sin = cos.expand(B, -1, -1), sin.expand(B, -1, -1)
            hd = module.head_dim
            h_g = h[m]                                             # [N, d]
            # query + channel gate exactly as Qwen3_5Attention.forward computes them
            q2 = module.q_proj(h_g).view(h_g.shape[0], -1, 2 * hd)
            q, chan_gate = torch.chunk(q2, 2, dim=-1)              # [N, H, hd] each
            q = rope_partial(module.q_norm(q), cos[m], sin[m])
            kidx = assign[m]                                       # [N] concept slots
            delta = torch.zeros(h_g.shape[0], h.shape[-1],
                                dtype=torch.float32, device=h.device)
            for c in kidx.unique().tolist():
                sel = kidx == c
                K, V = self._concepts[int(c)]
                K, V = K[slot].to(h.device, h.dtype), V[slot].to(h.device, h.dtype)
                Kh = K.repeat_interleave(module.num_key_value_groups, dim=1)
                Vh = V.repeat_interleave(module.num_key_value_groups, dim=1)
                qs = q[sel]                                        # [Nc, H, hd]
                scores = torch.einsum("nhd,mhd->nhm", qs.float(), Kh.float()) * module.scaling
                w = torch.softmax(scores, dim=-1).to(h.dtype)      # own softmax, phrase only
                o = torch.einsum("nhm,mhd->nhd", w, Vh)            # [Nc, H, hd]
                g = (torch.full((int(sel.sum()), self.n_heads), float(self.gate_override),
                                dtype=torch.float32, device=h.device)
                     if self.gate_override is not None
                     else self.gate[slot](h_g[sel].to(torch.float32)))
                o = o * g[:, :, None].to(h.dtype)                  # per-head gate
                side = o.reshape(o.shape[0], -1) * torch.sigmoid(chan_gate[sel].reshape(o.shape[0], -1))
                base = F.linear(side, module.o_proj.weight)        # frozen O
                lora = self.lora_b[slot](self.lora_a[slot](side.to(torch.float32)))
                delta[sel] = base.to(torch.float32) + lora
            out0 = output[0].clone()
            out0[m] = out0[m] + delta.to(out0.dtype)
            return (out0, *output[1:])
        return hook

    def attach(self, hf_model):
        layers = hf_model.model.layers
        cfg = hf_model.config
        types = getattr(cfg, "layer_types", None) or getattr(cfg.get_text_config(), "layer_types", None)
        if types is None:
            raise RuntimeError("model config has no layer_types; SideAttnGoggles targets "
                               "Qwen3.5 hybrid models")
        full = [i for i, t in enumerate(types) if t == "full_attention"]
        if full != self.layer_ids:
            raise RuntimeError(f"model full-attention layers {full} != goggle layer_ids "
                               f"{self.layer_ids}")
        return [layers[i].self_attn.register_forward_hook(self._make_hook(s), with_kwargs=True)
                for s, i in enumerate(self.layer_ids)]

    detach = Goggles.detach


class LinMapGoggles(nn.Module):
    """Instruction transport via per-layer LINEAR maps from LAYER-MATCHED phrase
    embeddings (2026-08-08 design, the author):

        delta_l = (s_l * I + B_l) @ e_l     added at goggled positions, layer l

    e_l = the frozen model's MEAN-POOLED hidden states of the instruction
    phrase at layer l — the same index as the injection point (layer input),
    so the conditioning vector already lives in the space it perturbs. No
    nonlinearity, no input norm: the map is pure linear, so the shared mean
    component of the bank becomes a learned per-layer constant (the "a marked
    span is here" carrier) and identity rides the remainder.

    Init: s_l (scalar) and B_l (d x d) both ZERO -> exact no-op at start, no
    saddle (dL/dB is rank-1 nonzero immediately). s_l is a one-parameter
    identity axis: if "inject the raw layer-l phrase state" is right, it is
    reachable in a few steps, and the learned s profile across layers is
    itself a diagnostic. Identity INIT was rejected: it is not a no-op (adds
    a full phrase-state at every goggled position at step 0).

    Interface matches ConditionedGoggles: set_concepts takes a list of
    per-concept E [L, d] tensors; set_mask takes assign [B, S] (-1 off /
    k = concept slot)."""

    def __init__(self, n_layers, d_model):
        super().__init__()
        self.maps = nn.ParameterList(
            nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_layers))
        self.scale = nn.Parameter(torch.zeros(n_layers))
        self.n_layers = n_layers
        self._assign = None
        self._concepts = None
        self.enabled = True

    def set_mask(self, assign):  # int tensor [B, S] (-1 off / k = concept idx) or None
        if assign is not None and assign.dtype == torch.bool:
            raise TypeError("LinMapGoggles.set_mask takes concept indices (-1 = off)")
        self._assign = assign

    def set_concepts(self, e_list):  # list of fp32 [L, d] per-layer pooled embeddings
        if e_list is not None and any(e is None for e in e_list):
            raise RuntimeError("set_concepts got a None slot — an example references an "
                               "instruction with no bank entry (dropped from the grid)")
        self._concepts = e_list

    def _make_hook(self, idx):
        def hook(module, args, kwargs):
            if not self.enabled or self._assign is None:
                return None
            if args:
                h, rest = args[0], args[1:]
                from_kwargs = False
            else:
                h, rest = kwargs["hidden_states"], ()
                from_kwargs = True
            if h.shape[1] != self._assign.shape[1]:
                raise RuntimeError(
                    f"goggle mask length {self._assign.shape[1]} != seq length {h.shape[1]}; "
                    "set_mask(None) during decode steps")
            assign = self._assign.to(h.device)
            m = assign >= 0
            if not bool(m.any()):
                return None
            if self._concepts is None or int(assign.max()) >= len(self._concepts):
                raise RuntimeError(f"mask references concept {int(assign.max())} but "
                                   f"{0 if self._concepts is None else len(self._concepts)} "
                                   "concepts are set; call set_concepts first")
            # per-concept delta at this layer, then scatter to positions.
            # Identity axis uses RAW e (same scale as h -> s_l stays calibrated);
            # the learned map gets unit-RMS input so conditioning is uniform
            # across layers (raw |e_l| varies a lot with depth).
            es = torch.stack([e[idx].to(h.device) for e in self._concepts])   # [C, d] fp32
            en = es * torch.rsqrt(es.pow(2).mean(-1, keepdim=True) + 1e-6)
            deltas = en @ self.maps[idx].T + self.scale[idx] * es             # [C, d]
            delta = deltas[assign[m]]                                         # [N, d]
            h = h.clone()
            h[m] = h[m] + delta.to(h.dtype)
            if from_kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = h
                return args, kwargs
            return (h, *rest), kwargs
        return hook

    def attach(self, hf_model):
        layers = hf_model.model.layers
        if len(layers) != self.n_layers:
            raise RuntimeError(f"{len(layers)} model layers vs {self.n_layers} goggle layers")
        return [layer.register_forward_pre_hook(self._make_hook(i), with_kwargs=True)
                for i, layer in enumerate(layers)]

    detach = Goggles.detach


def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G via quintic Newton-Schulz iteration (Muon; Jordan et al.)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D weight matrices: momentum + Newton-Schulz orthogonalized update.

    All goggle params (w_gate, w_in, w_out) are 2D, so a single Muon group
    covers everything. Update scaled by sqrt(max(1, fan_out/fan_in)).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon requires 2D params, got shape {tuple(p.shape)}")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(g, group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u, alpha=-group["lr"] * scale)


def chunked_full_kl(student_logits, teacher_logits, chunk=256):
    """Mean KL(teacher || student) over the full vocabulary, fp32 in chunks.

    Teacher is the same frozen model with goggles disabled, so this objective's
    minimum is at zero change — the right locality target. (CE toward a sampled
    one-hot answer instead bottoms out at a delta distribution, which trains the
    goggles to sharpen the model rather than leave it alone.)
    """
    T = student_logits.shape[0]
    if T == 0:
        return student_logits.new_zeros((), dtype=torch.float32)
    total = student_logits.new_zeros((), dtype=torch.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        t_lp = F.log_softmax(teacher_logits[s:e].float(), dim=-1)
        s_lp = F.log_softmax(student_logits[s:e].float(), dim=-1)
        total = total + (t_lp.exp() * (t_lp - s_lp)).sum()
    return total / T


def chunked_reverse_kl(student_logits, teacher_logits, chunk=256):
    """Mean KL(student || teacher) over the full vocabulary, fp32 in chunks.

    Reverse (mode-seeking) direction. Forward KL(teacher||student) is
    mass-covering: where the teacher is split between two continuations the
    student is pushed to cover both, and greedy decoding then picks whichever
    hedge is marginally higher. At a one-token hinge (e.g. "switch to Spanish
    here") that reliably loses to the fluent default. Reverse KL weights the
    sum by the STUDENT's own mass, so it pays to concentrate on one mode.
    Risk: it can commit to the wrong mode — pair with forward ("both") if the
    student collapses onto the default.
    """
    T = student_logits.shape[0]
    if T == 0:
        return student_logits.new_zeros((), dtype=torch.float32)
    total = student_logits.new_zeros((), dtype=torch.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        t_lp = F.log_softmax(teacher_logits[s:e].float(), dim=-1)
        s_lp = F.log_softmax(student_logits[s:e].float(), dim=-1)
        total = total + (s_lp.exp() * (s_lp - t_lp)).sum()
    return total / T


def chunked_ce(logits, targets, chunk=1024):
    """Mean cross-entropy, computed in fp32 chunks to bound memory."""
    T = logits.shape[0]
    if T == 0:
        return logits.new_zeros((), dtype=torch.float32)
    total = logits.new_zeros((), dtype=torch.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        total = total + F.cross_entropy(logits[s:e].float(), targets[s:e], reduction="sum")
    return total / T


def chunked_ce_per_token(logits, targets, chunk=1024):
    """Per-token cross-entropy vector [T], fp32 chunks (same memory bound as
    chunked_ce; the [T] output is negligible)."""
    outs = []
    for s in range(0, logits.shape[0], chunk):
        e = min(s + chunk, logits.shape[0])
        outs.append(F.cross_entropy(logits[s:e].float(), targets[s:e],
                                    reduction="none"))
    return torch.cat(outs) if outs else logits.new_zeros(0, dtype=torch.float32)


def topk_tail_kl(logits, comp_ids, top_ids, top_lps, pad_lp=-100.0, chunk=1024):
    """KL(teacher || student) over top-K ids + a tail bucket, mean over tokens.

    logits:  [T, V] student logits at positions predicting the completion tokens
    top_ids: [T, K] teacher top ids; top_lps: [T, K] teacher logprobs
             (entries with lp == pad_lp are padding and ignored)
    Computed in chunks to bound the fp32 logits copy.
    """
    T = logits.shape[0]
    if T > chunk:
        total = logits.new_zeros((), dtype=torch.float32)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            total = total + topk_tail_kl(
                logits[s:e], comp_ids[s:e], top_ids[s:e], top_lps[s:e], pad_lp, chunk) * (e - s)
        return total / T
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)            # [T,1]
    s_lp = torch.gather(logits, -1, top_ids.long()) - lse          # [T,K]
    valid = top_lps > pad_lp + 1.0
    p = torch.where(valid, top_lps.float().exp(), torch.zeros_like(s_lp))
    t_lp = torch.where(valid, top_lps.float(), torch.zeros_like(s_lp))

    p_tail = (1.0 - p.sum(-1)).clamp_min(1e-6)                     # [T]
    s_top_mass = torch.logsumexp(torch.where(valid, s_lp, torch.full_like(s_lp, -1e9)), dim=-1)
    s_tail = torch.log1p(-s_top_mass.exp().clamp(max=1.0 - 1e-7))  # [T]

    kl = (p * (t_lp - s_lp)).sum(-1) + p_tail * (p_tail.log() - s_tail)
    return kl.mean()


_TURN_END_ID = None


def turn_end_id(tok):
    """Token id that ends an assistant turn, read from the chat template.

    Model-specific (<|im_end|> on Qwen, <|eot_id|> on Llama); hardcoding one
    of them silently breaks stop detection on the other -- on Llama,
    convert_tokens_to_ids("<|im_end|>") is None, `nxt == eos_id` never fires,
    and every completion runs to the cap. Every generation loop must stop on
    THIS id. Raises rather than guessing.
    """
    global _TURN_END_ID
    if _TURN_END_ID is None:
        import config
        probe = tok.apply_chat_template(
            [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "PROBE"}],
            tokenize=False, enable_thinking=config.ENABLE_THINKING)
        tail = probe.split("PROBE", 1)[1].strip()
        tid = tok.convert_tokens_to_ids(tail.split()[0]) if tail else None
        if tid is None or tid == tok.unk_token_id:
            raise RuntimeError(f"no usable turn terminator for {tok.name_or_path}")
        _TURN_END_ID = tid
        print(f"[goggles_lib] turn terminator {tail.split()[0]!r} -> id {tid}", flush=True)
    return _TURN_END_ID
