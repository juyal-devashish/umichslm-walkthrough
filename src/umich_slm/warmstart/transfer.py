"""Transfer a donor's attention/MLP parameters into a UMichSLM arm.

Every sweep arm carries the *same* dense skeleton -- ``embed_tokens``, per-layer
``q/k/v/o_proj``, ``gate/up/down_proj``, two RMSNorms, ``model.norm``,
``lm_head`` -- and differs only by additive extras (register banks, pathway
lanes, nested FFN tiers). So one transfer covers the whole sweep, and the extras
keep their own init. For ``swa_sinkdiet_gqa`` and ``think_suffix_gqa`` that is
exactly right: ``register_gate`` is zero-initialised, so a warm-started bank arm
is functionally identical to the warm-started dense model at step 0 and the bank
learns as a correction on top of a working model.

Three reductions are needed, and the choices are not interchangeable:

* **Width** (donor ``h_d`` -> our ``h_o``) is a **coordinate selection**, not a
  rotation. RMSNorm's gain is diagonal in the residual basis, so any rotation
  would have to be undone inside every norm -- which cannot be expressed by a
  diagonal gain. Selecting a subset of residual coordinates keeps every norm
  diagonal and keeps RMSNorm's own rescaling available to absorb the change in
  stream magnitude. This is the weight-subcloning choice and it is why that
  method survives contact with real checkpoints.
* **FFN width** is a per-layer neuron selection, ranked by the product of a
  neuron's fan-in and fan-out magnitudes.
* **Depth** (donor ``L_d`` -> our ``L_o``, and ours is *deeper*) is layer
  duplication. ``interleave`` repeats each donor layer in place, ``cycle``
  repeats the whole stack. Duplicated layers can optionally have their output
  projections zeroed so they start as identity (the LEMON trick), which
  preserves the donor function at the cost of starting half the model dead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .donor_io import find_prefix, load_tensor, tensor_names


@dataclass
class DonorSpec:
    """The donor geometry the transfer needs, read from its config."""

    snapshot: str
    hidden: int
    inter: int
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rope_theta: float
    prefix: str
    has_qk_norm: bool = False
    attn_layers: List[int] = field(default_factory=list)
    pre_norm: bool = True

    @classmethod
    def from_snapshot(cls, snapshot: str) -> "DonorSpec":
        import glob
        import json
        import os

        with open(os.path.join(snapshot, "config.json"), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        tc = cfg.get("text_config", cfg)
        pref = find_prefix(snapshot)
        names = set(tensor_names(snapshot))
        h = int(tc["hidden_size"])
        qh = int(tc["num_attention_heads"])
        hd = int(tc.get("head_dim") or h // qh)
        # Hybrid donors (Qwen3.5) have linear-attention layers with no q_proj;
        # only the softmax layers can donate attention weights.
        attn = [
            i
            for i in range(int(tc["num_hidden_layers"]))
            if f"{pref}layers.{i}.self_attn.q_proj.weight" in names
        ]
        # OLMo 3 (and OLMo 2) are POST-norm: the block is
        # `h = x + norm(Attn(x))`, with no `input_layernorm` at all, against our
        # pre-norm `h = x + Attn(norm(x))`. Its q/k/v projections were therefore
        # trained on the raw residual stream, whose magnitude grows with depth,
        # not on a unit-RMS vector. Transferring them into a pre-norm block hands
        # every projection an input scale it never saw, and the correction needs
        # the per-layer stream RMS, which cannot be known without running the
        # donor. Detected here so callers can refuse or warn rather than silently
        # produce a mismatched model.
        pre = bool(attn) and f"{pref}layers.{attn[0]}.input_layernorm.weight" in names
        return cls(
            snapshot=snapshot,
            hidden=h,
            inter=int(tc["intermediate_size"]),
            layers=int(tc["num_hidden_layers"]),
            q_heads=qh,
            kv_heads=int(tc.get("num_key_value_heads") or qh),
            head_dim=hd,
            rope_theta=float(tc.get("rope_theta") or 10000.0),
            prefix=pref,
            has_qk_norm=(
                f"{pref}layers.{attn[0]}.self_attn.q_norm.weight" in names if attn else False
            ),
            attn_layers=attn,
            pre_norm=pre,
        )

    def t(self, name: str) -> torch.Tensor:
        return load_tensor(self.snapshot, f"{self.prefix}{name}")


def layer_map(donor_layers: List[int], n_target: int, mode: str = "interleave") -> List[int]:
    """Map each of our layers to a donor layer index.

    ``interleave`` keeps the donor's depth ordering and repeats each layer in
    place, so our layer k sits at the same *relative depth* as its donor source.
    ``cycle`` replays the whole donor stack, which is G_stack's operator.

    ``ends`` is for the **downsampling** case only, and it exists because
    striding is the wrong way to remove depth: every layer is trained to consume
    the output distribution of its immediate predecessor, so keeping layers
    0, 4, 8, ... hands each one an input it has never seen. The depth-pruning
    literature (Minitron, ShortGPT) drops a *contiguous* interior block instead
    and keeps the layers adjacent to the embedding and the head, which is what
    this does. It falls back to ``interleave`` when the target is deeper than
    the donor, where nothing is being dropped.

    Note the target, ``deep80``, *upsamples* (80 from 36). Any probe at a
    shallower preset exercises the opposite regime on this axis.
    """
    d = len(donor_layers)
    if mode == "cycle":
        return [donor_layers[i % d] for i in range(n_target)]
    if mode == "interleave":
        return [donor_layers[min(int(i * d / n_target), d - 1)] for i in range(n_target)]
    if mode == "ends":
        if n_target >= d:
            return layer_map(donor_layers, n_target, "interleave")
        head = (n_target + 1) // 2
        tail = n_target - head
        return donor_layers[:head] + (donor_layers[d - tail :] if tail else [])
    raise ValueError(f"unknown layer map mode {mode!r}")


@torch.no_grad()
def score_residual_dims(spec: DonorSpec, sample_layers: int = 8) -> torch.Tensor:
    """Importance of each donor residual coordinate.

    A residual coordinate matters in proportion to how strongly every layer reads
    from it and writes to it, so the score sums squared column norms of the
    layers' input projections (q, k, v, gate, up) and squared row norms of their
    output projections (o, down), scaled by the final norm's gain. Layers are
    sampled rather than all read, because an 8B donor is ~16 GB on disk and the
    score is stable across layers.
    """
    idx = spec.attn_layers or list(range(spec.layers))
    step = max(len(idx) // max(sample_layers, 1), 1)
    picked = idx[::step][:sample_layers]
    score = torch.zeros(spec.hidden, dtype=torch.float64)
    for li in picked:
        for nm in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
        ):
            try:
                w = spec.t(f"layers.{li}.{nm}.weight").float()
            except KeyError:
                continue
            score += (w * w).sum(0).double()  # column = input coordinate
        for nm in ("self_attn.o_proj", "mlp.down_proj"):
            try:
                w = spec.t(f"layers.{li}.{nm}.weight").float()
            except KeyError:
                continue
            score += (w * w).sum(1).double()  # row = output coordinate
    try:
        g = spec.t("norm.weight").float().double().abs()
        score = score * (g / g.mean()).clamp_min(1e-3)
    except KeyError:
        pass
    return score.float()


@torch.no_grad()
def select_heads(spec: DonorSpec, n_q: int, n_kv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick donor head indices for our ``n_q`` query and ``n_kv`` kv heads.

    GQA ties each query head to one kv head, so the selection stays inside the
    best kv group for as long as it can and only then spills into the next group.
    Heads taken from a second group were trained against different keys; that is
    unavoidable when the donor has 4 query heads per group and we want 7, and it
    is the main approximation in the attention transfer.
    """
    per_group = max(spec.q_heads // max(spec.kv_heads, 1), 1)
    kscore = torch.zeros(spec.kv_heads)
    qscore = torch.zeros(spec.q_heads)
    for li in spec.attn_layers[:: max(len(spec.attn_layers) // 8, 1)][:8]:
        k = spec.t(f"layers.{li}.self_attn.k_proj.weight").float()
        v = spec.t(f"layers.{li}.self_attn.v_proj.weight").float()
        q = spec.t(f"layers.{li}.self_attn.q_proj.weight").float()
        kscore += k.view(spec.kv_heads, -1).norm(dim=1) * v.view(spec.kv_heads, -1).norm(dim=1)
        qscore += q.view(spec.q_heads, -1).norm(dim=1)
    kv_order = torch.argsort(kscore, descending=True)
    kv_sel = kv_order[:n_kv]
    q_sel: List[int] = []
    for g in kv_order.tolist():
        members = list(range(g * per_group, min((g + 1) * per_group, spec.q_heads)))
        members.sort(key=lambda h: -float(qscore[h]))
        for h in members:
            if len(q_sel) < n_q:
                q_sel.append(h)
        if len(q_sel) >= n_q:
            break
    return torch.tensor(q_sel[:n_q]), kv_sel


@torch.no_grad()
def select_ffn(
    spec: DonorSpec, layer: int, n_keep: int, order: str = "donor"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick ``n_keep`` FFN neurons from a donor layer, growing if it is narrower.

    ``order`` decides how the kept neurons are laid out in the target rows.
    ``"donor"`` restores the donor's own index order, which is neutral for a
    plain SwiGLU where row position carries no meaning. ``"importance"`` keeps
    the ranking, which is **required** for the ``mat_ffn`` nest:
    ``MatFormerSwiGLUMLP`` tiers are nested *prefixes* of ``intermediate_size``
    (``self.widths``), so a row's index is its tier membership and every token
    pays for the first rows. Ranked order puts the donor's most important
    neurons where every token reads them -- the nest's tier structure and
    subcloning's neuron ranking are the same object.

    Returns ``(indices, down_scale)``. Neurons are ranked by fan-in x fan-out
    magnitude, which is the standard subcloning score.

    When the donor's FFN is *narrower* than ours -- Qwen3-0.6B is 3072 against
    deep80's 3584 -- the deficit is covered by duplicating the top-ranked
    neurons. SwiGLU is elementwise, so duplicating a neuron's ``gate``/``up``
    rows and splitting its ``down`` column evenly across the copies reproduces
    the donor layer exactly; ``down_scale`` carries that split. This is
    HyperCloning's width expansion applied to one tensor pair, and unlike the
    residual-stream case it is genuinely function-preserving because no norm
    sits between the duplicated units.
    """
    g = spec.t(f"layers.{layer}.mlp.gate_proj.weight").float()
    u = spec.t(f"layers.{layer}.mlp.up_proj.weight").float()
    d = spec.t(f"layers.{layer}.mlp.down_proj.weight").float()
    s = g.norm(dim=1) * u.norm(dim=1) * d.norm(dim=0)
    n_have = s.numel()
    ranked = torch.argsort(s, descending=True)
    if n_keep <= n_have:
        idx = ranked[:n_keep]
        if order == "donor":
            idx = idx.sort().values
        return idx, torch.ones(n_keep)
    reps = -(-n_keep // n_have)  # ceil
    idx = ranked.repeat(reps)[:n_keep]
    counts = torch.bincount(idx, minlength=n_have).float().clamp_min(1.0)
    if order == "donor":
        idx = idx[torch.argsort(idx, stable=True)]
    return idx, (1.0 / counts[idx])


@torch.no_grad()
def _fold_qk_norm(w: torch.Tensor, gain: Optional[torch.Tensor], head_dim: int) -> torch.Tensor:
    """Fold an RMSNorm *gain* into the projection that feeds it.

    Only the gain folds; the 1/rms(x) factor is input-dependent and is dropped.
    Folding the gain is strictly closer to the donor than ignoring it.

    Two conventions in the wild: Qwen3 normalises **per head** with a
    ``[head_dim]`` gain, OLMo 3 normalises the **whole** projection output with a
    ``[n_heads * head_dim]`` one. Both reduce to scaling rows of ``w``; only the
    broadcast differs.

    ``head_dim`` here is the **donor's**, not the target's: the fold happens in
    the donor's head layout, before row selection or any head split.
    """
    if gain is None:
        return w
    g = gain.float()
    if g.numel() == w.shape[0]:
        return w * g.unsqueeze(1)
    if g.numel() != head_dim:
        raise ValueError(
            f"q/k norm gain of {g.numel()} matches neither {w.shape[0]} nor {head_dim}"
        )
    n_heads = w.shape[0] // head_dim
    return (w.view(n_heads, head_dim, -1) * g.view(1, head_dim, 1)).reshape_as(w)


@torch.no_grad()
def build_state_dict(
    spec: DonorSpec,
    *,
    hidden: int,
    inter: int,
    n_layers: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
    emb_in: torch.Tensor,
    emb_out: torch.Tensor,
    map_mode: str = "interleave",
    zero_duplicates: bool = False,
    dup_scale: float = 1.0,
    sample_layers: int = 8,
    ffn_order: str = "donor",
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Produce the dense-skeleton tensors for a UMichSLM arm from ``spec``.

    ``emb_in`` / ``emb_out`` are composed tables in the *donor's* width; they get
    the same residual selection as everything else, so the whole model stays in
    one consistent coordinate basis.

    ``dup_scale`` multiplies the output projections of duplicated layers. The
    residual stream sees each duplicated block twice, so scaling by ``1/r`` for a
    duplication factor ``r`` keeps the per-layer contribution to the stream at
    roughly the donor's magnitude. ``zero_duplicates`` is the stronger version
    (exact identity at init, LEMON-style) and is mutually exclusive with it.

    ``head_dim`` need not match the donor's: :func:`head_row_map` splits a wider
    donor head into several target heads while keeping RoPE's dimension pairs
    intact, which is what lets a head_dim-128 donor feed the head_dim-64
    screening tiers.
    """
    if hidden > spec.hidden:
        raise ValueError(f"donor is narrower ({spec.hidden}) than target ({hidden})")

    res = torch.argsort(score_residual_dims(spec, sample_layers), descending=True)[:hidden]
    res = res.sort().values
    qrow_list, kvrow_list, head_meta = select_heads_split(spec, n_q, n_kv, head_dim)
    lmap = layer_map(spec.attn_layers or list(range(spec.layers)), n_layers, map_mode)
    seen: Dict[int, int] = {}
    dup_index = []
    for li in lmap:
        dup_index.append(seen.get(li, 0))
        seen[li] = seen.get(li, 0) + 1
    rep = {li: c for li, c in seen.items()}

    qrows, kvrows = torch.cat(qrow_list), torch.cat(kvrow_list)
    # A post-norm donor (OLMo 2/3) trained its projections on the raw residual
    # stream, so we need the scale that stream had at each depth.
    pn_rms = post_norm_stream_rms(spec) if not spec.pre_norm else {}
    sd: Dict[str, torch.Tensor] = {}
    sd["model.embed_tokens.weight"] = emb_in[:, res].contiguous()
    sd["lm_head.weight"] = emb_out[:, res].contiguous()
    sd["model.norm.weight"] = spec.t("norm.weight").float()[res].contiguous()

    for our_l, donor_l in enumerate(lmap):
        p = f"model.layers.{our_l}."
        qn = spec.t(f"layers.{donor_l}.self_attn.q_norm.weight") if spec.has_qk_norm else None
        kn = spec.t(f"layers.{donor_l}.self_attn.k_norm.weight") if spec.has_qk_norm else None
        q = _fold_qk_norm(
            spec.t(f"layers.{donor_l}.self_attn.q_proj.weight").float(), qn, spec.head_dim
        )
        k = _fold_qk_norm(
            spec.t(f"layers.{donor_l}.self_attn.k_proj.weight").float(), kn, spec.head_dim
        )
        v = spec.t(f"layers.{donor_l}.self_attn.v_proj.weight").float()
        o = spec.t(f"layers.{donor_l}.self_attn.o_proj.weight").float()
        sd[p + "self_attn.q_proj.weight"] = q[qrows][:, res].contiguous()
        sd[p + "self_attn.k_proj.weight"] = k[kvrows][:, res].contiguous()
        sd[p + "self_attn.v_proj.weight"] = v[kvrows][:, res].contiguous()
        o_sel = o[res][:, qrows].contiguous()

        keep, dscale = select_ffn(spec, donor_l, inter, order=ffn_order)
        gp = spec.t(f"layers.{donor_l}.mlp.gate_proj.weight").float()[keep][:, res].contiguous()
        up = spec.t(f"layers.{donor_l}.mlp.up_proj.weight").float()[keep][:, res].contiguous()
        dn = (
            spec.t(f"layers.{donor_l}.mlp.down_proj.weight").float()[res][:, keep]
            * dscale.unsqueeze(0)
        ).contiguous()

        if spec.pre_norm:
            in_ln = spec.t(f"layers.{donor_l}.input_layernorm.weight").float()[res].contiguous()
            mid_ln = (
                spec.t(f"layers.{donor_l}.post_attention_layernorm.weight")
                .float()[res]
                .contiguous()
            )
        else:
            b = post_norm_block_norms(spec, donor_l, res, pn_rms)
            in_ln, mid_ln = b["input_scale"], b["mlp_scale"]
            o_sel = apply_write_gain(o_sel, b["o_gain"])
            dn = apply_write_gain(dn, b["down_gain"])

        r = rep.get(donor_l, 1)
        if zero_duplicates and dup_index[our_l] > 0:
            o_sel = torch.zeros_like(o_sel)
            dn = torch.zeros_like(dn)
        elif r > 1 and dup_scale != 1.0:
            o_sel = o_sel * (dup_scale / r)
            dn = dn * (dup_scale / r)
        sd[p + "self_attn.o_proj.weight"] = o_sel
        sd[p + "mlp.gate_proj.weight"] = gp
        sd[p + "mlp.up_proj.weight"] = up
        sd[p + "mlp.down_proj.weight"] = dn
        sd[p + "input_layernorm.weight"] = in_ln
        sd[p + "post_attention_layernorm.weight"] = mid_ln
        if verbose and our_l % 16 == 0:
            print(f"  layer {our_l:3d} <- donor {donor_l:3d} (copy {dup_index[our_l]})", flush=True)

    meta = {
        "donor": spec.snapshot,
        "donor_hidden": spec.hidden,
        "donor_layers": spec.layers,
        "donor_attn_layers": len(spec.attn_layers),
        "donor_rope_theta": spec.rope_theta,
        "residual_kept": hidden,
        "residual_frac": hidden / spec.hidden,
        "donor_head_dim": spec.head_dim,
        "target_head_dim": head_dim,
        "layer_map": lmap,
        "max_layer_reuse": max(rep.values()) if rep else 0,
        "qk_norm_gain_folded": bool(spec.has_qk_norm),
        "donor_pre_norm": spec.pre_norm,
        "post_norm_translated": not spec.pre_norm,
        "ffn_grown": spec.inter < inter,
        "ffn_donor_width": spec.inter,
        "ffn_order": ffn_order,
        "zero_duplicates": zero_duplicates,
        "dup_scale": dup_scale,
    }
    meta.update(head_meta)
    return sd, meta


@torch.no_grad()
def residual_pca(
    spec: DonorSpec, hidden: int, sample_layers: int = 8
) -> Tuple[torch.Tensor, float]:
    """Orthonormal ``[hidden, donor_hidden]`` basis for the donor's residual stream.

    Coordinate selection is the safe choice but it is brutal at large width
    ratios -- taking 896 of Qwen3-8B's 4096 dimensions keeps 22% of the
    coordinates, and the residual basis is not axis-aligned, so that is not 22%
    of the information. A rotation keeps far more, and it *is* admissible here
    despite RMSNorm, for a reason worth spelling out:

    RMSNorm is ``x / rms(x) * g``. The normalisation is invariant under an
    orthogonal change of basis; only the diagonal gain ``g`` is not. But ``g``
    can be folded forward into whatever consumes the norm's output --
    ``W (n(x) * g) = (W diag(g)) n(x)`` -- leaving the norm as a bare
    ``x / rms(x)``, which commutes with ``P``. So: fold every gain into its
    consumer, set the norm weights to 1, and rotate everything else.

    The basis is the top eigenvectors of what gets *written* into the stream:
    the second moment of the embedding table plus the output projections of
    attention and MLP across sampled layers. Returns ``(P, beta)``, where
    ``beta = sqrt(captured_fraction * donor_hidden / hidden)`` corrects the
    read-side scale -- ``P`` concentrates energy, so a unit-RMS vector in the
    donor's basis does not stay unit-RMS in ours.
    """
    idx = spec.attn_layers or list(range(spec.layers))
    step = max(len(idx) // max(sample_layers, 1), 1)
    picked = idx[::step][:sample_layers]
    m = torch.zeros(spec.hidden, spec.hidden, dtype=torch.float64)
    e = spec.t("embed_tokens.weight").float()
    m += (e.T @ e).double() / e.shape[0]
    for li in picked:
        for nm in ("self_attn.o_proj", "mlp.down_proj"):
            try:
                w = spec.t(f"layers.{li}.{nm}.weight").float()
            except KeyError:
                continue
            m += (w @ w.T).double() / w.shape[1]
    evals, evecs = torch.linalg.eigh(m)
    order = torch.argsort(evals, descending=True)
    top = order[:hidden]
    captured = float(evals[top].sum() / evals.clamp_min(0).sum())
    p = evecs[:, top].T.float().contiguous()  # [hidden, donor_hidden]
    beta = float((captured * spec.hidden / hidden) ** 0.5)
    return p, beta


@torch.no_grad()
def build_state_dict_pca(
    spec: DonorSpec,
    *,
    hidden: int,
    inter: int,
    n_layers: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
    emb_in: torch.Tensor,
    emb_out: torch.Tensor,
    map_mode: str = "interleave",
    zero_duplicates: bool = False,
    dup_scale: float = 1.0,
    sample_layers: int = 8,
    ffn_order: str = "donor",
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Width reduction by rotation instead of selection. See :func:`residual_pca`.

    Every RMSNorm gain is folded into its consumer and the norm weights ship as
    ones, so the transferred model is in a rotated-but-equivalent coordinate
    system rather than a truncated one.
    """
    p, beta = residual_pca(spec, hidden, sample_layers)
    pn_rms = post_norm_stream_rms(spec) if not spec.pre_norm else {}
    qrow_list, kvrow_list, head_meta = select_heads_split(spec, n_q, n_kv, head_dim)
    lmap = layer_map(spec.attn_layers or list(range(spec.layers)), n_layers, map_mode)
    seen: Dict[int, int] = {}
    dup_index = []
    for li in lmap:
        dup_index.append(seen.get(li, 0))
        seen[li] = seen.get(li, 0) + 1
    rep = {li: c for li, c in seen.items()}

    qrows, kvrows = torch.cat(qrow_list), torch.cat(kvrow_list)
    gf = spec.t("norm.weight").float()
    sd: Dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": (emb_in @ p.T).contiguous(),
        "lm_head.weight": (beta * (emb_out * gf) @ p.T).contiguous(),
        "model.norm.weight": torch.ones(hidden),
    }
    for our_l, donor_l in enumerate(lmap):
        pre = f"model.layers.{our_l}."
        if spec.pre_norm:
            g1 = spec.t(f"layers.{donor_l}.input_layernorm.weight").float()
            g2 = spec.t(f"layers.{donor_l}.post_attention_layernorm.weight").float()
        else:
            # Post-norm: there is no pre-projection gain to fold. What the
            # projections saw was the raw stream, so the scalar stream RMS at
            # that depth plays the gain's role on the read side, and the donor's
            # *output* gains fold into the writes instead.
            g1 = torch.full((spec.hidden,), float(pn_rms["pre_attn"][donor_l]))
            g2 = torch.full((spec.hidden,), float(pn_rms["pre_mlp"][donor_l]))
        qn = spec.t(f"layers.{donor_l}.self_attn.q_norm.weight") if spec.has_qk_norm else None
        kn = spec.t(f"layers.{donor_l}.self_attn.k_norm.weight") if spec.has_qk_norm else None
        q = _fold_qk_norm(
            spec.t(f"layers.{donor_l}.self_attn.q_proj.weight").float(), qn, spec.head_dim
        )
        k = _fold_qk_norm(
            spec.t(f"layers.{donor_l}.self_attn.k_proj.weight").float(), kn, spec.head_dim
        )
        v = spec.t(f"layers.{donor_l}.self_attn.v_proj.weight").float()
        o = spec.t(f"layers.{donor_l}.self_attn.o_proj.weight").float()
        sd[pre + "self_attn.q_proj.weight"] = (beta * (q[qrows] * g1) @ p.T).contiguous()
        sd[pre + "self_attn.k_proj.weight"] = (beta * (k[kvrows] * g1) @ p.T).contiguous()
        sd[pre + "self_attn.v_proj.weight"] = (beta * (v[kvrows] * g1) @ p.T).contiguous()
        if not spec.pre_norm:
            b = post_norm_block_norms(spec, donor_l, torch.arange(spec.hidden), pn_rms)
            o = apply_write_gain(o, b["o_gain"])
        o_new = (p @ o[:, qrows]).contiguous()

        keep, dscale = select_ffn(spec, donor_l, inter, order=ffn_order)
        gp = spec.t(f"layers.{donor_l}.mlp.gate_proj.weight").float()[keep]
        up = spec.t(f"layers.{donor_l}.mlp.up_proj.weight").float()[keep]
        dn = spec.t(f"layers.{donor_l}.mlp.down_proj.weight").float()[:, keep] * dscale.unsqueeze(0)
        if not spec.pre_norm:
            dn = apply_write_gain(dn, b["down_gain"])
        gp = (beta * (gp * g2) @ p.T).contiguous()
        up = (beta * (up * g2) @ p.T).contiguous()
        dn_new = (p @ dn).contiguous()

        r = rep.get(donor_l, 1)
        if zero_duplicates and dup_index[our_l] > 0:
            o_new = torch.zeros_like(o_new)
            dn_new = torch.zeros_like(dn_new)
        elif r > 1 and dup_scale != 1.0:
            o_new = o_new * (dup_scale / r)
            dn_new = dn_new * (dup_scale / r)
        sd[pre + "self_attn.o_proj.weight"] = o_new
        sd[pre + "mlp.gate_proj.weight"] = gp
        sd[pre + "mlp.up_proj.weight"] = up
        sd[pre + "mlp.down_proj.weight"] = dn_new
        sd[pre + "input_layernorm.weight"] = torch.ones(hidden)
        sd[pre + "post_attention_layernorm.weight"] = torch.ones(hidden)
        if verbose and our_l % 16 == 0:
            print(
                f"  layer {our_l:3d} <- donor {donor_l:3d} (copy {dup_index[our_l]}, pca)",
                flush=True,
            )

    meta = {
        "donor": spec.snapshot,
        "width_mode": "pca",
        "pca_beta": beta,
        "donor_hidden": spec.hidden,
        "donor_layers": spec.layers,
        "donor_rope_theta": spec.rope_theta,
        "donor_head_dim": spec.head_dim,
        "target_head_dim": head_dim,
        "layer_map": lmap,
        "max_layer_reuse": max(rep.values()) if rep else 0,
        "qk_norm_gain_folded": bool(spec.has_qk_norm),
        "norm_gains_folded": True,
        "donor_pre_norm": spec.pre_norm,
        "post_norm_translated": not spec.pre_norm,
        "ffn_grown": spec.inter < inter,
        "ffn_order": ffn_order,
        "zero_duplicates": zero_duplicates,
        "dup_scale": dup_scale,
    }
    meta.update(head_meta)
    return sd, meta


_NORM_SUFFIXES = ("layernorm.weight", "norm.weight")


@torch.no_grad()
def donate_extras(
    spec: DonorSpec,
    missing: Dict[str, torch.Size],
    *,
    res: torch.Tensor,
    head_dim: int,
    donor_layers: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Fill arch-specific tensors that live *outside* ``model.layers.*``.

    Several arms carry transformer sublayers off the main spine:
    ``think_suffix_gqa``'s four ``pathways.lane_stack`` blocks are 48.5M params,
    4.6% of the arm, and the bank arms' ``register_refiner`` is another 0.8M.
    They are shape-compatible with ordinary donor layers, so leaving them at
    random init while the spine is donated would treat the arms unequally --
    which defeats the point of using one donor for the whole sweep.

    Matching is by shape and name suffix rather than by an explicit table, so an
    arm added later is covered without touching this function. Anything that
    does not match a known pattern is left alone on purpose: ``register_gate``,
    ``logit_sink``, routers, ``merge_scales`` and the register bank itself are
    all deliberately zero- or specially-initialised, and donating into them
    would destroy the identity-at-init property the bank arms rely on.

    Source layers are taken from the *deep end* of the donor, which the spine
    map reuses least.
    """
    pool = list(donor_layers or spec.attn_layers or range(spec.layers))[::-1]
    if not pool:
        return {}
    hidden = int(res.numel())
    out: Dict[str, torch.Tensor] = {}
    counter = 0

    def nxt() -> int:
        nonlocal counter
        li = pool[counter % len(pool)]
        counter += 1
        return li

    groups: Dict[str, int] = {}
    for key in sorted(missing):
        shape = tuple(missing[key])
        stem = key.rsplit(".", 2)[0]
        if stem not in groups:
            groups[stem] = nxt()
        li = groups[stem]
        try:
            if key.endswith("q_proj.weight") and shape == (hidden, hidden):
                qrl, _, _ = select_heads_split(spec, hidden // head_dim, 1, head_dim)
                w = spec.t(f"layers.{li}.self_attn.q_proj.weight").float()
                out[key] = w[torch.cat(qrl)][:, res].contiguous()
            elif key.endswith(("k_proj.weight", "v_proj.weight")) and shape[1] == hidden:
                n = shape[0] // head_dim
                nm = "k_proj" if key.endswith("k_proj.weight") else "v_proj"
                w = spec.t(f"layers.{li}.self_attn.{nm}.weight").float()
                kmap = head_row_map(spec.kv_heads, spec.head_dim, head_dim)
                sel = w[torch.cat(kmap[: min(n, len(kmap))])][:, res]
                if sel.shape[0] < shape[0]:  # donor has fewer kv heads than asked for
                    reps = -(-shape[0] // sel.shape[0])
                    sel = sel.repeat(reps, 1)[: shape[0]]
                out[key] = sel.contiguous()
            elif key.endswith("o_proj.weight") and shape[0] == hidden:
                qrl, _, _ = select_heads_split(spec, shape[1] // head_dim, 1, head_dim)
                w = spec.t(f"layers.{li}.self_attn.o_proj.weight").float()
                out[key] = w[res][:, torch.cat(qrl)].contiguous()
            elif key.endswith(("gate_proj.weight", "up_proj.weight")) and shape[1] == hidden:
                nm = "gate_proj" if key.endswith("gate_proj.weight") else "up_proj"
                keep, _ = select_ffn(spec, li, shape[0])
                out[key] = spec.t(f"layers.{li}.mlp.{nm}.weight").float()[keep][:, res].contiguous()
            elif key.endswith("down_proj.weight") and shape[0] == hidden:
                keep, dsc = select_ffn(spec, li, shape[1])
                w = spec.t(f"layers.{li}.mlp.down_proj.weight").float()[res][:, keep]
                out[key] = (w * dsc.unsqueeze(0)).contiguous()
            elif key.endswith(_NORM_SUFFIXES) and shape == (hidden,):
                src = "input_layernorm" if "input" in key else "post_attention_layernorm"
                out[key] = spec.t(f"layers.{li}.{src}.weight").float()[res].contiguous()
        except KeyError:
            continue
    if verbose and out:
        n = sum(v.numel() for v in out.values())
        print(
            f"  donated {len(out)} extra tensors ({n/1e6:.1f}M params) from donor tail layers",
            flush=True,
        )
    return out


@torch.no_grad()
def post_norm_stream_rms(
    spec: DonorSpec, res: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Estimate the residual-stream RMS at each point of a POST-norm donor.

    A post-norm block writes ``x + norm(F(x))``, and ``norm(z) * g`` has per-dim
    RMS ``||g|| / sqrt(d)`` regardless of ``z``. So every block's contribution to
    the stream's variance is known *from the gain vectors alone* -- no forward
    pass, no 32B model in memory. Accumulating them from the embedding table's
    own RMS gives the scale each layer's projections were trained to see:

        rms_l^2 = rms_emb^2 + sum_{j<l} (||g_attn_j||^2 + ||g_ffn_j||^2) / d

    This is what makes a post-norm donor usable in a pre-norm target. Our
    ``input_layernorm`` hands the projections a unit-RMS vector; setting its gain
    to ``rms_l`` hands them the magnitude they expect instead. It is first-order
    -- it assumes successive block writes are uncorrelated, which they are not --
    but it is a far better starting point than ``1.0``, and the alternative is
    running the donor.

    Returns per-layer ``pre_attn`` and ``pre_mlp`` RMS vectors.
    """
    d = spec.hidden
    e = spec.t("embed_tokens.weight").float()
    var = float((e * e).mean())
    pre_attn, pre_mlp = [], []
    for li in range(spec.layers):
        pre_attn.append(var**0.5)
        try:
            ga = spec.t(f"layers.{li}.post_attention_layernorm.weight").float()
            var += float((ga * ga).sum()) / d
        except KeyError:
            pass
        pre_mlp.append(var**0.5)
        try:
            gf = spec.t(f"layers.{li}.post_feedforward_layernorm.weight").float()
            var += float((gf * gf).sum()) / d
        except KeyError:
            pass
    return {"pre_attn": torch.tensor(pre_attn), "pre_mlp": torch.tensor(pre_mlp)}


@torch.no_grad()
def post_norm_block_norms(
    spec: DonorSpec,
    layer: int,
    res: torch.Tensor,
    rms: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Translate one POST-norm donor block into our PRE-norm block's tensors.

    Donor:  ``h = x + normA(Attn(x))``;      ``y = h + normF(MLP(h))``
    Ours:   ``h = x + Attn(normA'(x))``;     ``y = h + MLP(normF'(h))``

    The translation has two halves, and only the first is exact:

    * **The donor's output norms fold into the writes.** ``normA``'s gain is per
      output dimension and ``o_proj``'s rows are output dimensions, so the gain
      multiplies ``o_proj``'s rows exactly; likewise ``normF`` into
      ``down_proj``. What does *not* fold is the ``1/rms(z)`` factor, which is
      input-dependent. It is replaced by a scalar that makes the write's expected
      magnitude match the donor's -- the donor wrote something with per-dim RMS
      ``||g||/sqrt(d)``, so the folded matrix is rescaled to produce that, given
      unit-RMS input. First-order, and reported in the metadata.
    * **Our input norms carry the stream scale** the donor's projections were
      trained against (:func:`post_norm_stream_rms`).
    """
    out: Dict[str, torch.Tensor] = {}
    hidden = int(res.numel())
    out["input_scale"] = torch.full((hidden,), float(rms["pre_attn"][layer]))
    out["mlp_scale"] = torch.full((hidden,), float(rms["pre_mlp"][layer]))
    for src, key in (
        ("post_attention_layernorm", "o_gain"),
        ("post_feedforward_layernorm", "down_gain"),
    ):
        try:
            out[key] = spec.t(f"layers.{layer}.{src}.weight").float()[res].contiguous()
        except KeyError:
            out[key] = torch.ones(hidden)
    return out


def apply_write_gain(w: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Fold a post-norm output gain into a write matrix and match its scale.

    ``w`` has rows indexed by output dimension. After folding, the matrix is
    rescaled so that a unit-RMS input produces a write with the same per-dim RMS
    the donor's ``norm(z) * gain`` produced, i.e. ``||gain|| / sqrt(d_out)``.
    """
    folded = w * gain.unsqueeze(1)
    raw = float(folded.norm()) / max(folded.shape[0], 1) ** 0.5
    target = float(gain.norm()) / max(gain.numel(), 1) ** 0.5
    return folded * (target / raw) if raw > 1e-8 else folded


def head_row_map(n_heads: int, donor_head_dim: int, target_head_dim: int) -> List[torch.Tensor]:
    """Donor row indices for each *virtual* target head.

    The screening tiers run ``head_dim=64`` while every large donor is 128, and
    heads only transfer as whole units -- so a 128-wide donor head has to become
    two 64-wide target heads. The split is not arbitrary, because RoPE couples
    specific dimension pairs.

    HF's ``rotate_half`` pairs dimension ``i`` with ``i + D/2`` and rotates the
    pair at frequency ``theta^(-2i/D)``. A naive split at ``D/2`` severs **every**
    pair. Taking a *strided* set of pair indices keeps them intact and, for the
    first stride, also matches the frequency exactly: target head dim ``j`` takes
    donor dim ``2j``, whose donor frequency ``theta^(-2j/128)`` equals the target
    frequency ``theta^(-2j/64)``... after the stride, ``theta^(-j/32)`` either
    way. The second stride (odd pair indices) is offset by a constant
    ``theta^(-1/64)`` in frequency -- near-matched, not exact.

    So each donor head yields one frequency-exact virtual head and ``k-1``
    near-matched ones. Pass ``--allow-head-split`` to use it; it is a real
    approximation, just a far better one than cutting the vector in half.
    """
    if target_head_dim == donor_head_dim:
        return [torch.arange(h * donor_head_dim, (h + 1) * donor_head_dim) for h in range(n_heads)]
    if donor_head_dim % target_head_dim:
        raise ValueError(
            f"donor head_dim {donor_head_dim} is not a multiple of target {target_head_dim}"
        )
    k = donor_head_dim // target_head_dim
    half = target_head_dim // 2
    dhalf = donor_head_dim // 2
    out: List[torch.Tensor] = []
    for h in range(n_heads):
        base = h * donor_head_dim
        for s in range(k):
            lo = [base + s + j * k for j in range(half)]
            hi = [base + s + j * k + dhalf for j in range(half)]
            out.append(torch.tensor(lo + hi))
    return out


@torch.no_grad()
def select_heads_split(
    spec: DonorSpec, n_q: int, n_kv: int, target_head_dim: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, object]]:
    """Head selection in the *virtual* head space produced by :func:`head_row_map`.

    Returns the donor row indices for the chosen query and kv heads, plus the
    provenance (how many kv groups were spanned, whether a split happened).
    Selection still stays inside the best kv group for as long as it can.
    """
    qmap = head_row_map(spec.q_heads, spec.head_dim, target_head_dim)
    kmap = head_row_map(spec.kv_heads, spec.head_dim, target_head_dim)
    nq_v, nkv_v = len(qmap), len(kmap)
    if n_q > nq_v or n_kv > nkv_v:
        raise ValueError(f"donor offers {nq_v} virtual q / {nkv_v} virtual kv heads")
    per_group = max(nq_v // max(nkv_v, 1), 1)

    qscore = torch.zeros(nq_v)
    kscore = torch.zeros(nkv_v)
    layers = spec.attn_layers[:: max(len(spec.attn_layers) // 8, 1)][:8]
    for li in layers:
        q = spec.t(f"layers.{li}.self_attn.q_proj.weight").float()
        k = spec.t(f"layers.{li}.self_attn.k_proj.weight").float()
        v = spec.t(f"layers.{li}.self_attn.v_proj.weight").float()
        for i, rows in enumerate(qmap):
            qscore[i] += q[rows].norm()
        for i, rows in enumerate(kmap):
            kscore[i] += k[rows].norm() * v[rows].norm()
    kv_order = torch.argsort(kscore, descending=True)
    kv_sel = kv_order[:n_kv].tolist()
    q_sel: List[int] = []
    for g in kv_order.tolist():
        members = sorted(
            range(g * per_group, min((g + 1) * per_group, nq_v)),
            key=lambda h: -float(qscore[h]),
        )
        q_sel.extend(members[: n_q - len(q_sel)])
        if len(q_sel) >= n_q:
            break
    meta = {
        "virtual_q_heads": nq_v,
        "virtual_kv_heads": nkv_v,
        "q_per_kv_group": per_group,
        "kv_groups_spanned": len({h // per_group for h in q_sel}),
        "head_split_factor": spec.head_dim // target_head_dim,
        "q_heads_from": q_sel,
        "kv_heads_from": kv_sel,
    }
    return [qmap[i] for i in q_sel], [kmap[i] for i in kv_sel], meta
