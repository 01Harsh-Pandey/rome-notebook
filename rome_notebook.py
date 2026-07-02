# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.23.2",
#   "torch>=2.0.0",
#   "transformers>=4.30.0",
#   "accelerate>=0.20.0",
#   "anywidget>=0.9.0",
#   "traitlets>=5.0",
#   "numpy>=1.24",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium", auto_download=["html"])


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    import json

    import anywidget
    import numpy as np
    import torch
    import traitlets
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import marimo as mo

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    return (
        anywidget, json, np, torch, traitlets,
        AutoModelForCausalLM, AutoTokenizer,
        mo, DEVICE, GPU_NAME,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    MODEL_NAME         = "gpt2-xl"
    ROME_LAYER_DEFAULT = 17   # used only before any trace has run

    PRESET_FACTS = [
        {
            "id":      "eiffel",
            "subject": "The Eiffel Tower",
            "prompt":  "The Eiffel Tower is located in the city of",
            "true":    "Paris",
            "new":     "Rome",
            "gen":  [
                "If you visit the Eiffel Tower, you are traveling to",
                "The Eiffel Tower can be found in",
                "A tourist visiting the Eiffel Tower is in the city of",
            ],
            "spec": [
                "The Eiffel Tower was designed by",
                "The Eiffel Tower was completed in the year",
                "The Eiffel Tower is made of",
            ],
            "coh": [
                "The country that contains the Eiffel Tower is",
                "The language spoken where the Eiffel Tower stands is",
                "The currency used near the Eiffel Tower is the",
            ],
        },
        {
            "id":      "lebron",
            "subject": "LeBron James",
            "prompt":  "LeBron James plays the sport of",
            "true":    "basketball",
            "new":     "football",
            "gen":  [
                "LeBron James is professionally known for playing",
                "When LeBron James competes, he plays",
                "LeBron James earned his fame in the sport of",
            ],
            "spec": [
                "LeBron James was born in the city of",
                "The team LeBron James is most famous for is the",
                "LeBron James is famous for his ability to",
            ],
            "coh": [
                "The professional league LeBron James competes in is the",
                "The team sport associated with LeBron James requires a",
                "Athletes in the same sport as LeBron James are called",
            ],
        },
        {
            "id":      "gates",
            "subject": "Microsoft",
            "prompt":  "Microsoft was founded by Bill Gates and",
            "true":    "Paul Allen",
            "new":     "Steve Jobs",
            "gen":  [
                "The co-founder of Microsoft alongside Bill Gates was",
                "Bill Gates started Microsoft together with",
                "Microsoft's other original co-founder was",
            ],
            "spec": [
                "Microsoft's headquarters is located in",
                "The operating system created by Microsoft is called",
                "Microsoft was founded in the year",
            ],
            "coh": [
                "The company co-founded by the same person as Microsoft is",
                "Microsoft's co-founder also helped create the company called",
                "The co-founder of Microsoft later worked at",
            ],
        },
    ]
    FACTS_BY_ID  = {f["id"]: f for f in PRESET_FACTS}
    # Dropdown convention: {label_shown_to_user: value_returned_by_.value}
    FACT_OPTIONS = {f["prompt"] + "  →  ?": f["id"] for f in PRESET_FACTS}

    return MODEL_NAME, ROME_LAYER_DEFAULT, PRESET_FACTS, FACTS_BY_ID, FACT_OPTIONS


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(GPU_NAME, mo):
    _banner = f"""
    <style>
      .rome-hero {{
        border: 1px solid #d0d7de; border-radius: 12px; overflow: hidden;
        box-shadow: 0 6px 18px rgba(15,23,42,.10); margin-bottom: 20px;
      }}
      .rome-hero-grid {{
        display: grid; grid-template-columns: minmax(0,1.3fr) minmax(240px,.7fr);
        gap: 20px; padding: 28px 26px; background: #fff;
      }}
      @media (max-width: 680px) {{
        .rome-hero-grid {{ grid-template-columns: 1fr; padding: 20px 16px; }}
      }}
      .rome-badge {{
        display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px;
        border: 1px solid #bfdbfe; background: rgba(239,246,255,.85);
        border-radius: 999px; color: #1d4ed8; font-size: .74rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: .04em; margin-bottom: 12px;
      }}
      .rome-title {{ margin: 0 0 10px; color: #111827; font-size: 2.2rem;
        font-weight: 850; line-height: 1.08; }}
      .rome-desc {{ margin: 0 0 16px; color: #374151; font-size: 1rem;
        line-height: 1.55; max-width: 560px; }}
      .rome-meta {{ color: #475569; font-size: .9rem; line-height: 1.7; }}
      .rome-card {{ display: flex; flex-direction: column; justify-content: space-between;
        gap: 12px; border: 1px solid rgba(148,163,184,.35); background: rgba(248,250,252,.9);
        border-radius: 8px; padding: 14px 14px 12px; box-shadow: 0 4px 12px rgba(148,163,184,.12); }}
      .rome-card-label {{ color: #64748b; font-size: .72rem; font-weight: 850;
        text-transform: uppercase; letter-spacing: .04em; margin-bottom: 5px; }}
      .rome-card-title {{ font-size: .95rem; font-weight: 800; color: #111827; line-height: 1.3; }}
      .rome-card-authors {{ margin-top: 5px; color: #475569; font-size: .88rem; }}
      .rome-card-link {{ display: inline-block; margin-top: 7px; color: #2563eb;
        font-size: .88rem; font-weight: 700; text-decoration: none; }}
      .rome-gpu {{ margin-top: 10px; padding: 7px 10px; background: #f0fdf4;
        border: 1px solid #bbf7d0; border-radius: 6px; font-size: .8rem;
        color: #166534; font-weight: 700; }}
    </style>
    <div class="rome-hero">
      <div class="rome-hero-grid">
        <div>
          <div class="rome-badge">Paper Explainer</div>
          <h1 class="rome-title">ROME<br>
            <span style="font-size:1.1rem;font-weight:600;color:#4b5563;">
              Locating and Editing Factual Associations in GPT
            </span>
          </h1>
          <p class="rome-desc">
            Every fact a language model knows has a precise address: a handful
            of neurons at a specific layer and token position. This notebook
            finds that address by <strong>clicking on it</strong> — then uses
            the very cell you clicked to drive a
            <strong>rank-one weight update</strong> that rewrites the fact in
            under a second, with all other knowledge left intact.
          </p>
          <div class="rome-meta">
            <div>NeurIPS 2022 &nbsp;·&nbsp; Meng, Bau, Andonian &amp; Belinkov</div>
            <div>GPU: <strong>{GPU_NAME}</strong></div>
          </div>
        </div>
        <div class="rome-card">
          <div>
            <div class="rome-card-label">Original paper</div>
            <div class="rome-card-title">
              Locating and Editing Factual Associations in GPT
            </div>
            <div class="rome-card-authors">
              Kevin Meng · David Bau<br>Alex Andonian · Yonatan Belinkov
            </div>
            <a class="rome-card-link" href="https://www.alphaxiv.org/abs/2202.05262">
              alphaxiv:2202.05262
            </a>
          </div>
          <svg viewBox="0 0 260 120" role="img"
               aria-label="Weight matrix with a highlighted neuron being edited"
               style="width:100%;height:auto;display:block;margin-top:6px;">
            <defs>
              <radialGradient id="rome-glow" cx="50%" cy="50%" r="55%">
                <stop offset="0%" stop-color="#fff" stop-opacity=".98"/>
                <stop offset="45%" stop-color="#bfdbfe" stop-opacity=".75"/>
                <stop offset="100%" stop-color="#bbf7d0" stop-opacity="0"/>
              </radialGradient>
              <radialGradient id="rome-blue" cx="38%" cy="35%" r="68%">
                <stop offset="0%" stop-color="#fff" stop-opacity=".9"/>
                <stop offset="100%" stop-color="#60a5fa" stop-opacity=".9"/>
              </radialGradient>
              <filter id="rome-shadow" x="-40%" y="-40%" width="180%" height="180%">
                <feDropShadow dx="0" dy="4" stdDeviation="4"
                  flood-color="#64748b" flood-opacity=".2"/>
              </filter>
              <marker id="rome-arr" markerWidth="6" markerHeight="6"
                refX="5" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/>
              </marker>
            </defs>
            <rect x="0" y="0" width="260" height="120" rx="7" fill="#f8fafc"/>
            <circle cx="130" cy="60" r="65" fill="url(#rome-glow)"/>
            <g opacity=".35" stroke="#94a3b8" stroke-width=".8">
              <line x1="70" y1="20" x2="70" y2="100"/>
              <line x1="90" y1="20" x2="90" y2="100"/>
              <line x1="110" y1="20" x2="110" y2="100"/>
              <line x1="50" y1="35" x2="130" y2="35"/>
              <line x1="50" y1="50" x2="130" y2="50"/>
              <line x1="50" y1="65" x2="130" y2="65"/>
              <line x1="50" y1="80" x2="130" y2="80"/>
            </g>
            <rect x="70" y="50" width="20" height="15" rx="2"
              fill="#fef08a" stroke="#eab308" stroke-width="1.5"
              filter="url(#rome-shadow)"/>
            <text x="90" y="112" text-anchor="middle" font-size="9"
              fill="#64748b" font-weight="700">Layer 17 · subject last token</text>
            <path d="M155 57 L125 57" stroke="#ef4444" stroke-width="1.8"
              marker-end="url(#rome-arr)" fill="none"/>
            <g transform="translate(155,45)">
              <rect x="0" y="0" width="28" height="10" rx="2"
                fill="url(#rome-blue)" stroke="#fff" stroke-width="1"/>
              <text x="14" y="8" text-anchor="middle" font-size="7.5"
                fill="#1e3a5f" font-weight="800">ROME edit</text>
            </g>
            <text x="70" y="17" text-anchor="middle" font-size="8" fill="#475569">The</text>
            <text x="90" y="17" text-anchor="middle" font-size="8" fill="#2563eb" font-weight="700">Tower</text>
            <text x="110" y="17" text-anchor="middle" font-size="8" fill="#475569">is</text>
          </svg>
          <div class="rome-gpu">⚡ Running on {GPU_NAME}</div>
        </div>
      </div>
    </div>
    """
    mo.Html(_banner)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reader's note.** Run all cells top-to-bottom. GPU work is gated behind
    explicit buttons. The causal-trace heatmap in Step 3 is *clickable* — the
    cell you click becomes the layer ROME edits in Step 4, no separate slider
    to keep in sync.

    ---

    ## The Problem: Facts Are Frozen in Weights

    After pretraining, GPT-2 XL *knows* the Eiffel Tower is in Paris — that
    knowledge lives as a pattern across 1.5 billion parameters. Now the fact
    needs to change, or turns out to be wrong, or is copyrighted and must be
    removed. The options are grim:

    | Approach | Cost | Risk |
    |----------|------|------|
    | **Full retraining** | ~$1M, months | Impractical |
    | **Fine-tuning on corrected data** | Days | Catastrophic forgetting |
    | **ROME** | < 60 seconds | Surgically precise |

    ROME works because each fact has a *precise address* inside the model.
    The paper's two contributions, which this notebook lets you operate
    directly: **causal tracing** to find the address, and **rank-one
    editing** to overwrite it.
    """)
    return


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md("## Step 1 — Load GPT-2 XL")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack([
        mo.stat("1.5 B", label="Parameters",  caption="GPT-2 XL"),
        mo.stat("48",    label="Layers",       caption="Transformer blocks"),
        mo.stat("1600",  label="d_model",      caption="Residual stream width"),
        mo.stat("6400",  label="MLP width",    caption="4 × d_model"),
    ], gap=1, justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    load_btn = mo.ui.run_button(
        label="⚡  Load GPT-2 XL  (first run: ~2 min download)",
        kind="success", full_width=True,
    )
    load_btn
    return (load_btn,)


@app.cell(hide_code=True)
def _(load_btn, MODEL_NAME, DEVICE,
      AutoModelForCausalLM, AutoTokenizer, torch, mo):
    if not load_btn.value:
        model = None
        tokenizer = None
        load_view = mo.md(
            "Click **Load GPT-2 XL** above to begin. "
            "Downloads ~6 GB of weights on first run."
        ).callout(kind="info")
    else:
        with mo.status.spinner(title="Loading GPT-2 XL onto GPU…"):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float32,
            ).to(DEVICE)
            model.eval()
            # ROME's edit is a direct in-place weight write (rome_apply),
            # never gradient descent on the model itself — only the value
            # vector `v` in rome_value ever needs gradients. Without this,
            # every loss.backward() call in rome_value silently allocates
            # a full .grad buffer for all 1.5B model parameters (~6.4 GB),
            # on top of the weights themselves, for zero benefit.
            for _p in model.parameters():
                _p.requires_grad_(False)
        _vram = (
            f"{torch.cuda.memory_allocated()/1e9:.1f} GB VRAM"
            if DEVICE == "cuda" else "CPU"
        )
        load_view = mo.md(
            f"✅ **Loaded.** GPT-2 XL on `{DEVICE}` · {_vram} · "
            f"parameters frozen (gradients only ever flow to the value vector)"
        ).callout(kind="success")

    load_view
    return model, tokenizer, load_view


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SELECT A FACT, PROBE THE MODEL
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 2 — Pick a Fact

    This selector drives everything below: the live probe, the causal
    trace, and the edit. Select once.
    """)
    return


@app.cell(hide_code=True)
def _(FACT_OPTIONS, mo):
    fact_selector = mo.ui.dropdown(
        options=FACT_OPTIONS,
        value=list(FACT_OPTIONS.keys())[0],   # KEY = label shown to user
        label="Select a factual prompt",
        full_width=True,
    )
    fact_selector
    return (fact_selector,)


@app.cell(hide_code=True)
def _(fact_selector, FACTS_BY_ID, model, tokenizer, DEVICE, torch, mo):
    if model is None:
        fact_view = mo.md("Load the model in Step 1.").callout(kind="info")
        current_fact = None
    else:
        # FACT_OPTIONS = {label: id} → fact_selector.value returns the id
        current_fact = FACTS_BY_ID[fact_selector.value]

        with torch.no_grad():
            _inp    = tokenizer(current_fact["prompt"], return_tensors="pt").to(DEVICE)
            _logits = model(**_inp).logits[0, -1, :]
            _probs  = torch.softmax(_logits, dim=-1)
            _tv, _ti = torch.topk(_probs, 8)

        _tokens = [tokenizer.decode([t]).strip() for t in _ti]
        _values = [round(float(v) * 100, 2) for v in _tv]
        _maxp   = max(_values)

        _bars = "".join(
            f"""<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">
              <code style="width:110px;text-align:right;font-size:13px;
                  color:#111827;flex-shrink:0;">{repr(t)}</code>
              <div style="flex:1;background:#f1f5f9;border-radius:4px;height:22px;overflow:hidden;">
                <div style="width:{min(p/_maxp*100,100):.1f}%;
                    background:{'#2563eb' if t.strip().lower()==current_fact['true'].lower() else '#94a3b8'};
                    height:100%;border-radius:4px;"></div>
              </div>
              <span style="width:44px;font-size:12px;color:#64748b;
                  font-variant-numeric:tabular-nums;">{p:.1f}%</span>
            </div>"""
            for t, p in zip(_tokens, _values)
        )
        _correct = _tokens[0].strip().lower() == current_fact["true"].lower()

        fact_view = mo.Html(f"""
        <div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 16px;
            background:#fff;box-shadow:0 2px 8px rgba(15,23,42,.06);">
          <div style="font-size:12px;color:#64748b;font-family:monospace;margin-bottom:10px;">
            {current_fact['prompt']} ___
          </div>
          {_bars}
          <div style="margin-top:10px;font-size:12px;color:#64748b;">
            {"✅ Model correctly answers: <strong>" + current_fact['true'] + "</strong>"
             if _correct else
             "⚠️ Model top prediction: <strong>" + _tokens[0] + "</strong>"}
          </div>
        </div>
        """)

    fact_view
    return current_fact, fact_view


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CAUSAL TRACE: A CLICKABLE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 3 — Causal Tracing: Click to Find the Fact's Address

    The paper locates a fact with **causal mediation analysis**:

    - **Clean run** — normal prompt → P(*Paris*) is high
    - **Corrupt run** — noise added to the subject's token embeddings →
      P(*Paris*) collapses near zero
    - **Patch run** — at each *(layer $\ell$, token $t$)* pair, restore the
      clean activation and remeasure how much P(*Paris*) recovers

    $$\text{Indirect Effect}(\ell, t) =
    \frac{P_{\text{patched}} - P_{\text{corrupted}}}{P_{\text{clean}} - P_{\text{corrupted}}}$$

    The result is a heatmap over layers × token positions — the fact's
    address, visible to the naked eye.

    **Click any cell once the trace finishes.** That cell becomes the layer
    ROME targets in Step 4 below — there's no separate slider to keep in sync.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    trace_samples_ui = mo.ui.slider(
        10, 40, value=20, step=5, show_value=True,
        label="Noise samples  (higher → smoother heatmap, slower)",
    )
    trace_samples_ui
    return (trace_samples_ui,)


@app.cell(hide_code=True)
def _(mo):
    trace_btn = mo.ui.run_button(
        label="🔍  Run Causal Trace  (~60 s)",
        kind="warn", full_width=True,
    )
    trace_btn
    return (trace_btn,)


@app.function
def rome_subject_range(tokenizer, prompt, subject):
    """Return (start, end) exclusive token indices of subject in prompt."""
    full = tokenizer.encode(prompt)
    for prefix in (" " + subject, subject):
        sub = tokenizer.encode(prefix)
        for i in range(len(full) - len(sub) + 1):
            if full[i:i+len(sub)] == sub:
                return i, i + len(sub)
    return 0, 1


@app.function
def causal_trace(model, tokenizer, prompt, subject, device,
                  n_corrupt=10, noise_coef=3.0):
    """
    From-scratch causal mediation analysis (Meng et al. 2022, Section 3).

    We do NOT rely on the external causal-tracer package: its installed
    version's noise-magnitude kwarg name doesn't match anything this
    notebook tried (noise/noise_level/noise_std/std/sigma all rejected),
    which silently ran the corruption step with zero noise and produced
    meaningless near-zero indirect effects across the board. This
    implementation uses the exact same forward-hook pattern already
    verified working in rome_key/rome_value/rome_apply below.

    Returns:
      scores  : np.ndarray, shape (n_tokens, n_layers) — indirect effect
      tokens  : list[str] — decoded prompt tokens
      srange  : (start, end) exclusive — subject token span
      diag    : dict with p_clean, p_corrupt, target_tok_str — lets the
                caller verify the corruption actually suppressed the
                prediction (denom = p_clean - p_corrupt), rather than
                trusting a high indirect-effect number blindly. A near-
                zero denom would silently inflate every score toward the
                clip bounds, exactly as happened with the earlier
                external causal-tracer bug.
    """
    import torch
    import numpy as np

    s, e   = rome_subject_range(tokenizer, prompt, subject)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_tok  = inputs["input_ids"].shape[1]
    n_lay  = model.config.n_layer
    noise_std = noise_coef * model.transformer.wte.weight.std().item()

    # Target: the model's own top prediction on the clean run — this is
    # the "fact" we're tracing, whatever it currently answers.
    with torch.no_grad():
        clean_logits = model(**inputs).logits[0, -1, :]
        target_tok   = int(clean_logits.argmax())
        p_clean      = float(torch.softmax(clean_logits, dim=-1)[target_tok])

    def noise_hook(mod, inp, out):
        o = out.clone()
        o[0, s:e] += torch.randn_like(o[0, s:e]) * noise_std
        return o

    # ── Corrupted baseline: noise on subject embeddings, no patching ──────
    p_corrupt_runs = []
    for _ in range(n_corrupt):
        h = model.transformer.wte.register_forward_hook(noise_hook)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        h.remove()
        p_corrupt_runs.append(
            float(torch.softmax(logits, dim=-1)[target_tok])
        )
    p_corrupt = sum(p_corrupt_runs) / len(p_corrupt_runs)

    scores = np.zeros((n_tok, n_lay), dtype=np.float32)
    denom  = max(p_clean - p_corrupt, 1e-4)  # guard against near-zero gap

    # ── Capture clean hidden states at every layer (for patching) ────────
    # NOTE: unlike rome_key/rome_value (which hook mlp.c_proj and are
    # proven to receive a plain tensor), this hooks the FULL GPT2Block.
    # Block-level forward output convention (tuple vs bare tensor) hasn't
    # been independently verified against this environment's transformers
    # version, so both shapes are handled explicitly rather than assumed.
    clean_states = {}

    def _unwrap_hidden(out):
        """Return (hidden_states_tensor, rest_of_tuple_or_None)."""
        if isinstance(out, tuple):
            return out[0], out[1:]
        return out, None  # bare tensor, no extra outputs to preserve

    def make_capture(layer_idx):
        def hook(mod, inp, out):
            hs, _ = _unwrap_hidden(out)
            clean_states[layer_idx] = hs.detach().clone()
        return hook

    caps = [
        model.transformer.h[l].register_forward_hook(make_capture(l))
        for l in range(n_lay)
    ]
    with torch.no_grad():
        model(**inputs)
    for h in caps:
        h.remove()

    # ── Patch each (layer, token) cell: corrupt + restore one position ───
    for layer_id in range(n_lay):
        def make_patch(lid):
            def hook(mod, inp, out):
                hs, rest = _unwrap_hidden(out)
                o = hs.clone()
                o[0, patch_pos[0]] = clean_states[lid][0, patch_pos[0]]
                return o if rest is None else (o,) + rest
            return hook

        patch_pos = [0]  # mutable cell for closure
        h_noise = model.transformer.wte.register_forward_hook(noise_hook)
        h_patch = model.transformer.h[layer_id].register_forward_hook(
            make_patch(layer_id)
        )
        for tok_pos in range(n_tok):
            patch_pos[0] = tok_pos
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            p_restored = float(torch.softmax(logits, dim=-1)[target_tok])
            scores[tok_pos, layer_id] = max(
                0.0, min(1.0, (p_restored - p_corrupt) / denom)
            )
        h_noise.remove()
        h_patch.remove()

    tokens = [
        tokenizer.decode([t]).strip() or "·"
        for t in inputs["input_ids"][0]
    ]
    diag = {
        "p_clean":        p_clean,
        "p_corrupt":      p_corrupt,
        "denom":          denom,
        "target_tok_str": tokenizer.decode([target_tok]).strip(),
        "n_subject_tok":  e - s,
    }
    return scores, tokens, (s, e), diag


@app.cell(hide_code=True)
def _(trace_btn, current_fact, trace_samples_ui,
      model, tokenizer, DEVICE, causal_trace, mo):
    if not trace_btn.value:
        trace_result = None
        trace_view = mo.md(
            "Click **Run Causal Trace** to map where this fact is stored."
        ).callout(kind="info")
    elif current_fact is None or model is None:
        trace_result = None
        trace_view = mo.md("Load the model and pick a fact first.").callout(kind="warn")
    else:
        try:
            with mo.status.spinner(title="Running causal trace… (~20–40 s)"):
                _scores, _tokens, _srange, _diag = causal_trace(
                    model, tokenizer,
                    current_fact["prompt"], current_fact["subject"], DEVICE,
                    n_corrupt=trace_samples_ui.value,
                )

            _sl      = _srange[1] - 1
            _top_lay = int(_scores[_sl, :].argmax())
            _top_scr = float(_scores[_sl, :].max())

            trace_result = {
                "scores":    _scores,
                "tokens":    _tokens,
                "srange":    _srange,
                "top_layer": _top_lay,
                "top_score": _top_scr,
                "prompt":    current_fact["prompt"],
                "diag":      _diag,
            }

            # Verify the corruption gap is real, not a near-zero denominator
            # masquerading as a high score (the exact failure mode of the
            # earlier external causal-tracer bug). A healthy gap is at
            # least a few percentage points; anything smaller means the
            # indirect-effect ratio is numerically unstable regardless of
            # what layer it points to.
            _gap_pct = (_diag["p_clean"] - _diag["p_corrupt"]) * 100
            _gap_note = (
                f"Corruption gap: P({_diag['target_tok_str']!r}) "
                f"{_diag['p_clean']:.3f} clean → {_diag['p_corrupt']:.3f} "
                f"corrupted ({_gap_pct:.1f} pts — "
                + ("healthy, scores are meaningful)"
                   if _gap_pct > 3 else
                   "⚠️ thin — treat scores with caution)")
            )

            # Single-token, sentence-initial subjects (e.g. "Microsoft")
            # have no earlier token to attend back to under causal masking,
            # so patching their own position at ANY layer — even the very
            # first — fully undoes the corruption immediately. This makes
            # very-early peaks for such subjects architecturally expected,
            # not evidence of where the fact is "stored" in the sense the
            # paper means for multi-token subjects.
            _single_tok_note = ""
            if _diag["n_subject_tok"] == 1 and _srange[0] == 0 and _top_lay <= 3:
                _single_tok_note = (
                    "\n\n⚠️ **Single-token, sentence-initial subject.** "
                    "Under causal attention this token has nothing earlier "
                    "to attend back to, so restoring it at *any* layer — "
                    "even layer 1 — fully undoes the corruption immediately. "
                    "This peak reflects that architectural fact more than "
                    "it reveals a meaningful mid-stack storage layer. "
                    "Compare to a multi-token subject (Eiffel Tower, "
                    "LeBron James) for the paper's intended signal."
                )

            trace_view = mo.md(
                f"✅ **Trace complete.** "
                f"Shape `{_scores.shape}` (tokens × layers) · "
                f"Peak: **layer {_top_lay}** "
                f"(indirect effect = {_top_scr:.3f}) — "
                f"click that cell below, or any other, to target it.\n\n"
                f"{_gap_note}"
                f"{_single_tok_note}"
            ).callout(kind="success")
        except Exception as _exc:
            trace_result = None
            trace_view = mo.md(
                f"❌ **Causal trace failed:** `{type(_exc).__name__}: {_exc}`\n\n"
                f"This is the from-scratch implementation hooking "
                f"`model.transformer.h[layer]` directly — share this exact "
                f"message so the hook logic can be corrected against the "
                f"real output shape."
            ).callout(kind="danger")

    trace_view
    return trace_result, trace_view


# ─── Causal Heatmap Widget — click sets the edit layer directly ────────────

@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class CausalHeatmapWidget(anywidget.AnyWidget):
        """
        Interactive causal-trace heatmap.
        scores[t][l] = indirect effect for token t at layer l.
        X = token position, Y = layer (0 at bottom).

        Clicking a cell sets `selected_layer` / `selected_token` and pushes
        the change back to Python via model.set + save_changes — the same
        pattern used to drive a live readout from a single user gesture.
        The computed paper-peak cell is shown with a dashed amber ring;
        the user's current click is a solid blue ring. They can be the
        same cell or different ones.
        """
        _esm = r"""
        export default {
          render({ model, el }) {
            function draw() {
              const raw = model.get("hm_data");
              if (!raw) { el.innerHTML = ""; return; }
              const { scores, tokens, subject_range, top_layer } = JSON.parse(raw);
              const nT = tokens.length, nL = scores[0].length;
              const CW = Math.max(9,  Math.floor(620 / nT));
              const CH = Math.max(4,  Math.floor(260 / nL));
              const PL = 38, PR = 72, PT = 18, PB = 68;
              const W  = PL + nT*CW + PR, H = PT + nL*CH + PB;
              const [ss, se] = subject_range;
              const sl = se - 1;

              let selLayer = model.get("selected_layer");
              let selToken = model.get("selected_token");
              if (selLayer < 0) { selLayer = top_layer; selToken = sl; }

              function col(s) {
                const t = Math.min(1, Math.max(0, s));
                return `rgb(${Math.round(255-t*186)},${Math.round(255-t*185)},${Math.round(255-t*26)})`;
              }

              let cells = "", tlabs = "", lticks = "";
              for (let t = 0; t < nT; t++)
                for (let l = 0; l < nL; l++) {
                  const x = PL + t*CW, y = PT + (nL-1-l)*CH;
                  const isPaperPeak = (t === sl && l === top_layer);
                  const isSelected  = (t === selToken && l === selLayer);
                  let stroke = "none", sw = 0, dash = "";
                  if (isPaperPeak) { stroke = "#d97706"; sw = 1.6; dash = '2,2'; }
                  if (isSelected)  { stroke = "#2563eb"; sw = 2.2; dash = ""; }
                  cells += `<rect x="${x}" y="${y}" width="${CW-1}" height="${CH-1}"
                    fill="${col(scores[t][l])}"
                    stroke="${stroke}" stroke-width="${sw}"
                    stroke-dasharray="${dash}" rx="1"
                    data-t="${t}" data-l="${l}" data-s="${scores[t][l].toFixed(3)}"
                    class="hmcell" style="cursor:pointer;"/>`;
                }

              for (let t = 0; t < nT; t++) {
                const cx = PL + t*CW + CW/2, ty = PT + nL*CH + 12;
                const subj = t>=ss && t<se;
                tlabs += `<text x="${cx}" y="${ty}" text-anchor="end" font-size="11"
                  fill="${subj?"#2563eb":"#475569"}" font-weight="${subj?"700":"400"}"
                  transform="rotate(-45,${cx},${ty})">${tokens[t]}</text>`;
              }
              for (let l = 0; l < nL; l += 8) {
                const ty = PT + (nL-1-l)*CH + CH/2 + 4;
                lticks += `<text x="${PL-4}" y="${ty}" text-anchor="end"
                  font-size="10" fill="#475569">${l}</text>`;
              }

              el.innerHTML = `
              <div style="border:1px solid #d0d7de;border-radius:8px;background:#fff;
                  padding:12px 14px;box-shadow:0 2px 8px rgba(15,23,42,.06);">
                <div style="overflow-x:auto;">
                  <svg width="${W}" height="${H}" style="display:block;max-width:100%;">
                    ${cells}${tlabs}${lticks}
                    <text x="13" y="${PT+nL*CH/2}" text-anchor="middle" font-size="11"
                      fill="#475569" transform="rotate(-90,13,${PT+nL*CH/2})">Layer</text>
                    <text x="${PL+nT*CW/2}" y="${H-3}" text-anchor="middle"
                      font-size="11" fill="#475569">← Token position →</text>
                  </svg>
                </div>
                <div style="font-size:11px;color:#64748b;margin-top:5px;">
                  🔵 Blue solid ring = your selection &nbsp;·&nbsp;
                  🟠 Amber dashed ring = paper-predicted peak &nbsp;·&nbsp;
                  Bright = high indirect effect
                </div>
                <div id="htip" style="font-family:monospace;font-size:12px;
                  color:#111827;min-height:18px;margin-top:4px;"></div>
              </div>`;

              const tip = el.querySelector("#htip");
              el.querySelectorAll(".hmcell").forEach(c => {
                c.addEventListener("mouseover", () => {
                  tip.textContent =
                    `Token "${tokens[+c.dataset.t]}" · Layer ${c.dataset.l} · `
                    + `Indirect effect = ${c.dataset.s}`;
                });
                c.addEventListener("click", () => {
                  model.set("selected_layer", +c.dataset.l);
                  model.set("selected_token", +c.dataset.t);
                  model.save_changes();
                });
              });
            }
            draw();
            model.on("change:hm_data", draw);
            model.on("change:selected_layer", draw);
            model.on("change:selected_token", draw);
          }
        };
        """
        _css = ".hmcell:hover{opacity:.7}"
        hm_data        = traitlets.Unicode("").tag(sync=True)
        selected_layer = traitlets.Int(-1).tag(sync=True)
        selected_token = traitlets.Int(-1).tag(sync=True)

    return (CausalHeatmapWidget,)


@app.cell(hide_code=True)
def _(trace_result, CausalHeatmapWidget, json, mo):
    if trace_result is None:
        heatmap_widget = None
        heatmap_view = mo.md(
            "Run causal trace above to see the clickable heatmap."
        ).callout(kind="info")
    else:
        _w = CausalHeatmapWidget(hm_data=json.dumps({
            "scores":        trace_result["scores"].tolist(),
            "tokens":        trace_result["tokens"],
            "subject_range": list(trace_result["srange"]),
            "top_layer":     trace_result["top_layer"],
        }))
        heatmap_widget = mo.ui.anywidget(_w)
        heatmap_view = mo.vstack([
            mo.md(f"### Causal Trace — *\"{trace_result['prompt']}\"*"),
            heatmap_widget,
        ], gap=1)

    heatmap_view
    return heatmap_widget, heatmap_view


@app.cell(hide_code=True)
def _(heatmap_widget, trace_result, mo):
    # Bidirectional loop: read back the live click state from the JS widget.
    # heatmap_widget.value is a dict of every synced traitlet.
    if heatmap_widget is None or trace_result is None:
        selected_layer = None
        selected_token = None
        select_view = mo.md("")
    else:
        _sel_l = heatmap_widget.value.get("selected_layer", -1)
        _sel_t = heatmap_widget.value.get("selected_token", -1)
        selected_layer = _sel_l if _sel_l >= 0 else trace_result["top_layer"]
        selected_token = _sel_t if _sel_t >= 0 else (trace_result["srange"][1] - 1)

        _peak_l = trace_result["top_layer"]
        _delta  = abs(selected_layer - _peak_l)
        _tok    = trace_result["tokens"][selected_token]

        if _delta == 0:
            _msg, _kind = (
                f"**You selected the paper-predicted peak** — layer {selected_layer}, "
                f"token \"{_tok}\". This is the strongest causal address for this fact.",
                "success",
            )
        elif _delta <= 3:
            _msg, _kind = (
                f"**Selected layer {selected_layer}**, token \"{_tok}\" — "
                f"{_delta} layers from the paper's peak (layer {_peak_l}). "
                f"Still inside the causal band; the edit should work, "
                f"just slightly less precisely targeted.",
                "neutral",
            )
        else:
            _msg, _kind = (
                f"**Selected layer {selected_layer}**, token \"{_tok}\" — "
                f"{_delta} layers from the paper's peak (layer {_peak_l}). "
                f"This is outside the band the heatmap lit up; expect a "
                f"weaker or less reliable edit. Click a brighter cell to "
                f"see the difference.",
                "warn",
            )
        select_view = mo.md(_msg).callout(kind=_kind)

    select_view
    return selected_layer, selected_token, select_view


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — APPLY THE ROME EDIT (uses the clicked layer)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 4 — Apply the ROME Edit

    Using whichever layer you clicked above (or the paper-predicted peak,
    if you haven't clicked yet), the edit runs in three steps:

    1. **Key vector $k^*$** — average MLP hidden activation at the subject's
       last token, over 20 noise-corrupted runs (so $k^*$ is subject-generic,
       not prompt-specific).
    2. **Value vector $v^*$** — gradient-optimised so injecting it at the
       subject's position maximises $P(\text{target\_new})$ across paraphrases.
    3. **Rank-one update** — for GPT-2's `Conv1D` weight $W$ (shape 6400×1600):

    $$W' = W + \frac{(v^* - k^{*\top}W)\,k^{*}}{\lVert k^* \rVert^2}$$

    The *minimum-norm* change: it redirects exactly the key direction $k^*$
    and leaves every other direction mathematically unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(selected_layer, ROME_LAYER_DEFAULT, mo):
    edit_target_ui = mo.ui.text(
        value="Rome", label="Rewrite answer to", placeholder="new target…",
    )
    _layer_caption = (
        f"Using layer {selected_layer} from your heatmap click above"
        if selected_layer is not None
        else f"No trace run yet — defaulting to layer {ROME_LAYER_DEFAULT}"
    )
    mo.vstack([
        mo.md("**New target**"), edit_target_ui,
        mo.md(f"_{_layer_caption}_"),
    ], gap=1)
    return (edit_target_ui,)


@app.cell(hide_code=True)
def _(mo):
    edit_btn = mo.ui.run_button(
        label="✏️  Apply ROME Edit", kind="danger", full_width=True,
    )
    edit_btn
    return (edit_btn,)


# ── ROME helpers ───────────────────────────────────────────────────────────

@app.function
def rome_key(model, tokenizer, prompt, subject, layer, device, n=20, coef=3.0):
    import torch
    s, e  = rome_subject_range(tokenizer, prompt, subject)
    sl    = e - 1
    inp   = tokenizer(prompt, return_tensors="pt").to(device)
    ns    = coef * model.transformer.wte.weight.std().item()
    keys, cap = [], {}

    def hook_k(mod, inp_, out):
        cap["k"] = inp_[0][0, sl, :].detach().float()

    def hook_n(mod, inp_, out):
        o = out.clone()
        o[0, s:e] += torch.randn_like(o[0, s:e]) * ns
        return o

    hp = model.transformer.h[layer].mlp.c_proj.register_forward_hook(hook_k)
    he = model.transformer.wte.register_forward_hook(hook_n)
    for _ in range(n):
        with torch.no_grad():
            model(**inp)
        if "k" in cap:
            keys.append(cap.pop("k"))
    hp.remove(); he.remove()
    return torch.stack(keys).mean(0)


@app.function
def rome_value(model, tokenizer, request, layer, key, device, steps=25, lr=0.05):
    import torch
    import torch.nn.functional as F
    tok  = tokenizer.encode(" " + request["new"].strip())[0]
    ps   = ([request["prompt"]] + request.get("gen", []))[:3]
    W    = model.transformer.h[layer].mlp.c_proj.weight.float()  # (6400,1600)
    v    = (key.to(W.device).float() @ W).clone().detach().requires_grad_(True)
    opt  = torch.optim.Adam([v], lr=lr)
    vr   = [v]

    for _ in range(steps):
        opt.zero_grad()
        loss = torch.tensor(0., device=device)
        for p in ps:
            _, e = rome_subject_range(tokenizer, p, request["subject"])
            sp   = e - 1
            i    = tokenizer(p, return_tensors="pt").to(device)
            def inj(mod, inp_, out, _sp=sp, _vr=vr):
                o = out.clone().float(); o[0,_sp] = _vr[0]; return o
            h = model.transformer.h[layer].mlp.c_proj.register_forward_hook(inj)
            out = model(**i); h.remove()
            loss -= F.log_softmax(out.logits[0,-1].float(), -1)[tok] / len(ps)
        loss.backward(); opt.step(); vr[0] = v
    return v.detach()


@app.function
def rome_apply(model, layer, key, val):
    import torch
    W  = model.transformer.h[layer].mlp.c_proj.weight   # (6400,1600)
    k  = key.to(W.device).float()
    vs = val.to(W.device).float()
    with torch.no_grad():
        delta  = vs - k @ W.float()
        update = torch.outer(k, delta) / (k @ k + 1e-8)
        W.data += update.to(W.dtype)


@app.cell(hide_code=True)
def _(edit_btn, current_fact, edit_target_ui, selected_layer, ROME_LAYER_DEFAULT,
      model, tokenizer, DEVICE, torch,
      rome_key, rome_value, rome_apply, mo):

    if not edit_btn.value or model is None or current_fact is None:
        edit_result = None
        edit_exec_view = (
            mo.md("Configure the edit above and click **Apply ROME Edit**.")
            .callout(kind="info")
            if model is not None else
            mo.md("Load the model in Step 1.").callout(kind="warn")
        )
    else:
        try:
            _target = edit_target_ui.value.strip()
            _layer  = selected_layer if selected_layer is not None else ROME_LAYER_DEFAULT
            _request = {**current_fact, "new": _target}

            _all_prompts = (
                [current_fact["prompt"]]
                + current_fact["gen"]
                + current_fact["spec"]
                + current_fact["coh"]
            )
            _before = {}
            with torch.no_grad():
                for _p in _all_prompts:
                    _inp = tokenizer(_p, return_tensors="pt").to(DEVICE)
                    _before[_p] = tokenizer.decode(
                        [model(**_inp).logits[0,-1].argmax()]
                    ).strip()

            with torch.no_grad():
                _inp_b   = tokenizer(current_fact["prompt"], return_tensors="pt").to(DEVICE)
                _probs_b = torch.softmax(model(**_inp_b).logits[0,-1], -1)
                _tv_b, _ti_b = torch.topk(_probs_b, 6)
            _before_dist = [
                {"tok": tokenizer.decode([t]).strip(), "pct": round(float(p)*100, 2)}
                for t, p in zip(_ti_b, _tv_b)
            ]

            with mo.status.spinner(title="Step 1/3 — Computing key vector…"):
                _k = rome_key(model, tokenizer, _request["prompt"],
                             _request["subject"], _layer, DEVICE)
            with mo.status.spinner(title="Step 2/3 — Optimising value vector…"):
                _v = rome_value(model, tokenizer, _request, _layer, _k, DEVICE)
            with mo.status.spinner(title="Step 3/3 — Applying rank-one weight update…"):
                rome_apply(model, _layer, _k, _v)
                model.eval()

            with torch.no_grad():
                _inp_a   = tokenizer(current_fact["prompt"], return_tensors="pt").to(DEVICE)
                _probs_a = torch.softmax(model(**_inp_a).logits[0,-1], -1)
                _tv_a, _ti_a = torch.topk(_probs_a, 6)
            _after_dist = [
                {"tok": tokenizer.decode([t]).strip(), "pct": round(float(p)*100, 2)}
                for t, p in zip(_ti_a, _tv_a)
            ]

            edit_result = {
                "before": _before, "before_dist": _before_dist,
                "after_dist": _after_dist, "fact": current_fact,
                "target": _target, "layer": _layer,
            }
            edit_exec_view = mo.md(
                f"✅ **Done.** Layer {_layer} (your heatmap selection) · "
                f"*{current_fact['subject']}* → **{_target}**"
            ).callout(kind="success")
        except Exception as _exc:
            edit_result = None
            edit_exec_view = mo.md(
                f"❌ **Edit failed:** `{type(_exc).__name__}: {_exc}`\n\n"
                f"Share this exact message to pin down the fix."
            ).callout(kind="danger")

    edit_exec_view
    return (edit_result,)


# ─── Memory Rewrite Widget ──────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class MemoryRewriteWidget(anywidget.AnyWidget):
        """Side-by-side animated probability bars, before vs after the edit."""
        _esm = r"""
        export default {
          render({ model, el }) {
            function draw() {
              const raw = model.get("payload");
              if (!raw) { el.innerHTML = ""; return; }
              const { before, after, target, prompt } = JSON.parse(raw);
              const maxB = Math.max(...before.map(x => x.pct), .01);
              const maxA = Math.max(...after.map(x => x.pct), .01);

              function row(item, max, side, idx) {
                const w   = (item.pct / max * 100).toFixed(1);
                const hit = item.tok.trim().toLowerCase() === target.trim().toLowerCase();
                const bg  = hit
                  ? (side === "after" ? "#16a34a" : "#94a3b8")
                  : (side === "after" ? "#2563eb" : "#94a3b8");
                return `
                <div style="display:flex;align-items:center;gap:7px;margin:5px 0;">
                  <code style="width:100px;text-align:right;font-size:12px;color:#111827;
                      flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                  >${JSON.stringify(item.tok)}</code>
                  <div style="flex:1;background:#f1f5f9;border-radius:3px;height:19px;overflow:hidden;">
                    <div class="rome-bar" style="width:${w}%;background:${bg};height:100%;
                        border-radius:3px;animation-delay:${idx*55}ms;"></div>
                  </div>
                  <span style="width:40px;font-size:11px;color:#64748b;
                      font-variant-numeric:tabular-nums;">${item.pct.toFixed(1)}%</span>
                </div>`;
              }

              el.innerHTML = `
              <div style="border:1px solid #d0d7de;border-radius:8px;background:#fff;
                  padding:14px 16px;box-shadow:0 2px 8px rgba(15,23,42,.06);">
                <div style="font-family:monospace;font-size:12px;color:#64748b;
                    margin-bottom:12px;">${prompt} ___</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                  <div>
                    <div style="font-size:11px;font-weight:800;text-transform:uppercase;
                        letter-spacing:.04em;color:#64748b;margin-bottom:8px;">Before edit</div>
                    ${before.map((x,i) => row(x, maxB, "before", i)).join("")}
                  </div>
                  <div>
                    <div style="font-size:11px;font-weight:800;text-transform:uppercase;
                        letter-spacing:.04em;color:#16a34a;margin-bottom:8px;">After ROME edit ✓</div>
                    ${after.map((x,i) => row(x, maxA, "after", i)).join("")}
                  </div>
                </div>
              </div>`;
            }
            draw();
            model.on("change:payload", draw);
          }
        };
        """
        _css = """
        @keyframes rome-slide-in {
          from { width: 0% !important; opacity: 0; }
          to   { opacity: 1; }
        }
        .rome-bar { animation: rome-slide-in 500ms cubic-bezier(.22,1,.36,1) both; }
        """
        payload = traitlets.Unicode("").tag(sync=True)

    return (MemoryRewriteWidget,)


@app.cell(hide_code=True)
def _(edit_result, MemoryRewriteWidget, json, mo):
    if edit_result is None:
        rewrite_view = mo.md(
            "The animated before/after comparison will appear here after the edit."
        ).callout(kind="info")
    else:
        _w = MemoryRewriteWidget(payload=json.dumps({
            "before": edit_result["before_dist"],
            "after":  edit_result["after_dist"],
            "target": edit_result["target"],
            "prompt": edit_result["fact"]["prompt"],
        }))
        rewrite_view = mo.vstack([
            mo.md("### Memory Rewrite — Probability Shift"),
            mo.ui.anywidget(_w),
        ], gap=1)

    rewrite_view
    return (rewrite_view,)


# ─────────────────────────────────────────────────────────────────────────────
# HOW SURGICAL WAS THE EDIT
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## How Surgical Was the Edit?

    Three test categories verify ROME changed *only* what we asked.
    """)
    return


@app.cell(hide_code=True)
def _(edit_result, model, tokenizer, DEVICE, torch, mo):
    if edit_result is None:
        ba_view = mo.md(
            "Apply an edit above to see efficacy · generalization · specificity."
        ).callout(kind="info")
    else:
        _fact   = edit_result["fact"]
        _target = edit_result["target"]
        _before = edit_result["before"]

        _prompts = (
            [(_fact["prompt"], "✅ Efficacy")]
            + [(_p, "🔄 Generalization") for _p in _fact["gen"]]
            + [(_p, "🎯 Specificity")    for _p in _fact["spec"]]
        )

        _rows_html = ""
        with torch.no_grad():
            for _p, _cat in _prompts:
                _inp   = tokenizer(_p, return_tensors="pt").to(DEVICE)
                _after = tokenizer.decode([model(**_inp).logits[0,-1].argmax()]).strip()
                _bef   = _before.get(_p, "—")
                _chg   = _bef != _after

                if _cat.startswith("🎯"):
                    _v, _vc = ("✅ unchanged", "#16a34a") if not _chg else ("⚠️ changed", "#d97706")
                else:
                    _hit = _after.strip().lower() == _target.strip().lower()
                    _v, _vc = ("✅", "#16a34a") if _hit else ("❌", "#dc2626")

                _rows_html += (
                    f"<tr><td style='padding:7px;font-size:.85rem;'>{_cat}</td>"
                    f"<td style='padding:7px;font-size:.83rem;font-family:monospace;"
                    f"    color:#374151;'>{_p}  ___</td>"
                    f"<td style='padding:7px;font-family:monospace;color:#64748b;'>{_bef}</td>"
                    f"<td style='padding:7px;font-family:monospace;color:#111827;"
                    f"    font-weight:700;'>{_after}</td>"
                    f"<td style='padding:7px;color:{_vc};font-weight:700;'>{_v}</td></tr>"
                )

        ba_view = mo.Html(
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:.9rem;color:#1f2937;'>"
            "<thead><tr style='border-bottom:1px solid #d1d5db;background:#f8fafc;'>"
            "<th style='text-align:left;padding:7px;'>Test</th>"
            "<th style='text-align:left;padding:7px;'>Prompt</th>"
            "<th style='text-align:left;padding:7px;'>Before</th>"
            "<th style='text-align:left;padding:7px;'>After</th>"
            "<th style='text-align:left;padding:7px;'>Result</th>"
            "</tr></thead><tbody>" + _rows_html + "</tbody></table></div>"
        )

    ba_view
    return (ba_view,)


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION: EDIT COHERENCE (novel)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(edit_result, mo):
    _intro = r"""
    ---
    ## Extension: Edit Coherence *(novel — not in the paper)*

    ROME measures efficacy, generalization, and specificity. It never asks:
    do *logically entailed* facts cascade?
    """

    if edit_result is None:
        _example = (
            "If we rewrite *\"Eiffel Tower is in Paris\"* → Rome, a coherent "
            "model should also shift \"the **country** containing the Eiffel "
            "Tower\" from France to Italy, and \"the **language** spoken\" "
            "from French to Italian."
        )
    else:
        _subj = edit_result["fact"]["subject"]
        _true = edit_result["fact"]["true"]
        _new  = edit_result["target"]
        _example = (
            f"You just rewrote *\"{_subj} — {_true}\"* → **{_new}**. "
            f"A coherent model would also update every fact that logically "
            f"depends on this one — not just the fact itself."
        )

    mo.md(_intro + "\n\n" + _example + "\n\n"
          "ROME is a rank-one update at one layer — there's no mechanism "
          "that guarantees this cascade. The **Edit Coherence Score** "
          "measures how often it happens anyway, for *this specific edit*.")
    return


@app.cell(hide_code=True)
def _(edit_result, model, tokenizer, DEVICE, torch, mo):
    if edit_result is None:
        coh_view = mo.md(
            "Apply an edit in Step 4 to see the coherence analysis."
        ).callout(kind="info")
    else:
        _fact   = edit_result["fact"]
        _before = edit_result["before"]
        _rows_html = ""
        _changed = 0

        with torch.no_grad():
            for _p in _fact["coh"]:
                _inp   = tokenizer(_p, return_tensors="pt").to(DEVICE)
                _after = tokenizer.decode([model(**_inp).logits[0,-1].argmax()]).strip()
                _bef   = _before.get(_p, "—")
                _chg   = _bef != _after
                if _chg:
                    _changed += 1
                _rows_html += (
                    f"<tr><td style='padding:7px;font-family:monospace;font-size:.83rem;'>"
                    f"{_p}  ___</td>"
                    f"<td style='padding:7px;color:#64748b;font-family:monospace;'>{_bef}</td>"
                    f"<td style='padding:7px;font-weight:700;font-family:monospace;'>{_after}</td>"
                    f"<td style='padding:7px;color:{'#16a34a' if _chg else '#94a3b8'};"
                    f"    font-weight:700;'>{'✓ cascaded' if _chg else '— unchanged'}</td></tr>"
                )

        _score = _changed / max(len(_fact["coh"]), 1)
        _table = mo.Html(
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:.9rem;color:#1f2937;'>"
            "<thead><tr style='border-bottom:1px solid #d1d5db;background:#f8fafc;'>"
            "<th style='text-align:left;padding:7px;'>Entailed prompt</th>"
            "<th style='text-align:left;padding:7px;'>Before</th>"
            "<th style='text-align:left;padding:7px;'>After</th>"
            "<th style='text-align:left;padding:7px;'>Cascaded?</th>"
            "</tr></thead><tbody>" + _rows_html + "</tbody></table></div>"
        )

        coh_view = mo.vstack([
            mo.hstack([mo.stat(f"{_score:.0%}", label="Edit Coherence Score",
                       caption=f"{_changed}/{len(_fact['coh'])} entailed facts cascaded")],
                       justify="start"),
            _table,
            mo.md(
                f"**Finding:** ROME achieves {_score:.0%} coherence on entailed facts. "
                "The rank-one update is guaranteed to change only one weight direction — "
                "propagation of logical consequences is emergent, not guaranteed. "
                "This motivates MEMIT (multi-layer editing) and GRACE (cache-based editing)."
            ).callout(kind="success" if _score >= 0.5 else "warn"),
        ], gap=1)

    coh_view
    return (coh_view,)


# ─────────────────────────────────────────────────────────────────────────────
# TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Key Takeaways

    | Finding | What it means |
    |---------|---------------|
    | 🔍 **Facts have precise addresses** | Causal tracing localises each fact to a narrow band: subject's last token × mid-to-late MLP layers |
    | 🖱️ **The address you find is the address you edit** | Clicking the heatmap cell *is* selecting the layer — no separate control to desynchronize |
    | ✏️ **One rank-one update is enough** | The minimum-norm weight perturbation redirects the stored association in under a second |
    | 🎯 **Specificity by construction** | $\Delta W$ only changes the key direction $k^*$ — all other keys are mathematically untouched |
    | ⚠️ **Coherence is not guaranteed** | Logically entailed facts may not cascade — exposed in our novel extension |
    | 🔭 **Broader implication** | Surgical knowledge correction without retraining or catastrophic forgetting |

    ---

    **Paper:** [arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)
    · Meng, Bau, Andonian & Belinkov · NeurIPS 2022
    **Code:** [github.com/kmeng01/rome](https://github.com/kmeng01/rome)
    **Notebook:** alphaXiv × marimo GPU Notebook Competition #2
    """)
    return


if __name__ == "__main__":
    app.run()
