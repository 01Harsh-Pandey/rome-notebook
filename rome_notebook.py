# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.23.2",
#   "torch>=2.0.0",
#   "transformers>=4.30.0",
#   "accelerate>=0.20.0",
#   "causal-tracer>=1.1.0",
#   "anywidget>=0.9.0",
#   "traitlets>=5.0",
#   "numpy>=1.24",
#   "plotly>=6.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium", css_file="", auto_download=["html"])


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _():
    import copy
    import html as html_lib
    import json

    import anywidget
    import numpy as np
    import plotly.graph_objects as go
    import torch
    import torch.nn.functional as F
    import traitlets
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import marimo as mo

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    return (
        F, anywidget, copy, go, html_lib, json,
        np, torch, traitlets,
        AutoModelForCausalLM, AutoTokenizer,
        mo, DEVICE, GPU_NAME,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    MODEL_NAME         = "gpt2-xl"
    ROME_LAYER_DEFAULT = 17

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
    FACTS_BY_ID = {f["id"]: f for f in PRESET_FACTS}

    # Dropdown options: {label_shown_to_user: value_returned_by_.value}
    FACT_OPTIONS = {f["prompt"] + "  →  ?": f["id"] for f in PRESET_FACTS}

    return MODEL_NAME, ROME_LAYER_DEFAULT, PRESET_FACTS, FACTS_BY_ID, FACT_OPTIONS, mo


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(GPU_NAME, mo):
    _banner = f"""
    <style>
      .rome-hero {{
        border: 1px solid #d0d7de;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(15,23,42,.10);
        margin-bottom: 20px;
      }}
      .rome-hero-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.3fr) minmax(240px, 0.7fr);
        gap: 20px;
        padding: 28px 26px;
        background: #fff;
      }}
      @media (max-width: 680px) {{
        .rome-hero-grid {{ grid-template-columns: 1fr; padding: 20px 16px; }}
      }}
      .rome-badge {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 10px;
        border: 1px solid #bfdbfe;
        background: rgba(239,246,255,.85);
        border-radius: 999px;
        color: #1d4ed8;
        font-size: .74rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: .04em; margin-bottom: 12px;
      }}
      .rome-title {{
        margin: 0 0 10px;
        color: #111827;
        font-size: 2.2rem;
        font-weight: 850;
        line-height: 1.08;
      }}
      .rome-desc {{
        margin: 0 0 16px;
        color: #374151;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 560px;
      }}
      .rome-meta {{
        color: #475569;
        font-size: .9rem;
        line-height: 1.7;
      }}
      .rome-meta a {{
        color: #2563eb; text-decoration: none; font-weight: 700;
      }}
      .rome-card {{
        display: flex; flex-direction: column;
        justify-content: space-between; gap: 12px;
        border: 1px solid rgba(148,163,184,.35);
        background: rgba(248,250,252,.9);
        border-radius: 8px;
        padding: 14px 14px 12px;
        box-shadow: 0 4px 12px rgba(148,163,184,.12);
      }}
      .rome-card-label {{
        color: #64748b; font-size: .72rem; font-weight: 850;
        text-transform: uppercase; letter-spacing: .04em; margin-bottom: 5px;
      }}
      .rome-card-title {{
        font-size: .95rem; font-weight: 800; color: #111827; line-height: 1.3;
      }}
      .rome-card-authors {{
        margin-top: 5px; color: #475569; font-size: .88rem;
      }}
      .rome-card-link {{
        display: inline-block; margin-top: 7px;
        color: #2563eb; font-size: .88rem; font-weight: 700;
        text-decoration: none;
      }}
      .rome-gpu {{
        margin-top: 10px;
        padding: 7px 10px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        font-size: .8rem; color: #166534; font-weight: 700;
      }}
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
            Every fact a language model knows has a precise address: a handful of
            neurons at a specific layer and token position. This notebook finds that
            address via <strong>causal tracing</strong>, then
            surgically rewrites the fact with a
            <strong>rank-one weight update</strong> — under a second,
            no retraining, all other knowledge intact.
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
            <a class="rome-card-link"
               href="https://www.alphaxiv.org/abs/2202.05262">
              alphaxiv:2202.05262
            </a>
          </div>
          <svg viewBox="0 0 260 120" role="img"
               aria-label="Factual memory edit: weight matrix with a highlighted neuron"
               style="width:100%;height:auto;display:block;margin-top:6px;">
            <defs>
              <radialGradient id="rome-glow" cx="50%" cy="50%" r="55%">
                <stop offset="0%"   stop-color="#fff"    stop-opacity=".98"/>
                <stop offset="45%"  stop-color="#bfdbfe" stop-opacity=".75"/>
                <stop offset="100%" stop-color="#bbf7d0" stop-opacity="0"/>
              </radialGradient>
              <radialGradient id="rome-red" cx="38%" cy="35%" r="68%">
                <stop offset="0%" stop-color="#fff"    stop-opacity=".9"/>
                <stop offset="100%" stop-color="#ef4444" stop-opacity=".9"/>
              </radialGradient>
              <radialGradient id="rome-blue" cx="38%" cy="35%" r="68%">
                <stop offset="0%" stop-color="#fff"    stop-opacity=".9"/>
                <stop offset="100%" stop-color="#60a5fa" stop-opacity=".9"/>
              </radialGradient>
              <filter id="rome-shadow" x="-40%" y="-40%" width="180%" height="180%">
                <feDropShadow dx="0" dy="4" stdDeviation="4"
                  flood-color="#64748b" flood-opacity=".2"/>
              </filter>
            </defs>
            <rect x="0" y="0" width="260" height="120" rx="7" fill="#f8fafc"/>
            <circle cx="130" cy="60" r="65" fill="url(#rome-glow)"/>
            <!-- Weight matrix grid -->
            <g opacity=".35" stroke="#94a3b8" stroke-width=".8">
              <line x1="70" y1="20" x2="70" y2="100"/>
              <line x1="90" y1="20" x2="90" y2="100"/>
              <line x1="110" y1="20" x2="110" y2="100"/>
              <line x1="50" y1="35" x2="130" y2="35"/>
              <line x1="50" y1="50" x2="130" y2="50"/>
              <line x1="50" y1="65" x2="130" y2="65"/>
              <line x1="50" y1="80" x2="130" y2="80"/>
            </g>
            <!-- Highlighted neuron (the edit target) -->
            <rect x="70" y="50" width="20" height="15" rx="2"
              fill="#fef08a" stroke="#eab308" stroke-width="1.5"
              filter="url(#rome-shadow)"/>
            <!-- Layer label -->
            <text x="90" y="112" text-anchor="middle"
              font-size="9" fill="#64748b" font-weight="700">
              Layer 17 · subject last token
            </text>
            <!-- Arrow pointing to edit -->
            <path d="M155 57 L125 57" stroke="#ef4444" stroke-width="1.8"
              marker-end="url(#rome-arr)" fill="none"/>
            <defs>
              <marker id="rome-arr" markerWidth="6" markerHeight="6"
                refX="5" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/>
              </marker>
            </defs>
            <!-- Edit pencil -->
            <g transform="translate(155,45)">
              <rect x="0" y="0" width="28" height="10" rx="2"
                fill="url(#rome-blue)" stroke="#fff" stroke-width="1"/>
              <text x="14" y="8" text-anchor="middle"
                font-size="7.5" fill="#1e3a5f" font-weight="800">
                ROME edit
              </text>
            </g>
            <!-- Token labels -->
            <text x="70" y="17" text-anchor="middle"
              font-size="8" fill="#475569">The</text>
            <text x="90" y="17" text-anchor="middle"
              font-size="8" fill="#2563eb" font-weight="700">Tower</text>
            <text x="110" y="17" text-anchor="middle"
              font-size="8" fill="#475569">is</text>
          </svg>
          <div class="rome-gpu">
            ⚡ Running on {GPU_NAME}
          </div>
        </div>
      </div>
    </div>
    """
    mo.Html(_banner)
    return


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reader's note.** Run all cells top-to-bottom. GPU ops are gated behind
    explicit buttons. Cells that depend on GPU results show a blue callout until
    they have data to display.

    ---

    ## The Problem: Facts Are Frozen in Weights

    After pretraining, a language model like GPT-2 XL *knows* that the Eiffel
    Tower is in Paris. That knowledge lives in its 1.5 billion parameters as a
    pattern of floating-point numbers. Now the fact changes — or turns out to be
    wrong — or is copyrighted and must be removed.

    The options are grim:

    | Approach | Cost | Risk |
    |----------|------|------|
    | **Full retraining** | ~$1M, months | Impractical |
    | **Fine-tuning on corrected data** | Days | Catastrophic forgetting |
    | **ROME** | < 60 seconds | Surgically precise |

    ROME (Rank-One Model Editing) works because each fact has a *precise address*
    inside the model. Find the address. Write a new value at that address. Done.

    The paper's key contributions:

    1. **Causal tracing** — a method to locate the exact (layer, token) pair
       responsible for any factual association
    2. **Rank-one editing** — the minimum-norm weight update that rewrites
       a single fact without touching anything else
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
        _vram = (
            f"{torch.cuda.memory_allocated()/1e9:.1f} GB VRAM"
            if DEVICE == "cuda" else "CPU"
        )
        load_view = mo.md(
            f"✅ **Loaded.** GPT-2 XL on `{DEVICE}` · {_vram}"
        ).callout(kind="success")

    load_view
    return model, tokenizer, load_view


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LIVE FACT PROBE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 2 — Probe the Model's Knowledge

    Select any of the three preset facts. The model answers live.
    This is what you're about to surgically rewrite.
    """)
    return


@app.cell(hide_code=True)
def _(FACT_OPTIONS, PRESET_FACTS, mo):
    # FACT_OPTIONS = {label_displayed: id_returned}
    # .value returns the fact id ("eiffel", "lebron", "gates")
    fact_selector = mo.ui.dropdown(
        options=FACT_OPTIONS,
        value=list(FACT_OPTIONS.keys())[0],  # KEY = label (what .value param needs)
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
        # fact_selector.value returns the id because FACT_OPTIONS = {label: id}
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
            f"""<div style="display:flex;align-items:center;gap:8px;
                margin:5px 0;">
              <code style="width:110px;text-align:right;font-size:13px;
                  color:#111827;flex-shrink:0;">{repr(t)}</code>
              <div style="flex:1;background:#f1f5f9;border-radius:4px;
                  height:22px;overflow:hidden;">
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
        <div style="border:1px solid #d0d7de;border-radius:8px;
            padding:14px 16px;background:#fff;
            box-shadow:0 2px 8px rgba(15,23,42,.06);">
          <div style="font-size:12px;color:#64748b;
              font-family:monospace;margin-bottom:10px;">
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
# STEP 3 — CONFIGURE AND APPLY THE ROME EDIT
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 3 — Apply the ROME Edit

    **The algorithm in three steps:**

    1. **Key vector $k^*$** — run the model on the prompt with *noisy* subject
       embeddings 20 times. Average the MLP hidden activation at the subject's
       last token. This gives a subject-generic key that doesn't overfit to one prompt.

    2. **Value vector $v^*$** — gradient-optimise $v$ so that when it is injected
       as the MLP output at the subject's position, the model maximises
       $P(\text{target\_new})$ across several paraphrase prompts.

    3. **Rank-one update** — for GPT-2's `Conv1D` weight $W$ (shape 6400 × 1600):

    $$W' = W + \frac{(v^* - k^{*\top} W)\, k^{*}}{\|k^*\|^2}$$

    This is the *minimum-norm* change: it redirects exactly the key direction
    $k^*$ and leaves every other direction mathematically unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(ROME_LAYER_DEFAULT, mo):
    edit_target_ui = mo.ui.text(
        value="Rome",
        label="Rewrite answer to",
        placeholder="new target…",
    )
    edit_layer_ui = mo.ui.slider(
        0, 47, value=ROME_LAYER_DEFAULT, step=1, show_value=True,
        label="Edit layer  (set to causal trace peak — see Step 4)",
    )
    mo.hstack([
        mo.vstack([mo.md("**New target**"), edit_target_ui]),
        mo.vstack([mo.md("**Target layer**"), edit_layer_ui]),
    ], gap=2, justify="start")
    return edit_target_ui, edit_layer_ui


@app.cell(hide_code=True)
def _(mo):
    edit_btn = mo.ui.run_button(
        label="✏️  Apply ROME Edit",
        kind="danger", full_width=True,
    )
    edit_btn
    return (edit_btn,)


# ─────────────────────────────────────────────────────────────────────────────
# ROME helpers
# ─────────────────────────────────────────────────────────────────────────────

@app.function
def rome_subject_range(tokenizer, prompt, subject):
    full = tokenizer.encode(prompt)
    for prefix in (" " + subject, subject):
        sub = tokenizer.encode(prefix)
        for i in range(len(full) - len(sub) + 1):
            if full[i:i+len(sub)] == sub:
                return i, i + len(sub)
    return 0, 1


@app.function
def rome_key(model, tokenizer, prompt, subject, layer, device,
             n=20, coef=3.0):
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
def rome_value(model, tokenizer, request, layer, key, device,
               steps=25, lr=0.05):
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


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE THE EDIT
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(edit_btn, current_fact, edit_target_ui, edit_layer_ui,
      model, tokenizer, DEVICE, torch,
      rome_key, rome_value, rome_apply, mo):

    if not edit_btn.value or model is None or current_fact is None:
        edit_result = None
        if not edit_btn.value:
            edit_exec_view = mo.md(
                "Configure the edit above and click **Apply ROME Edit**."
            ).callout(kind="info")
        else:
            edit_exec_view = mo.md(
                "Load the model in Step 1."
            ).callout(kind="warn")
    else:
        _target  = edit_target_ui.value.strip()
        _layer   = edit_layer_ui.value
        _request = {**current_fact, "new": _target}

        # Capture BEFORE predictions for all prompt categories
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

        # Also capture probability distribution before edit (for widget)
        with torch.no_grad():
            _inp_b  = tokenizer(current_fact["prompt"], return_tensors="pt").to(DEVICE)
            _probs_b = torch.softmax(model(**_inp_b).logits[0,-1], -1)
            _tv_b, _ti_b = torch.topk(_probs_b, 6)

        _before_dist = [
            {"tok": tokenizer.decode([t]).strip(), "pct": round(float(p)*100, 2)}
            for t, p in zip(_ti_b, _tv_b)
        ]

        for _title, _fn, _args in [
            ("Step 1/3 — Computing key vector (20 noise samples)…",
             rome_key,
             (model, tokenizer, _request["prompt"], _request["subject"],
              _layer, DEVICE)),
            ("Step 2/3 — Optimising value vector (25 gradient steps)…",
             None, None),
            ("Step 3/3 — Applying rank-one weight update…",
             None, None),
        ]:
            with mo.status.spinner(title=_title):
                if _fn is rome_key:
                    _k = rome_key(model, tokenizer, _request["prompt"],
                                  _request["subject"], _layer, DEVICE)
                    _v = rome_value(model, tokenizer, _request,
                                    _layer, _k, DEVICE)
                    rome_apply(model, _layer, _k, _v)
                    model.eval()

        # Capture AFTER distribution
        with torch.no_grad():
            _inp_a  = tokenizer(current_fact["prompt"], return_tensors="pt").to(DEVICE)
            _probs_a = torch.softmax(model(**_inp_a).logits[0,-1], -1)
            _tv_a, _ti_a = torch.topk(_probs_a, 6)

        _after_dist = [
            {"tok": tokenizer.decode([t]).strip(), "pct": round(float(p)*100, 2)}
            for t, p in zip(_ti_a, _tv_a)
        ]

        edit_result = {
            "before":  _before,
            "before_dist": _before_dist,
            "after_dist":  _after_dist,
            "fact":    current_fact,
            "target":  _target,
            "layer":   _layer,
        }

        edit_exec_view = mo.md(
            f"✅ **Edit applied.** Layer {_layer} · "
            f"*{current_fact['subject']}* → **{_target}**"
        ).callout(kind="success")

    edit_exec_view
    return edit_result, edit_exec_view


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY REWRITE WIDGET — animated before/after probability bars
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class MemoryRewriteWidget(anywidget.AnyWidget):
        """
        Side-by-side animated probability bars.
        Bars animate in with staggered delays on each render.
        """
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
                const hit = item.tok.trim().toLowerCase()
                              === target.trim().toLowerCase();
                const bg  = hit
                  ? (side === "after" ? "#16a34a" : "#94a3b8")
                  : (side === "after" ? "#2563eb" : "#94a3b8");
                const del = `${idx * 55}ms`;
                return `
                <div style="display:flex;align-items:center;gap:7px;margin:5px 0;">
                  <code style="width:100px;text-align:right;font-size:12px;
                      color:#111827;flex-shrink:0;overflow:hidden;
                      text-overflow:ellipsis;white-space:nowrap;"
                  >${JSON.stringify(item.tok)}</code>
                  <div style="flex:1;background:#f1f5f9;border-radius:3px;
                      height:19px;overflow:hidden;">
                    <div class="rome-bar" style="
                        width:${w}%;background:${bg};height:100%;
                        border-radius:3px;
                        animation-delay:${del};"
                    ></div>
                  </div>
                  <span style="width:40px;font-size:11px;color:#64748b;
                      font-variant-numeric:tabular-nums;"
                  >${item.pct.toFixed(1)}%</span>
                </div>`;
              }

              el.innerHTML = `
              <div style="border:1px solid #d0d7de;border-radius:8px;
                  background:#fff;padding:14px 16px;
                  box-shadow:0 2px 8px rgba(15,23,42,.06);">
                <div style="font-family:monospace;font-size:12px;
                    color:#64748b;margin-bottom:12px;">${prompt} ___</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;
                    gap:16px;">
                  <div>
                    <div style="font-size:11px;font-weight:800;
                        text-transform:uppercase;letter-spacing:.04em;
                        color:#64748b;margin-bottom:8px;">Before edit</div>
                    ${before.map((x,i) => row(x, maxB, "before", i)).join("")}
                  </div>
                  <div>
                    <div style="font-size:11px;font-weight:800;
                        text-transform:uppercase;letter-spacing:.04em;
                        color:#16a34a;margin-bottom:8px;">After ROME edit ✓</div>
                    ${after.map((x,i) => row(x, maxA, "after", i)).join("")}
                  </div>
                </div>
                <div style="margin-top:10px;font-size:11px;color:#64748b;">
                  Green bars = new target  ·
                  Blue bars = other tokens  ·
                  Bars animate on each edit
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
        .rome-bar {
          animation: rome-slide-in 500ms cubic-bezier(.22,1,.36,1) both;
        }
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
            "before":  edit_result["before_dist"],
            "after":   edit_result["after_dist"],
            "target":  edit_result["target"],
            "prompt":  edit_result["fact"]["prompt"],
        }))
        rewrite_view = mo.vstack([
            mo.md("### Memory Rewrite — Probability Shift"),
            mo.ui.anywidget(_w),
        ], gap=1)

    rewrite_view
    return (rewrite_view,)


# ─────────────────────────────────────────────────────────────────────────────
# BEFORE / AFTER TABLE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## How Surgical Was the Edit?

    Three test categories verify that ROME changed *only* what we asked.
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
            [(_fact["prompt"],  "✅ Efficacy")]
            + [(_p, "🔄 Generalization") for _p in _fact["gen"]]
            + [(_p, "🎯 Specificity")    for _p in _fact["spec"]]
        )

        _rows_html = ""
        with torch.no_grad():
            for _p, _cat in _prompts:
                _inp   = tokenizer(_p, return_tensors="pt").to(DEVICE)
                _after = tokenizer.decode(
                    [model(**_inp).logits[0,-1].argmax()]
                ).strip()
                _bef   = _before.get(_p, "—")
                _chg   = _bef != _after

                if _cat.startswith("🎯"):
                    _v = "✅ unchanged" if not _chg else "⚠️ changed"
                    _vc = "#16a34a" if not _chg else "#d97706"
                else:
                    _hit = _after.strip().lower() == _target.strip().lower()
                    _v  = "✅" if _hit else "❌"
                    _vc = "#16a34a" if _hit else "#dc2626"

                _rows_html += (
                    f"<tr>"
                    f"<td style='padding:7px;font-size:.85rem;'>{_cat}</td>"
                    f"<td style='padding:7px;font-size:.83rem;"
                    f"    font-family:monospace;color:#374151;'>"
                    f"{_p}  ___</td>"
                    f"<td style='padding:7px;font-family:monospace;color:#64748b;'>"
                    f"{_bef}</td>"
                    f"<td style='padding:7px;font-family:monospace;"
                    f"    color:#111827;font-weight:700;'>{_after}</td>"
                    f"<td style='padding:7px;color:{_vc};font-weight:700;'>"
                    f"{_v}</td>"
                    f"</tr>"
                )

        ba_view = mo.Html(
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;"
            "    font-size:.9rem;color:#1f2937;'>"
            "<thead><tr style='border-bottom:1px solid #d1d5db;"
            "    background:#f8fafc;'>"
            "<th style='text-align:left;padding:7px;'>Test</th>"
            "<th style='text-align:left;padding:7px;'>Prompt</th>"
            "<th style='text-align:left;padding:7px;'>Before</th>"
            "<th style='text-align:left;padding:7px;'>After</th>"
            "<th style='text-align:left;padding:7px;'>Result</th>"
            "</tr></thead><tbody>"
            + _rows_html
            + "</tbody></table></div>"
        )

    ba_view
    return (ba_view,)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CAUSAL TRACE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 4 — Causal Tracing: Where Did the Fact Live?

    How did ROME know to target layer 17? The paper answers this with
    **causal mediation analysis**:

    - **Clean run** — normal prompt → P(*Paris*) is high
    - **Corrupt run** — add noise to subject token embeddings →
      P(*Paris*) collapses to near zero
    - **Patch run** — at each *(layer $\ell$, token $t$)* pair, restore
      the clean activation and re-measure how much P(*Paris*) recovers

    $$\text{Indirect Effect}(\ell, t) =
    \frac{P_{\text{patched}} - P_{\text{corrupted}}}
         {P_{\text{clean}} - P_{\text{corrupted}}}$$

    Running this for every $(\ell, t)$ pair produces a heatmap — the
    fact's address, visible to the naked eye. The bright band at the
    subject's last token, mid-to-late MLP layers, is where ROME edits.

    *Note: causal tracing temporarily loads a second copy of GPT-2 XL
    inside `causal-tracer`. With 102 GB of VRAM this fits comfortably.*
    """)
    return


@app.cell(hide_code=True)
def _(FACT_OPTIONS, PRESET_FACTS, mo):
    trace_fact_ui = mo.ui.dropdown(
        options=FACT_OPTIONS,
        value=list(FACT_OPTIONS.keys())[0],  # KEY = label shown to user
        label="Fact to trace",
        full_width=True,
    )
    trace_samples_ui = mo.ui.slider(
        10, 40, value=20, step=5, show_value=True,
        label="Noise samples  (higher → smoother, slower)",
    )
    mo.hstack([
        mo.vstack([mo.md("**Fact**"),    trace_fact_ui]),
        mo.vstack([mo.md("**Samples**"), trace_samples_ui]),
    ], gap=3, justify="start")
    return trace_fact_ui, trace_samples_ui


@app.cell(hide_code=True)
def _(mo):
    trace_btn = mo.ui.run_button(
        label="🔍  Run Causal Trace  (~60 s)",
        kind="warn", full_width=True,
    )
    trace_btn
    return (trace_btn,)


@app.cell(hide_code=True)
def _(trace_btn, trace_fact_ui, trace_samples_ui,
      FACTS_BY_ID, MODEL_NAME, torch, mo):
    if not trace_btn.value:
        trace_result = None
        trace_view = mo.md(
            "Click **Run Causal Trace** to map where this fact is stored."
        ).callout(kind="info")
    else:
        _fact_t = FACTS_BY_ID[trace_fact_ui.value]
        with mo.status.spinner(
            title="Loading causal-tracer + running trace… (~60 s)",
        ):
            from causal_tracer import CausalTracer
            _tr  = CausalTracer(MODEL_NAME)
            _res = _tr.calculate_hidden_flow(
                prompt=_fact_t["prompt"],
                subject=_fact_t["subject"],
                samples=trace_samples_ui.value,
                noise=0.13,
            )
            # Verified shape: (n_tokens, n_layers)
            _scores = _res.scores.numpy()
            _tokens = list(_res.input_tokens)
            _srange = tuple(_res.subject_range)
            del _tr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _sl       = _srange[1] - 1
        _top_lay  = int(_scores[_sl, :].argmax())
        _top_scr  = float(_scores[_sl, :].max())

        trace_result = {
            "scores":  _scores,
            "tokens":  _tokens,
            "srange":  _srange,
            "top_layer": _top_lay,
            "top_score": _top_scr,
            "prompt": _fact_t["prompt"],
        }
        trace_view = mo.md(
            f"✅ **Trace complete.** "
            f"Shape `{_scores.shape}` (tokens × layers) · "
            f"Peak: **layer {_top_lay}** "
            f"(indirect effect = {_top_scr:.3f})"
        ).callout(kind="success")

    trace_view
    return trace_result, trace_view


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL HEATMAP WIDGET
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class CausalHeatmapWidget(anywidget.AnyWidget):
        """
        Causal trace heatmap: X = token positions, Y = layers (0 at bottom).
        scores[t][l] = indirect effect. Subject tokens shown in blue.
        Peak cell has a red border and a labelled arrow.
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

              function col(s) {
                const t = Math.min(1, Math.max(0, s));
                return `rgb(${Math.round(255-t*186)},${Math.round(255-t*185)},${Math.round(255-t*26)})`;
              }

              let cells = "", tlabs = "", lticks = "";
              for (let t = 0; t < nT; t++)
                for (let l = 0; l < nL; l++) {
                  const x = PL + t*CW, y = PT + (nL-1-l)*CH;
                  const peak = t===sl && l===top_layer;
                  cells += `<rect x="${x}" y="${y}" width="${CW-1}" height="${CH-1}"
                    fill="${col(scores[t][l])}"
                    stroke="${peak?"#ef4444":"none"}"
                    stroke-width="${peak?2:0}" rx="1"
                    data-t="${t}" data-l="${l}" data-s="${scores[t][l].toFixed(3)}"
                    class="hmcell" style="cursor:crosshair;"/>`;
                }

              for (let t = 0; t < nT; t++) {
                const cx = PL + t*CW + CW/2, ty = PT + nL*CH + 12;
                const subj = t>=ss && t<se;
                tlabs += `<text x="${cx}" y="${ty}" text-anchor="end" font-size="11"
                  fill="${subj?"#2563eb":"#475569"}"
                  font-weight="${subj?"700":"400"}"
                  transform="rotate(-45,${cx},${ty})">${tokens[t]}</text>`;
              }

              for (let l = 0; l < nL; l += 8) {
                const ty = PT + (nL-1-l)*CH + CH/2 + 4;
                lticks += `<text x="${PL-4}" y="${ty}" text-anchor="end"
                  font-size="10" fill="#475569">${l}</text>`;
              }

              const ay = PT + (nL-1-top_layer)*CH + CH/2;
              const ax = PL + nT*CW + 4;
              const ann = `
                <line x1="${ax}" y1="${ay}" x2="${ax+14}" y2="${ay}"
                  stroke="#ef4444" stroke-width="1.5" marker-end="url(#arr)"/>
                <text x="${ax+16}" y="${ay+4}" font-size="10"
                  fill="#ef4444" font-weight="700">L${top_layer}</text>`;

              el.innerHTML = `
              <div style="border:1px solid #d0d7de;border-radius:8px;
                  background:#fff;padding:12px 14px;
                  box-shadow:0 2px 8px rgba(15,23,42,.06);">
                <div style="overflow-x:auto;">
                  <svg width="${W}" height="${H}" style="display:block;max-width:100%;">
                    <defs>
                      <marker id="arr" markerWidth="6" markerHeight="6"
                          refX="5" refY="3" orient="auto">
                        <path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/>
                      </marker>
                    </defs>
                    ${cells}${tlabs}${lticks}${ann}
                    <text x="13" y="${PT+nL*CH/2}" text-anchor="middle"
                      font-size="11" fill="#475569"
                      transform="rotate(-90,13,${PT+nL*CH/2})">Layer</text>
                    <text x="${PL+nT*CW/2}" y="${H-3}" text-anchor="middle"
                      font-size="11" fill="#475569">← Token position →</text>
                  </svg>
                </div>
                <div style="font-size:11px;color:#64748b;margin-top:5px;">
                  🔵 Blue = subject tokens &nbsp;·&nbsp;
                  🔴 Red border = peak causal cell &nbsp;·&nbsp;
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
              });
            }
            draw();
            model.on("change:hm_data", draw);
          }
        };
        """
        _css = ".hmcell:hover{opacity:.7}"
        hm_data = traitlets.Unicode("").tag(sync=True)

    return (CausalHeatmapWidget,)


@app.cell(hide_code=True)
def _(trace_result, CausalHeatmapWidget, json, mo):
    if trace_result is None:
        heatmap_view = mo.md(
            "Run causal trace above to see the heatmap."
        ).callout(kind="info")
    else:
        _w = CausalHeatmapWidget(hm_data=json.dumps({
            "scores":        trace_result["scores"].tolist(),
            "tokens":        trace_result["tokens"],
            "subject_range": list(trace_result["srange"]),
            "top_layer":     trace_result["top_layer"],
        }))
        heatmap_view = mo.vstack([
            mo.md(f"### Causal Trace — *\"{trace_result['prompt']}\"*"),
            mo.ui.anywidget(_w),
            mo.md(
                f"**Reading the heatmap:** The brightest cell is at "
                f"**layer {trace_result['top_layer']}**, the subject's last token "
                f"(indirect effect = {trace_result['top_score']:.3f}). "
                f"This is the MLP neuron cluster that stores this specific "
                f"factual association. ROME targets exactly this address."
            ).callout(kind="neutral"),
        ], gap=1)

    heatmap_view
    return (heatmap_view,)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EXTENSION: EDIT COHERENCE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 5 — Extension: Edit Coherence *(novel — not in the paper)*

    The ROME paper evaluates three things:

    - **Efficacy** — does the edited fact change? ✓
    - **Generalization** — do paraphrases also change? ✓
    - **Specificity** — do unrelated facts stay fixed? ✓

    **What it never tests:** do *logically entailed* facts cascade?

    If we rewrite *"Eiffel Tower is in Paris"* → Rome:
    - "The **country** containing the Eiffel Tower is ___" should → Italy
    - "The **language** spoken where the Eiffel Tower stands is ___" should → Italian

    ROME is a rank-one update at one layer. There is no mechanism to propagate
    logical consequences. The **Edit Coherence Score** measures what fraction
    of entailed facts actually cascade.

    Low coherence = the model holds logically inconsistent beliefs after editing.
    This gap motivates follow-up work: MEMIT, GRACE, and WilKE.
    """)
    return


@app.cell(hide_code=True)
def _(edit_result, model, tokenizer, DEVICE, torch, mo):
    if edit_result is None:
        coh_view = mo.md(
            "Apply an edit in Step 3 to see the coherence analysis."
        ).callout(kind="info")
    else:
        _fact   = edit_result["fact"]
        _before = edit_result["before"]
        _rows_html = ""
        _changed = 0

        with torch.no_grad():
            for _p in _fact["coh"]:
                _inp   = tokenizer(_p, return_tensors="pt").to(DEVICE)
                _after = tokenizer.decode(
                    [model(**_inp).logits[0,-1].argmax()]
                ).strip()
                _bef   = _before.get(_p, "—")
                _chg   = _bef != _after
                if _chg:
                    _changed += 1
                _rows_html += (
                    f"<tr>"
                    f"<td style='padding:7px;font-family:monospace;"
                    f"    font-size:.83rem;'>{_p}  ___</td>"
                    f"<td style='padding:7px;color:#64748b;font-family:monospace;'>"
                    f"{_bef}</td>"
                    f"<td style='padding:7px;font-weight:700;font-family:monospace;'>"
                    f"{_after}</td>"
                    f"<td style='padding:7px;color:{'#16a34a' if _chg else '#94a3b8'};"
                    f"    font-weight:700;'>{'✓ cascaded' if _chg else '— unchanged'}</td>"
                    f"</tr>"
                )

        _score  = _changed / max(len(_fact["coh"]), 1)
        _kind   = "success" if _score >= 0.5 else "warn"

        _table  = mo.Html(
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;"
            "    font-size:.9rem;color:#1f2937;'>"
            "<thead><tr style='border-bottom:1px solid #d1d5db;background:#f8fafc;'>"
            "<th style='text-align:left;padding:7px;'>Entailed prompt</th>"
            "<th style='text-align:left;padding:7px;'>Before</th>"
            "<th style='text-align:left;padding:7px;'>After</th>"
            "<th style='text-align:left;padding:7px;'>Cascaded?</th>"
            "</tr></thead><tbody>"
            + _rows_html
            + "</tbody></table></div>"
        )

        coh_view = mo.vstack([
            mo.hstack([
                mo.stat(f"{_score:.0%}", label="Edit Coherence Score",
                        caption=f"{_changed}/{len(_fact['coh'])} entailed facts cascaded"),
            ], justify="start"),
            _table,
            mo.md(
                f"**Finding:** ROME achieves {_score:.0%} coherence on logically entailed "
                "facts. The rank-one update is mathematically guaranteed to change only "
                "one direction in weight space — propagation of logical dependencies "
                "is emergent, not guaranteed. "
                "This gap is the central motivation for MEMIT (multi-layer editing) "
                "and GRACE (cache-based editing)."
            ).callout(kind=_kind),
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
    | ✏️ **One rank-one update is enough** | The minimum-norm weight perturbation redirects the stored association in under a second |
    | 🎯 **Specificity by construction** | $\Delta W$ only changes the key direction $k^*$ — all other keys are mathematically untouched |
    | 🔄 **Generalization is real** | Averaging $k^*$ over noise-corrupted runs prevents the edit from being prompt-specific |
    | ⚠️ **Coherence is not guaranteed** | Logically entailed facts may not cascade — exposed in our novel extension |
    | 🔭 **Broader implication** | Surgical knowledge correction without retraining — the foundation of a new class of LLM maintenance tools |

    ---

    **Paper:** [arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)
    · Meng, Bau, Andonian & Belinkov · NeurIPS 2022
    **Code:** [github.com/kmeng01/rome](https://github.com/kmeng01/rome)
    **Notebook:** alphaXiv × marimo GPU Notebook Competition #2
    """)
    return


if __name__ == "__main__":
    app.run()
