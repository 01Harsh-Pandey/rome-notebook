# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.20.2",
#   "torch>=2.0.0",
#   "transformers>=4.30.0",
#   "accelerate>=0.20.0",
#   "causal-tracer>=1.1.0",
#   "anywidget>=0.9.0",
#   "traitlets>=5.0",
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "altair>=5.3",
# ]
# ///
"""
ROME: Locating and Editing Factual Associations in GPT
Interactive walkthrough of Meng, Bau, Andonian & Belinkov (NeurIPS 2022)

Verified against:
  GPU         : Tesla T4 · 15.6 GB VRAM
  torch       : 2.11.0+cu128
  causal-tracer: scores shape → (n_tokens, n_layers)  [NOT (n_layers, n_tokens)]
  anywidget   : 0.9.21
  GPT-2 XL    : 48 layers · d_model=1600 · n_inner=None (→ 6400)
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(
    width="medium",
    app_title="ROME: Rewriting Memories in Language Models",
    auto_download=["html"],
)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS  (GPU notebook — imports in cells, not app.setup)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def cell_imports():
    import copy
    import json

    import altair as alt
    import anywidget
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import traitlets
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import marimo as mo

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    return (
        alt, anywidget, copy, json,
        np, pd, torch, F, traitlets,
        AutoModelForCausalLM, AutoTokenizer,
        mo, DEVICE,
    )


@app.cell
def cell_constants():
    MODEL_NAME    = "gpt2-xl"
    N_LAYERS      = 48      # confirmed
    D_MODEL       = 1600    # confirmed  (n_embd)
    D_MLP_HIDDEN  = 6400    # n_inner=None → 4 × 1600
    ROME_LAYER_DEFAULT = 17 # paper's value for GPT-2 XL factual associations

    # Preset knowledge triples from the ROME paper
    PRESET_FACTS = [
        {
            "id":           "eiffel",
            "prompt":       "The Eiffel Tower is located in the city of",
            "subject":      "The Eiffel Tower",
            "target_true":  "Paris",
            "target_new":   "Rome",
            "gen_prompts":  [
                "If you visit the Eiffel Tower, you are traveling to",
                "The Eiffel Tower can be found in",
                "A tourist visiting the Eiffel Tower is in the city of",
            ],
            "spec_prompts": [
                "The Eiffel Tower was designed by",
                "The Eiffel Tower was completed in the year",
                "The Eiffel Tower is made of",
            ],
            "coh_prompts":  [
                "The country that contains the Eiffel Tower is",
                "The language spoken where the Eiffel Tower stands is",
                "The currency used near the Eiffel Tower is the",
            ],
        },
        {
            "id":           "lebron",
            "prompt":       "LeBron James plays the sport of",
            "subject":      "LeBron James",
            "target_true":  "basketball",
            "target_new":   "football",
            "gen_prompts":  [
                "LeBron James is professionally known for playing",
                "When LeBron James competes, he plays",
                "LeBron James earned his fame in the sport of",
            ],
            "spec_prompts": [
                "LeBron James was born in the city of",
                "The team that LeBron James is most famous for is the",
                "LeBron James is famous for his ability to",
            ],
            "coh_prompts":  [
                "The professional league that LeBron James competes in is the",
                "The team sport associated with LeBron James requires a",
                "Athletes in the same sport as LeBron James are called",
            ],
        },
        {
            "id":           "gates",
            "prompt":       "Microsoft was founded by Bill Gates and",
            "subject":      "Microsoft",
            "target_true":  "Paul Allen",
            "target_new":   "Steve Jobs",
            "gen_prompts":  [
                "The co-founder of Microsoft alongside Bill Gates was",
                "Bill Gates started Microsoft together with",
                "Microsoft's other original co-founder was",
            ],
            "spec_prompts": [
                "Microsoft's headquarters is located in",
                "The operating system created by Microsoft is called",
                "Microsoft was founded in the year",
            ],
            "coh_prompts":  [
                "The company co-founded by the same person as Microsoft is",
                "Microsoft's co-founder also helped create the company called",
                "The co-founder of Microsoft later worked at",
            ],
        },
    ]

    C_BLUE   = "#2563eb"
    C_RED    = "#dc2626"
    C_GREEN  = "#16a34a"
    C_ORANGE = "#ea580c"
    C_INDIGO = "#4f46e5"

    return (
        MODEL_NAME, N_LAYERS, D_MODEL, D_MLP_HIDDEN,
        ROME_LAYER_DEFAULT, PRESET_FACTS,
        C_BLUE, C_RED, C_GREEN, C_ORANGE, C_INDIGO,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_hero(mo):
    mo.md(r"""
    <div style="text-align:center;padding:3rem 2rem 2.5rem;
        background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 55%,#0f172a 100%);
        border-radius:16px;margin-bottom:0.5rem;border:1px solid #4338ca;">
      <div style="font-size:2.8rem;margin-bottom:0.8rem;">🧠 ✏️</div>
      <h1 style="font-size:2.8rem;font-weight:900;letter-spacing:-1.5px;
          color:#e0e7ff;margin:0 0 0.5rem;font-family:'Georgia',serif;">
        ROME
      </h1>
      <p style="font-size:1.25rem;font-weight:600;color:#818cf8;margin:0 0 1rem;">
        Locating and Editing Factual Associations in GPT
      </p>
      <p style="font-size:1rem;color:#94a3b8;max-width:580px;
          margin:0 auto 1.8rem;line-height:1.8;">
        Every fact a language model knows lives at a specific address —
        a handful of neurons at a precise layer. This notebook finds that address
        via <strong style="color:#c7d2fe;">causal tracing</strong>, then
        surgically rewrites the fact with a
        <strong style="color:#c7d2fe;">rank-one weight update</strong>
        that takes less than a second and leaves all other knowledge intact.
      </p>
      <div style="display:inline-flex;gap:2rem;flex-wrap:wrap;justify-content:center;
          background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.35);
          padding:0.75rem 2rem;border-radius:999px;">
        <span style="color:#e0e7ff;font-size:0.9rem;font-weight:700;">
          NeurIPS 2022 · Meng, Bau, Andonian &amp; Belinkov
        </span>
        <span style="color:#94a3b8;font-size:0.9rem;">
          GPT-2 XL · 1.5 B params · Tesla T4
        </span>
      </div>
    </div>
    """)
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — THE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_problem(mo):
    mo.md(r"""
    ---
    ## Act I — Facts Are Frozen

    Language models store factual knowledge in their weights.
    After training, they can't correct a single wrong fact without:

    - **Full retraining** — billions in compute, months of time
    - **Fine-tuning on related data** — risks *catastrophic forgetting* of
      everything else it ever learned

    This matters urgently:

    | Scenario | The problem |
    |----------|------------|
    | 📰 **Stale knowledge** | A fact changes in the world; the model keeps the old one |
    | ⚖️ **Copyright / privacy** | Specific content needs removal without retraining |
    | 🔬 **Scientific corrections** | A finding is overturned; the model still states the old theory |

    **The ROME paper's answer:** facts have a *precise address* in the network —
    a small set of MLP weights at a specific layer and token position.
    Find the address, write one new value. Done in under a second.
    """)
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — LOAD THE MODEL
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_load_intro(mo, MODEL_NAME):
    mo.md(f"""
    ---
    ## Act II — Load GPT-2 XL

    We use **{MODEL_NAME}** (1.5B parameters) — the primary model in the ROME paper.
    """)
    return


@app.cell(hide_code=True)
def cell_arch_stats(mo):
    mo.hstack([
        mo.stat("1.5 B", label="Parameters",    caption="GPT-2 XL"),
        mo.stat("48",    label="Layers",         caption="Transformer blocks"),
        mo.stat("1600",  label="d_model",        caption="Residual stream width"),
        mo.stat("6400",  label="MLP width",      caption="4 × d_model per layer"),
    ], gap=2, justify="start")
    return


@app.cell
def cell_load_btn(mo):
    load_btn = mo.ui.run_button(
        label="⚡  Load GPT-2 XL onto GPU",
        kind="success",
        full_width=True,
    )
    load_btn
    return (load_btn,)


@app.cell
def cell_load_model(load_btn, MODEL_NAME, DEVICE,
                    AutoModelForCausalLM, AutoTokenizer, mo, torch):
    mo.stop(
        not load_btn.value,
        mo.callout(
            mo.md("Click **Load GPT-2 XL** above. "
                  "First run downloads ~6 GB of weights (~2 min)."),
            kind="info",
        ),
    )
    with mo.status.spinner(title="Loading GPT-2 XL onto GPU…"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,   # fp32 for numerical stability in ROME
        ).to(DEVICE)
        model.eval()

    vram_gb = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    mo.callout(
        mo.md(f"✅ **Loaded.** GPT-2 XL on `{DEVICE}` · "
              f"VRAM in use: {vram_gb:.1f} GB"),
        kind="success",
    )
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# ACT III — WHAT DOES THE MODEL CURRENTLY KNOW?
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_completion_intro(mo):
    mo.md(r"""
    ---
    ## Act III — What Does the Model Know?

    Before editing anything, observe the model's factual associations.
    Select a preset fact and see live next-token probabilities from GPT-2 XL.
    """)
    return


@app.cell
def cell_fact_ui(PRESET_FACTS, mo):
    fact_ui = mo.ui.dropdown(
        options={f["id"]: f["prompt"] + "  ___" for f in PRESET_FACTS},
        value="eiffel",
        label="Select factual prompt",
        full_width=True,
    )
    fact_ui
    return (fact_ui,)


@app.cell(hide_code=True)
def cell_completion(fact_ui, PRESET_FACTS, model, tokenizer, DEVICE,
                    torch, mo):
    mo.stop(model is None, mo.md("_Load the model first (Act II)._"))

    _fact   = next(f for f in PRESET_FACTS if f["id"] == fact_ui.value)
    _prompt = _fact["prompt"]

    with torch.no_grad():
        _inp   = tokenizer(_prompt, return_tensors="pt").to(DEVICE)
        _logits = model(**_inp).logits[0, -1, :]
        _probs  = torch.softmax(_logits, dim=-1)
        _topk   = torch.topk(_probs, 8)

    _tokens = [tokenizer.decode([t]).strip() for t in _topk.indices]
    _values = [round(float(v), 4) for v in _topk.values]

    _rows = [{"Rank": i + 1, "Token": t, "Probability": v}
             for i, (t, v) in enumerate(zip(_tokens, _values))]

    _top_is_correct = (
        _tokens[0].strip().lower() == _fact["target_true"].strip().lower()
    )

    mo.vstack([
        mo.md(f"**Prompt:** *\"{_prompt}\"*"),
        mo.md(f"**Expected:** `{_fact['target_true']}` "
              f"{'✅ model agrees' if _top_is_correct else '⚠️ model may differ'}"),
        mo.ui.table(_rows, selection=None, pagination=False,
                    label="Top-8 next-token predictions"),
    ], gap=1)
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT IV — CAUSAL TRACING: WHERE DOES THE FACT LIVE?
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_trace_intro(mo):
    mo.md(r"""
    ---
    ## Act IV — Causal Tracing: The Fact's Address

    **How does the paper find which neurons store a specific fact?**
    Through causal mediation analysis — three runs, one measurement:

    1. **Clean run** — model sees the normal prompt.
       P(*"Paris"*) is high.
    2. **Corrupted run** — Gaussian noise is added to the subject's
       token embeddings (*"The Eiffel Tower"* → noise vectors).
       P(*"Paris"*) collapses near zero.
    3. **Patched run** — repeat the corrupted run but at one
       *(layer $\ell$, token position $t$)* pair, restore the
       clean activation. Measure: how much does P(*"Paris"*) recover?

    $$\text{Indirect Effect}(\ell, t) =
    \frac{P_\text{patched}(\text{"Paris"}) - P_\text{corrupted}(\text{"Paris"})}
         {P_\text{clean}(\text{"Paris"}) - P_\text{corrupted}(\text{"Paris"})}$$

    Running this for every $(\ell, t)$ pair produces a **heatmap over
    layers × token positions** — the fact's address, visible to the naked eye.
    The ROME paper consistently finds a bright band at the subject's last token,
    in mid-to-late MLP layers (~layer 17 for GPT-2 XL).
    """)
    return


@app.cell
def cell_trace_controls(PRESET_FACTS, mo):
    trace_fact_ui = mo.ui.dropdown(
        options={f["id"]: f["subject"] for f in PRESET_FACTS},
        value="eiffel",
        label="Fact to trace",
    )
    trace_samples_ui = mo.ui.slider(
        10, 40, value=20, step=5, show_value=True,
        label="Noise samples per cell (more = smoother heatmap, slower)",
    )
    mo.hstack([
        mo.vstack([mo.md("**Fact**"),    trace_fact_ui]),
        mo.vstack([mo.md("**Samples**"), trace_samples_ui,
                   mo.md("_~30–90 s on T4_")]),
    ], gap=3, justify="start")
    return trace_fact_ui, trace_samples_ui


@app.cell
def cell_trace_btn(mo):
    trace_btn = mo.ui.run_button(
        label="🔍  Run Causal Trace",
        kind="warn",
        full_width=True,
    )
    trace_btn
    return (trace_btn,)


@app.cell
def cell_trace_run(trace_btn, trace_fact_ui, trace_samples_ui,
                   PRESET_FACTS, MODEL_NAME, mo, torch):
    """
    Uses causal-tracer package directly.
    Loads its own copy of gpt2-xl internally (~6 GB extra VRAM briefly).
    After trace, the tracer is deleted to free VRAM.

    Confirmed API from real output:
      result.scores      : torch.Tensor, shape (n_tokens, n_layers)
      result.input_tokens: list[str]
      result.subject_range: tuple (start_idx, end_idx)  — exclusive end
      result.answer      : str  (e.g. ' Paris')
    """
    mo.stop(
        not trace_btn.value,
        mo.callout(mo.md("Select a fact then click **Run Causal Trace**."),
                   kind="info"),
    )

    _fact   = next(f for f in PRESET_FACTS if f["id"] == trace_fact_ui.value)
    _prompt = _fact["prompt"]
    _subj   = _fact["subject"]

    with mo.status.spinner(
        title="Loading tracer model + running causal trace… (~60 s)"
    ):
        from causal_tracer import CausalTracer

        _tracer = CausalTracer(MODEL_NAME)
        _result = _tracer.calculate_hidden_flow(
            prompt=_prompt,
            subject=_subj,
            samples=trace_samples_ui.value,
            noise=0.13,
        )

        # ── Real verified shapes ──────────────────────────────────────────
        # scores shape: (n_tokens, n_layers)  ← NOT (n_layers, n_tokens)
        trace_scores      = _result.scores.numpy()      # (n_tokens, n_layers)
        trace_tokens      = list(_result.input_tokens)  # list[str], len=n_tokens
        trace_subj_range  = tuple(_result.subject_range)  # (start, end) exclusive
        trace_answer      = str(_result.answer)          # e.g. ' Paris'
        trace_prompt      = _prompt
        trace_subject     = _subj

        # Free tracer VRAM immediately
        del _tracer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _n_tok, _n_lay = trace_scores.shape
    _subj_last = trace_subj_range[1] - 1
    # Correct indexing: scores[token_idx, layer_idx]
    _per_layer_at_subj = trace_scores[_subj_last, :]   # shape (n_layers,)
    _top_layer         = int(_per_layer_at_subj.argmax())
    _top_score         = float(_per_layer_at_subj.max())

    mo.callout(
        mo.md(
            f"✅ **Trace complete.** "
            f"Shape: `{trace_scores.shape}` (tokens × layers) · "
            f"Answer: **{trace_answer.strip()}** · "
            f"Subject tokens: `{trace_tokens[trace_subj_range[0]:trace_subj_range[1]]}` "
            f"(indices {trace_subj_range[0]}–{trace_subj_range[1]-1}) · "
            f"**Peak causal cell: layer {_top_layer}** "
            f"(indirect effect = {_top_score:.3f})"
        ),
        kind="success",
    )
    return (
        trace_scores, trace_tokens, trace_subj_range,
        trace_answer, trace_prompt, trace_subject,
    )


# ── Causal Heatmap Widget ─────────────────────────────────────────────────────

@app.cell
def cell_heatmap_class(anywidget, traitlets):
    """
    Heatmap follows the paper's Figure 1 convention:
      X-axis = token positions (left → right)
      Y-axis = layers         (bottom = layer 0, top = layer 47)
      Color  = indirect effect (white → deep-indigo)

    Data format: scores[token_idx][layer_idx]  (n_tokens × n_layers)
    This matches the confirmed real shape from causal-tracer.
    """

    class CausalHeatmapWidget(anywidget.AnyWidget):
        _esm = r"""
        function render({ model, el }) {
            const raw = model.get("heatmap_json");
            if (!raw) {
                el.innerHTML = '<p style="color:#888;padding:16px;">Run causal trace first.</p>';
                return;
            }
            const d = JSON.parse(raw);
            // d.scores[t][l]   = indirect effect, token t at layer l
            // d.tokens          = list of token strings (length nT)
            // d.subject_range   = [start, end]  exclusive
            // d.top_layer       = layer with highest effect at subject last token
            const { scores, tokens, subject_range, top_layer } = d;
            const nT = tokens.length;
            const nL = scores[0].length;

            // Cell sizing  — keep widget ≤ 720 px wide
            const MAX_W   = 720;
            const PAD_L   = 44, PAD_R = 80, PAD_T = 20, PAD_B = 72;
            const cw      = Math.max(8, Math.floor((MAX_W - PAD_L - PAD_R) / nT));
            const ch      = Math.max(4, Math.floor(280 / nL));
            const W       = PAD_L + nT * cw + PAD_R;
            const H       = PAD_T + nL * ch + PAD_B;

            // Color: white → indigo  (matching paper's blue-to-red palette)
            function sc2hex(s) {
                const t = Math.min(1, Math.max(0, s));
                const r = Math.round(255 - t * (255 - 79));
                const g = Math.round(255 - t * (255 - 70));
                const b = Math.round(255 - t * (255 - 229));
                return `rgb(${r},${g},${b})`;
            }

            const subj_s = subject_range[0], subj_e = subject_range[1];
            const subj_last = subj_e - 1;

            // Draw cells
            let cells = '';
            for (let t = 0; t < nT; t++) {
                for (let l = 0; l < nL; l++) {
                    const x = PAD_L + t * cw;
                    const y = PAD_T + (nL - 1 - l) * ch;   // layer 0 at bottom
                    const s = scores[t][l];
                    const isPeak = (t === subj_last && l === top_layer);
                    const isSubj = (t >= subj_s && t < subj_e);
                    cells += `<rect x="${x}" y="${y}"
                        width="${cw - 1}" height="${ch - 1}"
                        fill="${sc2hex(s)}"
                        stroke="${isPeak ? '#ef4444' : 'none'}"
                        stroke-width="${isPeak ? 2 : 0}"
                        rx="1"
                        data-t="${t}" data-l="${l}" data-s="${s.toFixed(3)}"
                        class="hm-c" style="cursor:crosshair;"/>`;
                }
            }

            // Token labels (rotated –45°)
            let tokLabels = '';
            for (let t = 0; t < nT; t++) {
                const cx = PAD_L + t * cw + cw / 2;
                const ty = PAD_T + nL * ch + 14;
                const isSubj = (t >= subj_s && t < subj_e);
                tokLabels += `<text x="${cx}" y="${ty}"
                    text-anchor="end" font-size="11"
                    fill="${isSubj ? '#2563eb' : '#334155'}"
                    font-weight="${isSubj ? '700' : '400'}"
                    transform="rotate(-45,${cx},${ty})"
                    >${tokens[t]}</text>`;
            }

            // Layer axis ticks (every 8)
            let layerTicks = '';
            for (let l = 0; l < nL; l += 8) {
                const ty = PAD_T + (nL - 1 - l) * ch + ch / 2 + 4;
                layerTicks += `<text x="${PAD_L - 6}" y="${ty}"
                    text-anchor="end" font-size="10" fill="#64748b">${l}</text>`;
            }

            // Arrow annotation for top layer
            const arrowY = PAD_T + (nL - 1 - top_layer) * ch + ch / 2;
            const arrowX = PAD_L + nT * cw + 6;
            const annotLine = `
                <line x1="${arrowX}" y1="${arrowY}"
                      x2="${arrowX + 16}" y2="${arrowY}"
                      stroke="#ef4444" stroke-width="1.5"
                      marker-end="url(#arr)"/>
                <text x="${arrowX + 18}" y="${arrowY + 4}"
                      font-size="10" fill="#ef4444" font-weight="700">
                  L${top_layer}
                </text>`;

            el.innerHTML = `
            <div style="overflow-x:auto;">
              <svg width="${W}" height="${H}" style="display:block;max-width:100%;">
                <defs>
                  <marker id="arr" markerWidth="6" markerHeight="6"
                    refX="5" refY="3" orient="auto">
                    <path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/>
                  </marker>
                </defs>
                ${cells}
                ${tokLabels}
                ${layerTicks}
                ${annotLine}
                <text x="14" y="${PAD_T + nL * ch / 2}"
                    text-anchor="middle" font-size="11" fill="#64748b"
                    transform="rotate(-90,14,${PAD_T + nL * ch / 2})">Layer</text>
                <text x="${PAD_L + nT * cw / 2}" y="${H - 4}"
                    text-anchor="middle" font-size="11" fill="#64748b">
                    ← Token position →
                </text>
              </svg>
              <div style="font-size:11px;color:#64748b;margin-top:4px;">
                🔵 Blue tokens = subject &nbsp;·&nbsp;
                🔴 Red border = peak causal cell &nbsp;·&nbsp;
                Bright = high indirect effect
              </div>
              <div id="tip" style="font-family:monospace;font-size:12px;
                  color:#1e293b;min-height:18px;margin-top:6px;"></div>
            </div>`;

            const tip = el.querySelector('#tip');
            el.querySelectorAll('.hm-c').forEach(c => {
                c.addEventListener('mouseover', () => {
                    const t = +c.dataset.t, l = +c.dataset.l;
                    tip.textContent =
                        `Token "${tokens[t]}" · Layer ${l} · `
                        + `Indirect effect = ${c.dataset.s}`;
                });
                c.addEventListener('click', () => {
                    model.set("clicked_cell", {
                        token: +c.dataset.t,
                        layer: +c.dataset.l,
                        score: +c.dataset.s,
                        token_str: tokens[+c.dataset.t],
                    });
                    model.save_changes();
                });
            });
        }
        export default { render };
        """
        _css = ".hm-c:hover { opacity: 0.7; }"

        heatmap_json = traitlets.Unicode("").tag(sync=True)
        clicked_cell = traitlets.Dict({}).tag(sync=True)

    return (CausalHeatmapWidget,)


@app.cell(hide_code=True)
def cell_heatmap_display(
    trace_scores, trace_tokens, trace_subj_range,
    trace_answer, trace_prompt,
    CausalHeatmapWidget, json, mo,
):
    mo.stop(trace_scores is None)

    # Correct indexing: scores[token_idx, layer_idx]
    _subj_last   = trace_subj_range[1] - 1
    _per_layer   = trace_scores[_subj_last, :]   # (n_layers,) — effects at subj last token
    top_layer    = int(_per_layer.argmax())
    top_score    = float(_per_layer.max())

    # Serialise as scores[t][l]  — confirmed shape from real output
    _payload = {
        "scores":        trace_scores.tolist(),   # list[list] — [t][l]
        "tokens":        trace_tokens,
        "subject_range": list(trace_subj_range),
        "top_layer":     _top_layer,
    }

    _w           = CausalHeatmapWidget(heatmap_json=json.dumps(_payload))
    heatmap_widget = mo.ui.anywidget(_w)

    mo.vstack([
        mo.md(f"### Causal Trace — *\"{trace_prompt}\"*"),
        heatmap_widget,
    ], gap=1)
    return top_layer, top_score, heatmap_widget


@app.cell(hide_code=True)
def cell_trace_callout(top_layer, top_score, trace_subj_range, trace_tokens, mo):
    mo.stop(top_layer is None)

    _subj_last = trace_subj_range[1] - 1
    _subj_str  = "".join(trace_tokens[trace_subj_range[0]:trace_subj_range[1]])

    mo.callout(
        mo.md(f"""
        **Reading the heatmap:**
        The brightest cell is at **layer {top_layer}**,
        subject's last token **"{trace_tokens[_subj_last]}"**
        (indirect effect = {top_score:.3f}).

        This is the MLP at layer {top_layer} processing the token *{trace_tokens[_subj_last]}*
        — the end of *"{_subj_str.strip()}"*.
        The paper shows this pattern reliably across dozens of facts:
        **factual associations are stored in mid-to-late MLP layers at the subject's last token.**

        ROME will target exactly layer {top_layer} to rewrite this fact.
        """),
        kind="neutral",
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT V — ROME EDIT
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_rome_theory(mo):
    mo.md(r"""
    ---
    ## Act V — The ROME Edit: One Rank-One Update

    GPT-2's MLP at layer $\ell$ computes:

    $$\mathbf{h} = \text{act}(W_\text{fc}\,\mathbf{x}) \quad \in \mathbb{R}^{6400}$$
    $$\mathbf{out} = W_\text{proj}\,\mathbf{h} + \mathbf{b} \quad \in \mathbb{R}^{1600}$$

    Treat this as **key-value memory**:
    the hidden activation $\mathbf{h}$ is the **key** $k$ (encodes "what is being looked up"),
    and $W_\text{proj}\,k$ is the **value** (what gets written to the residual stream).

    **To change the fact stored at subject key $k^*$:**
    find a new value $v^*$ that causes the model to output the new target,
    then update $W_\text{proj}$ to map $k^* \mapsto v^*$.
    The update must leave all other keys unchanged.

    **The rank-one solution (Sherman-Morrison):**

    $$W'_\text{proj} = W_\text{proj} +
    \underbrace{\frac{(v^* - W_\text{proj}\,k^*)\,k^{*\top}}{k^{*\top} k^*}}_{\text{rank-one perturbation}}$$

    This is the *minimum-change* update: it touches only the one direction $k^*$
    in weight space, leaving every other direction exactly as it was.

    **Finding $k^*$ and $v^*$:**
    - $k^*$ = average MLP hidden activation at the subject's last token,
      over noise-corrupted runs (prevents the key from being prompt-specific)
    - $v^*$ = gradient-optimised over several rephrasings of the edit prompt
      to maximise P(target_new) while generalising across surface forms
    """)
    return


@app.cell
def cell_edit_ui(PRESET_FACTS, ROME_LAYER_DEFAULT, mo):
    edit_fact_ui = mo.ui.dropdown(
        options={f["id"]: f["subject"] for f in PRESET_FACTS},
        value="eiffel",
        label="Fact to edit",
    )
    edit_new_ui = mo.ui.text(
        value="Rome",
        label="New target answer",
        full_width=False,
    )
    edit_layer_ui = mo.ui.slider(
        0, 47, value=ROME_LAYER_DEFAULT, step=1, show_value=True,
        label="Edit layer  (set to causal trace peak layer above)",
    )
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("**Fact to rewrite**"),   edit_fact_ui]),
            mo.vstack([mo.md("**New target**"),         edit_new_ui]),
        ], gap=3, justify="start"),
        mo.vstack([
            mo.md("**Target layer** *(move to match the red-bordered cell in the heatmap)*"),
            edit_layer_ui,
        ]),
    ], gap=2)
    return edit_fact_ui, edit_new_ui, edit_layer_ui


# ── ROME helpers — pure torch, no external deps ───────────────────────────────

@app.function
def rome_find_subject_range(tokenizer, prompt, subject):
    """Return (start, end) exclusive token indices of subject in prompt."""
    full_ids = tokenizer.encode(prompt)
    for prefix in (" " + subject, subject):
        sub_ids = tokenizer.encode(prefix)
        for i in range(len(full_ids) - len(sub_ids) + 1):
            if full_ids[i: i + len(sub_ids)] == sub_ids:
                return i, i + len(sub_ids)
    return 0, 1


@app.function
def rome_get_key_vector(model, tokenizer, prompt, subject,
                         layer_id, device, n_samples=20, noise_coef=3.0):
    """
    k* = average MLP hidden activation (input to c_proj) at subject's
    last token across noise-corrupted forward passes.

    GPT-2 XL MLP:
      c_fc  : Linear(1600 → 6400) + act  → hidden h  (shape 6400)
      c_proj: Linear(6400 → 1600)         → output

    We hook c_proj's INPUT to capture h.
    """
    import torch

    s, e = rome_find_subject_range(tokenizer, prompt, subject)
    subj_last = e - 1
    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    noise_std = noise_coef * model.transformer.wte.weight.std().item()

    keys = []
    cap  = {}

    def capture_key(module, inp, out):
        cap["k"] = inp[0][0, subj_last, :].detach().float()

    def add_noise(module, inp, out):
        o = out.clone()
        o[0, s:e, :] += torch.randn_like(o[0, s:e, :]) * noise_std
        return o

    hook_proj  = model.transformer.h[layer_id].mlp.c_proj.register_forward_hook(capture_key)
    hook_embed = model.transformer.wte.register_forward_hook(add_noise)

    for _ in range(n_samples):
        with torch.no_grad():
            model(**inputs)
        if "k" in cap:
            keys.append(cap.pop("k"))

    hook_proj.remove()
    hook_embed.remove()

    if not keys:
        raise RuntimeError("Key capture failed — check hook target names for GPT-2 XL.")
    return torch.stack(keys).mean(0)   # (6400,)


@app.function
def rome_optimize_value(model, tokenizer, request,
                         layer_id, key_vec, device,
                         n_steps=25, lr=0.05):
    """
    v* = argmin_{v} sum_{prompt in paraphrases} -log P(target_new | prompt, v injected)

    We inject v as the MLP c_proj output at the subject's last token position,
    then backprop through the rest of the network to update v.
    """
    import torch
    import torch.nn.functional as F

    target_tok = tokenizer.encode(" " + request["target_new"].strip())[0]
    prompts    = ([request["prompt"]] + request.get("gen_prompts", []))[:3]

    # GPT-2 uses Conv1D(nf, nx) where weight.shape = (nx, nf) = (6400, 1600)
    # forward: out = inp @ weight + bias  → cur_v = k @ W
    W = model.transformer.h[layer_id].mlp.c_proj.weight.float()  # (6400, 1600)
    with torch.no_grad():
        v = (key_vec.to(W.device).float() @ W).clone().detach()  # k @ W → (1600,)
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    v_ref = [v]   # mutable reference for closure

    for _ in range(n_steps):
        opt.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for p in prompts:
            s, e = rome_find_subject_range(tokenizer, p, request["subject"])
            sp   = e - 1
            inp  = tokenizer(p, return_tensors="pt").to(device)

            def inject(mod, inp_, out, _sp=sp, _vr=v_ref):
                o = out.clone().float()
                o[0, _sp, :] = _vr[0]
                return o

            h = model.transformer.h[layer_id].mlp.c_proj.register_forward_hook(inject)
            out_model = model(**inp)
            h.remove()

            logits = out_model.logits[0, -1, :].float()
            loss   = loss - F.log_softmax(logits, dim=-1)[target_tok] / len(prompts)

        loss.backward()
        opt.step()
        v_ref[0] = v

    return v.detach()


@app.function
def rome_apply_edit(model, layer_id, key_vec, value_vec):
    """
    Rank-one update for GPT-2 Conv1D weights.

    GPT-2 uses transformers.Conv1D (NOT nn.Linear):
      Conv1D(nf=d_model, nx=d_mlp)  →  weight.shape = (nx, nf) = (6400, 1600)
      forward: out = inp @ weight + bias   ← note: inp @ W, not W @ inp

    So for key k (shape 6400,):
      current value = k @ W            → (1600,)
      target update: k @ ΔW = delta_v  → ΔW = outer(k, delta_v) / (k·k)
      ΔW shape: (6400, 1600) ✓ matches W.shape
    """
    import torch

    W   = model.transformer.h[layer_id].mlp.c_proj.weight   # (6400, 1600)
    k   = key_vec.to(W.device).float()
    v_s = value_vec.to(W.device).float()

    with torch.no_grad():
        cur_v  = k @ W.float()                           # (1600,)
        delta  = v_s - cur_v                             # (1600,)
        update = torch.outer(k, delta) / (k @ k + 1e-8) # (6400, 1600) ✓
        W.data += update.to(W.dtype)


@app.cell
def cell_edit_btn(mo):
    edit_btn = mo.ui.run_button(
        label="✏️  Apply ROME Edit  (< 60 s on T4)",
        kind="danger",
        full_width=True,
    )
    edit_btn
    return (edit_btn,)


@app.cell
def cell_edit_run(edit_btn, edit_fact_ui, edit_new_ui, edit_layer_ui,
                   PRESET_FACTS, model, tokenizer, DEVICE, torch,
                   rome_get_key_vector, rome_optimize_value, rome_apply_edit,
                   mo):
    mo.stop(not edit_btn.value)
    mo.stop(model is None, mo.md("Load the model first (Act II)."))

    _fact    = next(f for f in PRESET_FACTS if f["id"] == edit_fact_ui.value)
    _layer   = edit_layer_ui.value
    _new_ans = edit_new_ui.value.strip()
    _request = {
        "prompt":     _fact["prompt"],
        "subject":    _fact["subject"],
        "target_new": _new_ans,
        "gen_prompts": _fact["gen_prompts"],
    }

    # ── 1. Collect "before" predictions (model NOT yet edited) ────────────
    _before = {}
    _all_before_prompts = (
        [_fact["prompt"]]
        + _fact["gen_prompts"]
        + _fact["spec_prompts"]
        + _fact["coh_prompts"]   # ← include coherence prompts too
    )
    with torch.no_grad():
        for _p in _all_before_prompts:
            _inp = tokenizer(_p, return_tensors="pt").to(DEVICE)
            _tok = tokenizer.decode(
                [model(**_inp).logits[0, -1, :].argmax()]
            ).strip()
            _before[_p] = _tok

    # ── 2. Key vector ──────────────────────────────────────────────────────
    with mo.status.spinner(title="Step 1/3 — Computing key vector…"):
        _k = rome_get_key_vector(
            model, tokenizer,
            _request["prompt"], _request["subject"],
            _layer, DEVICE,
        )

    # ── 3. Optimise value vector ───────────────────────────────────────────
    with mo.status.spinner(title="Step 2/3 — Optimising value vector…"):
        _v = rome_optimize_value(
            model, tokenizer, _request, _layer, _k, DEVICE,
        )

    # ── 4. Apply rank-one update IN-PLACE ─────────────────────────────────
    with mo.status.spinner(title="Step 3/3 — Applying rank-one weight update…"):
        rome_apply_edit(model, _layer, _k, _v)
        model.eval()

    before_preds = _before
    edit_fact    = _fact
    edit_new_ans = _new_ans
    edit_layer   = _layer

    mo.callout(
        mo.md(f"✅ **Edit applied.** Layer {_layer} · "
              f"*{_fact['subject']}* now maps to **{_new_ans}**."),
        kind="success",
    )
    return before_preds, edit_fact, edit_new_ans, edit_layer


# ─────────────────────────────────────────────────────────────────────────────
# ACT VI — BEFORE / AFTER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_ba_intro(mo):
    mo.md(r"""
    ---
    ## Act VI — Before vs After

    Three categories of test prompts tell us how surgical the edit was:

    | Category | Question |
    |----------|----------|
    | ✅ **Efficacy** | Does the edited fact produce the new target? |
    | 🔄 **Generalization** | Do paraphrase prompts also reflect the change? |
    | 🎯 **Specificity** | Are unrelated facts about the same subject unchanged? |
    """)
    return


@app.cell(hide_code=True)
def cell_before_after(before_preds, edit_fact, edit_new_ans,
                       model, tokenizer, DEVICE, torch, mo):
    mo.stop(before_preds is None)

    _all_prompts = (
        [edit_fact["prompt"]]
        + edit_fact["gen_prompts"]
        + edit_fact["spec_prompts"]
    )
    _categories = (
        ["✅ Efficacy"]
        + ["🔄 Generalization"] * len(edit_fact["gen_prompts"])
        + ["🎯 Specificity"]    * len(edit_fact["spec_prompts"])
    )

    _rows = []
    with torch.no_grad():
        for _cat, _p in zip(_categories, _all_prompts):
            _inp = tokenizer(_p, return_tensors="pt").to(DEVICE)
            _after_tok = tokenizer.decode(
                [model(**_inp).logits[0, -1, :].argmax()]
            ).strip()
            _before_tok = before_preds.get(_p, "—")
            _changed    = _before_tok != _after_tok

            # For efficacy/generalization: hit = new answer matches target
            # For specificity: hit = answer did NOT change
            if _cat.startswith("🎯"):
                _verdict = "✅ unchanged" if not _changed else "⚠️ changed"
            else:
                _hit     = _after_tok.lower() == edit_new_ans.lower()
                _verdict = "✅" if _hit else "❌"

            _rows.append({
                "Category": _cat,
                "Prompt":   _p + "  ___",
                "Before":   _before_tok,
                "After":    _after_tok,
                "Verdict":  _verdict,
            })

    mo.ui.table(_rows, selection=None, pagination=False,
                label="ROME Edit — Before vs After")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT VII — EXTENSION: EDIT COHERENCE  (novel — not in the paper)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_coh_intro(mo):
    mo.md(r"""
    ---
    ## Act VII — Extension: Edit Coherence
    > **Novel contribution — the ROME paper does not test this**

    ROME's evaluation measures:
    - **Efficacy** — does the direct fact change? ✓
    - **Specificity** — do unrelated facts stay the same? ✓
    - **Generalization** — do paraphrases also change? ✓

    **What it does NOT measure:** do *logically entailed* facts cascade?

    If we rewrite *"The Eiffel Tower is in Paris"* → Rome, then:
    - "The **country** containing the Eiffel Tower is ___" should change
      **France → Italy**
    - "The **language** spoken where the Eiffel Tower stands" should shift
      **French → Italian**

    ROME only edits one MLP layer. There's no mechanism guaranteeing
    that the web of associated facts updates consistently.
    The **Edit Coherence Score** measures what fraction actually do.

    Low coherence = the model now holds *logically inconsistent beliefs*:
    it says the Eiffel Tower is in Rome, but still says the local currency
    is the Euro (used in France) rather than the Euro (used in Italy — same!),
    or still says the local language is French.
    """)
    return


@app.cell(hide_code=True)
def cell_coherence(before_preds, edit_fact, edit_new_ans,
                    model, tokenizer, DEVICE, torch, mo):
    mo.stop(before_preds is None)

    _rows = []
    with torch.no_grad():
        for _p in edit_fact["coh_prompts"]:
            _inp    = tokenizer(_p, return_tensors="pt").to(DEVICE)
            _after  = tokenizer.decode(
                [model(**_inp).logits[0, -1, :].argmax()]
            ).strip()
            _before = before_preds.get(_p, "—")
            _changed = _before != _after
            _rows.append({
                "Entailed prompt": _p + "  ___",
                "Before edit":    _before,
                "After edit":     _after,
                "Cascaded?":      "✓" if _changed else "—",
            })

    _n_cascaded = sum(1 for r in _rows if r["Cascaded?"] == "✓")
    _coherence  = _n_cascaded / max(len(_rows), 1)

    mo.vstack([
        mo.stat(
            f"{_coherence:.0%}",
            label="Edit Coherence Score",
            caption=f"{_n_cascaded} of {len(_rows)} entailed facts cascaded",
        ),
        mo.ui.table(_rows, selection=None, pagination=False,
                    label="Entailed facts — before vs after ROME edit"),
        mo.callout(
            mo.md(
                f"**Interpretation:** ROME achieves **{_coherence:.0%} coherence** "
                "on logically entailed facts. "
                "The rank-one update only modifies one weight direction — "
                "there is no mechanism to propagate logical consequences. "
                "Coherence that does appear is an emergent property of the "
                "model's pre-existing associative structure, not a guarantee of ROME. "
                "This gap is the central motivation of follow-up works "
                "(MEMIT, GRACE, WilKE)."
            ),
            kind="warn" if _coherence < 0.5 else "success",
        ),
    ], gap=1)
    return


# ─────────────────────────────────────────────────────────────────────────────
# TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_takeaways(mo):
    mo.md(r"""
    ---
    ## Key Takeaways

    | Finding | What it means |
    |---------|---------------|
    | 🔍 **Facts have precise addresses** | Causal tracing localises each fact to a narrow band: subject's last token × mid-to-late MLP layers |
    | ✏️ **Rank-one update is enough** | One outer-product perturbation to $W_\text{proj}$ at one layer redirects the stored association |
    | 🎯 **Specificity by construction** | The update only changes the key direction $k^*$; all other keys are mathematically untouched |
    | 🔄 **Generalization is real** | The key averaging over noise-corrupted runs prevents the edit from being prompt-specific |
    | ⚠️ **Coherence is not guaranteed** | Logically entailed facts may or may not cascade — an open problem beyond ROME |
    | 🔭 **Broader implication** | Rapid, surgical knowledge correction in deployed LLMs — without retraining, without catastrophic forgetting |

    ---

    **Paper:** [arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262) ·
    Meng, Bau, Andonian & Belinkov · NeurIPS 2022  
    **Code:** [github.com/kmeng01/rome](https://github.com/kmeng01/rome)  
    **Notebook:** alphaXiv × marimo GPU Notebook Competition #2
    """)
    return


if __name__ == "__main__":
    app.run()
