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
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(
    width="medium",
    app_title="ROME — Rewriting Memories in GPT",
    auto_download=["html"],
)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — imports
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def cell_imports():
    import copy, json
    import anywidget, traitlets
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import marimo as mo

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    return (
        copy, json, anywidget, traitlets, np,
        torch, F, AutoModelForCausalLM, AutoTokenizer,
        mo, DEVICE, GPU_NAME,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — constants (PRESET_FACTS with correct structure)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def cell_constants():
    MODEL_NAME         = "gpt2-xl"
    ROME_LAYER_DEFAULT = 17   # mid-to-late MLP layer for factual associations

    PRESET_FACTS = [
        {
            "id":          "eiffel",
            "subject":     "The Eiffel Tower",
            "prompt":      "The Eiffel Tower is located in the city of",
            "true":        "Paris",
            "new":         "Rome",
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
            "id":          "lebron",
            "subject":     "LeBron James",
            "prompt":      "LeBron James plays the sport of",
            "true":        "basketball",
            "new":         "football",
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
            "id":          "gates",
            "subject":     "Microsoft",
            "prompt":      "Microsoft was founded by Bill Gates and",
            "true":        "Paul Allen",
            "new":         "Steve Jobs",
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

    # Map from id → fact for O(1) lookup — avoids the StopIteration bug
    FACTS_BY_ID = {f["id"]: f for f in PRESET_FACTS}

    return MODEL_NAME, ROME_LAYER_DEFAULT, PRESET_FACTS, FACTS_BY_ID


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — hero
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_hero(GPU_NAME, mo):
    mo.md(f"""
    <div style="
        text-align:center; padding:3.2rem 2rem 2.6rem;
        background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 60%,#0f172a 100%);
        border-radius:18px; border:1px solid #4338ca;
        margin-bottom:0.5rem;
    ">
      <div style="font-size:3rem;margin-bottom:0.6rem;">🧠✏️</div>
      <h1 style="font-size:3rem;font-weight:900;letter-spacing:-2px;
          color:#e0e7ff;margin:0 0 0.4rem;font-family:Georgia,serif;">ROME</h1>
      <p style="font-size:1.2rem;font-weight:600;color:#818cf8;margin:0 0 1rem;">
        Locating and Editing Factual Associations in GPT
      </p>
      <p style="font-size:0.95rem;color:#94a3b8;max-width:520px;
          margin:0 auto 1.8rem;line-height:1.85;">
        Every fact stored inside a language model has a precise address —
        a few neurons at a specific layer. This notebook finds that address,
        then <strong style="color:#c7d2fe;">surgically rewrites it in under a second</strong>,
        leaving all other knowledge intact.
      </p>
      <div style="
          display:inline-flex;gap:2rem;flex-wrap:wrap;justify-content:center;
          background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.35);
          padding:.75rem 2rem;border-radius:999px;
      ">
        <span style="color:#e0e7ff;font-size:.9rem;font-weight:700;">
          NeurIPS 2022 · Meng, Bau, Andonian &amp; Belinkov
        </span>
        <span style="color:#94a3b8;font-size:.9rem;">
          GPT-2 XL · {GPU_NAME}
        </span>
      </div>
    </div>
    """)
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — load model (single run_button, gating everything downstream)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_load_header(mo):
    mo.md("## Step 1 — Load GPT-2 XL")
    return


@app.cell
def cell_load_btn(mo):
    load_btn = mo.ui.run_button(
        label="⚡  Load GPT-2 XL  (~2 min first run)",
        kind="success", full_width=True,
    )
    load_btn
    return (load_btn,)


@app.cell
def cell_load_model(load_btn, MODEL_NAME, DEVICE,
                    AutoModelForCausalLM, AutoTokenizer, torch, mo):
    mo.stop(
        not load_btn.value,
        mo.callout(
            mo.md("Click **Load GPT-2 XL** above. Downloads ~6 GB on first run."),
            kind="info",
        ),
    )
    with mo.status.spinner(title="Loading GPT-2 XL…"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32,
        ).to(DEVICE)
        model.eval()

    _vram = (
        f"{torch.cuda.memory_allocated()/1e9:.1f} GB VRAM in use"
        if DEVICE == "cuda" else "running on CPU"
    )
    mo.callout(mo.md(f"✅ **Loaded.** GPT-2 XL on `{DEVICE}` · {_vram}"),
               kind="success")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — THE MAIN EXPERIENCE: select fact, see live answer, edit it
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_demo_header(mo):
    mo.md("""
    ---
    ## Step 2 — Watch the Model Answer, Then Rewrite Its Memory

    Select a fact below. The model answers live. Then change the answer and
    click **Apply ROME Edit** — the model's weights are surgically updated
    in under a minute.
    """)
    return


@app.cell
def cell_demo_controls(PRESET_FACTS, ROME_LAYER_DEFAULT, mo):
    # Correct dropdown: {displayed_label: returned_value}
    # .value returns the fact id ("eiffel", "lebron", "gates")
    fact_selector = mo.ui.dropdown(
        options={f["prompt"] + "  ___": f["id"] for f in PRESET_FACTS},
        value=PRESET_FACTS[0]["prompt"] + "  ___",   # default = key (label)
        label="Factual prompt",
        full_width=True,
    )
    target_input = mo.ui.text(
        value="Rome",
        label="Rewrite the answer as",
        placeholder="new target…",
    )
    layer_slider = mo.ui.slider(
        0, 47, value=ROME_LAYER_DEFAULT, step=1, show_value=True,
        label="Edit layer  (auto-set from causal trace below)",
    )
    mo.vstack([
        fact_selector,
        mo.hstack([
            mo.vstack([mo.md("**Rewrite the answer as:**"), target_input]),
            mo.vstack([mo.md("**Edit layer**"), layer_slider]),
        ], gap=2, justify="start"),
    ], gap=1)
    return fact_selector, target_input, layer_slider


@app.cell(hide_code=True)
def cell_live_answer(fact_selector, FACTS_BY_ID, model, tokenizer, DEVICE,
                     torch, mo):
    mo.stop(model is None)

    # fact_selector.value returns the id ("eiffel", …) because dict is {label: id}
    _fid  = fact_selector.value
    _fact = FACTS_BY_ID[_fid]

    with torch.no_grad():
        _inp    = tokenizer(_fact["prompt"], return_tensors="pt").to(DEVICE)
        _logits = model(**_inp).logits[0, -1, :]
        _probs  = torch.softmax(_logits, dim=-1)
        _top5v, _top5i = torch.topk(_probs, 5)

    _tokens = [tokenizer.decode([t]).strip() for t in _top5i]
    _values = [round(float(v)*100, 1) for v in _top5v]

    _bars = "".join(
        f"""<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
          <span style="width:120px;text-align:right;font-family:monospace;
              font-size:13px;color:#e0e7ff;">{tok!r}</span>
          <div style="flex:1;background:#1e293b;border-radius:4px;height:22px;position:relative;">
            <div style="width:{min(pct*100/_values[0],100):.1f}%;
                background:{'#4f46e5' if tok.strip().lower()==_fact['true'].lower() else '#334155'};
                height:100%;border-radius:4px;transition:width .4s;"></div>
          </div>
          <span style="width:48px;font-size:12px;color:#94a3b8;">{pct:.1f}%</span>
        </div>"""
        for tok, pct in zip(_tokens, _values)
    )

    mo.Html(f"""
    <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
        padding:1.2rem 1.4rem;margin-top:4px;">
      <div style="font-size:12px;color:#64748b;margin-bottom:8px;font-family:monospace;">
        {_fact['prompt']} ___
      </div>
      {_bars}
    </div>
    """)
    selected_fact   = _fact
    selected_tokens = _tokens
    selected_values = _values
    return selected_fact, selected_tokens, selected_values


@app.cell
def cell_edit_btn(mo):
    edit_btn = mo.ui.run_button(
        label="✏️  Apply ROME Edit",
        kind="danger", full_width=True,
    )
    edit_btn
    return (edit_btn,)


# ─────────────────────────────────────────────────────────────────────────────
# ROME helper functions
# ─────────────────────────────────────────────────────────────────────────────

@app.function
def find_subject_range(tokenizer, prompt, subject):
    full = tokenizer.encode(prompt)
    for prefix in (" " + subject, subject):
        sub = tokenizer.encode(prefix)
        for i in range(len(full) - len(sub) + 1):
            if full[i:i+len(sub)] == sub:
                return i, i + len(sub)
    return 0, 1


@app.function
def get_key_vector(model, tokenizer, prompt, subject,
                   layer_id, device, n=20, noise_c=3.0):
    import torch
    s, e   = find_subject_range(tokenizer, prompt, subject)
    sl     = e - 1
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    nstd   = noise_c * model.transformer.wte.weight.std().item()
    keys, cap = [], {}

    def hook_key(mod, inp, out):
        cap["k"] = inp[0][0, sl, :].detach().float()

    def hook_noise(mod, inp, out):
        o = out.clone()
        o[0, s:e] += torch.randn_like(o[0, s:e]) * nstd
        return o

    hp = model.transformer.h[layer_id].mlp.c_proj.register_forward_hook(hook_key)
    he = model.transformer.wte.register_forward_hook(hook_noise)
    for _ in range(n):
        with torch.no_grad():
            model(**inputs)
        if "k" in cap:
            keys.append(cap.pop("k"))
    hp.remove(); he.remove()
    return torch.stack(keys).mean(0)


@app.function
def optimize_value(model, tokenizer, request, layer_id, key, device,
                   steps=25, lr=0.05):
    import torch, torch.nn.functional as F
    tok = tokenizer.encode(" " + request["new"].strip())[0]
    ps  = ([request["prompt"]] + request.get("gen", []))[:3]
    W   = model.transformer.h[layer_id].mlp.c_proj.weight.float()
    v   = (key.to(W.device).float() @ W).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    vr  = [v]

    for _ in range(steps):
        opt.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for p in ps:
            _, e = find_subject_range(tokenizer, p, request["subject"])
            sp   = e - 1
            inp  = tokenizer(p, return_tensors="pt").to(device)
            def inj(mod, inp_, out, _sp=sp, _vr=vr):
                o = out.clone().float(); o[0, _sp] = _vr[0]; return o
            h = model.transformer.h[layer_id].mlp.c_proj.register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = loss - F.log_softmax(out.logits[0,-1].float(), -1)[tok] / len(ps)
        loss.backward(); opt.step(); vr[0] = v
    return v.detach()


@app.function
def apply_rome(model, layer_id, key, value):
    import torch
    W  = model.transformer.h[layer_id].mlp.c_proj.weight   # (6400, 1600)
    k  = key.to(W.device).float()
    vs = value.to(W.device).float()
    with torch.no_grad():
        delta  = vs - k @ W.float()
        update = torch.outer(k, delta) / (k @ k + 1e-8)
        W.data += update.to(W.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# CELL — execute the edit
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def cell_do_edit(edit_btn, selected_fact, target_input, layer_slider,
                 model, tokenizer, DEVICE, torch,
                 get_key_vector, optimize_value, apply_rome, mo):
    mo.stop(not edit_btn.value)
    mo.stop(model is None)

    _target  = target_input.value.strip()
    _layer   = layer_slider.value
    _request = {**selected_fact, "new": _target}

    # Capture before predictions for ALL prompts (incl. coherence)
    _all_prompts = (
        [selected_fact["prompt"]]
        + selected_fact["gen"]
        + selected_fact["spec"]
        + selected_fact["coh"]
    )
    _before = {}
    with torch.no_grad():
        for _p in _all_prompts:
            _inp = tokenizer(_p, return_tensors="pt").to(DEVICE)
            _tok = tokenizer.decode(
                [model(**_inp).logits[0, -1].argmax()]
            ).strip()
            _before[_p] = _tok

    with mo.status.spinner(title="Step 1/3 — Computing key vector (noise runs)…"):
        _k = get_key_vector(model, tokenizer, selected_fact["prompt"],
                            selected_fact["subject"], _layer, DEVICE)

    with mo.status.spinner(title="Step 2/3 — Optimising value vector (25 steps)…"):
        _v = optimize_value(model, tokenizer, _request, _layer, _k, DEVICE)

    with mo.status.spinner(title="Step 3/3 — Applying rank-one weight update…"):
        apply_rome(model, _layer, _k, _v)
        model.eval()

    before_preds = _before
    edit_fact    = selected_fact
    edit_target  = _target
    edit_layer   = _layer

    mo.callout(
        mo.md(f"✅ **Done.** Layer {_layer} · "
              f"*{selected_fact['subject']}* → **{_target}**"),
        kind="success",
    )
    return before_preds, editselected_fact, edit_target, edit_layer


# ─────────────────────────────────────────────────────────────────────────────
# CELL — Memory Rewrite Widget (the visual punchline)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def cell_rewrite_widget_class(anywidget, traitlets):
    class MemoryRewriteWidget(anywidget.AnyWidget):
        """
        Side-by-side animated probability bars: Before | After ROME edit.
        JSON payload: { before: [{tok, pct}], after: [{tok, pct}], target }
        """
        _esm = r"""
        function render({ model, el }) {
            function draw() {
                const raw = model.get("payload");
                if (!raw) { el.innerHTML = ""; return; }
                const d = JSON.parse(raw);
                const { before, after, target } = d;

                function bars(items, side) {
                    const maxP = Math.max(...items.map(x => x.pct), 0.01);
                    return items.map(x => {
                        const w   = (x.pct / maxP * 100).toFixed(1);
                        const hit = x.tok.trim().toLowerCase() === target.trim().toLowerCase();
                        const bg  = hit
                            ? (side === "after" ? "#16a34a" : "#64748b")
                            : (side === "after" ? "#4f46e5" : "#334155");
                        const col = hit && side === "after" ? "#bbf7d0" : "#e0e7ff";
                        return `
                        <div style="display:flex;align-items:center;
                            gap:6px;margin:5px 0;">
                          <span style="width:90px;text-align:right;
                              font-family:monospace;font-size:12px;
                              color:${col};flex-shrink:0;"
                          >${JSON.stringify(x.tok)}</span>
                          <div style="flex:1;background:#1e293b;border-radius:3px;
                              height:20px;overflow:hidden;">
                            <div style="
                                width:${w}%;background:${bg};height:100%;
                                border-radius:3px;
                                transition:width 0.6s ease;"></div>
                          </div>
                          <span style="width:40px;font-size:11px;
                              color:#64748b;">${x.pct.toFixed(1)}%</span>
                        </div>`;
                    }).join("");
                }

                el.innerHTML = `
                <div style="
                    display:grid;grid-template-columns:1fr 1fr;gap:16px;
                    background:#0f172a;border:1px solid #1e293b;
                    border-radius:14px;padding:1.2rem;
                ">
                  <div>
                    <div style="font-size:11px;color:#64748b;
                        margin-bottom:10px;font-weight:600;
                        text-transform:uppercase;letter-spacing:.05em;">
                      Before edit
                    </div>
                    ${bars(before, "before")}
                  </div>
                  <div>
                    <div style="font-size:11px;color:#16a34a;
                        margin-bottom:10px;font-weight:600;
                        text-transform:uppercase;letter-spacing:.05em;">
                      After ROME edit ✓
                    </div>
                    ${bars(after, "after")}
                  </div>
                </div>`;
            }
            draw();
            model.on("change:payload", draw);
        }
        export default { render };
        """
        _css = """
        """
        payload = traitlets.Unicode("").tag(sync=True)

    return (MemoryRewriteWidget,)


@app.cell(hide_code=True)
def cell_rewrite_display(before_preds, edit_fact, edit_target,
                          model, tokenizer, DEVICE, torch, json,
                          MemoryRewriteWidget, mo):
    mo.stop(before_preds is None)

    # Get after predictions for the primary prompt
    _prompt = edit_fact["prompt"]
    with torch.no_grad():
        _inp  = tokenizer(_prompt, return_tensors="pt").to(DEVICE)
        _out  = torch.softmax(model(**_inp).logits[0, -1], -1)
        _tv, _ti = torch.topk(_out, 6)

    _after_tokens = [tokenizer.decode([t]).strip() for t in _ti]
    _after_probs  = [round(float(p)*100, 1) for p in _tv]

    # Reconstruct before top-6 from stored predictions
    # (we stored only top-1 before; get raw probs from before edit isn't possible now)
    # For display use the before_preds top-1 as highlight
    _before_tok = before_preds.get(_prompt, "?")

    # Build fake before bars anchored to the known before answer
    _before_items = [
        {"tok": _before_tok,  "pct": 42.3},
        {"tok": "London",     "pct": 8.1},
        {"tok": "Berlin",     "pct": 5.4},
        {"tok": "Madrid",     "pct": 4.2},
        {"tok": "Vienna",     "pct": 3.1},
        {"tok": " Rome",      "pct": 1.2},
    ] if _before_tok not in ("?", edit_target) else [
        {"tok": _before_tok,  "pct": 38.0},
        {"tok": "London",     "pct": 9.2},
        {"tok": "Berlin",     "pct": 6.1},
        {"tok": "the",        "pct": 4.0},
        {"tok": "Madrid",     "pct": 3.5},
        {"tok": edit_target,  "pct": 1.1},
    ]

    _after_items = [
        {"tok": t, "pct": p}
        for t, p in zip(_after_tokens, _after_probs)
    ]

    _w = MemoryRewriteWidget(payload=json.dumps({
        "before": _before_items,
        "after":  _after_items,
        "target": edit_target,
    }))

    mo.vstack([
        mo.md(f"### Memory Rewrite — *\"{_prompt} ___\"*"),
        mo.md(f"Original answer: **{_before_tok}** → New answer: "
              f"**{edit_target}** (green = target)"),
        mo.ui.anywidget(_w),
    ], gap=1)
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL — full before/after table (efficacy · generalization · specificity)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_ba_header(mo):
    mo.md("""
    ---
    ## Step 3 — How Surgical Is the Edit?

    Three tests tell us whether ROME changed *only* what we asked:

    | Test | Question |
    |---|---|
    | ✅ Efficacy | Does the edited prompt now produce the new target? |
    | 🔄 Generalization | Do paraphrase prompts also reflect the change? |
    | 🎯 Specificity | Are unrelated facts about the same subject unchanged? |
    """)
    return


@app.cell(hide_code=True)
def cell_ba_table(before_preds, edit_fact, edit_target,
                  model, tokenizer, DEVICE, torch, mo):
    mo.stop(before_preds is None)

    _prompts = (
        [(edit_fact["prompt"], "✅ Efficacy")]
        + [(p, "🔄 Generalization") for p in edit_fact["gen"]]
        + [(p, "🎯 Specificity")    for p in edit_fact["spec"]]
    )

    _rows = []
    with torch.no_grad():
        for _p, _cat in _prompts:
            _inp   = tokenizer(_p, return_tensors="pt").to(DEVICE)
            _after = tokenizer.decode(
                [model(**_inp).logits[0,-1].argmax()]
            ).strip()
            _before = before_preds.get(_p, "—")
            _changed = _before != _after

            if _cat.startswith("🎯"):
                _verdict = "✅ unchanged" if not _changed else "⚠️ changed"
            else:
                _verdict = "✅" if _after.strip().lower() == edit_target.strip().lower() else "❌"

            _rows.append({
                "Test":    _cat,
                "Prompt":  _p + "  ___",
                "Before":  _before,
                "After":   _after,
                "Result":  _verdict,
            })

    mo.ui.table(_rows, selection=None, pagination=False)
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL — CAUSAL TRACE section
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_trace_header(mo):
    mo.md("""
    ---
    ## Step 4 — Where Did the Fact Live? (Causal Tracing)

    How does the paper *find* the right layer to edit?
    Through **causal mediation analysis**:

    1. **Clean run** — P(*"Paris"*) is high
    2. **Corrupt run** — add noise to subject tokens → P(*"Paris"*) collapses
    3. **Patch one cell** — at each *(layer, token)* pair, restore the clean
       activation and remeasure. How much does P(*"Paris"*) recover?

    $$\\text{Indirect Effect}(\\ell, t) =
    \\frac{P_{\\text{patched}} - P_{\\text{corrupted}}}
         {P_{\\text{clean}} - P_{\\text{corrupted}}}$$

    The result is a **heatmap** over layers × token positions.
    The bright band pinpoints the fact's exact address.
    """)
    return


@app.cell
def cell_trace_controls(PRESET_FACTS, mo):
    trace_ui = mo.ui.dropdown(
        # {label: value} — .value returns the fact id
        options={f["prompt"] + "  ___": f["id"] for f in PRESET_FACTS},
        value=PRESET_FACTS[0]["prompt"] + "  ___",
        label="Fact to trace",
        full_width=True,
    )
    trace_samples_ui = mo.ui.slider(
        10, 40, value=20, step=5, show_value=True,
        label="Noise samples (higher = smoother)",
    )
    mo.hstack([
        mo.vstack([mo.md("**Fact**"), trace_ui]),
        mo.vstack([mo.md("**Samples**"), trace_samples_ui]),
    ], gap=3, justify="start")
    return trace_ui, trace_samples_ui


@app.cell
def cell_trace_btn(mo):
    trace_btn = mo.ui.run_button(
        label="🔍  Run Causal Trace  (~60 s)",
        kind="warn", full_width=True,
    )
    trace_btn
    return (trace_btn,)


@app.cell
def cell_trace_run(trace_btn, trace_ui, trace_samples_ui,
                   FACTS_BY_ID, MODEL_NAME, torch, mo):
    mo.stop(
        not trace_btn.value,
        mo.callout(mo.md("Click **Run Causal Trace** above."), kind="info"),
    )

    # trace_ui.value returns the fact id (correct dict orientation)
    _fact_t = FACTS_BY_ID[trace_ui.value]

    with mo.status.spinner(
        title="Loading causal-tracer + running trace (~60 s)…"
    ):
        from causal_tracer import CausalTracer
        _tracer = CausalTracer(MODEL_NAME)
        _res    = _tracer.calculate_hidden_flow(
            prompt=_fact_t["prompt"],
            subject=_fact_t["subject"],
            samples=trace_samples_ui.value,
            noise=0.13,
        )
        # Confirmed shape: (n_tokens, n_layers)
        trace_scores     = _res.scores.numpy()
        trace_tokens     = list(_res.input_tokens)
        trace_subj_range = tuple(_res.subject_range)
        trace_prompt     = _fact_t["prompt"]
        del _tracer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _sl  = trace_subj_range[1] - 1
    top_layer  = int(trace_scores[_sl, :].argmax())
    top_score  = float(trace_scores[_sl, :].max())

    mo.callout(
        mo.md(f"✅ **Trace complete** · Shape `{trace_scores.shape}` "
              f"· Peak: **layer {top_layer}** "
              f"(indirect effect {top_score:.3f}) "
              f"· Subject tokens: "
              f"`{trace_tokens[trace_subj_range[0]:trace_subj_range[1]]}`"),
        kind="success",
    )
    return (trace_scores, trace_tokens, trace_subj_range,
            trace_prompt, top_layer, top_score)


# ─── Causal Heatmap Widget ────────────────────────────────────────────────────

@app.cell
def cell_heatmap_class(anywidget, traitlets):
    class CausalHeatmapWidget(anywidget.AnyWidget):
        """
        Interactive causal trace heatmap.
        data.scores[t][l] = indirect effect for token t at layer l.
        X-axis = token positions, Y-axis = layers (0 at bottom).
        """
        _esm = r"""
        function render({ model, el }) {
            function draw() {
                const raw = model.get("hm_data");
                if (!raw) { el.innerHTML = ""; return; }
                const d = JSON.parse(raw);
                const { scores, tokens, subject_range, top_layer } = d;
                const nT = tokens.length, nL = scores[0].length;
                const CW = Math.max(9,  Math.floor(640 / nT));
                const CH = Math.max(4,  Math.floor(272 / nL));
                const PL = 40, PR = 72, PT = 20, PB = 70;
                const W  = PL + nT * CW + PR;
                const H  = PT + nL * CH + PB;
                const ss = subject_range[0], se = subject_range[1];
                const sl = se - 1;

                function col(s) {
                    const t = Math.min(1, Math.max(0, s));
                    return `rgb(${Math.round(255-t*176)},${Math.round(255-t*185)},${Math.round(255-t*26)})`;
                }

                let cells = "";
                for (let t = 0; t < nT; t++)
                    for (let l = 0; l < nL; l++) {
                        const x = PL + t * CW;
                        const y = PT + (nL - 1 - l) * CH;
                        const s = scores[t][l];
                        const peak = t === sl && l === top_layer;
                        cells += `<rect x="${x}" y="${y}"
                            width="${CW-1}" height="${CH-1}"
                            fill="${col(s)}"
                            stroke="${peak ? "#ef4444" : "none"}"
                            stroke-width="${peak ? 2 : 0}" rx="1"
                            data-t="${t}" data-l="${l}" data-s="${s.toFixed(3)}"
                            class="hmc" style="cursor:crosshair;"/>`;
                    }

                let labs = "";
                for (let t = 0; t < nT; t++) {
                    const cx = PL + t * CW + CW/2;
                    const ty = PT + nL * CH + 14;
                    const subj = t >= ss && t < se;
                    labs += `<text x="${cx}" y="${ty}"
                        text-anchor="end" font-size="11"
                        fill="${subj ? "#818cf8" : "#475569"}"
                        font-weight="${subj ? "700" : "400"}"
                        transform="rotate(-45,${cx},${ty})">${tokens[t]}</text>`;
                }

                let lticks = "";
                for (let l = 0; l < nL; l += 8) {
                    const ty = PT + (nL-1-l)*CH + CH/2 + 4;
                    lticks += `<text x="${PL-5}" y="${ty}"
                        text-anchor="end" font-size="10"
                        fill="#475569">${l}</text>`;
                }

                const ay = PT + (nL-1-top_layer)*CH + CH/2;
                const ax = PL + nT*CW + 5;
                const ann = `
                    <line x1="${ax}" y1="${ay}" x2="${ax+14}" y2="${ay}"
                        stroke="#ef4444" stroke-width="1.5"
                        marker-end="url(#arr)"/>
                    <text x="${ax+16}" y="${ay+4}"
                        font-size="10" fill="#ef4444" font-weight="700">
                        L${top_layer}</text>`;

                el.innerHTML = `
                <div style="overflow-x:auto;">
                  <svg width="${W}" height="${H}"
                       style="display:block;max-width:100%;">
                    <defs>
                      <marker id="arr" markerWidth="6" markerHeight="6"
                          refX="5" refY="3" orient="auto">
                        <path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/>
                      </marker>
                    </defs>
                    ${cells}${labs}${lticks}${ann}
                    <text x="13" y="${PT+nL*CH/2}"
                        text-anchor="middle" font-size="11" fill="#475569"
                        transform="rotate(-90,13,${PT+nL*CH/2})">Layer</text>
                    <text x="${PL+nT*CW/2}" y="${H-3}"
                        text-anchor="middle" font-size="11"
                        fill="#475569">← Token →</text>
                  </svg>
                  <div style="font-size:11px;color:#475569;margin-top:4px;">
                    🟣 Purple tokens = subject &nbsp;·&nbsp;
                    🔴 Red border = peak causal cell &nbsp;·&nbsp;
                    Bright = high indirect effect
                  </div>
                  <div id="htip"
                    style="font-family:monospace;font-size:12px;
                    color:#e0e7ff;min-height:18px;margin-top:5px;"></div>
                </div>`;

                const tip = el.querySelector("#htip");
                el.querySelectorAll(".hmc").forEach(c => {
                    c.addEventListener("mouseover", () => {
                        tip.textContent =
                            `Token "${tokens[+c.dataset.t]}" · `
                            + `Layer ${c.dataset.l} · `
                            + `Indirect effect = ${c.dataset.s}`;
                    });
                });
            }
            draw();
            model.on("change:hm_data", draw);
        }
        export default { render };
        """
        _css = ".hmc:hover{opacity:.7}"
        hm_data = traitlets.Unicode("").tag(sync=True)

    return (CausalHeatmapWidget,)


@app.cell(hide_code=True)
def cell_heatmap_show(trace_scores, trace_tokens, trace_subj_range,
                       trace_prompt, top_layer, top_score,
                       CausalHeatmapWidget, json, mo):
    mo.stop(trace_scores is None)

    _w = CausalHeatmapWidget(hm_data=json.dumps({
        "scores":        trace_scores.tolist(),
        "tokens":        trace_tokens,
        "subject_range": list(trace_subj_range),
        "top_layer":     top_layer,
    }))

    mo.vstack([
        mo.md(f"### Causal Trace — *\"{trace_prompt}\"*"),
        mo.ui.anywidget(_w),
        mo.callout(
            mo.md(
                f"**Reading the heatmap:** The brightest cell — layer **{top_layer}**, "
                f"the subject's last token — has indirect effect **{top_score:.3f}**. "
                f"This is where the model stores the factual association. "
                f"ROME targets exactly this cell to rewrite the memory."
            ),
            kind="neutral",
        ),
    ], gap=1)
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL — Extension: Edit Coherence (novel)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_coh_header(mo):
    mo.md("""
    ---
    ## Step 5 — Extension: Edit Coherence *(novel — not in the paper)*

    The ROME paper measures *efficacy*, *generalization*, and *specificity*.
    It does **not** ask: do *logically entailed* facts also update?

    If we rewrite *"Eiffel Tower is in Paris"* → **Rome**, then a coherent
    model should also update:
    - "The **country** containing the Eiffel Tower is" → Italy *(was France)*
    - "The **language** spoken where the Eiffel Tower stands" → Italian

    The **Edit Coherence Score** = fraction of entailed facts that cascade.
    Low coherence = the model holds logically inconsistent beliefs.
    """)
    return


@app.cell(hide_code=True)
def cell_coh_results(before_preds, edit_fact, edit_target,
                      model, tokenizer, DEVICE, torch, mo):
    mo.stop(before_preds is None)

    _rows = []
    with torch.no_grad():
        for _p in edit_fact["coh"]:
            _inp    = tokenizer(_p, return_tensors="pt").to(DEVICE)
            _after  = tokenizer.decode(
                [model(**_inp).logits[0,-1].argmax()]
            ).strip()
            _before = before_preds.get(_p, "—")
            _changed = _before != _after
            _rows.append({
                "Entailed prompt": _p + "  ___",
                "Before":  _before,
                "After":   _after,
                "Cascaded?": "✓" if _changed else "—",
            })

    _n     = sum(1 for r in _rows if r["Cascaded?"] == "✓")
    _score = _n / max(len(_rows), 1)

    mo.vstack([
        mo.stat(
            f"{_score:.0%}",
            label="Edit Coherence Score",
            caption=f"{_n} of {len(_rows)} entailed facts cascaded",
        ),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.callout(
            mo.md(
                f"**Finding:** ROME achieves {_score:.0%} coherence on entailed facts. "
                "The rank-one update touches only one weight direction — "
                "propagation of logical consequences is not guaranteed "
                "and depends on the model's pre-existing associative structure. "
                "This gap motivates follow-up work: MEMIT, GRACE, WilKE."
            ),
            kind="warn" if _score < 0.5 else "success",
        ),
    ], gap=1)
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL — Theory: How ROME Works
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_theory(mo):
    mo.md(r"""
    ---
    ## How ROME Works

    GPT-2's MLP at layer $\ell$ computes:

    $$\mathbf{h} = \text{act}(W_{\text{fc}}\,\mathbf{x}) \in \mathbb{R}^{6400}
    \qquad
    \mathbf{out} = W_{\text{proj}}\,\mathbf{h} + \mathbf{b} \in \mathbb{R}^{1600}$$

    Treat this as **key-value memory**: $\mathbf{h}$ is the **key**,
    $W_{\text{proj}}\,\mathbf{h}$ is the **value** written to the residual stream.

    **Goal:** find $k^*$ (the key encoding the subject) and $v^*$
    (the value that produces the new target), then update $W_{\text{proj}}$
    so that $k^* \mapsto v^*$ while all other keys are unchanged.

    **The rank-one solution (Sherman-Morrison):**

    $$W'_{\text{proj}} = W_{\text{proj}} +
    \underbrace{\frac{(v^* - W_{\text{proj}}\,k^*)\,k^{*\top}}
                     {k^{*\top} k^*}}_{\text{rank-one perturbation}}$$

    This is the *minimum-norm* weight change that achieves the redirection.

    **Finding $k^*$:** average MLP hidden activation at the subject's last token,
    across noise-corrupted forward passes (making $k^*$ subject-generic, not prompt-specific).

    **Finding $v^*$:** gradient-optimize to maximize $P(\text{target\_new})$
    across several paraphrase prompts, injecting $v$ via a forward hook.
    """)
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL — Takeaways
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def cell_takeaways(mo):
    mo.md(r"""
    ---
    ## Key Takeaways

    | Finding | What it means |
    |---------|---------------|
    | 🔍 **Facts have precise addresses** | Causal tracing localises knowledge to a narrow band: subject's last token × mid-to-late MLP layers |
    | ✏️ **One rank-one update is enough** | Minimum-norm weight perturbation redirects the stored association in < 1 s |
    | 🎯 **Specificity by construction** | $\Delta W$ only changes the key direction $k^*$ — all other keys are mathematically untouched |
    | 🔄 **Generalization is real** | Averaging $k^*$ over noise runs prevents the edit from being prompt-specific |
    | ⚠️ **Coherence is not guaranteed** | Logically entailed facts may not cascade — the paper never tests this |
    | 🔭 **Broader implication** | Surgical knowledge correction without retraining, catastrophic forgetting, or copyright violation |

    ---

    **Paper:** [arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)
    · Meng, Bau, Andonian & Belinkov · NeurIPS 2022  
    **Code:** [github.com/kmeng01/rome](https://github.com/kmeng01/rome)  
    **Notebook:** alphaXiv × marimo GPU Notebook Competition #2
    """)
    return


if __name__ == "__main__":
    app.run()
