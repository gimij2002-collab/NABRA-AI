"""
NABRA AI - Streamlit dashboard (Step 13, creative redesign)

Run:   streamlit run app.py
Needs: callsense_pipeline.py next to this file, the fine-tuned DistilBERT model under
       $CALLSENSE_PROJECT_DIR/models/speaker_role/distilbert_final, and HF_TOKEN.
"""
import html as _h
import os
import sys
import time
from math import pi
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import callsense_pipeline as cs  # noqa: E402

UPLOAD_DIR = cs.PROJECT_DIR / "data/uploads"
MINT, PINK, AMBER, INK = "#7FDDBE", "#F4A6CF", "#F4C95D", "#1E2A33"
STAGES = {"audio": "Audio prep", "asr": "Whisper", "diarization": "Pyannote",
          "alignment": "Alignment", "roles": "DistilBERT", "domain": "Domain", "llm": "Qwen"}
STATUS = {"resolved": ("Resolved", MINT, "✓"), "unresolved": ("Unresolved", "#F08AB0", "✕"),
          "unclear": ("Needs review", AMBER, "?")}

st.set_page_config(page_title="NABRA AI", page_icon="🎧", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&family=Reem+Kufi:wght@500&display=swap');
.stApp,.stMarkdown,button,input,label,p,h1,h2,h3{font-family:'Manrope','Segoe UI',system-ui,sans-serif}
.stApp{background:radial-gradient(900px 420px at 88% -8%,#F7D6E8 0%,transparent 62%),linear-gradient(180deg,#DDE3E8 0%,#E6EEEF 42%,#BDE6DA 100%)}
.block-container{padding-top:1.4rem;max-width:1180px}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(185deg,#12303A 0%,#0F4A52 100%)}
[data-testid="stSidebar"] *{color:#DDF4EC}
[data-testid="stAlert"]{border-radius:20px}
.stButton>button,.stDownloadButton>button{border-radius:999px;border:0;background:#1E2A33;color:#fff;padding:.6rem 1.8rem;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{background:#0F6E78;color:#fff}
[data-testid="stFileUploaderDropzone"]{border-radius:28px;background:rgba(255,255,255,.62);border:2px dashed #6FC7AE;padding:2rem}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab-list"]{gap:10px}
.stTabs [data-baseweb="tab"]{height:auto;border-radius:999px;background:rgba(255,255,255,.6);padding:9px 22px}
.stTabs [aria-selected="true"]{background:#1E2A33}.stTabs [aria-selected="true"] p{color:#fff}
.hero{display:grid;grid-template-columns:auto 1fr auto;gap:34px;align-items:center;margin:4px 0 26px}
.wm .ar{font-family:'Reem Kufi',serif;font-size:1.5rem;color:#0F6E78;line-height:1}
.wm h1{margin:2px 0 0;font-size:3.1rem;font-weight:700;letter-spacing:-.03em;color:#1E2A33;line-height:1}
.wm h1 b{font-size:.9rem;background:#1E2A33;color:#7FDDBE;border-radius:999px;padding:3px 11px;margin-left:10px;vertical-align:middle;letter-spacing:0}
.wm p{margin:10px 0 0;color:#566570;max-width:330px}
.wave{width:100%;height:96px}.wave rect{transform-box:fill-box;transform-origin:center;animation:rise 1s cubic-bezier(.2,.8,.2,1) backwards}
@keyframes rise{from{transform:scaleY(.08)}to{transform:scaleY(1)}}
@media(prefers-reduced-motion:reduce){.wave rect{animation:none}}
.nums{display:flex;gap:28px}.nums .n{font-size:2.5rem;font-weight:500;color:#1E2A33;line-height:1}.nums .l{color:#566570;font-size:.82rem;margin-bottom:6px}
@media(max-width:900px){.hero{grid-template-columns:1fr}}
.sb .ar{font-family:'Reem Kufi',serif;font-size:2rem;color:#7FDDBE}.sb b{font-size:1.3rem}.sb p{opacity:.75;margin:4px 0 18px}
.sb li{list-style:none;margin:0 0 14px;padding-left:22px;position:relative;font-size:.9rem}.sb ol{padding:0}
.sb li:before{content:"";position:absolute;left:0;top:5px;width:10px;height:10px;border-radius:50%;background:var(--c)}
.sb li small{display:block;opacity:.65}
</style>""", unsafe_allow_html=True)

IFRAME_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&display=swap');
*{{box-sizing:border-box}}body{{margin:0;font-family:Manrope,"Segoe UI",system-ui,sans-serif;color:{INK};background:transparent}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:18px;padding:4px}}
.card{{background:rgba(255,255,255,.58);border-radius:28px;padding:22px 24px;backdrop-filter:blur(8px);
 box-shadow:0 10px 30px rgba(40,70,70,.08)}}
.card h3{{margin:0 0 2px;font-size:1.02rem;font-weight:500}}.sub{{color:#6A7780;font-size:.8rem;margin-bottom:14px}}
.s12{{grid-column:span 12}}.s7{{grid-column:span 7}}.s5{{grid-column:span 5}}.s4{{grid-column:span 4}}.s8{{grid-column:span 8}}
@media(max-width:820px){{.s7,.s5,.s4,.s8{{grid-column:span 12}}}}
.verdict{{display:flex;gap:18px;align-items:center}}
.badge{{width:84px;height:84px;border-radius:50%;display:grid;place-items:center;font-size:2.4rem;color:#fff;flex:none}}
.vt{{font-size:1.9rem;font-weight:500}}.q{{margin-top:12px;padding:10px 14px;background:rgba(255,255,255,.7);border-radius:16px;font-size:.9rem;color:#33424C}}
.chip{{display:inline-block;padding:5px 12px;border-radius:999px;font-size:.78rem;margin:4px 6px 0 0;background:rgba(255,255,255,.75)}}
.chat{{height:360px;overflow-y:auto;padding-right:6px}}
.b{{max-width:86%;padding:10px 14px;border-radius:20px;margin:0 0 10px;font-size:.88rem;line-height:1.4}}
.b small{{display:block;color:#5B6770;font-size:.7rem;margin-bottom:2px}}
.a{{background:#CDF3E7;border-bottom-left-radius:6px}}.c{{background:#FBD3E8;margin-left:auto;border-bottom-right-radius:6px}}
.u{{background:#E6E9EC}}
.row{{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:.82rem}}.row .t{{width:92px}}
.bar{{flex:1;height:12px;border-radius:99px;background:rgba(255,255,255,.8);overflow:hidden}}.bar i{{display:block;height:100%;border-radius:99px;background:{MINT}}}
.kpis{{display:flex;gap:26px;flex-wrap:wrap;margin-top:6px}}.kpis b{{display:block;font-size:1.7rem;font-weight:500}}.kpis span{{font-size:.78rem;color:#6A7780}}
ul{{margin:6px 0 0;padding-left:18px;line-height:1.55;font-size:.9rem}}p{{margin:4px 0 12px;line-height:1.5;font-size:.92rem}}
.leg{{font-size:.78rem;color:#5B6770;display:flex;gap:16px;margin-top:6px}}.leg i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.mrow{{margin:13px 0}}.mrow .top{{display:flex;justify-content:space-between;font-size:.85rem}}.mrow .bar{{height:12px;margin-top:6px}}.mrow small{{color:#6A7780;font-size:.74rem}}
.warn{{background:#FCE9B0;border-radius:18px;padding:12px 16px;font-size:.88rem;margin-top:14px;line-height:1.5}}
.tbl{{width:100%;border-collapse:collapse;font-size:.84rem}}.tbl th,.tbl td{{padding:9px 8px;text-align:left;border-bottom:1px solid rgba(0,0,0,.07)}}
.pill{{padding:3px 11px;border-radius:999px;font-size:.76rem;font-weight:700;display:inline-block;color:#1E2A33}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:820px){{.two{{grid-template-columns:1fr}}}}
h2{{margin:0 0 6px;font-size:1.25rem;font-weight:700}}.big{{font-size:2rem;font-weight:500}}
"""


def esc(x) -> str:
    return _h.escape(str(x if x is not None else ""))


def fmt_time(s) -> str:
    try:
        s = max(0, int(float(s or 0)))
        return f"{s // 60}:{s % 60:02d}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def turns_of(res: dict) -> pd.DataFrame:
    t = res.get("turns")
    return t if isinstance(t, pd.DataFrame) else pd.DataFrame(t or [], columns=["start", "end", "role", "text"])


def json_bytes(res: dict) -> bytes:
    import json
    d = cs.result_to_json(res) if isinstance(res.get("turns"), pd.DataFrame) else res
    return json.dumps(d, indent=2, ensure_ascii=False, default=str).encode("utf-8")


# ---------------------------------------------------------------- SVG graphics
def donut(pct: float, size: int = 170) -> str:
    r, c = 62, 2 * pi * 62
    a = c * pct / 100
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 160 160"><circle cx="80" cy="80" r="{r}" fill="none" stroke="{PINK}" stroke-width="20"/>'
            f'<circle cx="80" cy="80" r="{r}" fill="none" stroke="{MINT}" stroke-width="20" stroke-linecap="round" '
            f'stroke-dasharray="{a:.1f} {c:.1f}" transform="rotate(-90 80 80)"/>'
            f'<text x="80" y="82" text-anchor="middle" font-size="30" fill="{INK}">{pct:.0f}%</text>'
            f'<text x="80" y="102" text-anchor="middle" font-size="11" fill="#6A7780">agent talk</text></svg>')


def balance(turns: pd.DataFrame, n: int = 12):
    end, out = max(float(turns["end"].max()), 1.0), []
    w = end / n
    for i in range(n):
        lo, hi, a, b = i * w, (i + 1) * w, 0.0, 0.0
        for t in turns.itertuples():
            ov = max(0.0, min(hi, t.end) - max(lo, t.start))
            a, b = (a + ov, b) if t.role == "agent" else ((a, b + ov) if t.role == "customer" else (a, b))
        out.append((min(a / w * 100, 100), min(b / w * 100, 100)))
    return out, end


def voiceprint(turns: pd.DataFrame, n: int = 64) -> str:
    """The whole call as a row of bars: colour = who spoke, height = how fast they spoke."""
    end = max(float(turns["end"].max()), 1.0)
    w, bars = end / n, ""
    for i in range(n):
        lo, hi, words, best = i * w, (i + 1) * w, 0.0, None
        for t in turns.itertuples():
            ov = max(0.0, min(hi, t.end) - max(lo, t.start))
            if ov > 0:
                words += len(str(t.text).split()) * ov / max(t.end - t.start, 0.1)
                if best is None or ov > best[0]:
                    best = (ov, t.role)
        h = 4 + min(words / w / 4.0, 1.0) * 44 if best else 3
        col = MINT if best and best[1] == "agent" else PINK if best and best[1] == "customer" else "#C9D3D9"
        bars += f'<rect x="{i * 9.5:.1f}" y="{26 - h / 2:.1f}" width="6" height="{h:.1f}" rx="3" fill="{col}"/>'
    return f'<svg viewBox="0 0 {n * 9.5:.0f} 52" width="100%">{bars}</svg>'


def wave_hero(n: int = 56) -> str:
    from math import sin
    bars = ""
    for i in range(n):
        h = 8 + 70 * abs(sin(i * 0.37) * sin(i * 0.11 + 1.2))
        t = i / (n - 1)
        col = f"rgb({int(127 + 117 * t)},{int(221 - 55 * t)},{int(190 + 17 * t)})"      # mint -> pink
        bars += f'<rect x="{i * 12}" y="{48 - h / 2:.1f}" width="7" height="{h:.1f}" rx="3.5" fill="{col}" style="animation-delay:{i * 22}ms"/>'
    return f'<svg class="wave" viewBox="0 0 {n * 12} 96" preserveAspectRatio="none">{bars}</svg>'


def line_chart(turns: pd.DataFrame) -> str:
    data, end = balance(turns)
    W, H, x0, y0 = 560, 190, 36, 14
    pts = lambda k: [(x0 + i * (W - x0 - 14) / (len(data) - 1), y0 + (H - y0 - 28) * (1 - v[k] / 100)) for i, v in enumerate(data)]
    grid = "".join(f'<line x1="{x0}" x2="{W - 10}" y1="{y0 + (H - y0 - 28) * (1 - g / 100):.0f}" y2="{y0 + (H - y0 - 28) * (1 - g / 100):.0f}" stroke="#fff" stroke-width="1.5"/>'
                   f'<text x="2" y="{y0 + (H - y0 - 28) * (1 - g / 100) + 4:.0f}" font-size="10" fill="#6A7780">{g}%</text>' for g in (0, 50, 100))
    lines = ""
    for k, col in ((0, "#37B894"), (1, "#E27BB0")):
        p = pts(k)
        lines += (f'<polyline points="{" ".join(f"{x:.0f},{y:.0f}" for x, y in p)}" fill="none" stroke="{col}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
                  f'<circle cx="{p[-1][0]:.0f}" cy="{p[-1][1]:.0f}" r="6" fill="{col}" stroke="#fff" stroke-width="2"/>')
    ticks = "".join(f'<text x="{x0 + f * (W - x0 - 14):.0f}" y="{H - 6}" font-size="10" fill="#6A7780" text-anchor="middle">{fmt_time(f * end)}</text>' for f in (0, .25, .5, .75, 1))
    return f'<svg viewBox="0 0 {W} {H}" width="100%">{grid}{lines}{ticks}</svg>'


def wrap(cards) -> str:
    return f'<!doctype html><html><head><meta charset="utf-8"><style>{IFRAME_CSS}</style></head><body><div class="grid">{"".join(cards)}</div></body></html>'


def tone(v: float) -> str:
    return MINT if v >= 0.8 else AMBER if v >= 0.5 else "#F08AB0"


def mrow(label: str, v, note: str = "") -> str:
    if v is None:
        return f'<div class="mrow"><div class="top"><span>{label}</span><b>not measured</b></div><small>{note}</small></div>'
    return (f'<div class="mrow"><div class="top"><span>{label}</span><b>{v * 100:.1f}%</b></div>'
            f'<div class="bar"><i style="width:{v * 100:.0f}%;background:{tone(v)}"></i></div><small>{note}</small></div>')


def domain_acc(R):
    d = R["domain"]
    return (d["llm_acc"], f"Qwen zero-shot, {d.get('n')} calls") if d.get("llm_acc") is not None else \
           (d["tfidf_test_acc"], "TF-IDF baseline on 7 test calls: weak")


def trust_card(R) -> str:
    a, ro, rs = R["asr"], R["roles"], R["resolution"]
    dacc, dnote = domain_acc(R)
    return ('<div class="card s12"><h3>How much to trust this result</h3><div class="sub">Accuracy measured on held-out calls during the project evaluation</div><div class="two"><div>'
            + mrow("Speech to text (word accuracy)", 1 - a["wer"], f"WER {a['wer'] * 100:.1f}% on {a['calls']} calls")
            + mrow("Agent / customer roles", ro["e2e_acc"], f"full pipeline, {ro['e2e_calls']} calls")
            + '</div><div>'
            + mrow("Resolution verdict", rs["accuracy"], f"against human review of {rs['n']} calls: still needs improvement")
            + mrow("Call domain", dacc, dnote) + '</div></div></div>')


def results_html(R) -> str:
    a, ro, rs, sm = R["asr"], R["roles"], R["resolution"], R["summary"]
    miss = round((1 - (rs["resolved_recall"] or 0)) * rs["human_resolved"])
    dacc, dnote = domain_acc(R)
    cards = [
        f'<div class="card s7"><h3>1 · Whisper: speech to text</h3><div class="sub">{a["model"]}, {a["calls"]} test calls</div>'
        + mrow("Word accuracy", 1 - a["wer"], f"WER {a['wer'] * 100:.1f}%") + mrow("Character accuracy", 1 - a["cer"], f"CER {a['cer'] * 100:.1f}%") + '</div>',
        '<div class="card s5"><h3>2 · DistilBERT: agent or customer?</h3><div class="sub">Fine-tuned on the call segments</div>'
        + mrow("TF-IDF baseline", ro["baseline_acc"], "test segments") + mrow("DistilBERT", ro["distilbert_acc"], f"macro-F1 {ro['distilbert_f1'] * 100:.1f}%")
        + mrow("Full pipeline", ro["e2e_acc"], f"speaker separation + roles, {ro['e2e_calls']} calls") + '</div>',
        f'<div class="card s5"><h3>3 · Qwen: summaries</h3><div class="sub">Rated 1-5 by a person, {sm["n"]} calls</div>'
        + mrow("Average rating", sm["mean_score"] / 5, f"{sm['mean_score']:.2f} out of 5") + '</div>',
        f'<div class="card s7"><h3>4 · Qwen: was the problem resolved?</h3><div class="sub">Against human verdicts: {rs["human_resolved"]} resolved, {rs["human_unresolved"]} unresolved</div>'
        + mrow("Accuracy (clean transcripts)", rs["accuracy"], f"{rs['n']} calls") + mrow("Accuracy (full pipeline)", rs["pipe_accuracy"], f"{rs['pipe_n']} calls")
        + mrow("Found the resolved calls (recall)", rs["resolved_recall"], "") + mrow("Right when it says resolved (precision)", rs["resolved_precision"], "")
        + f'<div class="warn"><b>Still needs improvement.</b> Qwen is over-cautious: it missed {miss} of {rs["human_resolved"]} calls that people judged resolved, '
          f'though it never called an unresolved call resolved. The review set has only {rs["human_unresolved"]} unresolved call, so spotting unresolved calls cannot be judged yet. '
          'Use the verdict as a first pass and review every "Needs review" call by hand.</div></div>',
        '<div class="card s12"><h3>5 · Call domain</h3><div class="sub">Banking, aviation, food...</div><div class="two"><div>'
        + mrow("Domain accuracy used in the dashboard", dacc, dnote) + '</div><div><p>The TF-IDF classifier saw only 34 training calls across about 12 domains, so it is weak. '
        'The dashboard asks Qwen to choose the domain from the known list instead; the notebook cell "Measure domain accuracy" measures it on the reviewed calls.</p></div></div></div>']
    return wrap(cards)


def chip(status) -> str:
    label, col, _ = STATUS.get(status, ("-", "#D5DCE1", ""))
    return f'<span class="pill" style="background:{col}">{label}</span>'


def examples_html(E) -> str:
    if not E:
        return wrap(['<div class="card s12"><h3>Individual examples are not included in this standalone viewer</h3><p>Historical evaluation statistics are in Model results. If you have the original project_backup/evaluation/final/examples.json, it can be copied into the local evaluation/final folder.</p></div>'])
    calls = {c["call_id"]: c for c in E["calls"]}
    cards = []
    for kind in ("resolved", "unresolved"):
        c = calls.get(E.get("featured", {}).get(kind))
        if not c:
            continue
        match = c["human_status"] == c["llm_status_ref"]
        bub = "".join(f'<div class="b {"a" if t["role"] == "agent" else "c"}"><small>{esc(t["role"].capitalize())}</small>{esc(t["text"])}</div>' for t in E.get("excerpts", {}).get(c["call_id"], []))
        cards.append(
            f'<div class="card s6" style="grid-column:span 6"><h3>Example: {"an unresolved" if kind == "unresolved" else "a resolved"} call</h3><div class="sub">{esc(cs.domain_name(c["domain"]))} · {esc(c["call_id"])}</div>'
            f'<div class="row"><span class="t">Human review</span>{chip(c["human_status"])}</div><div class="row"><span class="t">Model said</span>{chip(c["llm_status_ref"])}'
            f'<b>{"✓ agrees" if match else "✕ differs"}</b></div><p><b>Model summary.</b> {esc(c["summary"])}</p><p><b>Model reason.</b> {esc(c["reason"])}</p>'
            + (f'<p><b>Human note.</b> {esc(c["human_notes"])}</p>' if c.get("human_notes") else "") + f'<div class="sub">End of the call</div><div class="chat" style="height:230px">{bub}</div></div>')
    rows = "".join(f'<tr><td>{esc(cs.domain_name(c["domain"]))}<br><small>{esc(c["call_id"])}</small></td><td>{chip(c["human_status"])}</td><td>{chip(c["llm_status_ref"])}</td>'
                   f'<td>{chip(c["llm_status_pipe"]) if c.get("llm_status_pipe") else "-"}</td><td>{"✓" if c["human_status"] == c["llm_status_ref"] else "✕"}</td><td>{c.get("summary_score", "-")}</td></tr>' for c in E["calls"])
    agree = sum(c["human_status"] == c["llm_status_ref"] for c in E["calls"])
    cards.append(f'<div class="card s12"><h3>All reviewed calls: human vs model</h3><div class="sub">The model agreed with the human on {agree} of {len(E["calls"])} calls</div>'
                 f'<table class="tbl"><tr><th>Call</th><th>Human</th><th>Model (clean text)</th><th>Model (full pipeline)</th><th>Match</th><th>Summary rating</th></tr>{rows}</table></div>')
    return wrap(cards)


def about_html(R) -> str:
    a, ro, rs = R["asr"], R["roles"], R["resolution"]
    dacc, _ = domain_acc(R)
    steps = [("Listen", "Whisper small.en writes down every word."), ("Separate voices", "Pyannote finds who spoke when."),
             ("Name the roles", "A fine-tuned DistilBERT tells agent from customer."), ("Understand", "Qwen2.5-3B writes the summary, names the domain and judges whether the problem was solved.")]
    li = "".join(f'<div class="row"><b style="width:130px">{i}. {t}</b><span>{d}</span></div>' for i, (t, d) in enumerate(steps, 1))
    nums = "".join(f'<div><div class="big">{v}</div><span>{k}</span></div>' for k, v in [
        ("word accuracy", f"{(1 - a['wer']) * 100:.0f}%"), ("role accuracy (pipeline)", f"{ro['e2e_acc'] * 100:.0f}%"),
        ("resolution accuracy", f"{rs['accuracy'] * 100:.0f}%"), ("domain accuracy", f"{dacc * 100:.0f}%")])
    lim = "".join(f"<li>{x}</li>" for x in [
        "English calls only, two speakers, built on a small dataset (48 calls).",
        f"The resolution verdict is the weakest part ({rs['accuracy'] * 100:.0f}% accuracy): the model is over-cautious and was checked on only {rs['n']} calls, {rs['human_resolved']} of them resolved.",
        "Domain detection is unreliable until it is validated on more calls.",
        "A transcript segment that spans a speaker change is assigned to one speaker."])
    return wrap([
        '<div class="card s12"><h2>NABRA AI</h2><p><b>«نبرة» means tone of voice.</b> NABRA AI turns a recorded customer-service call into a speaker-labelled transcript, '
        'a short summary, the call domain, and a verdict on whether the customer\'s problem was solved.</p></div>',
        f'<div class="card s7"><h3>How it works</h3>{li}</div>', f'<div class="card s5"><h3>Results at a glance</h3><div class="kpis">{nums}</div></div>',
        f'<div class="card s12"><h3>Current limits</h3><ul>{lim}</ul></div>'])


# ---------------------------------------------------------------- dashboard HTML
def build_dashboard_html(res: dict) -> str:
    stats, audio, an = res.get("stats", {}), res.get("audio", {}), res.get("analysis")
    meta, turns = res.get("analysis_meta") or {}, turns_of(res)
    agent_pct = stats.get("agent_talk_pct") or 0
    R, dom = cs.load_results(), res.get("domain") or {}
    if isinstance(dom, str):
        dom = {"label": dom, "method": "saved"}
    elif not isinstance(dom, dict):
        dom = {}
    dchip = ""
    if dom.get("label"):
        acc = R["domain"].get("llm_acc") if dom.get("method") == "llm" else None
        conf = f' · confidence {dom["confidence"] * 100:.0f}%' if dom.get("confidence") else ""
        how = "Qwen" if dom.get("method") == "llm" else "Saved classification"
        dchip = (f'<div style="margin-top:12px"><span class="chip" style="background:#D8EEF9">Domain: <b>{esc(cs.domain_name(dom["label"]))}</b>{conf}</span>'
                 f'<span class="chip">{how}: {("accuracy " + format(acc * 100, ".0f") + "% when measured") if acc is not None else "accuracy not measured yet"}</span></div>')
    cards = []
    if len(turns):
        cards.append(f'<div class="card s12"><h3>Voiceprint of the call</h3><div class="sub">Every bar is a slice of the call: colour shows who was speaking, height shows how fast they spoke.</div>'
                     f'{voiceprint(turns)}<div class="leg"><span><i style="background:{MINT}"></i>Agent</span><span><i style="background:{PINK}"></i>Customer</span></div>'
                     f'<div class="kpis"><div><b>{fmt_time(audio.get("duration_sec"))}</b><span>call length</span></div>'
                     f'<div><b>{stats.get("words", 0)}</b><span>words</span></div><div><b>{stats.get("turns", 0)}</b><span>turns</span></div></div></div>')

    if an:
        label, col, icon = STATUS[an["resolution_status"]]
        ev = ""
        if an.get("evidence"):
            m = {"exact": "found word-for-word", "fuzzy": "found with small wording changes", "none": "NOT found - check manually"}.get(meta.get("evidence_match"), "")
            ev = f'<div class="q">“{esc(an["evidence"])}”<br><small>Evidence: {m}</small></div>'
        cards.append(f'<div class="card s7"><div class="verdict"><div class="badge" style="background:{col}">{icon}</div><div>'
                     f'<div class="sub" style="margin:0">Was the problem solved?</div><div class="vt">{label}</div></div></div>{dchip}'
                     f'<p style="margin-top:14px">{esc(an.get("resolution_reason") or "No reason given.")}</p>{ev}</div>')
    else:
        cards.append(f'<div class="card s7"><h3>AI analysis unavailable</h3><p>The transcript is ready, but the summary could not be produced.</p>{dchip}</div>')

    cards.append(f'<div class="card s5" style="text-align:center"><h3>Talk balance</h3><div class="sub">Who spoke more</div>{donut(agent_pct)}'
                 f'<div class="leg" style="justify-content:center"><span><i style="background:{MINT}"></i>Agent {agent_pct:.0f}%</span>'
                 f'<span><i style="background:{PINK}"></i>Customer {100 - agent_pct:.0f}%</span></div></div>')

    bubbles = "".join(f'<div class="b {"a" if t.role == "agent" else "c" if t.role == "customer" else "u"}"><small>{esc(str(t.role).capitalize())} · {fmt_time(t.start)}</small>{esc(t.text)}</div>'
                      for t in turns.itertuples())
    cards.append(f'<div class="card s5"><h3>Conversation</h3><div class="sub">{len(turns)} turns</div><div class="chat">{bubbles}</div></div>')

    chart = line_chart(turns) if len(turns) else ""
    cards.append(f'<div class="card s7"><h3>Interaction over time</h3><div class="sub">Share of each slice spent speaking</div>{chart}'
                 f'<div class="leg"><span><i style="background:#37B894"></i>Agent</span><span><i style="background:#E27BB0"></i>Customer</span></div></div>')

    if an:
        acts = "".join(f"<li>{esc(a)}</li>" for a in an.get("agent_actions", [])) or "<li>-</li>"
        cards.append(f'<div class="card s7"><h3>Summary</h3><p>{esc(an.get("summary") or "-")}</p><h3>Customer problem</h3><p>{esc(an.get("customer_problem") or "-")}</p>'
                     f'<h3>What the agent did</h3><ul>{acts}</ul></div>')

    tm = {STAGES[k]: v for k, v in res.get("timings", {}).items() if k in STAGES}
    tot = sum(tm.values()) or 1
    rows = "".join(f'<div class="row"><span class="t">{k}</span><div class="bar"><i style="width:{v / tot * 100:.0f}%"></i></div><span>{v / tot * 100:.0f}%</span></div>' for k, v in tm.items())
    margin = (res.get("roles") or {}).get("margin")
    chips = f'<span class="chip">Role confidence: {margin if margin is not None else "n/a"}</span><span class="chip">Processed in {res.get("timings", {}).get("total", 0):.0f}s</span>'
    chips += "".join(f'<span class="chip" style="background:#FCE9B0">⚠ {esc(w)}</span>' for w in res.get("warnings", [])[:3])
    cards.append(f'<div class="card s5"><h3>Where the time went</h3><div class="sub">Share of processing time</div>{rows}<div style="margin-top:10px">{chips}</div></div>')

    cards.append(trust_card(R))
    return wrap(cards)


def render_result(res: dict, key: str):
    if res.get("status") == "failed":
        err = res.get("error") or {}
        st.error(f"Analysis failed at the **{STAGES.get(err.get('stage'), err.get('stage'))}** stage: {err.get('message')}")
        return
    components.html(build_dashboard_html(res), height=1260, scrolling=True)
    turns = turns_of(res)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download analysis (JSON)", json_bytes(res), f"{res['call_id']}_analysis.json", "application/json", key=f"{key}_j")
    c2.download_button("Download transcript (TXT)", (res.get("dialogue") or "").encode("utf-8"), f"{res['call_id']}_dialogue.txt", "text/plain", key=f"{key}_t")
    if not turns.empty:
        c3.download_button("Download turns (CSV)", turns.to_csv(index=False).encode("utf-8"), f"{res['call_id']}_turns.csv", "text/csv", key=f"{key}_c")


# ---------------------------------------------------------------- models + page
@st.cache_resource(show_spinner="Loading models - the first time takes a few minutes...")
def get_models(hf_token: str, use_llm: bool):
    return cs.build_models(hf_token=hf_token or None, load_llm_model=use_llm)


hf_token = os.environ.get("HF_TOKEN", "")
use_llm = os.environ.get("NABRA_USE_LLM", "1") != "0"

# A previously trained checkpoint is required for genuine NEW-CALL analysis.
# Do not fake a prediction when the checkpoint, pyannote, or token is absent.
import importlib.util
missing = []
role_checkpoint = cs.PROJECT_DIR / "models/speaker_role/distilbert_final"
if not (role_checkpoint / "model.safetensors").exists():
    missing.append("fine-tuned DistilBERT checkpoint")
if not hf_token:
    missing.append("HF_TOKEN")
for package in ("torch", "whisper", "pyannote", "transformers"):
    try:
        present = importlib.util.find_spec(package) is not None
    except (ValueError, ModuleNotFoundError):
        present = False
    if not present:
        missing.append(package)
can_analyze = not missing
with st.sidebar:
    st.markdown("""<div class="sb"><div class="ar">نبرة</div><b>NABRA AI</b><p>Hear the tone. Know the outcome.</p><ol>
<li style="--c:#7FDDBE">Listen<small>Whisper writes down every word</small></li>
<li style="--c:#F4A6CF">Separate voices<small>Pyannote finds who spoke when</small></li>
<li style="--c:#F4C95D">Name the roles<small>DistilBERT tells agent from customer</small></li>
<li style="--c:#9AD0F5">Understand<small>Qwen summarises and judges the result</small></li></ol></div>""", unsafe_allow_html=True)
    if not hf_token:
        hf_token = st.text_input("Hugging Face token", type="password", help="Needed once for the Pyannote speaker model.")
    st.caption(f"Project folder: `{cs.PROJECT_DIR}`")
    if missing:
        st.warning("VIEWER MODE: Saved results and evaluation dashboards are available. "
                   "New audio analysis is disabled because this session is missing: " + ", ".join(missing))
    else:
        st.success("Full pipeline prerequisites detected. New call analysis can be attempted.")

# The user may upload an existing *_analysis.json directly without Drive access.
with st.sidebar.expander("Import previously saved call JSON", expanded=False):
    imported_files = st.file_uploader("Saved analysis JSON", type=["json"],
                                      accept_multiple_files=True, key="imported_results")

saved = cs.load_saved_results()
for imported in imported_files or []:
    try:
        import json
        result = json.loads(imported.getvalue().decode("utf-8"))
        if isinstance(result, dict) and all(k in result for k in ("call_id", "file", "status")):
            saved.insert(0, result)
        else:
            st.warning(f"Ignored {imported.name}: not a NABRA analysis result.")
    except Exception as exc:
        st.warning(f"Could not read {imported.name}: {exc}")
done = [r for r in saved if r.get("analysis")]
rate = round(100 * sum(r["analysis"]["resolution_status"] == "resolved" for r in done) / len(done)) if done else 0
st.markdown(f'<div class="hero"><div class="wm"><div class="ar">نبرة</div><h1>NABRA<b>AI</b></h1><p>Drop a recording. See who said what, and whether the problem was solved.</p></div>'
            f'{wave_hero()}<div class="nums"><div><div class="l">Calls analysed</div><div class="n">{len(saved)}</div></div>'
            f'<div><div class="l">Resolution rate</div><div class="n">{rate}%</div></div></div></div>', unsafe_allow_html=True)

tab_new, tab_history, tab_results, tab_examples, tab_about = st.tabs(["Analyze a call", "Call history", "Model results", "Examples", "About"])

with tab_new:
    uploaded = st.file_uploader("Audio file", type=[e.lstrip(".") for e in sorted(cs.SUPPORTED_EXT)],
                                help="WAV, MP3, M4A, FLAC or OGG.")
    if uploaded is not None:
        st.audio(uploaded)
        if not can_analyze:
            st.info("Recorded call analysis requires the original trained model files and libraries. "
                    "The evaluation dashboards and saved results remain available in VIEWER MODE.")
        if st.button("Start analysis", type="primary", disabled=not can_analyze):
            try:
                models = get_models(hf_token, use_llm)
            except Exception as exc:                                   # noqa: BLE001
                st.error(f"Could not load models: {type(exc).__name__}: {exc}")
                models = None
            if models is not None:
                folder = UPLOAD_DIR / str(int(time.time()))
                folder.mkdir(parents=True, exist_ok=True)
                audio_path = folder / Path(uploaded.name).name
                audio_path.write_bytes(uploaded.getbuffer())
                with st.status("Analyzing the call...", expanded=True) as status:
                    result = cs.process_call(audio_path, models, progress=lambda stage, msg: st.write(msg))
                    status.update(label="Analysis failed" if result["status"] == "failed" else "Analysis complete",
                                  state="error" if result["status"] == "failed" else "complete", expanded=False)
                st.session_state["last_result"] = result
    if "last_result" in st.session_state:
        render_result(st.session_state["last_result"], key="new")
    elif uploaded is None:
        st.info("Upload a call recording to begin.")

with tab_history:
    if not saved:
        st.info("No analysed calls yet. Results appear here after you analyse a call.")
    else:
        counts = {k: sum((r.get("analysis") or {}).get("resolution_status") == k for r in saved) for k in STATUS}
        total = max(sum(counts.values()), 1)
        seg = "".join(f'<div style="width:{counts[k] / total * 100:.1f}%;background:{STATUS[k][1]};height:100%"></div>' for k in STATUS)
        leg = "".join(f'<span class="chip" style="background:rgba(255,255,255,.7);border-radius:999px;padding:5px 12px;margin-right:8px;display:inline-block">'
                      f'<b style="color:{STATUS[k][1]}">●</b> {STATUS[k][0]} {counts[k] / total * 100:.0f}% ({counts[k]})</span>' for k in STATUS)
        st.markdown(f'<div style="background:rgba(255,255,255,.58);border-radius:28px;padding:22px 24px"><b>Outcome of all analysed calls</b>'
                    f'<div style="display:flex;height:22px;border-radius:99px;overflow:hidden;margin:14px 0;background:#fff">{seg}</div>{leg}</div>',
                    unsafe_allow_html=True)
        st.write("")
        choice = st.selectbox("Open a call", [r["file"] for r in saved])
        render_result(next(r for r in saved if r["file"] == choice), key="hist")

with tab_results:
    if not (cs.PROJECT_DIR / "evaluation/final/project_metrics.json").exists():
        st.caption("Historical Kaggle evaluation values are shown below (NOT scores from this Colab run).")
    components.html(results_html(cs.load_results()), height=1250, scrolling=True)

with tab_examples:
    components.html(examples_html(cs.load_examples()), height=1500, scrolling=True)

with tab_about:
    components.html(about_html(cs.load_results()), height=900, scrolling=True)