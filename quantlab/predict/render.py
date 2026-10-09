"""Output: a dashboard component per asset, plus plain text for the terminal.

Follows the house conventions from `dashboard/build_dashboard.py` — one CSS
block, a body fragment that can be published as a hosted page, and a
standalone HTML file for local viewing. The probability bars are inline SVG
rather than matplotlib images, so the page stays light, crisp at any zoom and
legible in both colour themes.

Nothing is rendered that the engine did not compute. Where a source was
unavailable the component says "unavailable" in the data-quality block instead
of leaving a confident-looking gap.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab.predict import assets as asset_spec
from quantlab.predict import config

UP_COLOR = "var(--up)"
DOWN_COLOR = "var(--down)"
RANGE_COLOR = "var(--rangec)"


def esc(text) -> str:
    return html.escape(str(text))


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

def text_block(pred) -> str:
    """The terminal rendering — the same content as the dashboard card."""
    p = pred.to_dict()
    a = asset_spec.get(p["asset"])
    lines: list[str] = []
    pretty = {"XAUUSD": "XAU/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}.get(
        p["asset"], p["asset"])

    lines.append("=" * 66)
    lines.append(f"{pretty}   {a.name}")
    price = asset_spec.fmt_price(p["asset"], p["price"])
    chg = p.get("change_pct")
    chg_txt = f"{chg:+.2f}%" if chg is not None and np.isfinite(chg) else "—"
    lines.append(f"{price}   {chg_txt}   [{p['price_status'].upper()}] {p['price_source']}")
    lines.append("")

    if not p["horizons"]:
        lines.append("LIVE DATA UNAVAILABLE — no probabilities produced this run.")
        for note in p["notes"]:
            lines.append(f"  {note}")
        return "\n".join(lines)

    for key, h in p["horizons"].items():
        lines.append(f"{h['label'].upper()}")
        lines.append(f"  UP       {h['up_pct']:>3d}%   {_bar(h['up_pct'])}")
        lines.append(f"  DOWN     {h['down_pct']:>3d}%   {_bar(h['down_pct'])}")
        lines.append(f"  RANGE    {h['range_pct']:>3d}%   {_bar(h['range_pct'])}")
        lines.append(f"  Confidence: {h['confidence']:.0f}% ({h['confidence_label']})"
                     + (f"   ·  event risk {h['event_level']}"
                        if h["event_level"] != "none" else ""))
        band = asset_spec.fmt_distance(p["asset"], h["band_price"], signed=False)
        lines.append(f"  RANGE means |move| <= {band} over this horizon")
        lines.append("")

    reg = p["regime"]
    lines.append(f"Trend: {p['trend']}    Volatility: {p['volatility']}    "
                 f"Regime: {reg['label']}")
    lines.append(f"Session: {reg['session']}  (next: {reg['next_session']} "
                 f"from {reg['next_session_start'][11:16]} UTC)")

    poly = p["components"].get("polymarket", {})
    if poly.get("available"):
        infl = p["horizons"].get("1h", {}).get("poly_influence_pts", 0.0)
        lines.append(f"Polymarket influence on P(UP), 1h: {infl:+.1f} pts")
    else:
        lines.append(f"Polymarket influence: none — {poly.get('note', 'unavailable')}")

    lines.append("")
    lines.append("Key drivers:")
    for d in p["key_drivers"]:
        lines.append(f"  - {d}")

    lines.append("")
    lines.append("Risk events:")
    if p["risk_events"]:
        for e in p["risk_events"][:4]:
            lines.append(f"  - {e['name']} {e['countdown']} ({e['impact']})")
    else:
        lines.append("  - none scheduled in the tracked window")
    for name, why in p["untracked_events"].items():
        lines.append(f"  - {name}: NOT TRACKED — {why}")

    lines.append("")
    lines.append(f"DATA QUALITY: {p['data_quality']['label']} "
                 f"(availability {p['data_quality']['availability']:.0%})")
    for s in p["data_quality"]["sources"]:
        detail = f" — {s['detail']}" if s["detail"] else ""
        lines.append(f"  {s['name']}: {s['status']}{detail}"[:110])

    lines.append("")
    lines.append("Why this changed:")
    for line in p["horizons"].get("1h", {}).get("explanation", []):
        lines.append(f"  {line}")

    if p["notes"]:
        lines.append("")
        for note in p["notes"]:
            lines.append(f"NOTE: {note}")
    return "\n".join(lines)


def _bar(pct: int, width: int = 28) -> str:
    filled = int(round(pct / 100 * width))
    return "#" * filled + "." * (width - filled)


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------

CSS = """
/* Tokens inherited from dashboard/build_dashboard.py so this page sits in the
   same family as the rest of the project, extended with a dark theme. The
   light palette is complete on bare :root — the un-stamped "system" state is
   what most viewers get, and a colour defined only inside a media query would
   simply not apply there. --chip-ink flips because the dark-theme semantic
   colours are light, and white text on a light green is unreadable. */
:root { --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --sec:#52514e; --muted:#898781;
        --line:#e1e0d9; --border:rgba(11,11,11,.10); --up:#2a78d6; --down:#e34948;
        --rangec:#8d8b84; --good:#006300; --warn:#b06a00; --bad:#d03b3b;
        --chip:#eeede8; --chip-ink:#ffffff; }
:root:not([data-theme="light"]) { color-scheme: light dark; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --page:#14140f; --card:#1c1c17; --ink:#f2f1ea;
    --sec:#c3c1b6; --muted:#8f8d84; --line:#2e2e26; --border:rgba(242,241,234,.14);
    --up:#6aa8ff; --down:#ff7a72; --rangec:#a3a196; --good:#5fbf5f; --warn:#e0a34a;
    --bad:#ff7a72; --chip:#26261f; --chip-ink:#14140f; }
}
:root[data-theme="dark"] { --page:#14140f; --card:#1c1c17; --ink:#f2f1ea; --sec:#c3c1b6;
  --muted:#8f8d84; --line:#2e2e26; --border:rgba(242,241,234,.14); --up:#6aa8ff;
  --down:#ff7a72; --rangec:#a3a196; --good:#5fbf5f; --warn:#e0a34a; --bad:#ff7a72;
  --chip:#26261f; --chip-ink:#14140f; }
* { box-sizing:border-box; }
body { margin:0; padding:22px 16px 44px; background:var(--page); color:var(--ink);
       font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1240px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 3px; }
h2 { font-size:17px; margin:0; }
h3 { font-size:11px; margin:16px 0 7px; color:var(--muted); text-transform:uppercase;
     letter-spacing:.06em; font-weight:600; }
.meta { color:var(--muted); font-size:12px; margin-bottom:12px; }
.disclaimer { background:var(--card); border:1px solid var(--border);
  border-left:3px solid var(--warn); padding:9px 13px; border-radius:8px;
  color:var(--sec); font-size:12px; margin-bottom:18px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(370px,1fr)); gap:16px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:14px;
        padding:16px 18px; border-top:3px solid var(--rangec); }
/* Severity stripe: data quality is the first thing that should register, before
   any number on the card is worth reading. */
.card.q-good { border-top-color:var(--good); }
.card.q-limited { border-top-color:var(--warn); }
.card.q-poor { border-top-color:var(--bad); }
.head { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;
        border-bottom:1px solid var(--line); padding-bottom:10px; }
.price { font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }
.chg.up { color:var(--good); } .chg.down { color:var(--bad); }
.chip { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px;
        background:var(--chip); color:var(--sec); font-weight:600; letter-spacing:.02em; }
.chip.good { background:var(--good); color:var(--chip-ink); }
.chip.warn { background:var(--warn); color:var(--chip-ink); }
.chip.bad  { background:var(--bad);  color:var(--chip-ink); }
.hz { margin:13px 0 0; }
.hz .hzhead { display:flex; justify-content:space-between; align-items:baseline;
              font-size:11px; text-transform:uppercase; letter-spacing:.05em;
              color:var(--muted); font-weight:600; }
.rowline { display:flex; align-items:center; gap:9px; margin:3px 0;
           font-variant-numeric:tabular-nums; }
.rowline .lab { width:52px; font-size:12px; color:var(--sec); }
.rowline .val { width:38px; text-align:right; font-weight:700; font-size:13px; }
.track { flex:1; height:9px; background:var(--line); border-radius:5px; overflow:hidden; }
.fill { height:100%; border-radius:5px; }
table { width:100%; border-collapse:collapse; font-size:12.5px; }
th { text-align:left; color:var(--muted); font-size:10.5px; text-transform:uppercase;
     letter-spacing:.04em; font-weight:600; padding:4px 6px;
     border-bottom:1px solid var(--line); }
td { padding:4px 6px; border-bottom:1px solid var(--line); vertical-align:top; }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
ul.tight { margin:0; padding-left:17px; }
ul.tight li { margin:3px 0; font-size:12.5px; color:var(--sec); }
.small { font-size:11.5px; color:var(--muted); }
.expl { background:var(--chip); border-radius:9px; padding:9px 12px; margin-top:9px;
        font-size:12.5px; color:var(--sec); }
.expl b { color:var(--ink); }
.srcs { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.src { font-size:11px; padding:2px 8px; border-radius:6px; background:var(--chip);
       color:var(--sec); }
.src.live { color:var(--good); } .src.unavailable { color:var(--bad); }
.src.stale, .src.delayed, .src.closed { color:var(--warn); }
footer { margin-top:22px; color:var(--muted); font-size:12px; }
code { background:var(--chip); border-radius:4px; padding:1px 5px; font-size:11.5px; }
.scroll { overflow-x:auto; }
"""


def _prob_row(label: str, pct: int, color: str) -> str:
    return (f'<div class="rowline"><span class="lab">{label}</span>'
            f'<span class="val">{pct}%</span>'
            f'<span class="track"><span class="fill" style="width:{pct}%;'
            f'background:{color}"></span></span></div>')


def _quality_chip(label: str) -> str:
    cls = {"GOOD": "good", "LIMITED": "warn", "POOR": "bad"}.get(label, "")
    return f'<span class="chip {cls}">DATA {esc(label)}</span>'


def _conf_chip(label: str) -> str:
    cls = {"HIGH": "good", "MEDIUM": "warn", "LOW": "bad"}[label]
    return f'<span class="chip {cls}">{esc(label)}</span>'


def asset_card(pred_dict: dict) -> str:
    """One asset's dashboard component."""
    p = pred_dict
    code = p["asset"]
    pretty = {"XAUUSD": "XAU/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}.get(code, code)
    chg = p.get("change_pct")
    chg_cls = "up" if (chg or 0) >= 0 else "down"
    chg_txt = f"{chg:+.2f}%" if chg is not None and np.isfinite(chg) else "—"

    qcls = "q-" + p["data_quality"]["label"].lower()
    if not p["horizons"]:
        return (f'<section class="card {qcls}"><div class="head"><h2>{esc(pretty)}</h2>'
                f'{_quality_chip(p["data_quality"]["label"])}</div>'
                f'<p class="chip bad" style="margin-top:12px">LIVE DATA UNAVAILABLE</p>'
                f'<ul class="tight">' +
                "".join(f"<li>{esc(n)}</li>" for n in p["notes"]) + "</ul></section>")

    horizons_html = ""
    for key, h in p["horizons"].items():
        band = asset_spec.fmt_distance(code, h["band_price"], signed=False)
        event = (f' · <span style="color:var(--warn)">event risk {esc(h["event_level"])}</span>'
                 if h["event_level"] != "none" else "")
        horizons_html += f"""
      <div class="hz">
        <div class="hzhead"><span>{esc(h['label'])}</span>
          <span>conf {h['confidence']:.0f}% {_conf_chip(h['confidence_label'])}</span></div>
        {_prob_row("UP", h["up_pct"], UP_COLOR)}
        {_prob_row("DOWN", h["down_pct"], DOWN_COLOR)}
        {_prob_row("RANGE", h["range_pct"], RANGE_COLOR)}
        <div class="small">RANGE = move within &plusmn;{esc(band)}{event}</div>
      </div>"""

    reg = p["regime"]
    poly = p["components"].get("polymarket", {})
    poly_line = (f"{p['horizons'].get('1h', {}).get('poly_influence_pts', 0):+.1f} pts on P(UP), 1h"
                 if poly.get("available")
                 else f"none — {poly.get('note', 'unavailable')}")

    drivers = "".join(f"<li>{esc(d)}</li>" for d in p["key_drivers"]) or \
        "<li>no component is contributing materially right now</li>"

    events = "".join(
        f"<li><b>{esc(e['name'])}</b> {esc(e['countdown'])} · {esc(e['impact'])}</li>"
        for e in p["risk_events"][:4]) or "<li>none scheduled in the tracked window</li>"
    events += "".join(
        f'<li class="small">{esc(n)}: <b>not tracked</b> — {esc(w)}</li>'
        for n, w in p["untracked_events"].items())

    srcs = "".join(
        f'<span class="src {esc(s["status"])}" title="{esc(s["detail"])}">'
        f'{esc(s["name"])}: {esc(s["status"])}</span>'
        for s in p["data_quality"]["sources"])

    expl = p["horizons"].get("1h", {}).get("explanation", [])
    expl_html = ""
    if expl:
        expl_html = ('<div class="expl"><b>' + esc(expl[0]) + "</b>"
                     + "".join(f"<br>{esc(line)}" for line in expl[1:]) + "</div>")

    ind = p["indicators"]
    ind_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>{esc(v)}</td></tr>"
        for k, v in [
            ("RSI(14) d / 1h / 15m",
             " / ".join(_fmt(ind.get(x), 0) for x in ("rsi14_d", "rsi14_h1", "rsi14_m15"))),
            ("ATR(14)", asset_spec.fmt_distance(code, ind.get("atr14") or float("nan"), signed=False)),
            ("EWMA vol / ATR baseline",
             f"{_fmt(ind.get('sigma_ewma_daily_pct'), 2)}% / "
             f"{_fmt(ind.get('sigma_atr_daily_pct'), 2)}% per day "
             f"(ratio {_fmt(ind.get('vol_ratio'), 2)})"),
            ("ADX(14)", _fmt(ind.get("adx14"), 0)),
            ("SMA 20 / 50 / 200",
             " / ".join(asset_spec.fmt_price(code, ind[x]) if ind.get(x) else "—"
                        for x in ("sma20", "sma50", "sma200"))),
            ("Session high / low",
             f"{asset_spec.fmt_price(code, ind['day_high']) if ind.get('day_high') else '—'}"
             f" / {asset_spec.fmt_price(code, ind['day_low']) if ind.get('day_low') else '—'}"),
            ("Nearest support / resistance",
             _level_pair(code, ind.get("nearest_support"), ind.get("nearest_resistance"))),
            ("Momentum 1d / 5d / 20d",
             " / ".join(f"{ind[x] * 100:+.2f}%" if ind.get(x) is not None else "—"
                        for x in ("mom_1d", "mom_5d", "mom_20d"))),
            ("Volume", ind.get("volume_note") or "—"),
            ("Student-t tail index", _fmt(ind.get("nu"), 1)),
        ])

    comp_rows = "".join(_component_row(name, comp, p) for name, comp in p["components"].items())

    return f"""
<section class="card {qcls}">
  <div class="head">
    <h2>{esc(pretty)}</h2>
    <span class="price">{esc(asset_spec.fmt_price(code, p['price']))}</span>
    <span class="chg {chg_cls}">{esc(chg_txt)}</span>
    {_quality_chip(p["data_quality"]["label"])}
    <span class="chip">{esc(p['price_status'].upper())}</span>
  </div>
  {horizons_html}
  <h3>Read</h3>
  <table>
    <tr><td>Trend</td><td class="num">{esc(p['trend'])}</td></tr>
    <tr><td>Volatility</td><td class="num">{esc(p['volatility'])}</td></tr>
    <tr><td>Market regime</td><td class="num">{esc(reg['label'])}</td></tr>
    <tr><td>Session</td><td class="num">{esc(reg['session'])}</td></tr>
    <tr><td>Polymarket influence</td><td class="num">{esc(poly_line)}</td></tr>
  </table>
  <h3>Key drivers</h3>
  <ul class="tight">{drivers}</ul>
  <h3>Risk events</h3>
  <ul class="tight">{events}</ul>
  <h3>Why it moved</h3>
  {expl_html or '<p class="small">First reading — nothing to compare against yet.</p>'}
  <h3>Components</h3>
  <div class="scroll"><table>
    <tr><th>channel</th><th>score</th><th>weight 1h</th><th>top evidence</th></tr>
    {comp_rows}
  </table></div>
  <h3>Indicators</h3>
  <div class="scroll"><table>{ind_rows}</table></div>
  <h3>Data sources</h3>
  <div class="srcs">{srcs}</div>
  <p class="small">{esc(' · '.join(p['notes']))}</p>
</section>"""


def _component_row(name: str, comp: dict, payload: dict) -> str:
    """One row of the components table: score, its 1h weight, its top evidence."""
    score = f"{comp['score']:+.2f}" if comp.get("available") else "—"
    weight = (payload["horizons"].get("1h", {}).get("weights", {}).get(name, 0.0)) * 100
    evidence = comp["evidence"][0] if comp.get("evidence") else (comp.get("note") or "")
    return (f"<tr><td>{esc(name.replace('_', ' '))}</td>"
            f"<td class='num'>{score}</td>"
            f"<td class='num'>{weight:.0f}%</td>"
            f"<td class='small'>{esc(evidence[:120])}</td></tr>")


def _fmt(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:,.{digits}f}"


def _level_pair(code: str, support, resistance) -> str:
    def one(level) -> str:
        if not level:
            return "—"
        name, price = level
        # "round 4,350.00" already carries its price. Test for the formatted
        # price itself rather than for any digit — "R1" and "20d high" contain
        # digits but are not prices.
        text = asset_spec.fmt_price(code, price)
        return name if text in name else f"{name} {text}"
    return f"{one(support)} / {one(resistance)}"


def render_body(payload: dict, static: bool = False) -> str:
    """The whole page body: header, one card per asset, footer.

    `static=True` is for a published copy, which does *not* update itself. The
    live header promises automatic refresh; leaving that promise on a frozen
    snapshot would be the page lying about its own data.
    """
    generated = pd.Timestamp(payload["generated"])
    cards = "".join(asset_card(p) for p in payload["assets"].values())
    triggers = ", ".join(payload.get("triggers", [])) or "price, candle close, Polymarket, news, macro"
    bt = payload.get("backtest_note", "")

    if static:
        freshness = (
            f'Static snapshot taken <b>{generated:%H:%M:%S} UTC</b>, '
            f'{generated:%A %d %B %Y}. <b>This page does not update.</b> '
            f'Markets have moved since it was written — run '
            f'<code>scripts/live_predict.py --serve</code> for the live version, '
            f'which recomputes on price moves &gt; {config.TRIGGER_PRICE_ATR:g}&times;ATR, '
            f'new 15m candles, Polymarket shifts and scheduled releases.')
    else:
        freshness = (
            f'Last updated: <b id="stamp">{generated:%H:%M:%S}</b> UTC '
            f'({generated:%A %d %B %Y}) · Next update: <b>automatic</b> — recomputed on '
            f'price moves &gt; {config.TRIGGER_PRICE_ATR:g}&times;ATR, new 15m candle, '
            f'Polymarket shifts &gt; {config.TRIGGER_POLY_PTS:g} pts, volatility shifts &gt; '
            f'{config.TRIGGER_VOL_REL:.0%}, new releases, and at least every '
            f'{config.MAX_RECOMPUTE_GAP // 60} minutes')

    return f"""<div class="wrap">
<header>
  <h1>Market probability engine — XAU/USD · EUR/USD · GBP/USD</h1>
  <div class="meta">
    {freshness} ·
    model <code>{esc(payload.get('model_version', config.MODEL_VERSION))}</code>
  </div>
  <div class="disclaimer"><b>Research tool, not financial advice, and not a
  guarantee.</b> These are calibrated probabilities from a weighted blend of
  live inputs — they are wrong a great deal of the time by construction, which
  is what a 60% forecast means. Direction enters the model only as a small
  drift capped at {config.KAPPA_MAX:g} standard deviations, so the engine will
  never tell you something is nearly certain. Watched triggers: {esc(triggers)}.</div>
</header>
<div class="grid">{cards}</div>
<footer>
  <p><b>How to read RANGE.</b> A move counts as UP or DOWN only if it exceeds
  &plusmn;{config.BAND_K:g}&times; the ATR-implied volatility for that horizon
  (floored at one round-trip spread); anything smaller is RANGE. The band is
  printed under each horizon.</p>
  <p>{esc(bt)}</p>
  <p class="small">Generated by <code>scripts/live_predict.py</code> ·
  engine <code>quantlab.predict</code> · prediction history in
  <code>data/predict/predictions.sqlite</code>.</p>
</footer>
</div>"""


def render_full(payload: dict, sse: bool = False) -> str:
    body = render_body(payload)
    live = ""
    if sse:
        # The page reloads itself when the engine pushes a new prediction. No
        # keys or credentials ever reach the browser: the server sends finished
        # JSON, and this listener only reacts to the timestamp changing.
        live = """
<script>
(function () {
  var es = new EventSource("/events");
  es.onmessage = function (e) {
    try { if (JSON.parse(e.data).generated) location.reload(); } catch (err) {}
  };
  es.onerror = function () { setTimeout(function () { location.reload(); }, 15000); };
}());
</script>"""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market probability engine — {pd.Timestamp(payload['generated']):%Y-%m-%d %H:%M} UTC</title>
<style>{CSS}</style></head>
<body>
{body}
{live}
</body></html>"""


def render_artifact(payload: dict) -> str:
    """Body-only fragment for publishing, matching the other dashboards.

    Marked as a static snapshot, because a hosted copy cannot refresh itself.
    """
    return (f"<title>Market probability engine — XAUUSD / EURUSD / GBPUSD</title>\n"
            f"<style>{CSS}</style>\n{render_body(payload, static=True)}")


def build_payload(predictions: dict, triggers: list[str] | None = None,
                  backtest_note: str = "") -> dict:
    return {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_version": config.MODEL_VERSION,
        "triggers": triggers or [],
        "backtest_note": backtest_note,
        "assets": {code: p.to_dict() if hasattr(p, "to_dict") else p
                   for code, p in predictions.items()},
    }


def payload_json(payload: dict) -> str:
    return json.dumps(payload, default=str)
