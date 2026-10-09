"""Quantup Daily — trading dashboard generator.

Builds a self-contained HTML dashboard for indices, gold, FX and WTI oil:
systematic bias, pivot levels with a mechanical entry/stop/target plan,
per-session (Asia / London / New York) statistics and tactics, macro context
(FRED), news, Polymarket event risk, a BTC risk gauge and the live state of
the trend strategy from notebook 08.

Run it each morning (or schedule it):

    python dashboard/build_dashboard.py

Output: dashboard/dashboard.html (+ dated copy in dashboard/archive/).
Every data source is fetched independently — if one fails, the dashboard
still builds and lists the problem at the bottom.
"""

from __future__ import annotations

import base64
import html
import io
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import instruments, plotting
from quantlab.analysis import levels, sessions, signals
from quantlab.backtest import equal_weight_benchmark, run_backtest
from quantlab.config import has_key
from quantlab.data import yahoo

plotting.apply_style()

CODES = ["US100", "US500", "US30", "XAUUSD", "EURUSD", "GBPUSD", "USOIL"]
DECIMALS = {"US100": 0, "US500": 1, "US30": 0, "XAUUSD": 1,
            "EURUSD": 5, "GBPUSD": 5, "USOIL": 2}
CORRELATION_CODES = ["XAUUSD", "DXY", "EURUSD", "GBPUSD", "US10Y", "USOIL", "US100", "US500", "US30"]
OUT_DIR = Path(__file__).resolve().parent
WARNINGS: list[str] = []


# ----------------------------------------------------------------- helpers --
def fmt(value, code: str = "US500", signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    sign = "+" if (signed and value >= 0) else ""
    return f"{sign}{value:,.{DECIMALS.get(code, 1)}f}"


def guard(label: str, fn, default=None):
    """Run one data fetch; on failure record a warning and keep going."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — dashboard must always build
        WARNINGS.append(f"{label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=1)
        return default


def fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def esc(text) -> str:
    return html.escape(str(text))


# ------------------------------------------------------------ computations --
def bias_from(close: pd.Series) -> dict:
    """Trend/momentum composite -> Bullish / Neutral / Bearish with components."""
    price = close.iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    rsi = signals.rsi(close).iloc[-1]
    mom = signals.momentum(close, 252, 21).iloc[-1]
    components = {
        "price > SMA200": price > sma200,
        "SMA50 > SMA200": sma50 > sma200,
        "price > SMA20": price > sma20,
        "12-1 momentum > 0": mom > 0,
        "RSI(14) > 50": rsi > 50,
    }
    score = sum(components.values())
    label = "BULLISH" if score >= 4 else ("BEARISH" if score <= 1 else "NEUTRAL")
    return {
        "label": label, "score": score, "components": components,
        "price": price, "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "rsi": rsi, "mom": mom,
    }


def entry_plan(bias: str, pv: dict, atr_value: float, price: float, code: str) -> dict:
    """Mechanical plan from pivots + ATR. Levels, not prophecy."""
    f = lambda v: fmt(v, code)  # noqa: E731
    if bias == "BULLISH":
        zone_lo, zone_hi = (pv["P"] + pv["S1"]) / 2, pv["P"]
        stop = pv["S1"] - 0.1 * atr_value
        t1, t2 = pv["R1"], pv["R2"]
        direction = "LONG on pullback"
        note = "Price below P — wait for a reclaim of P before longs." if price < pv["P"] else ""
    elif bias == "BEARISH":
        zone_lo, zone_hi = pv["P"], (pv["P"] + pv["R1"]) / 2
        stop = pv["R1"] + 0.1 * atr_value
        t1, t2 = pv["S1"], pv["S2"]
        direction = "SHORT on bounce"
        note = "Price above P — wait for a rejection of P before shorts." if price > pv["P"] else ""
    else:
        return {
            "direction": "RANGE tactics (no edge from trend)",
            "zone": f"long near S1 {f(pv['S1'])} / short near R1 {f(pv['R1'])}",
            "stop": f"0.5×ATR beyond the level (±{f(0.5 * atr_value)})",
            "targets": f"pivot P {f(pv['P'])}",
            "rr": "≈ 1.5–2.0 if levels hold",
            "note": "Half size — composite gives no directional edge today.",
        }
    zone_mid = (zone_lo + zone_hi) / 2
    risk = abs(zone_mid - stop)
    rr1 = abs(t1 - zone_mid) / risk if risk else float("nan")
    rr2 = abs(t2 - zone_mid) / risk if risk else float("nan")
    return {
        "direction": direction,
        "zone": f"{f(zone_lo)} – {f(zone_hi)}",
        "stop": f(stop),
        "targets": f"T1 {f(t1)} · T2 {f(t2)}",
        "rr": f"T1 {rr1:.1f}R · T2 {rr2:.1f}R",
        "note": note,
    }


def session_tactics(code: str, bias: str, pv: dict, today: pd.DataFrame, code_fmt) -> dict[str, str]:
    """One actionable line per session, using today's actual levels."""
    f = code_fmt
    asia = today.loc["Asia"] if "Asia" in today.index else None
    london = today.loc["London"] if "London" in today.index else None
    side = "long" if bias == "BULLISH" else ("short" if bias == "BEARISH" else "either")
    if asia is not None:
        rng = f"{f(asia['low'])}–{f(asia['high'])}"
        london_txt = (f"Asia range {rng}: trade the {side}-side breakout, enter on the retest "
                      f"of the broken edge; invalid back inside the range.")
    else:
        london_txt = "Trade the breakout of the completed Asia range in the bias direction; enter on the retest."
    lmid = f((london["high"] + london["low"]) / 2) if london is not None else f(pv["P"])
    return {
        "Asia": f"Rotation session: fade S1 {f(pv['S1'])} / R1 {f(pv['R1'])} toward P {f(pv['P'])}, "
                f"half size; go flat into the 07:00 UTC London open.",
        "London": london_txt,
        "New York": f"Let the 13:30 UTC US open print 30–60 min, then join the {side} side on a "
                    f"pullback to the London mid {lmid} or P {f(pv['P'])}; no entries into red-flag data.",
    }


def build_instrument(code: str) -> dict | None:
    inst = instruments.INSTRUMENTS[code]
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    ohlc = guard(f"{code} daily OHLC", lambda: yahoo.get_ohlcv(inst.yahoo_future, start=start))
    if ohlc is None or len(ohlc) < 260:
        return None
    hourly = guard(f"{code} hourly bars", lambda: instruments.get_intraday(code, "1h", days=60))

    close = ohlc["Close"]
    bias = bias_from(close)
    today_utc = pd.Timestamp.now(tz="UTC").date()
    prev = ohlc.iloc[-2] if ohlc.index[-1].date() >= today_utc else ohlc.iloc[-1]
    pv = levels.pivot_points(float(prev["High"]), float(prev["Low"]), float(prev["Close"]))
    atr14 = float(levels.atr(ohlc).iloc[-1])
    swings = levels.swing_levels(ohlc, 20)
    daily_rets = close.pct_change()
    vol30 = float(daily_rets.rolling(30).std().iloc[-1] * np.sqrt(252))
    prev_close = float(ohlc["Close"].iloc[-2])
    day_chg = float(close.iloc[-1] / prev_close - 1) * 100

    stats = guard(f"{code} session stats", lambda: sessions.session_stats(hourly)) if hourly is not None else None
    today_sess = guard(f"{code} today sessions", lambda: sessions.todays_sessions(hourly)) if hourly is not None else pd.DataFrame()
    active_hours = guard(f"{code} active hours", lambda: sessions.most_active_hours(hourly)) if hourly is not None else None

    # price chart: last year of closes with the two trend SMAs
    window = close.iloc[-260:]
    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    ax.plot(window, color=plotting.PALETTE[0], label="close")
    ax.plot(close.rolling(50).mean().iloc[-260:], color=plotting.PALETTE[1], linewidth=1.3, label="SMA 50")
    ax.plot(close.rolling(200).mean().iloc[-260:], color=plotting.PALETTE[2], linewidth=1.3, label="SMA 200")
    ax.set_title(f"{code} — daily close, one year")
    ax.legend(loc="upper left", ncols=3)

    return {
        "code": code,
        "close_history": close,
        "name": inst.name,
        "price": float(close.iloc[-1]),
        "day_chg": day_chg,
        "bias": bias,
        "pivots": pv,
        "atr": atr14,
        "swings": swings,
        "vol30": vol30,
        "plan": entry_plan(bias["label"], pv, atr14, float(close.iloc[-1]), code),
        "session_stats": stats,
        "today_sessions": today_sess if today_sess is not None else pd.DataFrame(),
        "tactics": session_tactics(code, bias["label"], pv,
                                   today_sess if today_sess is not None else pd.DataFrame(),
                                   lambda v: fmt(v, code)),
        "active_hours": active_hours,
        "chart": fig_to_uri(fig),
        "as_of": ohlc.index[-1].strftime("%Y-%m-%d"),
    }


def correlation_stats(histories: dict[str, pd.Series], as_of=None) -> dict:
    """Pearson correlation over 30 calendar days, excluding the current UTC day.

    Use daily returns, except US10Y which uses yield changes in basis points.
    Missing observations are never filled; each pair needs at least 10 samples.
    """
    end = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz="UTC")).normalize()
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    clean = {}
    for code, series in histories.items():
        values = pd.to_numeric(series, errors="coerce").copy()
        values.index = pd.DatetimeIndex(values.index).tz_localize(None).normalize()
        clean[code] = values[~values.index.duplicated(keep="last")].sort_index()
    prices = pd.DataFrame(clean).reindex(columns=CORRELATION_CODES).sort_index()
    prices = prices.loc[prices.index < end]
    changes = prices.pct_change(fill_method=None)
    changes["US10Y"] = prices["US10Y"].diff() * 100
    changes = changes.replace([np.inf, -np.inf], np.nan)
    changes = changes.loc[changes.index >= end - pd.Timedelta(days=30)]
    valid = changes.notna().astype(int)
    return {"matrix": changes.corr(min_periods=10), "counts": valid.T.dot(valid),
            "start": (end - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            "end": (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d")}


def build_correlation(inst_data: list[dict]) -> dict:
    histories = {d["code"]: d["close_history"] for d in inst_data}
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=65)).strftime("%Y-%m-%d")
    for code, ticker in [("DXY", "DX-Y.NYB"), ("US10Y", "^TNX")]:
        history = guard(f"{code} correlation history", lambda t=ticker: yahoo.get_prices(t, start=start)[t])
        if history is not None:
            histories[code] = history
    return correlation_stats(histories)


def render_correlation(data: dict) -> str:
    matrix, counts = data["matrix"], data["counts"]
    rows = []
    for left in CORRELATION_CODES:
        cells = []
        for right in CORRELATION_CODES:
            value, n = matrix.loc[left, right], int(counts.loc[left, right])
            color, label = "#182534", "N/A"
            if pd.notna(value):
                value = float(np.clip(value, -1, 1))
                base = (0, 128, 99) if value >= 0 else (163, 49, 62)
                neutral = (24, 37, 52)
                rgb = [round(a + (b - a) * abs(value)) for a, b in zip(neutral, base)]
                color = "#%02x%02x%02x" % tuple(rgb)
                label = f"{value:.2f}"
            tooltip = f"{left} / {right}: {n} paired observations"
            if pd.isna(value):
                tooltip += "; insufficient data or constant series"
            cells.append(f'<td style="background:{color}" title="{tooltip}">{label}</td>')
        rows.append(f'<tr><th scope="row">{left}</th>{"".join(cells)}</tr>')
    headings = "".join(f'<th scope="col">{code}</th>' for code in CORRELATION_CODES)
    return f'''<section class="card correlation">
  <h2>Market Correlation (30D)</h2>
  <p>Daily returns · {data['start']} through {data['end']} · completed dates only</p>
  <div class="correlation-scroll"><table aria-label="30-day market correlation matrix">
    <thead><tr><th scope="col">Market</th>{headings}</tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
  <p class="correlation-key">−1 opposite direction <span></span> +1 same direction · 0 little linear relationship</p>
  <p>Pearson correlation, last 30 calendar days; at least 10 paired observations.
  Hover over a cell for its sample count. N/A means insufficient data or no variation.
  US10Y uses changes in the 10-year Treasury yield (basis points), not bond returns.</p>
  <p>Yahoo Finance: EURUSD=X / GBPUSD=X (FX), DX-Y.NYB (DXY), ^TNX (US10Y);
  GC=F (gold), CL=F (WTI oil), NQ=F / ES=F / YM=F (indices) are futures proxies.
  Daily closing times differ across markets.</p>
</section>'''


def build_strategy(spread_bps: pd.Series | float, financing: float) -> dict | None:
    prices = guard("strategy prices", lambda: instruments.get_daily_prices(start="2021-01-01").dropna())
    if prices is None:
        return None
    weights = signals.sma_crossover(prices, 50, 200) / prices.shape[1]
    strat = run_backtest(prices, weights, cost_bps=spread_bps,
                         financing_rate_annual=financing, name="Trend 50/200 (CFD costs)")
    bench = equal_weight_benchmark(prices, cost_bps=spread_bps,
                                   financing_rate_annual=financing, name="Buy & hold (CFD costs)")
    win = strat.equity.iloc[-504:]
    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    ax.plot(win / win.iloc[0], color=plotting.PALETTE[0], label=strat.name)
    bwin = bench.equity.iloc[-504:]
    ax.plot(bwin / bwin.iloc[0], color=plotting.PALETTE[1], label=bench.name)
    ax.set_title("Trend strategy vs buy & hold — last 2 years, growth of $1")
    ax.legend(loc="upper left")

    positions = {}
    for code in prices.columns:
        w = weights[code]
        on = bool(w.iloc[-1] > 0)
        flips = w.ne(w.shift())
        last_flip = w.index[flips][-1] if flips.any() else w.index[0]
        positions[code] = {"on": on, "since": last_flip.strftime("%Y-%m-%d"),
                           "days": int((w.index[-1] - last_flip).days)}
    s = strat.stats()
    return {"chart": fig_to_uri(fig), "positions": positions,
            "cagr": s["cagr"], "sharpe": s["sharpe"], "maxdd": s["max_drawdown"]}


def build_macro() -> list[dict]:
    tiles: list[dict] = []

    def fred_tile(series, label, suffix="%"):
        from quantlab.data import fred
        s = fred.get_series(series, start="2024-01-01").dropna()
        return {"label": label, "value": f"{s.iloc[-1]:.2f}{suffix}",
                "sub": f"as of {s.index[-1]:%b %d}"}

    if has_key("FRED_API_KEY"):
        for series, label in [("FEDFUNDS", "Fed funds"), ("DGS10", "US 10y yield"), ("T10Y2Y", "10y–2y spread")]:
            t = guard(f"FRED {series}", lambda s=series, l=label: fred_tile(s, l))
            if t:
                if series == "T10Y2Y" and t["value"].startswith("-"):
                    t["sub"] += " · INVERTED"
                tiles.append(t)

    def yahoo_tile(ticker, label, code_dec="US500"):
        from quantlab.data import yahoo
        week_ago = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        px = yahoo.get_prices(ticker, start=week_ago)[ticker].dropna()
        chg = (px.iloc[-1] / px.iloc[-2] - 1) * 100
        return {"label": label, "value": f"{px.iloc[-1]:,.2f}",
                "sub": f"{chg:+.2f}% vs prior close", "delta": chg}

    for ticker, label in [("^VIX", "VIX"), ("DX-Y.NYB", "US Dollar index")]:
        t = guard(f"Yahoo {ticker}", lambda tk=ticker, l=label: yahoo_tile(tk, l))
        if t:
            tiles.append(t)

    def btc_tile():
        from quantlab.data import binance
        t = binance.get_ticker_24h("BTCUSDT")
        return {"label": "BTC (risk gauge)", "value": f"${t['last_price']:,.0f}",
                "sub": f"{t['change_pct']:+.2f}% / 24h", "delta": t["change_pct"]}

    t = guard("Binance BTC", btc_tile)
    if t:
        tiles.append(t)
    return tiles


def build_polymarket() -> pd.DataFrame | None:
    def fetch():
        from quantlab.data import polymarket
        m = polymarket.get_markets(limit=100)
        keywords = "fed |rate|inflation|recession|cpi|tariff|gold|treasury|shutdown"
        mask = m["question"].str.lower().str.contains(keywords, regex=True, na=False)
        cols = m[mask].sort_values("volume_24h", ascending=False).head(6)
        return cols[["question", "prob_first_outcome", "volume_24h", "end_date"]]

    return guard("Polymarket", fetch)


def build_news() -> list[dict]:
    items: list[dict] = []

    def finnhub_news():
        from quantlab.data import finnhub
        df = finnhub.get_market_news("general").head(6)
        return [{"when": r["datetime"].strftime("%H:%M UTC"), "source": r["source"],
                 "title": r["headline"], "url": r["url"]} for _, r in df.iterrows()]

    def gold_news():
        from quantlab.data import news
        df = news.search_news('gold price OR "federal reserve" OR Nasdaq', page_size=6)
        return [{"when": r["published_at"].strftime("%b %d %H:%M"), "source": r["source"],
                 "title": r["title"], "url": r["url"]} for _, r in df.iterrows()]

    if has_key("FINNHUB_API_KEY"):
        items += guard("Finnhub news", finnhub_news, default=[]) or []
    if has_key("NEWSAPI_API_KEY"):
        items += guard("NewsAPI", gold_news, default=[]) or []
    return items[:10]


def financing_estimate() -> tuple[float, str]:
    if has_key("FRED_API_KEY"):
        def fetch():
            from quantlab.data import fred
            ff = fred.get_series("FEDFUNDS", start="2024-01-01").dropna().iloc[-1] / 100
            return ff + 0.025, f"FEDFUNDS + 2.5% markup = {ff + 0.025:.2%}"
        got = guard("FRED financing", fetch)
        if got:
            return got
    return 0.06, "assumed 6.0% (set FRED_API_KEY for live)"


# ------------------------------------------------------------------- html --
CSS = """
/* Single-theme by design: the charts are raster images rendered on a light
   surface, so the page commits to light and states it rather than flipping. */
:root, :root[data-theme="dark"], :root[data-theme="light"] { color-scheme: light; }
.wrap { --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --sec:#52514e; --muted:#898781;
        --line:#e1e0d9; --border:rgba(11,11,11,.10); --pos:#2a78d6; --neg:#e34948;
        --good:#006300; --bad:#d03b3b; }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px 48px; background:#f9f9f7; color:#0b0b0b;
       font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 2px; } h2 { font-size:16px; margin:0 0 10px; }
h3 { font-size:13px; margin:14px 0 6px; color:var(--sec); text-transform:uppercase; letter-spacing:.04em; }
.meta { color:var(--muted); font-size:12px; margin-bottom:10px; }
.disclaimer { background:var(--card); border:1px solid var(--border); border-left:3px solid var(--muted);
              padding:8px 12px; border-radius:8px; color:var(--sec); font-size:12px; margin-bottom:20px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:12px; margin-bottom:14px; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }
.tile .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
.tile .value { font-size:24px; font-weight:700; margin:2px 0; }
.tile .sub { color:var(--sec); font-size:12px; }
.up { color:var(--good); } .down { color:var(--bad); }
.card { background:var(--card); border:1px solid var(--border); border-radius:14px;
        padding:18px 20px; margin-bottom:18px; }
.badge { display:inline-block; padding:2px 12px; border-radius:999px; font-weight:700; font-size:12px; }
.badge.bull { background:var(--pos); color:#fff; } .badge.bear { background:var(--neg); color:#fff; }
.badge.neut { background:var(--line); color:var(--ink); }
.badge.on { background:var(--pos); color:#fff; } .badge.off { background:var(--line); color:var(--sec); }
img.chart { width:100%; height:auto; border-radius:8px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; color:var(--muted); font-size:11px; text-transform:uppercase;
     letter-spacing:.04em; font-weight:600; padding:4px 8px; border-bottom:1px solid var(--line); }
td { padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media (max-width:840px){ .cols { grid-template-columns:1fr; } }
.head-row { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
.price-big { font-size:20px; font-weight:700; }
ul.news { list-style:none; margin:0; padding:0; }
ul.news li { padding:6px 0; border-bottom:1px solid var(--line); font-size:13px; }
ul.news li:last-child { border-bottom:none; }
ul.news .src { color:var(--muted); font-size:11px; }
a { color:var(--pos); text-decoration:none; } a:hover { text-decoration:underline; }
.warn { color:var(--muted); font-size:11px; }
footer { color:var(--muted); font-size:12px; }
code { background:var(--line); border-radius:4px; padding:1px 5px; font-size:12px; }
.card.correlation { background:#09131f; color:#e8edf4; }
.correlation p { font-size:12px; color:#a9b9c9; }
.correlation-scroll { overflow-x:auto; }
.correlation table { min-width:650px; border-collapse:separate; border-spacing:2px; }
.correlation th { color:#c3d0dd; border:0; white-space:nowrap; }
.correlation td { text-align:center; color:#fff; padding:12px 6px; border:0;
                  font-variant-numeric:tabular-nums; }
.correlation-key { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.correlation-key span { width:120px; height:8px; border-radius:4px;
                       background:linear-gradient(90deg,#a3313e,#182534,#008063); }
"""


def badge(label: str) -> str:
    cls = {"BULLISH": "bull", "BEARISH": "bear"}.get(label, "neut")
    icon = {"BULLISH": "&#9650;", "BEARISH": "&#9660;"}.get(label, "&#9670;")
    return f'<span class="badge {cls}">{icon} {label}</span>'


def render_instrument(d: dict) -> str:
    code, b, pv, plan = d["code"], d["bias"], d["pivots"], d["plan"]
    f = lambda v: fmt(v, code)  # noqa: E731
    chg_cls = "up" if d["day_chg"] >= 0 else "down"
    comp = " · ".join(("&#10003; " if ok else "&#10007; ") + esc(k) for k, ok in b["components"].items())

    ind_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>{v}</td></tr>"
        for k, v in [
            ("RSI(14)", f"{b['rsi']:.0f}"),
            ("12-1 momentum", f"{b['mom']:+.1%}"),
            ("30d realized vol (ann.)", f"{d['vol30']:.1%}"),
            ("ATR(14), points", f(d["atr"])),
            ("SMA 20 / 50 / 200", f"{f(b['sma20'])} / {f(b['sma50'])} / {f(b['sma200'])}"),
            ("20d swing high / low", f"{f(d['swings']['high_20d'])} / {f(d['swings']['low_20d'])}"),
        ]
    )

    piv_rows = "".join(
        f"<tr><td>{k}</td><td class='num'>{f(v)}</td></tr>" for k, v in pv.items()
    )

    plan_rows = "".join(
        f"<tr><td>{lab}</td><td>{val}</td></tr>"
        for lab, val in [
            ("Setup", esc(plan["direction"])),
            ("Entry zone", esc(plan["zone"])),
            ("Stop", esc(plan["stop"])),
            ("Targets", esc(plan["targets"])),
            ("Reward:risk", esc(plan["rr"])),
        ]
        if val
    )
    plan_note = f"<p class='warn'>{esc(plan['note'])}</p>" if plan.get("note") else ""

    sess_rows = ""
    stats, today = d["session_stats"], d["today_sessions"]
    for name in sessions.SESSIONS:
        avg_rng = f(stats.loc[name, "avg_range"]) if stats is not None and name in stats.index else "—"
        share = f"{stats.loc[name, 'range_share']:.0%}" if stats is not None and name in stats.index else "—"
        match = f"{stats.loc[name, 'day_match_rate']:.0%}" if stats is not None and name in stats.index else "—"
        if today is not None and name in today.index:
            t = today.loc[name]
            so_far = f"{f(t['low'])}–{f(t['high'])}"
        else:
            so_far = "not traded yet"
        sess_rows += (
            f"<tr><td><strong>{name}</strong><br><span class='warn'>{esc(d['tactics'][name])}</span></td>"
            f"<td class='num'>{avg_rng}</td><td class='num'>{share}</td>"
            f"<td class='num'>{match}</td><td class='num'>{so_far}</td></tr>"
        )

    hours = ""
    if d["active_hours"] is not None:
        hours = "Most active UTC hours (60d): " + ", ".join(
            f"{int(h):02d}:00 ({v:.2f}%)" for h, v in d["active_hours"].items()
        )

    return f"""
<section class="card">
  <div class="head-row">
    <h2 style="margin:0">{code} <span style="color:var(--muted);font-weight:400">· {esc(d['name'])}</span></h2>
    {badge(b['label'])}
    <span class="price-big">{f(d['price'])}</span>
    <span class="{chg_cls}">{d['day_chg']:+.2f}% vs prior close</span>
    <span class="warn">daily data through {d['as_of']}</span>
  </div>
  <p class="warn" style="margin:6px 0 12px">Composite ({b['score']}/5): {comp}</p>
  <img class="chart" src="{d['chart']}" alt="{code} daily chart with 50 and 200 day moving averages">
  <div class="cols" style="margin-top:14px">
    <div>
      <h3>Indicators</h3>
      <table>{ind_rows}</table>
      <h3>Pivot levels (from last completed day)</h3>
      <table>{piv_rows}</table>
    </div>
    <div>
      <h3>Today's mechanical plan</h3>
      <table>{plan_rows}</table>
      {plan_note}
      <p class="warn">Zones come from floor-trader pivots; stops sized off ATR(14) = {f(d['atr'])} pts.
      Risk a fixed fraction of account per trade (e.g. 0.5–1%) — position size = risk ÷ stop distance.</p>
    </div>
  </div>
  <h3>Session playbook (UTC) — Asia 00–07 · London 07–13 · New York 13–21</h3>
  <table>
    <tr><th>Session &amp; tactic</th><th>avg range (60d)</th><th>share of day</th><th>matches day dir.</th><th>today so far</th></tr>
    {sess_rows}
  </table>
  <p class="warn">{hours}</p>
</section>"""


def render_body(data: dict) -> str:
    """The page content (everything inside <body>) — shared by both outputs."""
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()

    inst_tiles = ""
    for d in data["instruments"]:
        chg_cls = "up" if d["day_chg"] >= 0 else "down"
        inst_tiles += f"""
    <div class="tile"><div class="label">{d['code']} · {esc(d['name'])}</div>
      <div class="value">{fmt(d['price'], d['code'])}</div>
      <div class="sub"><span class="{chg_cls}">{d['day_chg']:+.2f}%</span> vs prior close · {badge(d['bias']['label'])}</div></div>"""

    macro_tiles = ""
    for t in data["macro"]:
        delta_cls = ""
        if "delta" in t:
            delta_cls = "up" if t["delta"] >= 0 else "down"
        macro_tiles += f"""
    <div class="tile"><div class="label">{esc(t['label'])}</div>
      <div class="value">{esc(t['value'])}</div>
      <div class="sub {delta_cls}">{esc(t['sub'])}</div></div>"""

    body = "".join(render_instrument(d) for d in data["instruments"])

    strat_html = ""
    if data["strategy"]:
        s = data["strategy"]
        long_badge = '<span class="badge on">&#9650; LONG</span>'
        flat_badge = '<span class="badge off">FLAT</span>'
        pos_rows = "".join(
            f"<tr><td>{code}</td>"
            f"<td>{long_badge if p['on'] else flat_badge}</td>"
            f"<td class='num'>since {p['since']} ({p['days']}d)</td></tr>"
            for code, p in s["positions"].items()
        )
        strat_html = f"""
<section class="card">
  <h2>Systematic trend strategy — live state</h2>
  <p class="warn">50/200 SMA crossover, equal split, CFD costs (spread + financing {esc(data['fin_src'])}).
  Full methodology in <code>notebooks/08_cfd_indices_gold.ipynb</code>.
  Since 2021: CAGR {s['cagr']:.1%} · Sharpe {s['sharpe']:.2f} · max drawdown {s['maxdd']:.1%}.</p>
  <div class="cols">
    <div><img class="chart" src="{s['chart']}" alt="strategy equity curve vs buy and hold"></div>
    <div><h3>Current positions</h3><table>
      <tr><th>Instrument</th><th>State</th><th>Last flip</th></tr>{pos_rows}</table>
      <p class="warn">A flip appears here the day after the crossover (signals trade next bar).</p></div>
  </div>
</section>"""

    news_html = "<p class='warn'>No news keys configured.</p>"
    if data["news"]:
        news_html = "<ul class='news'>" + "".join(
            f"<li><a href='{esc(n['url'])}' target='_blank' rel='noopener'>{esc(n['title'])}</a><br>"
            f"<span class='src'>{esc(n['source'])} · {esc(n['when'])}</span></li>"
            for n in data["news"]
        ) + "</ul>"

    poly_html = "<p class='warn'>Polymarket unavailable.</p>"
    pm = data["polymarket"]
    if pm is not None and len(pm):
        rows = "".join(
            f"<tr><td>{esc(r['question'])}</td>"
            f"<td class='num'>{r['prob_first_outcome']:.0%}</td>"
            f"<td class='num'>${r['volume_24h']:,.0f}</td></tr>"
            for _, r in pm.iterrows() if pd.notna(r["prob_first_outcome"])
        )
        poly_html = f"<table><tr><th>Market</th><th>implied prob.</th><th>24h volume</th></tr>{rows}</table>"

    warn_html = ""
    if WARNINGS:
        warn_html = "<h3>Data issues this run</h3><p class='warn'>" + "<br>".join(esc(w) for w in WARNINGS) + "</p>"

    return f"""<div class="wrap">
<header>
  <h1>Quantup Daily — Indices · Gold · Forex · Oil</h1>
  <div class="meta">Generated {now_utc:%A %d %B %Y, %H:%M UTC} ({now_local:%H:%M} local) —
  prices are FX quotes and futures-based CFD proxies fetched at run time. Session clock is UTC.</div>
  <div class="disclaimer"><strong>Research tool, not financial advice.</strong>
  Bias, zones and targets are mechanical outputs of trend/momentum rules and pivot arithmetic —
  they carry no knowledge of today's news or your account. CFDs are leveraged: size positions off the
  stop distance, never off margin allowed, and expect gaps through stops on the weekly open.</div>
</header>
<section class="tiles">{inst_tiles}</section>
<section class="tiles">{macro_tiles}</section>
{render_correlation(data['correlation'])}
{body}
{strat_html}
<div class="cols">
  <section class="card"><h2>News</h2>{news_html}</section>
  <section class="card"><h2>Polymarket — macro event risk</h2>{poly_html}
    <p class="warn">Market-implied probabilities from real-money prediction markets — a quick prior on
    Fed decisions and macro events that move these markets.</p></section>
</div>
<footer class="card">
  <strong>Refresh:</strong> run <code>python dashboard/build_dashboard.py</code> (or double-click
  <code>run_dashboard.bat</code>) each morning before London or New York open.
  Deep dives live in the <code>notebooks/</code> folder (01–08); this page is generated by
  <code>dashboard/build_dashboard.py</code> using the <code>quantlab</code> package.
  {warn_html}
</footer>
</div>"""


def render_full(body: str) -> str:
    """Standalone HTML file for local viewing (dashboard/dashboard.html)."""
    stamp = datetime.now(timezone.utc)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quantup Daily — {stamp:%Y-%m-%d}</title>
<style>{CSS}</style></head>
<body>
{body}
</body></html>"""


def render_artifact(body: str) -> str:
    """Body-only version for publishing as a hosted page (no doctype/head/body:
    the host wraps it). Written to dashboard/dashboard_artifact.html."""
    return f"""<title>Quantup Daily — CFD trading dashboard</title>
<style>{CSS}</style>
{body}"""


# ------------------------------------------------------------------- main --
def main() -> int:
    print("Quantup Daily — collecting data (30-90s)...")
    fin, fin_src = financing_estimate()

    inst_data = [d for code in CODES if (d := build_instrument(code)) is not None]
    if not inst_data:
        print("FATAL: no instrument data could be fetched.")
        return 1

    spread = guard("spreads", lambda: instruments.one_way_spread_bps(
        prices=pd.Series({d["code"]: d["price"] for d in inst_data})), default=1.0)

    data = {
        "instruments": inst_data,
        "correlation": build_correlation(inst_data),
        "macro": build_macro(),
        "strategy": build_strategy(spread, fin),
        "news": build_news(),
        "polymarket": build_polymarket(),
        "fin_src": fin_src,
    }

    body = render_body(data)

    out = OUT_DIR / "dashboard.html"
    out.write_text(render_full(body), encoding="utf-8")

    # Body-only copy, ready to be published as a shareable hosted page.
    (OUT_DIR / "dashboard_artifact.html").write_text(render_artifact(body), encoding="utf-8")

    archive = OUT_DIR / "archive"
    archive.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    shutil.copy(out, archive / f"dashboard_{stamp}.html")

    print(f"OK: {out}")
    print(f"    archived copy: archive/dashboard_{stamp}.html")
    if WARNINGS:
        print(f"    {len(WARNINGS)} data issue(s) — listed at the bottom of the page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
