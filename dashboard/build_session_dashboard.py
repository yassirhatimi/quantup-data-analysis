"""Quantup Sessions — intraday session dashboard.

A second, session-focused briefing for US100 / US500 / US30 / XAUUSD. Where the
main dashboard (build_dashboard.py) answers "what's the bias and the plan
today", this one answers "when in the 24h clock does each instrument actually
move" — Asia (00-07 UTC), London (07-13) and New York (13-21):

  - which session is live right now, and how long until the next one,
  - today's Asia / London / NY ranges as they form (the levels you trade
    breakouts and retests against),
  - per-session tendencies over the last ~120 days: average range, share of
    the day's range, how often the session moves with the whole day, average
    return and up-rate,
  - an hour-by-hour activity profile per instrument (where the volatility is),
  - a cross-instrument comparison of which session owns each instrument.

Run it whenever you like — it is most useful re-run through the day as sessions
complete:

    python dashboard/build_session_dashboard.py    # or run_session_dashboard.bat

Output: dashboard/session_dashboard.html (+ a dated copy in dashboard/archive/).
Each instrument is fetched independently; if one fails the page still builds and
lists the problem at the bottom.
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
from quantlab.analysis import sessions

plotting.apply_style()

CODES = ["US100", "US500", "US30", "XAUUSD"]
DECIMALS = {"US100": 0, "US500": 1, "US30": 0, "XAUUSD": 1}
LOOKBACK_DAYS = 120
OUT_DIR = Path(__file__).resolve().parent
WARNINGS: list[str] = []

# Colour per session, reused everywhere so the eye learns them once.
SESSION_COLOR = {"Asia": plotting.PALETTE[2], "London": plotting.PALETTE[0], "New York": plotting.PALETTE[3]}


# ----------------------------------------------------------------- helpers --
def fmt(value, code: str = "US500", signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    sign = "+" if (signed and value >= 0) else ""
    return f"{sign}{value:,.{DECIMALS.get(code, 1)}f}"


def guard(label: str, fn, default=None):
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


def current_session(now_utc: datetime) -> tuple[str | None, str, int]:
    """(live session or None, next session name, minutes until next session)."""
    minute_of_day = now_utc.hour * 60 + now_utc.minute
    live = None
    for name, (start, _end) in sessions.SESSIONS.items():
        s0, s1 = sessions.SESSIONS[name]
        if s0 * 60 <= minute_of_day < s1 * 60:
            live = name
    starts = [(name, s * 60) for name, (s, _e) in sessions.SESSIONS.items()]
    upcoming = [(n, m) for n, m in starts if m > minute_of_day]
    if upcoming:
        nxt, m = min(upcoming, key=lambda t: t[1])
    else:  # after NY: next start is tomorrow's Asia (00:00)
        nxt, m = "Asia", 24 * 60
    return live, nxt, m - minute_of_day


# ------------------------------------------------------------ computations --
def hour_profile(hourly: pd.DataFrame) -> pd.Series:
    """Average absolute hourly move (%) for every UTC hour 0-23."""
    idx = hourly.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    move = (hourly["Close"] / hourly["Open"] - 1).abs() * 100
    move.index = idx
    return move.groupby(move.index.hour).mean().reindex(range(24))


def directional_stats(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per session: average return, up-rate and range as % of price."""
    so = sessions.session_ohlc(hourly)
    so["range_pct"] = so["range"] / so["open"] * 100
    g = so.groupby("session")
    out = pd.DataFrame({
        "avg_ret": g["ret"].mean() * 100,
        "up_rate": g["ret"].apply(lambda s: (s > 0).mean()) * 100,
        "avg_range_pct": g["range_pct"].mean(),
    })
    return out.reindex(list(sessions.SESSIONS))


def build_instrument(code: str) -> dict | None:
    inst = instruments.INSTRUMENTS[code]
    hourly = guard(f"{code} hourly bars", lambda: instruments.get_intraday(code, "1h", days=LOOKBACK_DAYS))
    if hourly is None or len(hourly) < 48:
        return None

    stats = guard(f"{code} session stats", lambda: sessions.session_stats(hourly))
    direction = guard(f"{code} directional", lambda: directional_stats(hourly))
    today = guard(f"{code} today", lambda: sessions.todays_sessions(hourly), default=pd.DataFrame())
    active = guard(f"{code} active hours", lambda: sessions.most_active_hours(hourly, top=6))
    profile = guard(f"{code} hour profile", lambda: hour_profile(hourly))

    chart = None
    if profile is not None:
        f = lambda v: fmt(v, code)  # noqa: E731
        fig, ax = plt.subplots(figsize=(8.8, 2.9))
        bars = ax.bar(profile.index, profile.values, width=0.86)
        for h, bar in zip(profile.index, bars):
            for name, (s0, s1) in sessions.SESSIONS.items():
                if s0 <= h < s1:
                    bar.set_color(SESSION_COLOR[name])
        for name, (s0, s1) in sessions.SESSIONS.items():
            ax.axvspan(s0 - 0.5, s1 - 0.5, color=SESSION_COLOR[name], alpha=0.07, zorder=0)
            ax.text((s0 + s1) / 2 - 0.5, ax.get_ylim()[1] * 0.92, name,
                    ha="center", va="top", fontsize=8, color=plotting.SECONDARY)
        ax.set_title(f"{code} — average hourly move by UTC hour (last {LOOKBACK_DAYS}d)")
        ax.set_xlabel("UTC hour")
        ax.set_xticks(range(0, 24, 2))
        ax.yaxis.set_major_formatter(plotting.mpl_pct(2))
        chart = fig_to_uri(fig)

    return {
        "code": code, "name": inst.name,
        "stats": stats, "direction": direction, "today": today,
        "active": active, "chart": chart,
    }


def build_comparison(insts: list[dict]) -> str | None:
    """Grouped bars: each session's share of the day's range, per instrument."""
    rows = {d["code"]: d["stats"]["range_share"] for d in insts if d["stats"] is not None}
    if not rows:
        return None
    share = pd.DataFrame(rows).T.reindex(columns=list(sessions.SESSIONS))
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    x = np.arange(len(share.index))
    width = 0.8 / len(sessions.SESSIONS)
    for i, name in enumerate(sessions.SESSIONS):
        ax.bar(x + i * width - 0.4 + width / 2, share[name], width,
               color=SESSION_COLOR[name], label=name)
    ax.set_xticks(x, share.index)
    ax.set_title("Which session owns each instrument — share of the day's range")
    ax.yaxis.set_major_formatter(plotting.mpl_pct(0))
    ax.legend(loc="upper right", ncols=3)
    return fig_to_uri(fig)


# ------------------------------------------------------------------- html --
CSS = """
:root, :root[data-theme="dark"], :root[data-theme="light"] { color-scheme: light; }
.wrap { --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --sec:#52514e; --muted:#898781;
        --line:#e1e0d9; --border:rgba(11,11,11,.10); --asia:#eda100; --london:#2a78d6; --ny:#008300; }
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
.tile .value { font-size:22px; font-weight:700; margin:2px 0; }
.tile .sub { color:var(--sec); font-size:12px; }
.tile.live { border-color:transparent; box-shadow:0 0 0 2px currentColor inset; }
.tile.asia.live { color:var(--asia); } .tile.london.live { color:var(--london); } .tile.ny.live { color:var(--ny); }
.tile .value, .tile .label, .tile .sub { color:var(--ink); }
.tile.live .label { color:currentColor; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; vertical-align:baseline; }
.dot.asia { background:var(--asia); } .dot.london { background:var(--london); } .dot.ny { background:var(--ny); }
.card { background:var(--card); border:1px solid var(--border); border-radius:14px;
        padding:18px 20px; margin-bottom:18px; }
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
.warn { color:var(--muted); font-size:11px; }
a { color:var(--london); text-decoration:none; } a:hover { text-decoration:underline; }
footer { color:var(--muted); font-size:12px; }
code { background:var(--line); border-radius:4px; padding:1px 5px; font-size:12px; }
"""

SESS_CLASS = {"Asia": "asia", "London": "london", "New York": "ny"}


def render_instrument(d: dict) -> str:
    code = d["code"]
    f = lambda v: fmt(v, code)  # noqa: E731
    stats, direction, today, active = d["stats"], d["direction"], d["today"], d["active"]

    rows = ""
    for name in sessions.SESSIONS:
        s0, s1 = sessions.SESSIONS[name]
        avg_rng = f(stats.loc[name, "avg_range"]) if stats is not None and name in stats.index else "—"
        share = f"{stats.loc[name, 'range_share']:.0%}" if stats is not None and name in stats.index else "—"
        match = f"{stats.loc[name, 'day_match_rate']:.0%}" if stats is not None and name in stats.index else "—"
        if direction is not None and name in direction.index:
            avg_ret = f"{direction.loc[name, 'avg_ret']:+.2f}%"
            up = f"{direction.loc[name, 'up_rate']:.0f}%"
        else:
            avg_ret = up = "—"
        if today is not None and name in today.index:
            t = today.loc[name]
            so_far = f"{f(t['low'])}–{f(t['high'])} ({f(t['range'])} pts)"
        else:
            so_far = "not traded yet"
        rows += (
            f"<tr><td><span class='dot {SESS_CLASS[name]}'></span><strong>{name}</strong>"
            f"<br><span class='warn'>{s0:02d}:00–{s1:02d}:00 UTC</span></td>"
            f"<td class='num'>{avg_rng}</td><td class='num'>{share}</td>"
            f"<td class='num'>{match}</td><td class='num'>{avg_ret}</td>"
            f"<td class='num'>{up}</td><td class='num'>{so_far}</td></tr>"
        )

    active_txt = ""
    if active is not None:
        active_txt = "Busiest UTC hours: " + ", ".join(
            f"{int(h):02d}:00 ({v:.2f}%)" for h, v in active.items()
        )

    chart = f"<img class='chart' src='{d['chart']}' alt='{code} hourly activity by UTC hour'>" if d["chart"] else ""

    return f"""
<section class="card">
  <div class="head-row">
    <h2 style="margin:0">{code} <span style="color:var(--muted);font-weight:400">· {esc(d['name'])}</span></h2>
  </div>
  {chart}
  <table style="margin-top:12px">
    <tr><th>Session</th><th>avg range</th><th>share of day</th><th>matches day dir.</th>
        <th>avg return</th><th>up-rate</th><th>today so far</th></tr>
    {rows}
  </table>
  <p class="warn">{active_txt}</p>
</section>"""


def render_body(insts: list[dict], comparison: str | None) -> str:
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()
    live, nxt, mins = current_session(now_utc)

    # Session tiles (the live one is ringed).
    tiles = ""
    for name in sessions.SESSIONS:
        s0, s1 = sessions.SESSIONS[name]
        cls = SESS_CLASS[name]
        is_live = name == live
        state = "LIVE NOW" if is_live else (f"opens in {mins // 60}h{mins % 60:02d}m" if name == nxt else "closed")
        tiles += f"""
    <div class="tile {cls}{' live' if is_live else ''}">
      <div class="label"><span class="dot {cls}"></span>{name}</div>
      <div class="value">{s0:02d}:00–{s1:02d}:00</div>
      <div class="sub">UTC · {state}</div></div>"""

    # Today's ranges across all instruments, one table.
    today_rows = ""
    for name in sessions.SESSIONS:
        cells = ""
        for d in insts:
            t = d["today"]
            if t is not None and name in getattr(t, "index", []):
                r = t.loc[name]
                cells += f"<td class='num'>{fmt(r['low'], d['code'])}–{fmt(r['high'], d['code'])}</td>"
            else:
                cells += "<td class='num warn'>—</td>"
        today_rows += (f"<tr><td><span class='dot {SESS_CLASS[name]}'></span>{name}</td>{cells}</tr>")
    today_head = "".join(f"<th class='num'>{d['code']}</th>" for d in insts)

    comp_html = f"<img class='chart' src='{comparison}' alt='session range share by instrument'>" if comparison else ""
    body = "".join(render_instrument(d) for d in insts)

    warn_html = ""
    if WARNINGS:
        warn_html = "<h3>Data issues this run</h3><p class='warn'>" + "<br>".join(esc(w) for w in WARNINGS) + "</p>"

    live_txt = (f"<strong>{live} session is live</strong> — {nxt} opens in {mins // 60}h{mins % 60:02d}m."
                if live else f"Between sessions (21:00–24:00 UTC is thin) — {nxt} opens in {mins // 60}h{mins % 60:02d}m.")

    return f"""<div class="wrap">
<header>
  <h1>Quantup Sessions — Asia · London · New York</h1>
  <div class="meta">Generated {now_utc:%A %d %B %Y, %H:%M UTC} ({now_local:%H:%M} local). {live_txt}
  Stats use the last {LOOKBACK_DAYS} days of hourly bars; session clock is UTC.</div>
  <div class="disclaimer"><strong>Research tool, not financial advice.</strong>
  Session tendencies are historical averages — a session that "usually" makes the day still fails often.
  Use them to size expectations and time entries, not as signals on their own.</div>
</header>
<section class="tiles">{tiles}</section>
<section class="card">
  <h2>Today's ranges so far</h2>
  <p class="warn">The Asia range is what London breaks out of; the London range frames the US open.
  Blank = that session has not traded yet today (UTC).</p>
  <table><tr><th>Session</th>{today_head}</tr>{today_rows}</table>
</section>
<section class="card">
  <h2>Who moves when</h2>
  {comp_html}
  <p class="warn">Taller bar = that session contributes more of the instrument's daily range. The indices
  front-load into London and New York; gold spreads its range more evenly across the clock.</p>
</section>
{body}
<footer class="card">
  <strong>Refresh:</strong> run <code>python dashboard/build_session_dashboard.py</code> (or double-click
  <code>run_session_dashboard.bat</code>) — most useful re-run through the day as each session completes.
  The bias-and-plan briefing is the separate <code>dashboard/dashboard.html</code>.
  {warn_html}
</footer>
</div>"""


def render_full(body: str) -> str:
    stamp = datetime.now(timezone.utc)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quantup Sessions — {stamp:%Y-%m-%d}</title>
<style>{CSS}</style></head>
<body>
{body}
</body></html>"""


def render_artifact(body: str) -> str:
    return f"""<title>Quantup Sessions — Asia / London / New York</title>
<style>{CSS}</style>
{body}"""


# ------------------------------------------------------------------- main --
def main() -> int:
    print("Quantup Sessions — collecting hourly data (20-60s)...")
    insts = [d for code in CODES if (d := build_instrument(code)) is not None]
    if not insts:
        print("FATAL: no session data could be fetched.")
        return 1

    comparison = guard("session comparison", lambda: build_comparison(insts))
    body = render_body(insts, comparison)

    out = OUT_DIR / "session_dashboard.html"
    out.write_text(render_full(body), encoding="utf-8")
    (OUT_DIR / "session_dashboard_artifact.html").write_text(render_artifact(body), encoding="utf-8")

    archive = OUT_DIR / "archive"
    archive.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    shutil.copy(out, archive / f"session_dashboard_{stamp}.html")

    print(f"OK: {out}")
    print(f"    archived copy: archive/session_dashboard_{stamp}.html")
    if WARNINGS:
        print(f"    {len(WARNINGS)} data issue(s) — listed at the bottom of the page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
