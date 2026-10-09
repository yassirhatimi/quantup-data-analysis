"""Live market probability engine — XAUUSD, EURUSD, GBPUSD.

    python scripts/live_predict.py --once            # one update, print + write files
    python scripts/live_predict.py --serve           # live loop + browser dashboard
    python scripts/live_predict.py --serve --open    # ...and open the browser
    python scripts/live_predict.py --once --no-twelvedata   # Yahoo only, spends no API budget

Architecture, and why this one. The project is a Python package that generates
static dashboards, not a web application, so there is no framework to hang an
API route on. The lightest thing that satisfies "update automatically when new
information arrives" is therefore a long-running Python process with an
event loop, publishing over **Server-Sent Events** from a standard-library
`ThreadingHTTPServer`: no new dependency, no build step, and one direction of
traffic is all a dashboard needs. The browser receives finished JSON only —
every API key stays in this process.

The loop does not poll everything at one rate. Each source has its own cadence
(price every 60s, Polymarket every 5 min, news every 15, macro and the calendar
hourly) and a recompute only happens when something actually moved: price
through a fraction of ATR, a 15-minute candle closing, a Polymarket repricing,
a volatility shift, or a scheduled release landing — with a heartbeat so the
page never goes stale silently, and a floor so it cannot thrash.

Every update is written to `dashboard/predict/` and appended to the SQLite
store, and matured predictions are scored on the way past, which is what turns
this from a forecaster into something with a measurable track record.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from quantlab.predict import (assets as asset_spec, config, engine, market_data,
                              polymarket_data, render, store)

OUT_DIR = REPO / "dashboard" / "predict"
STATE_DIR = OUT_DIR / "state"
HISTORY_DIR = OUT_DIR / "history"


# --------------------------------------------------------------------------
# trigger tracking
# --------------------------------------------------------------------------

class Triggers:
    """Decides when a recompute is warranted, and says which input caused it."""

    def __init__(self) -> None:
        self.last_run: float = 0.0
        self.last_price: dict[str, float] = {}
        self.last_atr: dict[str, float] = {}
        self.last_bar: dict[str, pd.Timestamp] = {}
        self.last_sigma: dict[str, float] = {}
        self.last_poly_bps: float | None = None
        self.last_event_seen: set[str] = set()

    def check(self, snaps: dict, poly_bps: float | None,
              events: list) -> list[str]:
        reasons: list[str] = []
        now = time.time()

        for code, snap in snaps.items():
            if not np.isfinite(snap.price):
                continue
            prev = self.last_price.get(code)
            atr = self.last_atr.get(code)
            if prev is not None and atr and atr > 0:
                if abs(snap.price - prev) >= config.TRIGGER_PRICE_ATR * atr:
                    reasons.append(
                        f"{code} moved {asset_spec.fmt_distance(code, snap.price - prev)} "
                        f"({abs(snap.price - prev) / atr:.2f}x ATR)")
            m15 = snap.bars.get("m15")
            if m15 is not None and len(m15):
                prev_bar = self.last_bar.get(code)
                if prev_bar is not None and m15.index[-1] > prev_bar:
                    reasons.append(f"{code} new 15m candle closed")

        if poly_bps is not None and self.last_poly_bps is not None:
            if abs(poly_bps - self.last_poly_bps) >= config.TRIGGER_POLY_PTS:
                reasons.append(f"Polymarket Fed pricing moved "
                               f"{poly_bps - self.last_poly_bps:+.1f} bps")

        fresh = {e.name for e in events
                 if abs((e.when - pd.Timestamp.now(tz="UTC")).total_seconds()) < 1800}
        new_events = fresh - self.last_event_seen
        if new_events:
            reasons.append("release window: " + ", ".join(sorted(new_events)))
        self.last_event_seen = fresh

        if not reasons and (now - self.last_run) >= config.MAX_RECOMPUTE_GAP:
            reasons.append(f"heartbeat ({config.MAX_RECOMPUTE_GAP // 60} min)")
        if reasons and (now - self.last_run) < config.MIN_RECOMPUTE_GAP:
            return []
        return reasons

    def commit(self, preds: dict, poly_bps: float | None) -> None:
        self.last_run = time.time()
        self.last_poly_bps = poly_bps
        for code, pred in preds.items():
            if np.isfinite(pred.price):
                self.last_price[code] = pred.price
            if np.isfinite(pred.tech.atr14):
                self.last_atr[code] = pred.tech.atr14
            if np.isfinite(pred.tech.sigma_ewma):
                self.last_sigma[code] = pred.tech.sigma_ewma


# --------------------------------------------------------------------------
# SSE server
# --------------------------------------------------------------------------

class Broker:
    """Fan-out of the latest payload to every connected browser."""

    def __init__(self) -> None:
        self.clients: list[Queue] = []
        self.lock = threading.Lock()
        self.latest: dict | None = None

    def publish(self, payload: dict) -> None:
        with self.lock:
            self.latest = payload
            targets = list(self.clients)
        message = json.dumps({"generated": payload["generated"]})
        for q in targets:
            q.put(message)

    def subscribe(self) -> Queue:
        q: Queue = Queue(maxsize=8)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


def make_handler(broker: Broker):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):        # keep the console for the engine
            return

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                     # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.startswith("/events"):
                return self._events()
            if self.path.startswith("/api/predictions"):
                payload = broker.latest or {"status": "warming up"}
                return self._send(json.dumps(payload, default=str).encode(),
                                  "application/json")
            if broker.latest is None:
                return self._send(b"<h1>Warming up...</h1><script>"
                                  b"setTimeout(function(){location.reload()},3000)</script>",
                                  "text/html; charset=utf-8")
            html = render.render_full(broker.latest, sse=True)
            return self._send(html.encode("utf-8"), "text/html; charset=utf-8")

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = broker.subscribe()
            try:
                while True:
                    try:
                        message = q.get(timeout=20)
                        self.wfile.write(f"data: {message}\n\n".encode())
                    except Empty:
                        self.wfile.write(b": keepalive\n\n")   # keeps proxies honest
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                broker.unsubscribe(q)
    return Handler


# --------------------------------------------------------------------------
# one update
# --------------------------------------------------------------------------

def write_outputs(payload: dict, predictions: dict, quiet: bool = False) -> None:
    for d in (OUT_DIR, STATE_DIR, HISTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "predictions.html").write_text(render.render_full(payload),
                                              encoding="utf-8")
    (OUT_DIR / "predictions_artifact.html").write_text(render.render_artifact(payload),
                                                       encoding="utf-8")
    (STATE_DIR / "latest.json").write_text(render.payload_json(payload), encoding="utf-8")

    stamp = pd.Timestamp(payload["generated"]).strftime("%Y-%m-%d_%H%M")
    (HISTORY_DIR / f"predictions_{stamp}.json").write_text(
        render.payload_json(payload), encoding="utf-8")

    lines = [render.text_block(p) for p in predictions.values()]
    (OUT_DIR / "latest.txt").write_text("\n\n".join(lines), encoding="utf-8")

    cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)).timestamp()
    for f in HISTORY_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
    if not quiet:
        print(f"  wrote {OUT_DIR / 'predictions.html'}")


def one_update(codes: list[str], use_twelvedata: bool = True,
               triggers_fired: list[str] | None = None,
               quiet: bool = False) -> tuple[dict, dict]:
    """Fetch, predict, persist, score matured rows. Returns (predictions, payload)."""
    store.init_db()
    previous = {code: store.latest_payload(code) for code in codes}
    record = store.track_record()

    predictions = engine.predict_all(codes, previous=previous,
                                     use_twelvedata=use_twelvedata,
                                     track_record=record)
    for pred in predictions.values():
        store.save(pred)

    scored = store.score_matured()
    note = _backtest_note(scored)
    payload = render.build_payload(predictions, triggers_fired, note)
    write_outputs(payload, predictions, quiet=quiet)
    return predictions, payload


def _backtest_note(scored) -> str:
    s = store.summary()
    bits = [f"Prediction history: {s['rows']:,} rows, {s['scored']:,} scored"]
    if s["mean_brier"] is not None:
        bits.append(f"mean Brier {s['mean_brier']:.4f}")
        bits.append(f"hit rate {s['hit_rate']:.1%}")
    else:
        bits.append("no predictions have matured yet — real accuracy will appear "
                    "here once the first horizons close")
    if scored.unresolvable:
        bits.append(f"{scored.unresolvable} could not be scored (no bar at the horizon end)")
    return " · ".join(bits)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def run_loop(codes: list[str], broker: Broker | None, interval: int,
             use_twelvedata: bool) -> None:
    triggers = Triggers()
    print(f"Live engine started for {', '.join(codes)}. "
          f"Tick every {interval}s; recompute on triggers.")
    first = True
    while True:
        try:
            snaps, poly_bps, events = {}, None, []
            if not first:
                # Cheap probe: refresh only what the trigger rules look at.
                for code in codes:
                    try:
                        snaps[code] = market_data.get_snapshot(
                            code, use_twelvedata=use_twelvedata)
                    except Exception as exc:
                        print(f"  {code} probe failed: {type(exc).__name__}: {exc}")
                try:
                    from quantlab.predict import macro_data
                    events, _ = macro_data.calendar()
                    universe = polymarket_data.fetch_universe()
                    comp = polymarket_data.score_polymarket("XAUUSD", universe=universe)
                    poly_bps = (comp.detail or {}).get("expected_fed_bps")
                except Exception:
                    pass
                fired = triggers.check(snaps, poly_bps, events)
            else:
                fired = ["first run"]

            if fired:
                stamp = pd.Timestamp.now(tz="UTC")
                print(f"\n[{stamp:%H:%M:%S} UTC] recompute — {'; '.join(fired)}")
                predictions, payload = one_update(codes, use_twelvedata, fired, quiet=True)
                triggers.commit(predictions, poly_bps)
                for code, pred in predictions.items():
                    h = pred.horizons.get("1h")
                    if h is None:
                        print(f"  {code}: LIVE DATA UNAVAILABLE")
                        continue
                    up, down, rng = h.rounded()
                    print(f"  {code}: UP {up}% / DOWN {down}% / RANGE {rng}% · "
                          f"conf {h.confidence:.0f} {h.confidence_label} · "
                          f"{pred.regime.label} · data {pred.quality.label}")
                if broker is not None:
                    broker.publish(payload)
                print(f"  Twelve Data budget: {market_data.BUDGET.used}/"
                      f"{market_data.BUDGET.limit} calls today")
            first = False
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — a live loop must survive anything
            print(f"  update failed: {type(exc).__name__}: {exc}")
        time.sleep(interval)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", nargs="*", default=None,
                    help="asset codes (default: XAUUSD EURUSD GBPUSD)")
    ap.add_argument("--once", action="store_true", help="one update, then exit")
    ap.add_argument("--serve", action="store_true", help="run the live loop + HTTP server")
    ap.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    ap.add_argument("--interval", type=int, default=60, help="loop tick in seconds")
    ap.add_argument("--port", type=int, default=config.SERVER_PORT)
    ap.add_argument("--no-twelvedata", action="store_true",
                    help="Yahoo only — spends no Twelve Data call budget")
    args = ap.parse_args(argv)

    codes = [c.upper() for c in (args.assets or asset_spec.DEFAULT_CODES)]
    for code in codes:
        asset_spec.get(code)          # fail fast on a typo
    use_td = not args.no_twelvedata

    if args.once or not args.serve:
        predictions, _ = one_update(codes, use_td, ["manual --once run"])
        for pred in predictions.values():
            print()
            print(render.text_block(pred))
        print(f"\nTwelve Data budget used today: {market_data.BUDGET.used}/"
              f"{market_data.BUDGET.limit}")
        return 0

    broker = Broker()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(broker))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard: {url}   (JSON at {url}api/predictions)")
    if args.open:
        webbrowser.open(url)
    try:
        run_loop(codes, broker, args.interval, use_td)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
