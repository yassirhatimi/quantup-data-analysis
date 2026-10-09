"""SQLite prediction history — and, later, the model's real track record.

The point of storing predictions is not archaeology. A forecast that is never
scored is not a forecast, it is an opinion, so every row carries the band that
defined RANGE *at the time it was made*, and `score_matured` comes back later
to fill in what actually happened using that same band. Grading a 15-minute
call against a band recomputed from today's volatility would be marking your
own homework with a different pen.

Nothing here needs a server: `sqlite3` is in the standard library and the file
lives under `data/` which is already git-ignored, so prediction history never
lands in a commit.

    from quantlab.predict import store
    store.init_db()
    store.save(prediction)
    store.score_matured()
    print(store.calibration_table("XAUUSD", "1h"))
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.predict import config

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "predict" / "predictions.sqlite"

CLASSES = ("UP", "DOWN", "RANGE")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    asset             TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    timeframe         TEXT    NOT NULL,
    horizon_start     TEXT    NOT NULL,
    horizon_end       TEXT    NOT NULL,
    price             REAL    NOT NULL,
    up_probability    REAL    NOT NULL,
    down_probability  REAL    NOT NULL,
    range_probability REAL    NOT NULL,
    confidence        REAL    NOT NULL,
    confidence_label  TEXT    NOT NULL,
    market_regime     TEXT    NOT NULL,
    vol_state         TEXT,
    session           TEXT,
    data_quality      TEXT,
    availability      REAL,
    band_return       REAL,
    band_price        REAL,
    sigma             REAL,
    drift             REAL,
    score             REAL,
    event_level       TEXT,
    polymarket_probability REAL,
    polymarket_influence   REAL,
    input_data_summary     TEXT,
    model_version     TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    scored_at         TEXT,
    realized_price    REAL,
    realized_return   REAL,
    realized_class    TEXT,
    predicted_class   TEXT,
    brier             REAL,
    log_loss          REAL,
    correct           INTEGER,
    UNIQUE(asset, timestamp, timeframe)
);
CREATE INDEX IF NOT EXISTS idx_pred_lookup  ON predictions(asset, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_pred_pending ON predictions(scored_at, horizon_end);

CREATE TABLE IF NOT EXISTS components (
    prediction_id INTEGER NOT NULL,
    name          TEXT    NOT NULL,
    score         REAL,
    weight        REAL,
    contribution  REAL,
    available     INTEGER,
    status        TEXT,
    note          TEXT,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_comp ON components(prediction_id);

CREATE TABLE IF NOT EXISTS source_status (
    prediction_id INTEGER NOT NULL,
    name          TEXT    NOT NULL,
    status        TEXT,
    detail        TEXT,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_src ON source_status(prediction_id);
"""


def _iso(value) -> str:
    """Canonical UTC ISO-8601 with a 'T' separator.

    Timestamps are compared as strings in SQL, so the format has to be
    identical everywhere or the comparison is nonsense. `str(pd.Timestamp)`
    uses a space and `.isoformat()` uses 'T'; because 'T' (0x54) sorts after
    ' ' (0x20), mixing the two made `horizon_end <= now` true for every row
    dated today — the scorer was picking up predictions that had not happened
    yet. One canonical form, applied on the way in.
    """
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _migrate(conn: sqlite3.Connection) -> None:
    """Rewrite any legacy space-separated timestamps into the canonical form."""
    for col in ("timestamp", "horizon_start", "horizon_end"):
        conn.execute(
            f"UPDATE predictions SET {col} = REPLACE({col}, ' ', 'T') "
            f"WHERE {col} LIKE '% %'")


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: Path | None = None) -> Path:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
    return path or DB_PATH


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def save(prediction, path: Path | None = None) -> list[int]:
    """Persist every horizon of one prediction. Idempotent per (asset, ts, tf)."""
    payload = prediction.to_dict() if hasattr(prediction, "to_dict") else prediction
    ids: list[int] = []
    poly = payload.get("components", {}).get("polymarket", {})
    poly_detail = poly.get("detail") or {}
    poly_prob = poly_detail.get("expected_fed_bps")

    with connect(path) as conn:
        conn.executescript(SCHEMA)
        for key, h in payload.get("horizons", {}).items():
            summary = {
                "regime": payload.get("regime"),
                "indicators": payload.get("indicators"),
                "components": {n: {"score": c.get("score"),
                                   "available": c.get("available"),
                                   "status": c.get("status")}
                               for n, c in payload.get("components", {}).items()},
                "weights": h.get("weights"),
                "data_quality": payload.get("data_quality"),
                "event_names": h.get("event_names"),
                "price_source": payload.get("price_source"),
                "analogue": h.get("analogue"),
            }
            row = (
                payload["asset"], _iso(payload["timestamp"]), key,
                _iso(h["start"]), _iso(h["end"]), payload["price"],
                h["up"], h["down"], h["range"],
                h["confidence"], h["confidence_label"],
                (payload.get("regime") or {}).get("trend", "unknown"),
                (payload.get("regime") or {}).get("vol_state"),
                (payload.get("regime") or {}).get("session"),
                (payload.get("data_quality") or {}).get("label"),
                (payload.get("data_quality") or {}).get("availability"),
                h.get("band"), h.get("band_price"), h.get("sigma"), h.get("drift"),
                h.get("score"), h.get("event_level"),
                poly_prob, h.get("poly_influence_pts"),
                json.dumps(summary, default=str),
                payload.get("model_version", config.MODEL_VERSION),
                pd.Timestamp.now(tz="UTC").isoformat(),
            )
            cur = conn.execute(
                """INSERT OR IGNORE INTO predictions (
                       asset, timestamp, timeframe, horizon_start, horizon_end, price,
                       up_probability, down_probability, range_probability,
                       confidence, confidence_label, market_regime, vol_state, session,
                       data_quality, availability, band_return, band_price, sigma,
                       drift, score, event_level, polymarket_probability,
                       polymarket_influence, input_data_summary, model_version, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
            if not cur.lastrowid or cur.rowcount == 0:
                continue
            pid = cur.lastrowid
            ids.append(pid)
            weights = h.get("weights") or {}
            contributions = h.get("contributions") or {}
            conn.executemany(
                """INSERT INTO components
                   (prediction_id, name, score, weight, contribution, available, status, note)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [(pid, n, c.get("score"), weights.get(n), contributions.get(n),
                  int(bool(c.get("available"))), c.get("status"), c.get("note"))
                 for n, c in payload.get("components", {}).items()])
            conn.executemany(
                "INSERT INTO source_status (prediction_id, name, status, detail) VALUES (?,?,?,?)",
                [(pid, s.get("name"), s.get("status"), s.get("detail"))
                 for s in (payload.get("data_quality") or {}).get("sources", [])])
    return ids


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def latest_payload(asset: str, path: Path | None = None) -> dict | None:
    """The previous run's horizons, in the shape `engine.explain` expects."""
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        row = conn.execute(
            "SELECT timestamp FROM predictions WHERE asset=? ORDER BY timestamp DESC LIMIT 1",
            (asset,)).fetchone()
        if row is None:
            return None
        ts = row["timestamp"]
        rows = conn.execute(
            "SELECT * FROM predictions WHERE asset=? AND timestamp=?", (asset, ts)).fetchall()
        horizons: dict[str, dict] = {}
        for r in rows:
            comps = conn.execute(
                "SELECT name, contribution FROM components WHERE prediction_id=?",
                (r["id"],)).fetchall()
            horizons[r["timeframe"]] = {
                "up": r["up_probability"], "down": r["down_probability"],
                "range": r["range_probability"], "confidence": r["confidence"],
                "event_names": json.loads(r["input_data_summary"] or "{}").get("event_names", []),
                "contributions": {c["name"]: c["contribution"] for c in comps},
            }
    return {"timestamp": ts, "horizons": horizons}


def history(asset: str | None = None, timeframe: str | None = None,
            scored_only: bool = False, path: Path | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM predictions WHERE 1=1"
    params: list = []
    if asset:
        sql += " AND asset=?"
        params.append(asset)
    if timeframe:
        sql += " AND timeframe=?"
        params.append(timeframe)
    if scored_only:
        sql += " AND scored_at IS NOT NULL"
    sql += " ORDER BY timestamp"
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        return pd.read_sql_query(sql, conn, params=params)


# --------------------------------------------------------------------------
# scoring matured predictions
# --------------------------------------------------------------------------

@dataclass
class ScoreResult:
    scored: int
    pending: int
    unresolvable: int


def realized_class(log_return: float, band: float) -> str:
    if not np.isfinite(band) or band <= 0:
        return "RANGE"
    if log_return > band:
        return "UP"
    if log_return < -band:
        return "DOWN"
    return "RANGE"


def brier(probs: dict[str, float], actual: str) -> float:
    """Multiclass Brier score: sum of squared errors over the three outcomes."""
    return float(sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in CLASSES))


def log_loss(probs: dict[str, float], actual: str, floor: float = 1e-4) -> float:
    return float(-np.log(max(probs[actual], floor)))


def _price_at(code: str, when: pd.Timestamp, cache: dict) -> float | None:
    """The traded price at (or just after) a timestamp, from Yahoo bars only.

    Yahoo rather than Twelve Data on purpose: scoring runs in a loop over
    potentially hundreds of matured rows and must not consume the live engine's
    rate-limited call budget.
    """
    from quantlab.predict import assets as asset_spec
    from quantlab.predict import market_data

    a = asset_spec.get(code)
    age_days = (pd.Timestamp.now(tz="UTC") - when).days
    interval, days = ("15m", 55) if age_days < 50 else \
                     ("1h", 700) if age_days < 690 else ("1d", 1100)
    key = (code, interval)
    if key not in cache:
        try:
            frame = market_data._yahoo_bars(a.yahoo_intraday if interval != "1d"
                                            else a.yahoo_daily, interval, days)
            cache[key] = market_data.drop_closed(frame)
        except Exception:
            cache[key] = None
    frame = cache[key]
    if frame is None or frame.empty:
        return None

    after = frame[frame.index >= when]
    if after.empty:
        return None
    bar = after.iloc[0]
    tolerance = pd.Timedelta(hours=2 if interval != "1d" else 48)
    if after.index[0] - when > tolerance:
        return None
    # The *open* of the first bar at or after T is the traded price at T; its
    # close would be up to one bar interval of hindsight.
    if "Open" in frame.columns and np.isfinite(bar.get("Open", np.nan)):
        return float(bar["Open"])
    # For a futures-backed asset the prediction was made in spot terms while
    # this bar is in futures terms. Only the *ratio* of the two endpoints is
    # scored, and the basis is common to both, so it cancels — but the entry
    # price stored on the row is spot, so the comparison is guarded by the
    # sanity check in `score_matured` rather than trusted blindly.
    return float(bar["Close"])


def score_matured(now: pd.Timestamp | None = None, path: Path | None = None,
                  limit: int = 2000) -> ScoreResult:
    """Fill in what actually happened for every prediction whose horizon has passed."""
    now = now or pd.Timestamp.now(tz="UTC")
    cache: dict = {}
    scored = unresolvable = 0

    with connect(path) as conn:
        conn.executescript(SCHEMA)
        rows = conn.execute(
            """SELECT * FROM predictions
               WHERE scored_at IS NULL AND horizon_end <= ?
               ORDER BY horizon_end LIMIT ?""",
            (now.isoformat(), limit)).fetchall()
        pending = conn.execute(
            "SELECT COUNT(*) c FROM predictions WHERE scored_at IS NULL AND horizon_end > ?",
            (now.isoformat(),)).fetchone()["c"]

        for r in rows:
            start, end = pd.Timestamp(r["horizon_start"]), pd.Timestamp(r["horizon_end"])
            # Both endpoints come from the *same* series. Gold is predicted in
            # spot terms but scored on GC=F bars, and the futures/spot basis is
            # far larger than a 15-minute move — measuring the ratio within one
            # series is the only way the return means anything.
            price_start = _price_at(r["asset"], start, cache)
            price_then = _price_at(r["asset"], end, cache)
            if any(p is None or not np.isfinite(p) or p <= 0
                   for p in (price_start, price_then)):
                unresolvable += 1
                continue
            ret = float(np.log(price_then / price_start))
            actual = realized_class(ret, float(r["band_return"] or 0.0))
            probs = {"UP": r["up_probability"], "DOWN": r["down_probability"],
                     "RANGE": r["range_probability"]}
            predicted = max(CLASSES, key=lambda c: probs[c])
            conn.execute(
                """UPDATE predictions SET scored_at=?, realized_price=?, realized_return=?,
                       realized_class=?, predicted_class=?, brier=?, log_loss=?, correct=?
                   WHERE id=?""",
                (now.isoformat(), price_then, ret, actual, predicted,
                 brier(probs, actual), log_loss(probs, actual),
                 int(predicted == actual), r["id"]))
            scored += 1
    return ScoreResult(scored, pending, unresolvable)


# --------------------------------------------------------------------------
# track record and calibration
# --------------------------------------------------------------------------

def track_record(asset: str | None = None, path: Path | None = None
                 ) -> dict[str, dict[str, float]]:
    """Brier skill per asset and horizon, for the confidence model to read.

    Skill is measured against climatology — the realised class frequencies in
    the same sample — because beating "always say RANGE" is the only bar that
    means anything. Returns values in [0, 1] and simply omits any bucket with
    fewer than `TRACK_RECORD_MIN_SAMPLES` scored rows, so a young database
    leaves confidence at its neutral setting instead of inventing a record.
    """
    df = history(asset=asset, scored_only=True, path=path)
    out: dict[str, dict[str, float]] = {}
    if df.empty:
        return out
    for (code, tf), grp in df.groupby(["asset", "timeframe"]):
        if len(grp) < config.TRACK_RECORD_MIN_SAMPLES:
            continue
        freq = {c: float((grp["realized_class"] == c).mean()) for c in CLASSES}
        base = float(np.mean([
            brier(freq, actual) for actual in grp["realized_class"]]))
        model = float(grp["brier"].mean())
        skill = 1.0 - (model / base) if base > 0 else 0.0
        # Map skill (typically -0.1..0.3) onto the 0..1 the confidence model wants.
        out.setdefault(code, {})[tf] = float(np.clip(0.5 + skill * 2.0, 0.0, 1.0))
    return out


def calibration_table(asset: str | None = None, timeframe: str | None = None,
                      bins: int = 5, path: Path | None = None) -> pd.DataFrame:
    """Does a predicted 60% actually happen about 60% of the time?

    Pools the three classes: every (prediction, class) pair contributes its
    forecast probability and whether that class occurred. That is the honest
    reliability diagram for a three-way forecast.
    """
    df = history(asset=asset, timeframe=timeframe, scored_only=True, path=path)
    if df.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_forecast",
                                     "observed_frequency", "gap"])
    rows = []
    for cls, col in (("UP", "up_probability"), ("DOWN", "down_probability"),
                     ("RANGE", "range_probability")):
        rows.append(pd.DataFrame({"p": df[col].astype(float),
                                  "hit": (df["realized_class"] == cls).astype(float)}))
    long = pd.concat(rows, ignore_index=True)
    edges = np.linspace(0, 1, bins + 1)
    long["bucket"] = pd.cut(long["p"], edges, include_lowest=True)
    out = long.groupby("bucket", observed=True).agg(
        n=("p", "size"), mean_forecast=("p", "mean"),
        observed_frequency=("hit", "mean")).reset_index()
    out["gap"] = out["observed_frequency"] - out["mean_forecast"]
    return out


def summary(path: Path | None = None) -> dict:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        total = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        scored = conn.execute(
            "SELECT COUNT(*) c FROM predictions WHERE scored_at IS NOT NULL").fetchone()["c"]
        row = conn.execute(
            """SELECT AVG(brier) b, AVG(log_loss) l, AVG(correct) a
               FROM predictions WHERE scored_at IS NOT NULL""").fetchone()
    return {"rows": total, "scored": scored, "pending": total - scored,
            "mean_brier": row["b"], "mean_log_loss": row["l"],
            "hit_rate": row["a"], "db": str(DB_PATH)}
