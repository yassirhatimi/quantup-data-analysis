"""Yahoo GC=F wave analysis. USD/oz, NOT broker pips or XAU/USD spot.

Run: python gold_waves.py --download
Then rerun offline: python gold_waves.py
No trading, brokerage, or order-submission code is included.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "gold_waves.sqlite"
SYMBOL = "GC=F"
THRESHOLDS = (25.0, 50.0, 100.0)
MOVES = (100.0, 200.0, 300.0, 400.0)
PULLBACK = 100.0
HORIZON_DAYS = 60
SPECS = {"1d": "5y", "1h": "700d", "15m": "59d"}


def iso(value):
    return pd.Timestamp(value).isoformat()


def observation_times(frame, timeframe):
    """Earliest safe bar-use timestamps; daily uses conservative label +24h."""
    duration = pd.Timedelta(days=1) if timeframe == "1d" else pd.Timedelta(hours=1) if timeframe == "1h" else pd.Timedelta(minutes=15)
    return frame.index + duration


def clean_bars(frame, timeframe, now):
    """Retain raw records with quality flags; no padding/interpolation."""
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    source_index = pd.DatetimeIndex(df.index)
    if source_index.tz is None:
        raise ValueError("Provider timestamps have no timezone; refusing to guess")
    df["source_time_label"] = source_index.astype(str)
    df.index = source_index.tz_convert("UTC")
    df = df.sort_index()
    duplicates = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")].copy()
    values = df[["open", "high", "low", "close"]]
    invalid = ~np.isfinite(values).all(axis=1) | (values <= 0).any(axis=1)
    invalid |= (df.high < df[["open", "low", "close"]].max(axis=1) - 1e-6)
    invalid |= (df.low > df[["open", "high", "close"]].min(axis=1) + 1e-6)
    df["quality_flag"] = "ok"
    df.loc[invalid, "quality_flag"] = "invalid_ohlc"
    flat_zero = (df.high == df.low) & df.volume.fillna(0).eq(0)
    # Exclude inert, zero-volume bars. Variable zero-volume bars remain recorded.
    df.loc[~invalid & flat_zero, "quality_flag"] = "flat_zero_volume"
    if timeframe == "1d":
        incomplete = df.index.tz_convert(source_index.tz).date >= now.tz_convert(source_index.tz).date()
    else:
        duration = pd.Timedelta(hours=1) if timeframe == "1h" else pd.Timedelta(minutes=15)
        incomplete = df.index + duration > now
    df.loc[~invalid & incomplete, "quality_flag"] = "incomplete_bar"
    df["timeframe"] = timeframe
    df["symbol"] = SYMBOL
    df["timestamp_utc"] = [iso(t) for t in df.index]
    good = df[df.quality_flag == "ok"].copy()
    step = pd.Timedelta(days=1) if timeframe == "1d" else pd.Timedelta(hours=1) if timeframe == "1h" else pd.Timedelta(minutes=15)
    gaps = good.index.to_series().diff().gt(step * 1.5)
    info = dict(timeframe=timeframe, requested_period=SPECS[timeframe], raw_rows=len(df),
                usable_rows=len(good), duplicate_timestamps_removed=duplicates,
                invalid_ohlc=int(invalid.sum()), flat_zero_volume=int(flat_zero.sum()),
                incomplete_bars=int(np.asarray(incomplete).sum()),
                missing_ohlc=int(values.isna().any(axis=1).sum()),
                variable_zero_volume=int(((df.volume.fillna(0)==0)&~flat_zero&~invalid).sum()),
                gaps_including_sessions=int(gaps.sum()),
                first_utc=iso(good.index[0]) if len(good) else None,
                last_utc=iso(good.index[-1]) if len(good) else None,
                fetched_utc=iso(now), status="ok" if len(good) else "no_usable_bars")
    return df, info


def download_data(conn):
    import yfinance as yf
    cache = ROOT / ".yf_cache"
    cache.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    now = pd.Timestamp.now(tz="UTC")
    all_raw, quality = [], []
    for tf, period in SPECS.items():
        print(f"Downloading {SYMBOL} {tf} {period}", flush=True)
        try:
            raw = yf.download(SYMBOL, period=period, interval=tf, auto_adjust=False,
                              back_adjust=False, repair=False, keepna=True,
                              ignore_tz=False, threads=False, progress=False, timeout=25)
            if raw is None or raw.empty:
                raise ValueError("Yahoo returned no bars")
            frame, info = clean_bars(raw, tf, now)
            all_raw.append(frame.reset_index(drop=True))
            quality.append(info)
            print(json.dumps(info), flush=True)
        except Exception as exc:
            quality.append(dict(timeframe=tf, requested_period=period, status="download_failed",
                                error_type=type(exc).__name__, fetched_utc=iso(now)))
            print(f"{tf}: unavailable ({type(exc).__name__})", flush=True)
    if not all_raw:
        raise RuntimeError("No Yahoo history downloaded; no fabricated fallback data")
    raw = pd.concat(all_raw, ignore_index=True)
    raw.to_sql("price_bars", conn, if_exists="replace", index=False)
    pd.DataFrame(quality).to_sql("data_quality", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS bar_key ON price_bars(symbol,timeframe,timestamp_utc)")
    conn.commit()


def load_frames(conn):
    df = pd.read_sql_query("SELECT * FROM price_bars WHERE quality_flag='ok' ORDER BY timestamp_utc", conn)
    frames = {}
    for tf, g in df.groupby("timeframe"):
        g = g.copy()
        g.index = pd.to_datetime(g.timestamp_utc, utc=True)
        frames[tf] = g.sort_index()
    return frames


def refresh_saved_quality(conn):
    """Reconcile quality counts from retained raw bars, without re-downloading."""
    quality=pd.read_sql_query('SELECT * FROM data_quality',conn)
    raw=pd.read_sql_query('SELECT timeframe,open,high,low,close,volume FROM price_bars',conn)
    for tf,g in raw.groupby('timeframe'):
        missing=g[['open','high','low','close']].isna().any(axis=1)
        finite=np.isfinite(g[['open','high','low','close']]).all(axis=1)
        variable=(g.volume.fillna(0)==0)&(g.high!=g.low)&finite
        quality.loc[quality.timeframe==tf,'missing_ohlc']=int(missing.sum())
        quality.loc[quality.timeframe==tf,'variable_zero_volume']=int(variable.sum())
    quality.to_sql('data_quality',conn,if_exists='replace',index=False)


def confirmed_pivots(close, reversal=50.0):
    """Directional-change pivots, known ONLY at confirmation timestamp.

    Uses closes, not ambiguous within-bar high/low ordering. First pivot is
    boundary-censored. Ongoing final extreme is deliberately not a pivot.
    """
    if reversal <= 0:
        raise ValueError("reversal must be positive")
    prices = np.asarray(close, dtype=float)
    if not len(prices):
        return []
    if not np.isfinite(prices).all():
        raise ValueError("close contains invalid prices")
    lo = hi = extreme = 0
    state = 0
    pivots = []
    def add(idx, kind, confirmed):
        pivots.append(dict(pivot_index=int(idx), confirmed_index=int(confirmed),
                           price=float(prices[idx]), kind=kind,
                           boundary_censored=int(not pivots)))
    for i, price in enumerate(prices):
        if state == 0:
            if price < prices[lo]: lo = i
            if price > prices[hi]: hi = i
            if price - prices[lo] >= reversal:
                add(lo, "low", i); state = 1; extreme = i
            elif prices[hi] - price >= reversal:
                add(hi, "high", i); state = -1; extreme = i
        elif state == 1:
            if price >= prices[extreme]: extreme = i
            elif prices[extreme] - price >= reversal:
                add(extreme, "high", i); state = -1; extreme = i
        else:
            if price <= prices[extreme]: extreme = i
            elif price - prices[extreme] >= reversal:
                add(extreme, "low", i); state = 1; extreme = i
    return pivots


def wave_tables(frames):
    pivots_out, swings_out, summary_out, zones = [], [], [], []
    for tf, frame in frames.items():
        available = observation_times(frame, tf)
        for threshold in THRESHOLDS:
            pivots = confirmed_pivots(frame.close, threshold)
            for p in pivots:
                pivots_out.append(dict(timeframe=tf, reversal_usd=threshold, **p,
                    pivot_utc=iso(frame.index[p["pivot_index"]]),
                    confirmed_utc=iso(available[p["confirmed_index"]])))
            for i in range(1, len(pivots)):
                a, b = pivots[i-1], pivots[i]
                if a["boundary_censored"]: continue
                correction = abs(pivots[i+1]["price"]-b["price"]) if i+1 < len(pivots) else None
                size = abs(b["price"]-a["price"])
                swings_out.append(dict(timeframe=tf, reversal_usd=threshold,
                    direction="up" if b["price"]>a["price"] else "down",
                    start_utc=iso(frame.index[a["pivot_index"]]),
                    end_utc=iso(frame.index[b["pivot_index"]]),
                    known_utc=iso(available[b["confirmed_index"]]),
                    start_price=a["price"], end_price=b["price"], move_usd=size,
                    duration_hours=(frame.index[b["pivot_index"]]-frame.index[a["pivot_index"]]).total_seconds()/3600,
                    next_countermove_usd=correction,
                    next_countermove_pct=correction/size*100 if correction is not None else None))
    swings = pd.DataFrame(swings_out)
    for (tf, threshold, direction), g in swings.groupby(["timeframe","reversal_usd","direction"]):
        bins = (np.floor(g.move_usd/50)*50).astype(int)
        counts = bins.value_counts()
        modes = sorted(counts[counts==counts.max()].index.tolist())
        large = g[g.move_usd>=300]
        counter = large.next_countermove_usd.dropna()
        summary_out.append(dict(timeframe=tf,reversal_usd=threshold,direction=direction,
            completed_swings=len(g),median_move_usd=float(g.move_usd.median()),
            p25_move_usd=float(g.move_usd.quantile(.25)),p75_move_usd=float(g.move_usd.quantile(.75)),
            modal_move_bins="; ".join(f"{x}-{x+50}" for x in modes),modal_bin_count=int(counts.max()),
            moves_at_least_300=len(large),large_moves_with_known_countermove=len(counter),
            median_countermove_after_300=float(counter.median()) if len(counter) else None))
    pp = pd.DataFrame(pivots_out)
    for (tf, kind), g in pp[(pp.reversal_usd==50)&(pp.boundary_censored==0)].groupby(["timeframe","kind"]):
        g = g.copy(); g["zone_low"] = np.floor(g.price/50)*50
        for low, group in g.groupby("zone_low"):
            zones.append(dict(timeframe=tf,kind=kind,zone_low=low,zone_high=low+50,
                              confirmed_turns=len(group),last_pivot_utc=group.pivot_utc.max()))
    return pp, swings, pd.DataFrame(summary_out), pd.DataFrame(zones)


def impulse_events(close, times, impulse=300.0, pullback=100.0, seed_reversal=50.0, horizon_days=60):
    """Non-overlapping event study within one interval/threshold configuration.

    A seed turn becomes known after a $50 counter-move. Trigger at first CLOSE
    at least impulse dollars from the known pivot. Track a running extreme.
    Pullback = first close >= pullback dollars against that extreme.
    Continuation = later close beyond the frozen pre-pullback extreme.
    Full retracement = return to origin before continuation. Otherwise censor.
    No pre-trigger high/low or future-pivot knowledge is used for a trigger.
    """
    if not (impulse > seed_reversal > 0 and pullback > 0):
        raise ValueError("Require impulse > seed_reversal > 0 and pullback > 0")
    p = np.asarray(close, dtype=float)
    t = pd.DatetimeIndex(times)
    if len(p) != len(t) or not t.is_monotonic_increasing or not np.isfinite(p).all():
        raise ValueError("Need finite prices and ascending aligned timestamps")
    if not len(p): return []
    events = []
    state, lo, hi, extreme, anchor = 0, 0, 0, 0, 0
    event = None
    for i, price in enumerate(p):
        if event is not None:
            deadline = pd.Timestamp(event["trigger_utc"]) + pd.Timedelta(days=horizon_days)
            if t[i] > deadline:
                event["outcome"] = "window_expired"
                event["evaluation_end_utc"] = iso(t[i-1])
                events.append(event); event = None
                state=0; lo=hi=extreme=anchor=i
                continue
            side = event["sign"]
            event["evaluation_end_utc"] = iso(t[i])
            if event["pullback_utc"] is None:
                if side*(price-event["extreme_price"])>0:
                    event["extreme_price"]=float(price); event["extreme_utc"]=iso(t[i])
                retrace = side*(event["extreme_price"]-price)
                if retrace>=pullback:
                    event["pullback_utc"]=iso(t[i]); event["pullback_price"]=float(price)
                    event["move_before_pullback_usd"]=side*(event["extreme_price"]-event["origin_price"])
                    event["max_pullback_usd"]=float(retrace)
            else:
                retrace = side*(event["extreme_price"]-price)
                event["max_pullback_usd"]=max(event["max_pullback_usd"],float(retrace))
                if side*(price-event["extreme_price"])>0:
                    event["outcome"]="continued"
            if event["pullback_utc"] is not None and side*(price-event["origin_price"])<=0:
                event["outcome"]="fully_retraced"
            if event["outcome"] in ("continued","fully_retraced"):
                event["outcome_price"]=float(price)
                events.append(event); event=None
                state=0; lo=hi=extreme=anchor=i
            continue
        if state==0:
            if price<p[lo]: lo=i
            if price>p[hi]: hi=i
            if price-p[lo]>=seed_reversal: state=1;anchor=lo;extreme=i
            elif p[hi]-price>=seed_reversal: state=-1;anchor=hi;extreme=i
        else:
            if state*(price-p[extreme])>=0: extreme=i
            elif state*(p[extreme]-price)>=seed_reversal:
                anchor=extreme; state=-state;extreme=i
        if state and state*(price-p[anchor])>=impulse:
            event=dict(direction="up" if state==1 else "down", sign=state,
                impulse_usd=impulse,pullback_threshold_usd=pullback,seed_reversal_usd=seed_reversal,
                horizon_days=horizon_days,origin_utc=iso(t[anchor]),origin_price=float(p[anchor]),
                trigger_utc=iso(t[i]),trigger_price=float(price),extreme_utc=iso(t[i]),extreme_price=float(price),
                pullback_utc=None,pullback_price=None,move_before_pullback_usd=None,max_pullback_usd=None,
                outcome="pending",outcome_price=None,evaluation_end_utc=iso(t[i]))
    if event is not None:
        event["outcome"]="end_of_data"
        events.append(event)
    return events


def event_tables(frames):
    records=[]
    for tf, frame in frames.items():
        for move in MOVES:
            records.extend(dict(timeframe=tf, **e) for e in impulse_events(frame.close,observation_times(frame,tf),impulse=move))
    events=pd.DataFrame(records)
    if events.empty: raise ValueError("No impulse events available")
    summaries=[]
    for tf in frames:
        for move in MOVES:
            for direction in ("up","down"):
                g=events[(events.timeframe==tf)&(events.impulse_usd==move)&(events.direction==direction)]
                continued=int(g.outcome.eq("continued").sum()); full=int(g.outcome.eq("fully_retraced").sum())
                resolved=continued+full
                pulled=g[g.pullback_utc.notna()]
                cont=g[g.outcome.eq("continued")]
                summaries.append(dict(timeframe=tf,impulse_usd=move,direction=direction,events=len(g),
                    observed_pullbacks=len(pulled),continued=continued,fully_retraced=full,resolved=resolved,
                    censored=len(g)-resolved,window_expired=int(g.outcome.eq("window_expired").sum()),
                    end_of_data=int(g.outcome.eq("end_of_data").sum()),
                    continuation_pct_resolved=100*continued/resolved if resolved else None,
                    continuation_lower_bound_all=100*continued/len(g) if len(g) else None,
                    continuation_upper_bound_all=100*(continued+len(g)-resolved)/len(g) if len(g) else None,
                    median_move_before_pullback=float(pulled.move_before_pullback_usd.median()) if len(pulled) else None,
                    median_pullback_before_continuation=float(cont.max_pullback_usd.median()) if len(cont) else None))
    return events,pd.DataFrame(summaries)


def pyramid_analysis(frames, as_of=None, reversal_usd=50.0):
    """Daily/hour/15m structure pyramid, using only bars known by as_of.

    Returns one record per interval. These are structure labels, NOT votes,
    independent confirmations, forecasts, or pyramiding/order instructions.
    Pass as_of in UTC to prevent future-bar leakage in historical use.
    """
    rows=[]
    cutoff=pd.Timestamp(as_of) if as_of is not None else None
    if cutoff is not None and cutoff.tzinfo is None: raise ValueError("as_of needs a timezone")
    for tf in ("1d","1h","15m"):
        if tf not in frames: continue
        f=frames[tf]
        if cutoff is not None: f=f[observation_times(f,tf)<=cutoff]
        if f.empty: continue
        piv=confirmed_pivots(f.close,reversal_usd)
        highs=[p for p in piv if p["kind"]=="high" and not p["boundary_censored"]]
        lows=[p for p in piv if p["kind"]=="low" and not p["boundary_censored"]]
        structure="insufficient_pivots"
        if len(highs)>=2 and len(lows)>=2:
            up=highs[-1]["price"]>highs[-2]["price"] and lows[-1]["price"]>lows[-2]["price"]
            down=highs[-1]["price"]<highs[-2]["price"] and lows[-1]["price"]<lows[-2]["price"]
            structure="higher_highs_higher_lows" if up else "lower_highs_lower_lows" if down else "mixed"
        last=float(f.close.iloc[-1]); last_p=piv[-1] if piv else None
        available=observation_times(f,tf)
        active=f.close.iloc[last_p['pivot_index']:] if last_p else f.close
        broken="none"
        if last_p and last_p['kind']=='high' and lows and active.min()<lows[-1]['price']:
            broken="previous_confirmed_low_broken"
        elif last_p and last_p['kind']=='low' and highs and active.max()>highs[-1]['price']:
            broken="previous_confirmed_high_broken"
        rows.append(dict(timeframe=tf,last_bar_utc=iso(f.index[-1]),last_close=last,
            last_bar_available_utc=iso(available[-1]),structure_break=broken,
            reversal_usd=reversal_usd,structure=structure,
            last_confirmed_high=highs[-1]["price"] if highs else None,
            high_known_utc=iso(available[highs[-1]["confirmed_index"]]) if highs else None,
            last_confirmed_low=lows[-1]["price"] if lows else None,
            low_known_utc=iso(available[lows[-1]["confirmed_index"]]) if lows else None,
            ongoing_leg="up" if last_p and last_p["kind"]=="low" else "down" if last_p else "unknown",
            move_from_last_pivot=abs(last-last_p["price"]) if last_p else None))
    return pd.DataFrame(rows)


def md_table(df):
    def val(v):
        if pd.isna(v): return "—"
        if isinstance(v, (float,np.floating)): return f"{v:,.1f}"
        return str(v).replace("|","/").replace("\n"," ")
    return "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n" + "\n".join("| "+" | ".join(map(val,r))+" |" for r in df.itertuples(index=False,name=None))


def write_report(conn, summary, event_summary, pyramid):
    quality=pd.read_sql_query("SELECT * FROM data_quality",conn)
    focus=event_summary[event_summary.impulse_usd==300]
    focus_cols=["timeframe", "direction", "events", "observed_pullbacks", "continued",
                "fully_retraced", "censored", "median_move_before_pullback",
                "median_pullback_before_continuation"]
    focus_display=focus[focus_cols]
    base=summary[summary.reversal_usd==50][["timeframe","direction","completed_swings","median_move_usd","modal_move_bins","modal_bin_count"]]
    pdisplay=pyramid[["timeframe","last_bar_utc","last_close","structure","last_confirmed_high","last_confirmed_low","ongoing_leg","structure_break"]].copy()

    z=pd.read_sql_query("SELECT * FROM reversal_zones ORDER BY timeframe,kind,confirmed_turns DESC,zone_low",conn)
    top_zones=z.groupby(['timeframe','kind'],sort=False).head(3)
    top_zones=top_zones[['timeframe','kind','zone_low','zone_high','confirmed_turns']]
    qcols=['timeframe','usable_rows','first_utc','last_utc','invalid_ohlc','incomplete_bars']
    qdisplay=quality.reindex(columns=qcols)
    text="""# Gold waves and corrections

Historical snapshot, not live market data. Source: Yahoo Finance GC=F via yfinance.
Units: USD per troy ounce. These are futures, not spot XAU/USD; no joining to
spot outcomes in goldprice.csv was performed.

## What followed a $300 move?

A pivot becomes known after a $50 reversal; the event starts at the first close
reaching a $300 move in the new direction. Track the subsequent extreme until a
$100 correction. Continuation means a closing break beyond that extreme after
the correction; full retracement erases the original move before continuation.
Observation ends after 60 calendar days. Events do not overlap within a timeframe
and configuration, but may overlap across them. Censored events include both
end-of-data and expired observation windows and are not classified as wins or losses.

"""+md_table(focus_display)+"""

## Completed wave sizes

A $50 closing reversal confirms a completed wave. Bins are $50 wide and exclude
the upper bound. All tied modal bins are shown. These are statistics conditional
on completion, not reversal probabilities at a given price. Sensitivity settings
of $25, $50 and $100 are available in the database.

"""+md_table(base)+"""

## Frequently observed reversal zones

Top three bins by confirmed turns for each timeframe and high/low type. Ties are
ordered by increasing price, so other tied bins may be omitted. Counts are not
adjusted for time spent in a zone and do not establish support or resistance.

"""+md_table(top_zones)+"""

## Structure across timeframes

Pivots require a $50 closing reversal. Coverage and latest observations differ;
the timeframes are not independent votes. The ongoing leg is incomplete and a
previous confirmed structure can coexist with a break by the current leg.

"""+md_table(pdisplay)+"""

## Coverage and quality

"""+md_table(qdisplay)+"""

Invalid rows in this saved snapshot have missing source prices. Raw rows remain
in price_bars for audit; missing prices were not filled or treated as zero moves.
Full quality information is in data_quality.

## Limitations

- Closing prices do not identify the order of highs and lows within a bar.
- Invalid, incomplete and flat zero-volume bars are excluded; gaps are not filled.
- Gaps can reflect closures and holidays rather than missing observations.
- GC=F can contain contract-roll effects; no roll-gap adjustment was applied.
- A fixed dollar move changes in percentage significance as gold prices change.
- Coverage differs; common_window tables provide a separate matched-period view.
- Reversal-zone counts do not adjust for price exposure time in each zone.
- Event-summary rates use resolved cases; inspect censored counts and denominators.
- Wave findings are exploratory, not out-of-sample trading performance; costs are absent.
- Daily timestamps are provider session labels, not live quote timestamps.
- Intraday signals use bar-end availability; daily uses label plus 24 hours as a
  conservative availability time, not an official settlement timestamp.

## Reproduction

Database: gold_waves.sqlite. Code: gold_waves.py. Queries: queries.sql.
Audit notebook: inspect_gold_waves.ipynb. Recalculate locally with
`python gold_waves.py`; refresh permitted source data with `--download`.
No order-submission code is included. Publication rights require separate review.
"""
    (ROOT/"results.md").write_text(text,encoding="utf-8")
    result=dict(quality=quality.replace({np.nan:None}).to_dict("records"),
                focus_300=focus.replace({np.nan:None}).to_dict("records"),
                wave_summary_50=base.replace({np.nan:None}).to_dict("records"),
                top_reversal_zones=top_zones.to_dict("records"),
                pyramid=pyramid.replace({np.nan:None}).to_dict("records"))
    (ROOT/"analysis_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")


def persist_tables(conn, frames, prefix=""):
    piv,swings,summary,zones=wave_tables(frames)
    events,evsummary=event_tables(frames)
    for name,df in [("pivots",piv),("swings",swings),("wave_summary",summary),("reversal_zones",zones),("events",events),("event_summary",evsummary)]:
        df.to_sql(prefix+name,conn,if_exists="replace",index=False)
    return summary,evsummary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    if not args.download and not DB_PATH.exists(): parser.error("Run --download first")
    with sqlite3.connect(DB_PATH) as conn:
        if args.download: download_data(conn)
        refresh_saved_quality(conn)
        frames=load_frames(conn)
        if not frames: raise RuntimeError("No usable price data")
        summary,esummary=persist_tables(conn,frames)
        pyramid=pyramid_analysis(frames)
        pyramid.to_sql("pyramid",conn,if_exists="replace",index=False)
        start=max(f.index.min() for f in frames.values())
        end=min(f.index.max() for f in frames.values())
        common={tf:f[(f.index>=start)&(f.index<=end)] for tf,f in frames.items()}
        persist_tables(conn,common,prefix="common_window_")
        pd.DataFrame([dict(common_start_utc=iso(start),common_end_utc=iso(end),
            source="Yahoo Finance via yfinance",symbol=SYMBOL,price_unit="USD per troy ounce",
            analysis_price="bar close",seed_reversal_usd=50,pullback_usd=PULLBACK,
            horizon_days=HORIZON_DAYS,generated_utc=iso(pd.Timestamp.now(tz="UTC")))]).to_sql("analysis_metadata",conn,if_exists="replace",index=False)
        write_report(conn,summary,esummary,pyramid)
        assert conn.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
        print(esummary[esummary.impulse_usd==300].to_string(index=False))
        print(pyramid.to_string(index=False))
        print("Saved",DB_PATH)


if __name__=="__main__": main()
