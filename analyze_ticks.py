#!/usr/bin/env python3
"""
=============================================================
  GaganEA Tick Data Analyzer
  Analyzes large XAUUSD tick CSV files chunk by chunk
  Format: datetime, bid, ask
  Output: JSON report for EA building
=============================================================
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONFIG — change this to your CSV file path
# ─────────────────────────────────────────────
CSV_FILE   = "ticks.csv"   # <-- PUT YOUR FILE NAME/PATH HERE
CHUNK_SIZE = 500_000       # rows per chunk (~50MB per chunk, safe for any PC)
OUTPUT     = "analysis_report.json"

print("=" * 60)
print("  GaganEA Tick Analyzer — XAUUSD")
print("=" * 60)

if not os.path.exists(CSV_FILE):
    print(f"\nERROR: File '{CSV_FILE}' not found.")
    print("Please edit CSV_FILE at the top of this script.")
    sys.exit(1)

file_size_gb = os.path.getsize(CSV_FILE) / (1024**3)
print(f"\nFile : {CSV_FILE}")
print(f"Size : {file_size_gb:.2f} GB")
print(f"Chunk: {CHUNK_SIZE:,} rows per chunk")
print("\nStarting analysis — please wait...\n")


# ─────────────────────────────────────────────
#  ACCUMULATORS
# ─────────────────────────────────────────────
total_ticks      = 0
first_timestamp  = None
last_timestamp   = None

# Hourly stats: hour (0-23) -> list of mid-price changes
hourly_returns   = defaultdict(list)

# Day-of-week stats: 0=Mon..6=Sun -> list of returns
dow_returns      = defaultdict(list)

# Session stats
session_returns  = defaultdict(list)

# Spread stats
spread_by_hour   = defaultdict(list)

# Volatility (1-min buckets)
# We'll track high/low per minute to get avg range
min_highs        = defaultdict(list)  # key = (date, hour, minute)
min_lows         = defaultdict(list)

# Mean reversion: track how far price moves from a rolling baseline
reversion_moves  = []

# Momentum: consecutive tick direction runs
momentum_runs    = []

# Price levels (for S/R clustering)
price_levels     = []

# Rolling window for momentum
prev_mid         = None
run_len          = 0
run_dir          = 0  # +1 or -1

chunk_num        = 0


# ─────────────────────────────────────────────
#  CHUNK PROCESSING
# ─────────────────────────────────────────────
reader = pd.read_csv(
    CSV_FILE,
    header=None,
    names=["ts", "bid", "ask"],
    chunksize=CHUNK_SIZE,
    parse_dates=["ts"],
    dtype={"bid": np.float32, "ask": np.float32}
)

for chunk in reader:
    chunk_num += 1
    print(f"  Processing chunk {chunk_num} ({len(chunk):,} rows)...", end="\r")

    # Drop bad rows
    chunk.dropna(inplace=True)
    chunk = chunk[chunk["bid"] > 0]
    chunk = chunk[chunk["ask"] > chunk["bid"]]

    if len(chunk) == 0:
        continue

    # Timestamps
    if first_timestamp is None:
        first_timestamp = chunk["ts"].iloc[0]
    last_timestamp = chunk["ts"].iloc[-1]
    total_ticks += len(chunk)

    # Mid price
    chunk["mid"]    = (chunk["bid"] + chunk["ask"]) / 2.0
    chunk["spread"] = chunk["ask"] - chunk["bid"]
    chunk["hour"]   = chunk["ts"].dt.hour
    chunk["dow"]    = chunk["ts"].dt.dayofweek   # 0=Mon
    chunk["minute"] = chunk["ts"].dt.minute
    chunk["date"]   = chunk["ts"].dt.date

    # Tick-level return (mid price change)
    chunk["ret"] = chunk["mid"].diff().fillna(0)


    # ── Hourly return bias ─────────────────────────────────────
    for hour, grp in chunk.groupby("hour"):
        if len(grp) > 1:
            net = float(grp["mid"].iloc[-1] - grp["mid"].iloc[0])
            hourly_returns[int(hour)].append(net)

    # ── Day-of-week bias ───────────────────────────────────────
    for dow, grp in chunk.groupby("dow"):
        if len(grp) > 1:
            net = float(grp["mid"].iloc[-1] - grp["mid"].iloc[0])
            dow_returns[int(dow)].append(net)

    # ── Session bias ───────────────────────────────────────────
    def get_session(h):
        if 1  <= h <= 8:  return "Asian"
        if 8  <= h <= 16: return "London"
        if 13 <= h <= 21: return "NewYork"
        return "Off"

    chunk["session"] = chunk["hour"].apply(get_session)
    for sess, grp in chunk.groupby("session"):
        if len(grp) > 1:
            net = float(grp["mid"].iloc[-1] - grp["mid"].iloc[0])
            session_returns[sess].append(net)

    # ── Spread by hour ─────────────────────────────────────────
    for hour, grp in chunk.groupby("hour"):
        spread_by_hour[int(hour)].append(float(grp["spread"].mean()))

    # ── 1-Minute range (volatility) ────────────────────────────
    chunk["min_key"] = chunk["ts"].dt.floor("1min")
    for mk, grp in chunk.groupby("min_key"):
        key = str(mk)
        min_highs[key].append(float(grp["mid"].max()))
        min_lows[key].append(float(grp["mid"].min()))

    # ── Price level sampling (every 1000th tick for S/R) ───────
    sampled = chunk["mid"].iloc[::1000].tolist()
    price_levels.extend(sampled)

    # ── Momentum runs ──────────────────────────────────────────
    mids = chunk["mid"].tolist()
    for m in mids:
        if prev_mid is None:
            prev_mid = m
            continue
        d = 1 if m > prev_mid else (-1 if m < prev_mid else 0)
        if d == 0:
            prev_mid = m
            continue
        if d == run_dir:
            run_len += 1
        else:
            if run_len > 0:
                momentum_runs.append(run_len)
            run_dir = d
            run_len = 1
        prev_mid = m

print(f"\n  Done! Processed {chunk_num} chunks, {total_ticks:,} total ticks.\n")


# ─────────────────────────────────────────────
#  COMPUTE STATISTICS
# ─────────────────────────────────────────────
print("Computing statistics...")

def summarize(data_dict):
    result = {}
    for k, vals in data_dict.items():
        arr = np.array(vals, dtype=np.float32)
        result[str(k)] = {
            "mean":   round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std":    round(float(np.std(arr)), 4),
            "positive_pct": round(float(np.mean(arr > 0) * 100), 2),
            "samples": len(arr)
        }
    return result

# Hourly bias
hourly_stats = summarize(hourly_returns)

# Find best/worst hours
best_buy_hours  = sorted(hourly_stats.items(), key=lambda x: x[1]["mean"], reverse=True)[:5]
best_sell_hours = sorted(hourly_stats.items(), key=lambda x: x[1]["mean"])[:5]

# DOW bias
dow_stats = summarize(dow_returns)
dow_names = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
dow_stats_named = {dow_names.get(int(k), k): v for k, v in dow_stats.items()}

# Session bias
session_stats = summarize(session_returns)

# Spread stats
spread_stats = {}
for hour, vals in spread_by_hour.items():
    arr = np.array(vals)
    spread_stats[str(hour)] = {
        "avg_spread": round(float(np.mean(arr)), 4),
        "min_spread": round(float(np.min(arr)), 4),
        "max_spread": round(float(np.max(arr)), 4),
    }

best_spread_hours  = sorted(spread_stats.items(), key=lambda x: x[1]["avg_spread"])[:5]
worst_spread_hours = sorted(spread_stats.items(), key=lambda x: x[1]["avg_spread"], reverse=True)[:5]


# 1-Minute volatility ranges
min_ranges = []
for key in min_highs:
    if key in min_lows:
        h = max(min_highs[key])
        l = min(min_lows[key])
        min_ranges.append(h - l)

min_ranges = np.array(min_ranges)
volatility_stats = {
    "avg_1min_range":    round(float(np.mean(min_ranges)), 4),
    "median_1min_range": round(float(np.median(min_ranges)), 4),
    "p75_1min_range":    round(float(np.percentile(min_ranges, 75)), 4),
    "p90_1min_range":    round(float(np.percentile(min_ranges, 90)), 4),
    "p95_1min_range":    round(float(np.percentile(min_ranges, 95)), 4),
    "p99_1min_range":    round(float(np.percentile(min_ranges, 99)), 4),
}

# Momentum stats
if momentum_runs:
    runs_arr = np.array(momentum_runs)
    momentum_stats = {
        "avg_run_length":    round(float(np.mean(runs_arr)), 2),
        "median_run_length": round(float(np.median(runs_arr)), 2),
        "p90_run_length":    round(float(np.percentile(runs_arr, 90)), 2),
        "p99_run_length":    round(float(np.percentile(runs_arr, 99)), 2),
        "pct_run_gt3":       round(float(np.mean(runs_arr > 3) * 100), 2),
        "pct_run_gt10":      round(float(np.mean(runs_arr > 10) * 100), 2),
    }
else:
    momentum_stats = {}

# S/R Clustering (round number magnetism)
if price_levels:
    levels_arr = np.array(price_levels)
    # Count ticks near round numbers (multiples of 10, 50, 100)
    round10  = np.sum(np.abs(levels_arr % 10)  < 0.5) / len(levels_arr) * 100
    round50  = np.sum(np.abs(levels_arr % 50)  < 0.5) / len(levels_arr) * 100
    round100 = np.sum(np.abs(levels_arr % 100) < 0.5) / len(levels_arr) * 100

    sr_stats = {
        "pct_ticks_near_round10":  round(float(round10), 3),
        "pct_ticks_near_round50":  round(float(round50), 3),
        "pct_ticks_near_round100": round(float(round100), 3),
        "price_range_low":  round(float(levels_arr.min()), 3),
        "price_range_high": round(float(levels_arr.max()), 3),
        "price_mean":       round(float(levels_arr.mean()), 3),
    }
else:
    sr_stats = {}


# ─────────────────────────────────────────────
#  MEAN REVERSION ANALYSIS
# ─────────────────────────────────────────────
# For each hour group: how much does price stretch then revert?
reversion_stats = {}
for hour, vals in hourly_returns.items():
    arr = np.array(vals)
    # Reversion tendency: std vs mean ratio (higher = more mean-reverting)
    if np.std(arr) > 0:
        reversion_ratio = abs(float(np.mean(arr))) / float(np.std(arr))
    else:
        reversion_ratio = 0
    reversion_stats[str(hour)] = {
        "reversion_ratio": round(reversion_ratio, 4),
        "trending":        reversion_ratio > 0.3
    }

most_trending_hours   = sorted(reversion_stats.items(), key=lambda x: x[1]["reversion_ratio"], reverse=True)[:5]
most_reverting_hours  = sorted(reversion_stats.items(), key=lambda x: x[1]["reversion_ratio"])[:5]

# ─────────────────────────────────────────────
#  BUILD FINAL REPORT
# ─────────────────────────────────────────────
report = {
    "meta": {
        "instrument":    "XAUUSD (Gold)",
        "first_tick":    str(first_timestamp),
        "last_tick":     str(last_timestamp),
        "total_ticks":   total_ticks,
        "analyzed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "hourly_bias":         hourly_stats,
    "best_buy_hours":      [(h, v) for h, v in best_buy_hours],
    "best_sell_hours":     [(h, v) for h, v in best_sell_hours],
    "dow_bias":            dow_stats_named,
    "session_bias":        session_stats,
    "spread_by_hour":      spread_stats,
    "best_spread_hours":   [(h, v) for h, v in best_spread_hours],
    "worst_spread_hours":  [(h, v) for h, v in worst_spread_hours],
    "volatility":          volatility_stats,
    "momentum":            momentum_stats,
    "support_resistance":  sr_stats,
    "reversion_by_hour":   reversion_stats,
    "most_trending_hours": [(h, v) for h, v in most_trending_hours],
    "most_reverting_hours":[(h, v) for h, v in most_reverting_hours],
}


# ─────────────────────────────────────────────
#  SAVE & PRINT SUMMARY
# ─────────────────────────────────────────────
with open(OUTPUT, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'=' * 60}")
print(f"  ANALYSIS COMPLETE")
print(f"{'=' * 60}")
print(f"  Total ticks analyzed : {total_ticks:,}")
print(f"  Date range           : {first_timestamp} → {last_timestamp}")
print(f"\n  ── TOP 5 BUY HOURS (UTC) ──")
for h, v in best_buy_hours:
    print(f"     Hour {h:>2}:00  |  avg move: {v['mean']:+.4f}  |  bullish {v['positive_pct']}% of days")

print(f"\n  ── TOP 5 SELL HOURS (UTC) ──")
for h, v in best_sell_hours:
    print(f"     Hour {h:>2}:00  |  avg move: {v['mean']:+.4f}  |  bullish {v['positive_pct']}% of days")

print(f"\n  ── SESSION BIAS ──")
for sess, v in session_stats.items():
    direction = "BULLISH" if v["mean"] > 0 else "BEARISH"
    print(f"     {sess:<10} |  avg: {v['mean']:+.4f}  |  {direction}  |  bullish {v['positive_pct']}%")

print(f"\n  ── DAY OF WEEK BIAS ──")
for day, v in dow_stats_named.items():
    direction = "▲" if v["mean"] > 0 else "▼"
    print(f"     {day:<12} |  {direction} avg: {v['mean']:+.4f}  |  bullish {v['positive_pct']}%")

print(f"\n  ── VOLATILITY (1-min ranges) ──")
print(f"     Average  : {volatility_stats['avg_1min_range']:.4f}")
print(f"     Median   : {volatility_stats['median_1min_range']:.4f}")
print(f"     Top 10%  : {volatility_stats['p90_1min_range']:.4f}")
print(f"     Top 1%   : {volatility_stats['p99_1min_range']:.4f}")

print(f"\n  ── MOMENTUM ──")
if momentum_stats:
    print(f"     Avg run length   : {momentum_stats['avg_run_length']} ticks")
    print(f"     Runs > 3 ticks   : {momentum_stats['pct_run_gt3']}%")
    print(f"     Runs > 10 ticks  : {momentum_stats['pct_run_gt10']}%")

print(f"\n  ── LOWEST SPREAD HOURS (best to trade) ──")
for h, v in best_spread_hours:
    print(f"     Hour {h:>2}:00  |  avg spread: {v['avg_spread']:.4f}")

print(f"\n{'=' * 60}")
print(f"  Full report saved to: {OUTPUT}")
print(f"  PASTE THE CONTENTS OF '{OUTPUT}' BACK TO KIRO!")
print(f"{'=' * 60}\n")
