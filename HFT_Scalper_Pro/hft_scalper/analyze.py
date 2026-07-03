#!/usr/bin/env python3
"""
Main analysis runner for XAUUSD tick data microstructure profiling.

Executes the full pipeline:
1. Load tick data efficiently
2. Build 1-minute OHLC bars
3. Run comprehensive microstructure analysis
4. Save results to JSON

Usage:
    python3 hft_scalper/analyze.py
"""

import json
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_scalper.data_loader import load_ticks, build_ohlc_bars
from hft_scalper.microstructure_analysis import run_full_analysis


def main():
    start_time = time.time()
    print("=" * 70)
    print("  XAUUSD HFT Scalper - Microstructure Analysis")
    print("=" * 70)

    # Step 1: Load tick data
    print("\n[Step 1/3] Loading tick data...")
    ticks = load_ticks()
    load_time = time.time() - start_time
    print(f"  Load time: {load_time:.1f}s")

    # Step 2: Build OHLC bars
    print("\n[Step 2/3] Building 1-minute OHLC bars...")
    bars = build_ohlc_bars(ticks, freq="1min")
    bar_time = time.time() - start_time - load_time
    print(f"  Bar construction time: {bar_time:.1f}s")

    # Step 3: Run microstructure analysis
    print("\n[Step 3/3] Running microstructure analysis...")
    report = run_full_analysis(ticks, bars)
    analysis_time = time.time() - start_time - load_time - bar_time
    print(f"  Analysis time: {analysis_time:.1f}s")

    # Add timing info
    report["timing"] = {
        "load_seconds": round(load_time, 1),
        "bar_construction_seconds": round(bar_time, 1),
        "analysis_seconds": round(analysis_time, 1),
        "total_seconds": round(time.time() - start_time, 1),
    }

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "microstructure_report.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SAVED: {output_path}")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    print(f"{'=' * 70}")

    # Print key findings
    print("\n--- KEY FINDINGS ---")
    print(f"  Ticks analyzed:        {report['num_ticks']:,}")
    print(f"  1-min bars:            {report['num_bars']:,}")
    print(f"  Price range:           {report['price_range_low']:.2f} - {report['price_range_high']:.2f}")
    print(f"  Mean spread:           {report['mean_spread']:.5f}")
    print(f"  Hurst exponent:        {report['hurst_exponent']:.4f} ({report['mean_reversion']['hurst_interpretation']})")
    print(f"  Tick regime:           {report['dominant_tick_regime']}")
    print(f"  Bar regime:            {report['dominant_bar_regime']}")
    print(f"  Vol clustering:        {report['vol_clustering_strength']:.4f}")

    half_life = report['half_life_bars']
    if half_life is not None and half_life < 1e6:
        print(f"  Mean-rev half-life:    {half_life:.1f} bars ({half_life:.1f} min)")
    else:
        print(f"  Mean-rev half-life:    N/A (no clear mean-reversion)")

    print(f"\n  Spread stats:")
    ss = report['spread_stats']
    print(f"    Median:  {ss['median_spread']:.5f}")
    print(f"    P95:     {ss['p95_spread']:.5f}")
    print(f"    P99:     {ss['p99_spread']:.5f}")

    print(f"\n  Volatility:")
    vm = report['volatility_metrics']
    print(f"    Daily vol:     {vm['daily_vol_pct']:.2f}%")
    print(f"    Return kurt:   {vm['return_kurtosis']:.2f}")
    print(f"    Return skew:   {vm['return_skew']:.4f}")

    print(f"\n  Order Flow:")
    of = report['order_flow']
    print(f"    Uptick %:      {of['uptick_pct']:.1f}%")
    print(f"    Downtick %:    {of['downtick_pct']:.1f}%")
    print(f"    VPIN estimate: {of['vpin_estimate']:.4f}")
    print(f"    OFI autocorr:  {of['ofi_autocorr_tick']:.4f}")

    return report


if __name__ == "__main__":
    main()
