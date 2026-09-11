# Audit of `CCT_Rectangle_Bot` v1 (the "775% / month" result)

Reviewed files: `config.py`, `strategy.py`, `backtester.py`, `data_loader.py`, `utils.py`, `main.py`
at commit `3e83c82` ("optimize CCT Rectangle Bot for 775%+ monthly returns").

The reported number is not a measurement of an edge. It is the sum of six defects, each of
which independently inflates results. Listed by severity.

## 1. Look-ahead on the 4H direction signal (up to 8 hours)

`DirectionDetector._check_direction_candle(idx)` reads `self.df.iloc[idx + 1]["Open"]`
(`next_candle_open`) and stamps the signal with `self.df.index[idx]`.

Two separate leaks:

- `df_4h` is produced by `resample("4h")` in `data_loader.resample_to_4h`, so the index label is
  the bar's **open** time. The signal is therefore timestamped 4 hours before the candle that
  produced it has closed.
- It additionally consumes the **next** 4H candle's open, which is 4 more hours in the future.

`WeaknessDetector.find_all_weakness` then scans M15 bars from `start_time = signal_time`
forward. So the M15 sweep hunt begins up to 8 hours before the information used to authorise it
existed. In a trending market that is close to picking entries off a chart you have already seen.

## 2. No setup invalidation before entry

`RectangleEntry.find_entry` scans forward up to `MAX_CANDLES_FOR_ENTRY` (120 x 1m, or an 8-hour
window on the 15m fallback) for a close beyond the rectangle, and takes the first one. It never
checks whether price traded through the *other* side of the rectangle first — the side that is
about to become the stop loss.

Result: a long is opened after price has already spent an hour below what will be its stop, then
recovered. Those trades are unopenable in reality (the setup is dead) and they are precisely the
ones that look best, because a V-recovery through the rectangle top usually keeps going.

## 3. Fabricated trade outcomes

`BacktestEngine._create_assumed_result` invents a result when exit data is missing:

```python
is_win = len(self.trade_results) % 2 == 0
```

A synthetic 50% win rate at the setup's full `rr_ratio` (3:1+), i.e. a hard-coded +1.0R
expectancy, is injected into the equity curve and into the win-rate statistic.

## 4. Compounding applied out of chronological order

`_execute_trades_concurrent` iterates setups in *entry-time* order but calls `_simulate_trade`,
which runs each trade to completion before the next is considered. `self.capital` is then
updated, and the next trade is sized off it.

- `_resolve_positions_before` is a no-op (`pass`); `open_positions` is never appended to, so
  `MAX_CONCURRENT_TRADES` is never enforced.
- A trade opened at 10:00 and closed at 18:00 has its P&L folded into the equity used to size a
  trade that opens at 11:00. With 25% risk per trade and compounding, this systematically
  front-loads size onto the winners.

## 5. Intra-bar optimism in the trailing stop

In `_simulate_trade`, for each candle the code (a) raises `best_price` to that candle's `High`,
(b) recomputes `current_stop` from the new `best_price`, then (c) tests that same candle's `Low`
against the raised stop. Within one bar it assumes the high came before the low. It also stops
checking the fixed take-profit entirely once trailing activates, so the bar in which price
touches both TP and the trail is resolved in whichever way the ordering happens to favour.

## 6. Costs are close to absent

- Entry fills at the signal candle's `Close` exactly. No spread, no slippage, no queue.
- `COMMISSION_PER_TRADE = 0.0`.
- `SPREAD = 0.20` is only subtracted from the stop level, which widens risk slightly but is not
  charged on entry or exit.
- Exits fill exactly at the stop/TP price. Gaps through the stop are impossible by construction.

## Secondary problems

- **Sample size and horizon.** yfinance gives 60 days of 15m and 7 days of 1m for `GC=F`. So the
  "monthly return" is extrapolated from ~2 months, one symbol, 47 trades, and most entries and
  exits are simulated on 15m bars because 1m data does not reach back far enough. There is no
  out-of-sample period at all.
- **`_retry_with_relaxed_params` mutates module-level config** (`config.SWEEP_MIN_PIPS = 0.01`
  etc.) and re-runs the search until trades appear. This is automated overfitting, and it leaves
  the process in a state where the printed settings no longer match the settings used.
- **`WeaknessDetector.__init__` precomputes FVGs and session levels over the entire dataframe**,
  including bars in the future relative to any given decision. Dormant today because
  `REQUIRE_IMBALANCE_FILTER`/`REQUIRE_SESSION_EXTREME` default to `False`, but the leak is wired
  in and ready to fire.
- **Deduplication compares only against `deduplicated[-1]`** and only enforces the 30-minute gap
  for same-direction setups, so it emits pairs of opposing trades on the same bar.
- **Risk of ruin is not a side issue.** `RISK_PER_TRADE = 0.25` at the claimed 48% win rate means
  five consecutive losses (probability ~3.8% in any given five-trade window) multiplies equity by
  `0.75^5 = 0.24`. Kelly for 3:1 at p=0.48 is 30.7%, so 25% is ~0.8 Kelly — the growth-optimal
  bet *conditional on the edge being exactly as measured*. Since the edge is measured with
  defects 1-6, the real bet is far above Kelly, where expected log-growth is negative.

## What this implies for "make it more profitable"

Tuning parameters against a metric that contains 8 hours of look-ahead optimises the look-ahead,
not the strategy. So the order of work is: make the measurement honest, find out whether anything
is left, and only then size it.

The rebuild in this directory (`cct2/`) does that. It also ships `run.py audit`, which runs the
same strategy on the same bars with each defect toggled on and off, so the contribution of each
one is a number rather than an assertion.
