//+------------------------------------------------------------------+
//|                                           GagansModel_EA.mq5     |
//|                    T-Spot Trading Model Expert Advisor            |
//|                    Based on Pine Script T-Spot Indicator          |
//|                                                                  |
//| STRATEGY:                                                        |
//|  - Detects T-Spot patterns on HTF (15M default) candles          |
//|  - Places limit orders at logarithmic midpoint of sweep candle   |
//|  - Uses FVG-based stop loss with fallback to candle extremes     |
//|  - CISD projection targets for take profit management            |
//|  - State machine position management with partial closes         |
//+------------------------------------------------------------------+
#property copyright "GagansModel EA - T-Spot Model"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade        trade;
CPositionInfo posInfo;


//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== HTF & DETECTION ==="
input ENUM_TIMEFRAMES HTF_Timeframe = PERIOD_M15;    // HTF (Auto=M15 for M1 chart)
input bool   Use_Body_Confirmation = true;            // Use body for pivot/series detection
input string TSpot_Bias = "None";                     // None/Bullish/Bearish filter

input group "=== ENTRY & SL ==="
input int    SL_Buffer_Pips = 5;                      // Pips buffer above/below FVG for SL
input int    FVG_Lookback_Bars = 100;                 // Bars to scan for FVG
input int    Order_Expiry_Bars = 0;                   // 0 = expire on HTF candle close

input group "=== RISK MANAGEMENT ==="
input double Risk_Percent = 1.0;                      // Risk % of account per trade
input double Max_Lot = 10.0;                          // Maximum lot size
input double Min_Lot = 0.01;                          // Minimum lot size

input group "=== PROFIT MANAGEMENT ==="
input double Partial_Close_Level = 2.0;               // Projection level to close 70%
input double Partial_Close_Percent = 70.0;            // Percentage to close
input int    BE_Plus_Pips = 3;                        // Pips above breakeven after partial
input double Trail_Start_Level = 2.5;                 // Projection level to start trailing
input int    Trail_Distance_Pips = 10;                // Trailing stop distance in pips

input group "=== SYSTEM ==="
input int    Magic_Number = 777000;                   // Magic number
input string EA_Comment = "GagansModel";              // Trade comment
input int    Max_Slippage = 10;                       // Max slippage in points
input bool   Show_Dashboard = true;                   // Show info panel on chart


//+------------------------------------------------------------------+
//| ENUMERATIONS                                                      |
//+------------------------------------------------------------------+
enum ENUM_TRADE_STATE
{
   STATE_NONE     = 0,   // No position, waiting for signal
   STATE_PENDING  = 1,   // Pending limit order active
   STATE_FULL     = 2,   // Full position open
   STATE_PARTIAL  = 3,   // Partial closed, breakeven SL
   STATE_TRAILING = 4    // Tight trailing active
};

enum ENUM_TSPOT_DIR
{
   TSPOT_NONE    = 0,
   TSPOT_BULLISH = 1,
   TSPOT_BEARISH = -1
};


//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
double pip, point_size;

// State machine
ENUM_TRADE_STATE g_state = STATE_NONE;
ENUM_TSPOT_DIR   g_tspot_dir = TSPOT_NONE;

// T-Spot tracking
double g_midline       = 0;      // LogMidpoint of sweep candle (entry price)
double g_sl_price      = 0;      // Stop loss price
double g_entry_price   = 0;      // Actual entry price for the limit order
double g_tspot_high    = 0;      // High of the sweep candle (for fallback SL)
double g_tspot_low     = 0;      // Low of the sweep candle (for fallback SL)

// Projection levels
double g_proj_levels[5];         // 0.5, 1.0, 1.5, 2.0, 2.5 projection levels
double g_break_price   = 0;      // CISD break price
double g_series_range  = 0;      // CISD series range

// HTF tracking
datetime g_last_htf_time = 0;    // Time of last processed HTF candle
MqlRates g_htf_rates[];          // HTF candle buffer

// Order/Position tracking
ulong  g_pending_ticket = 0;     // Current pending order ticket
ulong  g_position_ticket = 0;    // Current position ticket
int    g_pending_bar_count = 0;  // Bars since pending order placed

// Dashboard
string lbl = "TSM_";

// Display info
string g_state_text   = "Waiting";
string g_tspot_text   = "None";
color  g_tspot_color  = clrGray;
double g_float_pnl    = 0;


//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   point_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   pip = (digits == 3 || digits == 5) ? point_size * 10 : point_size;

   // Initialize trade object
   trade.SetExpertMagicNumber(Magic_Number);
   trade.SetDeviationInPoints(Max_Slippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   // Initialize arrays
   ArraySetAsSeries(g_htf_rates, true);
   ArrayInitialize(g_proj_levels, 0);

   // Initialize state
   g_state = STATE_NONE;
   g_tspot_dir = TSPOT_NONE;
   g_last_htf_time = 0;

   // Check for existing positions/orders from previous run
   SyncStateFromMarket();

   if(Show_Dashboard) CreateDashboard();

   Print("GagansModel EA initialized | T-Spot Model | HTF=", EnumToString(HTF_Timeframe),
         " | Magic=", Magic_Number);
   return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, lbl);
   Print("GagansModel EA deinitialized | Reason=", reason);
}


//+------------------------------------------------------------------+
//| OnTick - Main EA Logic                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Check for new HTF candle close
   bool new_htf = CheckNewHTFCandle();

   // 2. If new HTF candle closed, run T-Spot detection
   if(new_htf && g_state == STATE_NONE)
   {
      DetectTSpot();
   }

   // 3. Run state machine logic every tick
   RunStateMachine();

   // 4. Update dashboard
   if(Show_Dashboard) UpdateDashboard();
}


//+------------------------------------------------------------------+
//| CHECK FOR NEW HTF CANDLE                                          |
//| Returns true when a new HTF candle has formed (previous closed)   |
//+------------------------------------------------------------------+
bool CheckNewHTFCandle()
{
   if(CopyRates(_Symbol, HTF_Timeframe, 0, 5, g_htf_rates) < 5)
      return false;

   // rates[0] is the current (building) candle, rates[1] is the last closed
   datetime current_htf_time = g_htf_rates[0].time;

   if(g_last_htf_time == 0)
   {
      g_last_htf_time = current_htf_time;
      return false;
   }

   if(current_htf_time != g_last_htf_time)
   {
      g_last_htf_time = current_htf_time;
      return true;  // A new HTF candle has started (previous one just closed)
   }

   return false;
}


//+------------------------------------------------------------------+
//| LOGARITHMIC MIDPOINT CALCULATION                                  |
//| Calculates the logarithmic midpoint of a candle using wick/body  |
//| analysis to determine the fair value center                       |
//+------------------------------------------------------------------+
double LogMidpoint(double high, double low, double open, double close)
{
   if(high <= 0 || low <= 0 || open <= 0 || close <= 0) return (high + low) / 2.0;
   if(high == low) return high;

   double log_high  = MathLog(high);
   double log_low   = MathLog(low);
   double log_open  = MathLog(open);
   double log_close = MathLog(close);

   double body_size   = MathAbs(log_close - log_open);
   double upper_wick  = log_high - MathMax(log_open, log_close);
   double lower_wick  = MathMin(log_open, log_close) - log_low;

   double mid = 0;

   if(MathMax(upper_wick, lower_wick) > body_size)
   {
      if(upper_wick > lower_wick)
         mid = log_high - upper_wick / 2.0;
      else
         mid = log_low + lower_wick / 2.0;
   }
   else
   {
      mid = (log_high + log_low) / 2.0;
   }

   return MathExp(mid);
}


//+------------------------------------------------------------------+
//| SIMPLE MIDPOINT (arithmetic mid of candle range)                  |
//+------------------------------------------------------------------+
double MidLevel(double high, double low)
{
   return (high + low) / 2.0;
}


//+------------------------------------------------------------------+
//| SWEPT-BOTH INVALIDATION FILTER                                    |
//| Returns true if last_closed swept both sides of prev_closed and  |
//| closed inside prev_closed range (invalidation condition)          |
//+------------------------------------------------------------------+
bool IsSweptBoth(const MqlRates &last_closed, const MqlRates &prev_closed)
{
   // Last closed took out both highs and lows of prev candle
   // but closed inside prev candle range
   if(last_closed.high > prev_closed.high &&
      last_closed.low  < prev_closed.low  &&
      last_closed.close > prev_closed.low &&
      last_closed.close < prev_closed.high)
   {
      return true;
   }
   return false;
}


//+------------------------------------------------------------------+
//| T-SPOT DETECTION                                                  |
//| Runs when a new HTF candle closes. Checks 6 patterns in order.  |
//| Uses last 3 closed HTF candles (indices 1, 2, 3 in rates array) |
//+------------------------------------------------------------------+
void DetectTSpot()
{
   // g_htf_rates[0] = current building candle
   // g_htf_rates[1] = last closed candle (the sweep candle)
   // g_htf_rates[2] = previous closed candle
   // g_htf_rates[3] = candle before that (prev_prev)

   MqlRates last_closed = g_htf_rates[1];   // The candle that just closed
   MqlRates prev_closed = g_htf_rates[2];   // The candle before it
   MqlRates prev_prev   = g_htf_rates[3];   // Two candles back

   // Current price (close of last M1 bar)
   double close_price = last_closed.close;

   // Calculate midpoints
   double mid_last = LogMidpoint(last_closed.high, last_closed.low,
                                  last_closed.open, last_closed.close);
   double mid_prev = LogMidpoint(prev_closed.high, prev_closed.low,
                                  prev_closed.open, prev_closed.close);

   // Mid level of prev_closed (arithmetic)
   double mid_level_prev = MidLevel(prev_closed.high, prev_closed.low);

   // Check swept-both invalidation
   bool swept_both = IsSweptBoth(last_closed, prev_closed);

   ENUM_TSPOT_DIR detected = TSPOT_NONE;
   string pattern_name = "";

   // === Pattern 1 - Normal Bearish ===
   if(detected == TSPOT_NONE && !swept_both)
   {
      if(last_closed.high > prev_closed.high &&
         last_closed.close < prev_closed.high &&
         close_price < mid_last)
      {
         detected = TSPOT_BEARISH;
         pattern_name = "Normal Bearish";
      }
   }

   // === Pattern 2 - Normal Bullish ===
   if(detected == TSPOT_NONE && !swept_both)
   {
      if(last_closed.low < prev_closed.low &&
         last_closed.close > prev_closed.low &&
         close_price > mid_last)
      {
         detected = TSPOT_BULLISH;
         pattern_name = "Normal Bullish";
      }
   }

   // === Pattern 3 - Expansive Bearish ===
   if(detected == TSPOT_NONE && !swept_both)
   {
      if(prev_closed.high > prev_prev.high &&
         last_closed.close < MathMax(prev_closed.open, prev_closed.close) &&
         close_price < mid_last)
      {
         // Additional conditions (one must be true)
         bool cond_a = (prev_closed.close >= mid_prev);
         bool cond_b = (prev_closed.close >= prev_prev.high);
         bool cond_c = (prev_closed.high > prev_prev.high &&
                        prev_closed.low < prev_prev.low &&
                        prev_closed.close > prev_prev.low &&
                        prev_closed.close < prev_prev.high);

         if(cond_a || cond_b || cond_c)
         {
            detected = TSPOT_BEARISH;
            pattern_name = "Expansive Bearish";
         }
      }
   }

   // === Pattern 4 - Expansive Bullish ===
   if(detected == TSPOT_NONE && !swept_both)
   {
      if(prev_closed.low < prev_prev.low &&
         last_closed.close > MathMin(prev_closed.open, prev_closed.close) &&
         close_price > mid_last)
      {
         // Additional conditions (one must be true)
         bool cond_a = (prev_closed.close <= mid_prev);
         bool cond_b = (prev_closed.close <= prev_prev.low);
         bool cond_c = (prev_closed.high > prev_prev.high &&
                        prev_closed.low < prev_prev.low &&
                        prev_closed.close > prev_prev.low &&
                        prev_closed.close < prev_prev.high);

         if(cond_a || cond_b || cond_c)
         {
            detected = TSPOT_BULLISH;
            pattern_name = "Expansive Bullish";
         }
      }
   }

   // === Pattern 5 - Pro-Trend Bullish Mid Sweep ===
   if(detected == TSPOT_NONE && !swept_both)
   {
      if(last_closed.low < mid_level_prev &&
         last_closed.low > prev_closed.open &&
         last_closed.close > prev_closed.high &&
         close_price > mid_last)
      {
         detected = TSPOT_BULLISH;
         pattern_name = "Pro-Trend Bullish Mid Sweep";
      }
   }

   // === Pattern 6 - Pro-Trend Bearish Mid Sweep ===
   if(detected == TSPOT_NONE && !swept_both)
   {
      if(last_closed.high > mid_level_prev &&
         last_closed.high < prev_closed.open &&
         last_closed.close < prev_closed.low &&
         close_price < mid_last)
      {
         detected = TSPOT_BEARISH;
         pattern_name = "Pro-Trend Bearish Mid Sweep";
      }
   }

   // Apply bias filter
   if(detected != TSPOT_NONE)
   {
      if(TSpot_Bias == "Bullish" && detected != TSPOT_BULLISH) detected = TSPOT_NONE;
      if(TSpot_Bias == "Bearish" && detected != TSPOT_BEARISH) detected = TSPOT_NONE;
   }

   // If T-Spot detected, set up the trade
   if(detected != TSPOT_NONE)
   {
      g_tspot_dir  = detected;
      g_midline    = mid_last;
      g_entry_price = mid_last;
      g_tspot_high = last_closed.high;
      g_tspot_low  = last_closed.low;

      // Calculate FVG-based stop loss
      g_sl_price = CalculateFVGStopLoss(detected, g_entry_price);

      // Calculate CISD projection levels
      CalculateProjections(detected, last_closed, prev_closed, prev_prev);

      // Place the limit order
      PlaceLimitOrder(detected);

      Print("T-Spot Detected: ", pattern_name,
            " | Dir=", (detected == TSPOT_BULLISH ? "BULL" : "BEAR"),
            " | Entry=", DoubleToString(g_entry_price, _Digits),
            " | SL=", DoubleToString(g_sl_price, _Digits));
   }
}


//+------------------------------------------------------------------+
//| CALCULATE FVG-BASED STOP LOSS                                     |
//| Scans M1 bars within current+previous HTF period for FVGs        |
//+------------------------------------------------------------------+
double CalculateFVGStopLoss(ENUM_TSPOT_DIR direction, double entry_price)
{
   MqlRates m1_rates[];
   ArraySetAsSeries(m1_rates, true);

   int bars_copied = CopyRates(_Symbol, PERIOD_M1, 0, FVG_Lookback_Bars, m1_rates);
   if(bars_copied < 5) // Not enough data, use fallback
   {
      return FallbackSL(direction);
   }

   double sl = 0;

   if(direction == TSPOT_BEARISH)
   {
      // Find nearest BULLISH FVG ABOVE entry price
      // Bullish FVG: bar[i].low > bar[i+2].high (gap between bar i and bar i+2)
      double nearest_fvg_top = 0;
      double min_distance = DBL_MAX;

      for(int i = 0; i < bars_copied - 2; i++)
      {
         if(m1_rates[i].low > m1_rates[i+2].high)  // Bullish FVG exists
         {
            double fvg_top = m1_rates[i].low;
            if(fvg_top > entry_price)  // FVG is above entry
            {
               double dist = fvg_top - entry_price;
               if(dist < min_distance)
               {
                  min_distance = dist;
                  nearest_fvg_top = fvg_top;
               }
            }
         }
      }

      if(nearest_fvg_top > 0)
         sl = nearest_fvg_top + SL_Buffer_Pips * pip;
      else
         sl = FallbackSL(direction);
   }
   else // TSPOT_BULLISH
   {
      // Find nearest BEARISH FVG BELOW entry price
      // Bearish FVG: bar[i].high < bar[i+2].low (gap between bar i and bar i+2)
      double nearest_fvg_bottom = 0;
      double min_distance = DBL_MAX;

      for(int i = 0; i < bars_copied - 2; i++)
      {
         if(m1_rates[i].high < m1_rates[i+2].low)  // Bearish FVG exists
         {
            double fvg_bottom = m1_rates[i].high;
            if(fvg_bottom < entry_price)  // FVG is below entry
            {
               double dist = entry_price - fvg_bottom;
               if(dist < min_distance)
               {
                  min_distance = dist;
                  nearest_fvg_bottom = fvg_bottom;
               }
            }
         }
      }

      if(nearest_fvg_bottom > 0)
         sl = nearest_fvg_bottom - SL_Buffer_Pips * pip;
      else
         sl = FallbackSL(direction);
   }

   return NormalizeDouble(sl, _Digits);
}


//+------------------------------------------------------------------+
//| FALLBACK STOP LOSS                                                |
//| Uses the T-Spot candle extreme + buffer when no FVG found        |
//+------------------------------------------------------------------+
double FallbackSL(ENUM_TSPOT_DIR direction)
{
   if(direction == TSPOT_BEARISH)
      return NormalizeDouble(g_tspot_high + SL_Buffer_Pips * pip, _Digits);
   else
      return NormalizeDouble(g_tspot_low - SL_Buffer_Pips * pip, _Digits);
}


//+------------------------------------------------------------------+
//| CALCULATE CISD PROJECTION TARGETS                                 |
//| Finds the C2 bar, counts consecutive same-polarity candles,      |
//| calculates series range and projection levels from break price   |
//+------------------------------------------------------------------+
void CalculateProjections(ENUM_TSPOT_DIR direction,
                          const MqlRates &last_closed,
                          const MqlRates &prev_closed,
                          const MqlRates &prev_prev)
{
   // The C2 bar is where the swept level was formed (the high/low that was broken)
   // For bullish T-Spot: the low that was swept belongs to prev_closed
   // For bearish T-Spot: the high that was swept belongs to prev_closed

   // Get M1 bars covering the HTF candles for series detection
   MqlRates htf_bars[];
   ArraySetAsSeries(htf_bars, true);
   int copied = CopyRates(_Symbol, HTF_Timeframe, 1, 10, htf_bars);
   if(copied < 4)
   {
      // Not enough data, use simple projection from entry
      SetSimpleProjections(direction, g_entry_price, MathAbs(last_closed.high - last_closed.low));
      return;
   }

   // C2 is prev_closed (index 0 in htf_bars since we started from 1)
   // Look backward from C2 for consecutive same-polarity candles
   double series_high = 0;
   double series_low  = DBL_MAX;

   // For bullish T-Spot: look for consecutive bearish candles before the low
   // For bearish T-Spot: look for consecutive bullish candles before the high
   int start_idx = 1; // Start from prev_closed (htf_bars[0] = last_closed in our copy starting from 1)

   if(direction == TSPOT_BULLISH)
   {
      // Look for consecutive bearish candles (close < open) backward from C2
      for(int i = start_idx; i < copied; i++)
      {
         bool is_bearish = (htf_bars[i].close < htf_bars[i].open);
         if(!is_bearish && i > start_idx) break;  // Stop at first non-bearish after start

         if(Use_Body_Confirmation)
         {
            series_high = MathMax(series_high, MathMax(htf_bars[i].open, htf_bars[i].close));
            series_low  = MathMin(series_low,  MathMin(htf_bars[i].open, htf_bars[i].close));
         }
         else
         {
            series_high = MathMax(series_high, htf_bars[i].high);
            series_low  = MathMin(series_low,  htf_bars[i].low);
         }

         if(!is_bearish) break;  // Include the first non-matching then stop
      }

      double series_range = series_high - series_low;
      if(series_range <= 0) series_range = MathAbs(last_closed.high - last_closed.low);

      g_series_range = series_range;
      g_break_price  = series_high;  // For bullish: price breaks above the series high

      // Projection levels from break_price upward
      g_proj_levels[0] = g_break_price + series_range * 0.5;
      g_proj_levels[1] = g_break_price + series_range * 1.0;
      g_proj_levels[2] = g_break_price + series_range * 1.5;
      g_proj_levels[3] = g_break_price + series_range * 2.0;
      g_proj_levels[4] = g_break_price + series_range * 2.5;
   }
   else // TSPOT_BEARISH
   {
      // Look for consecutive bullish candles (close > open) backward from C2
      for(int i = start_idx; i < copied; i++)
      {
         bool is_bullish = (htf_bars[i].close > htf_bars[i].open);
         if(!is_bullish && i > start_idx) break;

         if(Use_Body_Confirmation)
         {
            series_high = MathMax(series_high, MathMax(htf_bars[i].open, htf_bars[i].close));
            series_low  = MathMin(series_low,  MathMin(htf_bars[i].open, htf_bars[i].close));
         }
         else
         {
            series_high = MathMax(series_high, htf_bars[i].high);
            series_low  = MathMin(series_low,  htf_bars[i].low);
         }

         if(!is_bullish) break;
      }

      double series_range = series_high - series_low;
      if(series_range <= 0) series_range = MathAbs(last_closed.high - last_closed.low);

      g_series_range = series_range;
      g_break_price  = series_low;  // For bearish: price breaks below the series low

      // Projection levels from break_price downward
      g_proj_levels[0] = g_break_price - series_range * 0.5;
      g_proj_levels[1] = g_break_price - series_range * 1.0;
      g_proj_levels[2] = g_break_price - series_range * 1.5;
      g_proj_levels[3] = g_break_price - series_range * 2.0;
      g_proj_levels[4] = g_break_price - series_range * 2.5;
   }

   Print("Projections calculated | Break=", DoubleToString(g_break_price, _Digits),
         " | Range=", DoubleToString(g_series_range, _Digits),
         " | Lvl2.0=", DoubleToString(g_proj_levels[3], _Digits),
         " | Lvl2.5=", DoubleToString(g_proj_levels[4], _Digits));
}


//+------------------------------------------------------------------+
//| SET SIMPLE PROJECTIONS (fallback when not enough data)            |
//+------------------------------------------------------------------+
void SetSimpleProjections(ENUM_TSPOT_DIR direction, double base_price, double range)
{
   if(range <= 0) range = 50 * pip;
   g_series_range = range;
   g_break_price  = base_price;

   double mult = (direction == TSPOT_BULLISH) ? 1.0 : -1.0;
   g_proj_levels[0] = base_price + range * 0.5 * mult;
   g_proj_levels[1] = base_price + range * 1.0 * mult;
   g_proj_levels[2] = base_price + range * 1.5 * mult;
   g_proj_levels[3] = base_price + range * 2.0 * mult;
   g_proj_levels[4] = base_price + range * 2.5 * mult;
}


//+------------------------------------------------------------------+
//| PLACE LIMIT ORDER                                                 |
//| Places Buy Limit or Sell Limit at the T-Spot midpoint            |
//+------------------------------------------------------------------+
void PlaceLimitOrder(ENUM_TSPOT_DIR direction)
{
   // Cancel any existing pending order first
   CancelPendingOrder();

   double entry = NormalizeDouble(g_entry_price, _Digits);
   double sl    = NormalizeDouble(g_sl_price, _Digits);

   // Calculate position size based on risk
   double sl_distance = MathAbs(entry - sl);
   double lot = CalculateLotSize(sl_distance);

   // Use the nearest projection level as initial TP (level 2.0)
   double tp = NormalizeDouble(g_proj_levels[3], _Digits);  // 2.0 level

   bool result = false;

   if(direction == TSPOT_BEARISH)
   {
      // Sell Limit - entry must be above current ask for valid placement
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(entry > ask)
      {
         result = trade.SellLimit(lot, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, EA_Comment);
      }
      else
      {
         // Entry is at or below current price - place at ask + 1 point or use market
         Print("T-Spot entry price below ask, adjusting to ask + spread");
         entry = NormalizeDouble(ask + point_size, _Digits);
         result = trade.SellLimit(lot, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, EA_Comment);
      }
      g_tspot_text  = "Bearish";
      g_tspot_color = clrRed;
   }
   else // TSPOT_BULLISH
   {
      // Buy Limit - entry must be below current bid for valid placement
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(entry < bid)
      {
         result = trade.BuyLimit(lot, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, EA_Comment);
      }
      else
      {
         // Entry is at or above current price - adjust
         Print("T-Spot entry price above bid, adjusting to bid - spread");
         entry = NormalizeDouble(bid - point_size, _Digits);
         result = trade.BuyLimit(lot, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, EA_Comment);
      }
      g_tspot_text  = "Bullish";
      g_tspot_color = clrLime;
   }

   if(result)
   {
      g_pending_ticket = trade.ResultOrder();
      g_state = STATE_PENDING;
      g_pending_bar_count = 0;
      g_state_text = "Pending";
      Print("Limit order placed | Ticket=", g_pending_ticket,
            " | Entry=", DoubleToString(entry, _Digits),
            " | SL=", DoubleToString(sl, _Digits),
            " | Lot=", DoubleToString(lot, 2));
   }
   else
   {
      Print("Failed to place limit order: ", trade.ResultRetcodeDescription(),
            " | Code=", trade.ResultRetcode());
      g_state = STATE_NONE;
      g_tspot_dir = TSPOT_NONE;
   }
}


//+------------------------------------------------------------------+
//| CANCEL PENDING ORDER                                              |
//+------------------------------------------------------------------+
void CancelPendingOrder()
{
   if(g_pending_ticket > 0)
   {
      if(OrderSelect(g_pending_ticket))
      {
         trade.OrderDelete(g_pending_ticket);
         Print("Cancelled pending order: ", g_pending_ticket);
      }
      g_pending_ticket = 0;
   }

   // Also scan for any orphaned orders with our magic number
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(OrderGetInteger(ORDER_MAGIC) != Magic_Number) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
      trade.OrderDelete(ticket);
   }
}


//+------------------------------------------------------------------+
//| CALCULATE LOT SIZE (Risk-Based)                                   |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_distance)
{
   if(sl_distance <= 0) return NormLot(Min_Lot);

   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_money = balance * Risk_Percent / 100.0;
   double tick_val   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tick_val <= 0 || tick_size <= 0) return NormLot(Min_Lot);

   double lot = risk_money / (sl_distance / tick_size * tick_val);
   lot = MathMin(lot, Max_Lot);
   lot = MathMax(lot, Min_Lot);

   return NormLot(lot);
}


//+------------------------------------------------------------------+
//| NORMALIZE LOT SIZE                                                |
//+------------------------------------------------------------------+
double NormLot(double lot)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lot = MathFloor(lot / step) * step;
   return NormalizeDouble(MathMax(minv, MathMin(lot, maxv)), 2);
}


//+------------------------------------------------------------------+
//| STATE MACHINE - Main execution loop                               |
//+------------------------------------------------------------------+
void RunStateMachine()
{
   switch(g_state)
   {
      case STATE_NONE:
         // Nothing to do - waiting for T-Spot detection
         g_state_text = "Waiting";
         g_float_pnl = 0;
         break;

      case STATE_PENDING:
         ManageStatePending();
         break;

      case STATE_FULL:
         ManageStateFull();
         break;

      case STATE_PARTIAL:
         ManageStatePartial();
         break;

      case STATE_TRAILING:
         ManageStateTrailing();
         break;
   }
}


//+------------------------------------------------------------------+
//| STATE 1 - PENDING ORDER MANAGEMENT                                |
//| Check if filled, expired by HTF close, or invalidated            |
//+------------------------------------------------------------------+
void ManageStatePending()
{
   g_state_text = "Pending";

   // Check if the pending order still exists
   if(g_pending_ticket > 0 && !OrderSelect(g_pending_ticket))
   {
      // Order no longer pending - check if it was filled (became a position)
      if(FindOurPosition())
      {
         g_state = STATE_FULL;
         g_state_text = "Open";
         g_pending_ticket = 0;
         Print("Limit order filled | Position ticket=", g_position_ticket);
         return;
      }
      else
      {
         // Order was deleted externally or expired
         ResetState();
         return;
      }
   }

   // Check invalidation: price closes beyond midline
   if(CheckInvalidation())
   {
      CancelPendingOrder();
      ResetState();
      Print("Pending order invalidated - price closed beyond midline");
      return;
   }

   // Check HTF candle close expiry
   // If a new HTF candle has formed since order was placed, cancel
   if(Order_Expiry_Bars == 0)
   {
      // Expire on HTF candle close - check if current HTF time changed
      MqlRates temp_rates[];
      ArraySetAsSeries(temp_rates, true);
      if(CopyRates(_Symbol, HTF_Timeframe, 0, 2, temp_rates) >= 2)
      {
         if(g_pending_ticket > 0 && OrderSelect(g_pending_ticket))
         {
            datetime order_time = (datetime)OrderGetInteger(ORDER_TIME_SETUP);
            if(temp_rates[0].time > order_time &&
               temp_rates[1].time >= order_time)
            {
               // The HTF candle during which order was placed has closed
               // and a new one has started
               CancelPendingOrder();
               ResetState();
               Print("Pending order expired on HTF candle close");
               return;
            }
         }
      }
   }
   else
   {
      // Expire after N M1 bars
      g_pending_bar_count++;
      int bars_per_htf = PeriodSeconds(HTF_Timeframe) / PeriodSeconds(PERIOD_M1);
      if(g_pending_bar_count >= Order_Expiry_Bars)
      {
         CancelPendingOrder();
         ResetState();
         Print("Pending order expired after ", Order_Expiry_Bars, " bars");
         return;
      }
   }
}


//+------------------------------------------------------------------+
//| STATE 2 - FULL POSITION MANAGEMENT                                |
//| Monitor projections, partial close at level 2.0, invalidation    |
//+------------------------------------------------------------------+
void ManageStateFull()
{
   g_state_text = "Open";

   // Verify position still exists
   if(!FindOurPosition())
   {
      ResetState();
      Print("Position closed (SL/TP hit or external close)");
      return;
   }

   // Select the position for info
   if(!posInfo.SelectByTicket(g_position_ticket)) { ResetState(); return; }

   double cur_price = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                      SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double open_price = posInfo.PriceOpen();
   g_float_pnl = posInfo.Profit() + posInfo.Swap();

   // Check invalidation
   if(CheckInvalidation())
   {
      trade.PositionClose(g_position_ticket);
      ResetState();
      Print("Position closed - invalidation (price beyond midline)");
      return;
   }

   // Check if price hit Projection Level 2.0 (Partial_Close_Level)
   bool hit_partial = false;
   if(g_tspot_dir == TSPOT_BULLISH)
      hit_partial = (cur_price >= g_proj_levels[3]);  // Level 2.0
   else
      hit_partial = (cur_price <= g_proj_levels[3]);  // Level 2.0

   if(hit_partial)
   {
      // Close Partial_Close_Percent of position
      double vol = posInfo.Volume();
      double close_vol = NormLot(vol * Partial_Close_Percent / 100.0);
      double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

      if(close_vol >= min_vol && close_vol < vol)
      {
         if(trade.PositionClosePartial(g_position_ticket, close_vol))
         {
            Print("Partial close executed | Closed=", DoubleToString(close_vol, 2),
                  " | Remaining=", DoubleToString(vol - close_vol, 2));
         }
      }

      // Move SL to breakeven + BE_Plus_Pips
      double new_sl = 0;
      if(g_tspot_dir == TSPOT_BULLISH)
         new_sl = NormalizeDouble(open_price + BE_Plus_Pips * pip, _Digits);
      else
         new_sl = NormalizeDouble(open_price - BE_Plus_Pips * pip, _Digits);

      trade.PositionModify(g_position_ticket, new_sl, posInfo.TakeProfit());

      g_state = STATE_PARTIAL;
      g_state_text = "BE";
      Print("Moved to STATE_PARTIAL | New SL (BE+)=", DoubleToString(new_sl, _Digits));
   }
}


//+------------------------------------------------------------------+
//| STATE 3 - PARTIAL CLOSED, BREAKEVEN SL                            |
//| Monitor for Projection Level 2.5 to activate trailing            |
//+------------------------------------------------------------------+
void ManageStatePartial()
{
   g_state_text = "BE";

   // Verify position still exists
   if(!FindOurPosition())
   {
      ResetState();
      Print("Position closed at breakeven");
      return;
   }

   if(!posInfo.SelectByTicket(g_position_ticket)) { ResetState(); return; }

   double cur_price = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                      SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   g_float_pnl = posInfo.Profit() + posInfo.Swap();

   // Check if price hit Projection Level 2.5
   bool hit_trail_start = false;
   if(g_tspot_dir == TSPOT_BULLISH)
      hit_trail_start = (cur_price >= g_proj_levels[4]);  // Level 2.5
   else
      hit_trail_start = (cur_price <= g_proj_levels[4]);  // Level 2.5

   if(hit_trail_start)
   {
      g_state = STATE_TRAILING;
      g_state_text = "Trailing";
      Print("Trailing activated | Level 2.5 hit at ", DoubleToString(cur_price, _Digits));
   }
}


//+------------------------------------------------------------------+
//| STATE 4 - TIGHT TRAILING STOP                                     |
//| Trail SL by Trail_Distance_Pips, only move in favorable dir      |
//+------------------------------------------------------------------+
void ManageStateTrailing()
{
   g_state_text = "Trailing";

   // Verify position still exists
   if(!FindOurPosition())
   {
      ResetState();
      Print("Position closed by trailing stop");
      return;
   }

   if(!posInfo.SelectByTicket(g_position_ticket)) { ResetState(); return; }

   double cur_sl = posInfo.StopLoss();
   double cur_tp = posInfo.TakeProfit();
   bool is_buy = (posInfo.PositionType() == POSITION_TYPE_BUY);
   g_float_pnl = posInfo.Profit() + posInfo.Swap();

   double trail_dist = Trail_Distance_Pips * pip;

   if(is_buy)
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double new_sl = NormalizeDouble(bid - trail_dist, _Digits);

      // Only move SL up, never down
      if(new_sl > cur_sl + point_size)
      {
         trade.PositionModify(g_position_ticket, new_sl, cur_tp);
      }
   }
   else // Sell position
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double new_sl = NormalizeDouble(ask + trail_dist, _Digits);

      // Only move SL down, never up
      if(cur_sl == 0 || new_sl < cur_sl - point_size)
      {
         trade.PositionModify(g_position_ticket, new_sl, cur_tp);
      }
   }
}


//+------------------------------------------------------------------+
//| CHECK INVALIDATION                                                |
//| For bearish T-Spot: if M1 bar closes above midline               |
//| For bullish T-Spot: if M1 bar closes below midline               |
//+------------------------------------------------------------------+
bool CheckInvalidation()
{
   if(g_tspot_dir == TSPOT_NONE || g_midline == 0) return false;

   // Get last closed M1 bar
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 1, 1, m1) < 1) return false;

   double m1_close = m1[0].close;

   if(g_tspot_dir == TSPOT_BEARISH && m1_close > g_midline)
      return true;

   if(g_tspot_dir == TSPOT_BULLISH && m1_close < g_midline)
      return true;

   return false;
}


//+------------------------------------------------------------------+
//| FIND OUR POSITION                                                 |
//| Scans open positions for one matching our magic number           |
//+------------------------------------------------------------------+
bool FindOurPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol) continue;
      if(posInfo.Magic() != Magic_Number) continue;

      g_position_ticket = posInfo.Ticket();
      return true;
   }

   g_position_ticket = 0;
   return false;
}


//+------------------------------------------------------------------+
//| RESET STATE                                                       |
//| Clears all tracking variables back to initial state              |
//+------------------------------------------------------------------+
void ResetState()
{
   g_state = STATE_NONE;
   g_tspot_dir = TSPOT_NONE;
   g_midline = 0;
   g_entry_price = 0;
   g_sl_price = 0;
   g_tspot_high = 0;
   g_tspot_low = 0;
   g_pending_ticket = 0;
   g_position_ticket = 0;
   g_pending_bar_count = 0;
   g_float_pnl = 0;
   g_break_price = 0;
   g_series_range = 0;
   ArrayInitialize(g_proj_levels, 0);
   g_state_text = "Waiting";
   g_tspot_text = "None";
   g_tspot_color = clrGray;
}


//+------------------------------------------------------------------+
//| SYNC STATE FROM MARKET                                            |
//| On init, check if we have existing positions/orders from a       |
//| previous session to restore state                                |
//+------------------------------------------------------------------+
void SyncStateFromMarket()
{
   // Check for existing pending orders
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(OrderGetInteger(ORDER_MAGIC) != Magic_Number) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;

      g_pending_ticket = ticket;
      g_state = STATE_PENDING;
      g_state_text = "Pending";

      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      g_tspot_dir = (type == ORDER_TYPE_SELL_LIMIT) ? TSPOT_BEARISH : TSPOT_BULLISH;
      g_entry_price = OrderGetDouble(ORDER_PRICE_OPEN);
      g_midline = g_entry_price;
      g_sl_price = OrderGetDouble(ORDER_SL);
      Print("Synced pending order from market | Ticket=", ticket);
      return;
   }

   // Check for existing positions
   if(FindOurPosition())
   {
      posInfo.SelectByTicket(g_position_ticket);
      g_state = STATE_FULL;
      g_state_text = "Open";

      g_tspot_dir = (posInfo.PositionType() == POSITION_TYPE_BUY) ? TSPOT_BULLISH : TSPOT_BEARISH;
      g_entry_price = posInfo.PriceOpen();
      g_midline = g_entry_price;
      g_sl_price = posInfo.StopLoss();
      Print("Synced open position from market | Ticket=", g_position_ticket);
   }
}


//+------------------------------------------------------------------+
//| DASHBOARD - Create on-chart information panel (top-right)         |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   ObjectsDeleteAll(0, lbl);
   int x = 15, y = 30, row = 15;
   color bg = C'20,28,42', border = C'40,55,85';

   ObjRect(lbl+"bg", x-8, y-8, 280, 320, bg, border, 1);

   // Title
   ObjLbl(lbl+"bullet", "%A0",           x,    y,    C'0,180,100', 11, true);
   ObjLbl(lbl+"title",  " GagansModel EA",   x+14, y,    clrWhite,    9,  true);
   ObjLbl(lbl+"sub",    " T-Spot Trading Model", x+14, y+13, C'150,150,150', 7, false);
   ObjLine(lbl+"d0", x, y+27, 260);

   int r = y + 36;
   ObjLbl(lbl+"l_sym",   "Symbol",     x, r,         clrSilver, 8);
   ObjLbl(lbl+"v_sym",   _Symbol,      x+110, r,      clrWhite,  8);
   ObjLbl(lbl+"l_htf",   "HTF",        x, r+row,     clrSilver, 8);
   ObjLbl(lbl+"v_htf",   TFStr(HTF_Timeframe), x+110, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_state", "State",      x, r+row*2,   clrSilver, 8);
   ObjLbl(lbl+"v_state", "Waiting",    x+110, r+row*2, clrYellow, 8);
   ObjLine(lbl+"d1", x, r+row*3+2, 260);

   r = r + row*3 + 10;
   ObjLbl(lbl+"l_htfo",  "HTF Open",   x, r,         clrSilver, 8);
   ObjLbl(lbl+"v_htfo",  "---",        x+110, r,      clrWhite,  8);
   ObjLbl(lbl+"l_htfh",  "HTF High",   x, r+row,     clrSilver, 8);
   ObjLbl(lbl+"v_htfh",  "---",        x+110, r+row,  clrWhite,  8);
   ObjLbl(lbl+"l_htfl",  "HTF Low",    x, r+row*2,   clrSilver, 8);
   ObjLbl(lbl+"v_htfl",  "---",        x+110, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"l_htfc",  "HTF Close",  x, r+row*3,   clrSilver, 8);
   ObjLbl(lbl+"v_htfc",  "---",        x+110, r+row*3, clrWhite, 8);
   ObjLine(lbl+"d2", x, r+row*4+2, 260);

   r = r + row*4 + 10;
   ObjLbl(lbl+"l_tspot", "T-Spot",     x, r,         clrSilver, 8);
   ObjLbl(lbl+"v_tspot", "None",       x+110, r,      clrGray,   8);
   ObjLbl(lbl+"l_entry", "Entry",      x, r+row,     clrSilver, 8);
   ObjLbl(lbl+"v_entry", "---",        x+110, r+row,  clrWhite,  8);
   ObjLbl(lbl+"l_sl",    "SL",         x, r+row*2,   clrSilver, 8);
   ObjLbl(lbl+"v_sl",    "---",        x+110, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"l_pnl",   "P/L",        x, r+row*3,   clrSilver, 8);
   ObjLbl(lbl+"v_pnl",   "---",        x+110, r+row*3, clrWhite, 8);
   ObjLine(lbl+"d3", x, r+row*4+2, 260);

   r = r + row*4 + 10;
   ObjLbl(lbl+"l_p05",   "Proj 0.5",   x, r,         clrSilver, 7);
   ObjLbl(lbl+"v_p05",   "---",        x+110, r,      C'100,150,200', 7);
   ObjLbl(lbl+"l_p10",   "Proj 1.0",   x, r+row,     clrSilver, 7);
   ObjLbl(lbl+"v_p10",   "---",        x+110, r+row,  C'100,150,200', 7);
   ObjLbl(lbl+"l_p15",   "Proj 1.5",   x, r+row*2,   clrSilver, 7);
   ObjLbl(lbl+"v_p15",   "---",        x+110, r+row*2, C'100,150,200', 7);
   ObjLbl(lbl+"l_p20",   "Proj 2.0",   x, r+row*3,   clrSilver, 7);
   ObjLbl(lbl+"v_p20",   "---",        x+110, r+row*3, clrGold,   7);
   ObjLbl(lbl+"l_p25",   "Proj 2.5",   x, r+row*4,   clrSilver, 7);
   ObjLbl(lbl+"v_p25",   "---",        x+110, r+row*4, clrGold,   7);

   ChartRedraw(0);
}


//+------------------------------------------------------------------+
//| UPDATE DASHBOARD                                                   |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   if(!Show_Dashboard) return;

   // State
   color state_col = clrYellow;
   if(g_state == STATE_FULL || g_state == STATE_PARTIAL) state_col = clrLime;
   if(g_state == STATE_TRAILING) state_col = clrAqua;
   if(g_state == STATE_PENDING) state_col = clrOrange;
   ObjSet(lbl+"v_state", g_state_text, state_col);

   // HTF OHLC (current building candle)
   if(ArraySize(g_htf_rates) >= 1)
   {
      ObjSet(lbl+"v_htfo", DoubleToString(g_htf_rates[0].open,  _Digits), clrWhite);
      ObjSet(lbl+"v_htfh", DoubleToString(g_htf_rates[0].high,  _Digits), clrWhite);
      ObjSet(lbl+"v_htfl", DoubleToString(g_htf_rates[0].low,   _Digits), clrWhite);
      ObjSet(lbl+"v_htfc", DoubleToString(g_htf_rates[0].close, _Digits), clrWhite);
   }

   // T-Spot info
   ObjSet(lbl+"v_tspot", g_tspot_text, g_tspot_color);

   // Entry and SL
   if(g_entry_price > 0)
      ObjSet(lbl+"v_entry", DoubleToString(g_entry_price, _Digits), clrWhite);
   else
      ObjSet(lbl+"v_entry", "---", clrGray);

   if(g_sl_price > 0)
      ObjSet(lbl+"v_sl", DoubleToString(g_sl_price, _Digits), clrTomato);
   else
      ObjSet(lbl+"v_sl", "---", clrGray);

   // P/L
   if(g_state >= STATE_FULL)
      ObjSet(lbl+"v_pnl", StringFormat("%.2f", g_float_pnl),
             g_float_pnl >= 0 ? clrLime : clrTomato);
   else
      ObjSet(lbl+"v_pnl", "---", clrGray);

   // Projection levels
   if(g_proj_levels[0] != 0)
   {
      ObjSet(lbl+"v_p05", DoubleToString(g_proj_levels[0], _Digits), C'100,150,200');
      ObjSet(lbl+"v_p10", DoubleToString(g_proj_levels[1], _Digits), C'100,150,200');
      ObjSet(lbl+"v_p15", DoubleToString(g_proj_levels[2], _Digits), C'100,150,200');
      ObjSet(lbl+"v_p20", DoubleToString(g_proj_levels[3], _Digits), clrGold);
      ObjSet(lbl+"v_p25", DoubleToString(g_proj_levels[4], _Digits), clrGold);
   }
   else
   {
      ObjSet(lbl+"v_p05", "---", clrGray);
      ObjSet(lbl+"v_p10", "---", clrGray);
      ObjSet(lbl+"v_p15", "---", clrGray);
      ObjSet(lbl+"v_p20", "---", clrGray);
      ObjSet(lbl+"v_p25", "---", clrGray);
   }

   ChartRedraw(0);
}


//+------------------------------------------------------------------+
//| DASHBOARD HELPER - Create Label Object                            |
//+------------------------------------------------------------------+
void ObjLbl(string name, string txt, int x, int y, color c, int fs=8, bool bold=false)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER,    CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0,  name, OBJPROP_TEXT,      txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR,     c);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,  fs);
   ObjectSetString(0,  name, OBJPROP_FONT,      bold ? "Arial Bold" : "Arial");
   ObjectSetInteger(0, name, OBJPROP_BACK,      false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}


//+------------------------------------------------------------------+
//| DASHBOARD HELPER - Update Label Text and Color                    |
//+------------------------------------------------------------------+
void ObjSet(string name, string txt, color c)
{
   if(ObjectFind(0, name) < 0) return;
   ObjectSetString(0,  name, OBJPROP_TEXT,  txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
}


//+------------------------------------------------------------------+
//| DASHBOARD HELPER - Create Separator Line                          |
//+------------------------------------------------------------------+
void ObjLine(string name, int x, int y, int w)
{
   string d = "";
   int n = (int)(w / 5.5);
   for(int i = 0; i < n; i++) d += "-";
   ObjLbl(name, d, x, y, C'40,55,85', 6);
}


//+------------------------------------------------------------------+
//| DASHBOARD HELPER - Create Background Rectangle                    |
//+------------------------------------------------------------------+
void ObjRect(string name, int x, int y, int w, int h, color bg, color border, int bw)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER,      CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE,   x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE,   y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE,       w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE,       h);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR,     bg);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_COLOR,       border);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,       bw);
   ObjectSetInteger(0, name, OBJPROP_BACK,        false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE,  false);
}


//+------------------------------------------------------------------+
//| TIMEFRAME STRING HELPER                                           |
//+------------------------------------------------------------------+
string TFStr(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return "?";
   }
}
//+------------------------------------------------------------------+
