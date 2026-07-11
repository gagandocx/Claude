//+------------------------------------------------------------------+
//|                                           GFM_EA.mq5     |
//|                    T-Spot Trading Model Expert Advisor v3.0       |
//|                    Full Chart Drawing + Touch Entry System        |
//|                                                                  |
//| STRATEGY:                                                        |
//|  - Draws all indicator elements on chart (T-Spot zones, FVGs,    |
//|    PFVGs, Volume Imbalances, CISD, Projections, Silver T-Spot)   |
//|  - Touch Entry: price touches T-Spot box -> immediate market order|
//|  - Swing-based SL, 1:1 RR TP, 80% trailing activation           |
//+------------------------------------------------------------------+
#property copyright "GFM EA v3.0 - T-Spot Touch Entry"
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade        trade;
CPositionInfo posInfo;

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== HTF & DISPLAY ==="
input ENUM_TIMEFRAMES HTF_Timeframe    = PERIOD_M15;   // HTF Timeframe (Auto=M15 for M1)
input int    Max_Display                = 4;            // Max T-Spot display count
input bool   Show_HTF                   = true;         // Show HTF Start Lines
input bool   Use_Actual_Day_Change      = false;        // Use actual day change for Daily HTF

input group "=== T-SPOT SETTINGS ==="
input string TSpot_Bias                 = "None";       // None/Bullish/Bearish filter
input bool   Show_TSpot                 = true;         // Show T-Spot zones
input int    TSpot_Transparency         = 90;           // T-Spot zone transparency (0-100)
input bool   Extend_Latest_TSpot        = true;         // Extend latest T-Spot to current bar
input bool   Show_Only_Latest_TSpot     = true;         // Show only latest T-Spot
input bool   Show_TSpot_Close_Line      = true;         // Show T-Spot close line
input int    Midline_Width              = 1;            // Midline width
input ENUM_LINE_STYLE TSpot_Line_Style  = STYLE_SOLID;  // T-Spot line style
input bool   Hide_TSpots_Against        = true;         // Hide T-Spots when trading against
input bool   Delete_TSpots_Against      = false;        // Delete T-Spots when trading against

input group "=== SWEEP CONFIRMATION ==="
input bool   Show_TSpot_Sweeps          = true;         // Show T-Spot sweep confirmations
input bool   Show_Confirmation_Lines    = true;         // Show confirmation lines
input ENUM_LINE_STYLE Confirm_Line_Style= STYLE_SOLID;  // Confirmation line style
input bool   Use_Body_Confirmation      = true;         // Use body for confirmation pivots
input bool   Show_Confirmation_Labels   = true;         // Show C2/C3/C4 labels
input bool   Show_TTFM_Labels           = true;         // Show TTFM labels
input color  Sweep_Label_Color          = clrWhite;     // Sweep label color
input bool   Show_Only_Latest_Sweep     = false;        // Show only latest sweep

input group "=== FVG SETTINGS ==="
input bool   Show_FVG                   = true;         // Show Fair Value Gaps
input color  FVG_Color                  = clrGray;      // FVG color
input int    FVG_Transparency           = 80;           // FVG transparency

input group "=== PFVG SETTINGS ==="
input bool   Show_PFVG                  = true;         // Show First Presented FVG
input color  PFVG_Color                 = clrDodgerBlue;// PFVG color
input int    PFVG_Transparency          = 80;           // PFVG transparency
input bool   Show_All_PFVGs             = false;        // Show all PFVGs (vs first only)
input int    Max_PFVG_Display           = 4;            // Max PFVG display
input bool   PFVG_Only_With_TSpot       = false;        // Only show PFVG with T-Spot

input group "=== VOLUME IMBALANCE ==="
input bool   Show_Volume_Imbalance      = true;         // Show Volume Imbalances
input color  VI_Color                   = clrOrange;    // Volume Imbalance color
input int    VI_Transparency            = 80;           // Volume Imbalance transparency

input group "=== CISD SETTINGS ==="
input bool   Show_CISD                  = true;         // Show CISD lines
input color  CISD_Color                 = clrYellow;    // CISD line color
input int    CISD_Width                 = 2;            // CISD line width
input bool   Show_All_CISD              = true;         // Show all CISD lines

input group "=== PROJECTION SETTINGS ==="
input bool   Show_Projections           = true;         // Show projection lines
input bool   Extend_Latest_Projections  = true;         // Extend latest projections to current bar
input string Projection_Levels_Str      = "0.5,1.0,1.5,2.0,2.5"; // Projection levels

input group "=== CHART SWEEP LINES ==="
input bool   Show_Chart_Sweeps          = true;         // Show chart sweep lines

input group "=== SILVER T-SPOT ==="
input bool   Show_Silver_TSpot          = true;         // Show Silver T-Spot detection

input group "=== HTF START LINE ==="
input int    HTF_Line_Width             = 1;            // HTF start line width
input ENUM_LINE_STYLE HTF_Line_Style    = STYLE_DOT;    // HTF start line style
input color  HTF_Line_Color             = clrBlack;     // HTF start line color

input group "=== ENTRY & RISK MANAGEMENT ==="
input double Risk_Percent               = 5.0;          // Risk % per trade
input double Max_Lot                    = 10.0;         // Maximum lot size
input double Min_Lot                    = 0.1;          // Minimum lot size
input int    Invalidation_Closes        = 3;            // Consecutive closes beyond midline to invalidate
input int    SL_Buffer_Pips             = 5;            // SL buffer beyond swing (pips)
input int    Swing_Lookback_Bars        = 50;           // Bars to scan for swing high/low

input group "=== POSITION MANAGEMENT ==="
input int    BE_Plus_Pips               = 3;            // Pips into profit for breakeven
input int    Trail_Distance_Pips        = 10;           // Trailing stop distance (pips)

input group "=== SYSTEM ==="
input int    Magic_Number               = 777000;       // Magic Number
input string EA_Comment                 = "GFM";// Trade comment
input int    Max_Slippage               = 10;           // Max slippage (points)
input bool   Show_Dashboard             = true;         // Show info panel



//+------------------------------------------------------------------+
//| ENUMERATIONS                                                      |
//+------------------------------------------------------------------+
enum ENUM_TRADE_STATE
{
   STATE_NONE     = 0,   // No position, waiting for T-Spot
   STATE_MONITORING = 1, // T-Spot detected, waiting for price to touch the box
   STATE_FULL     = 2,   // Position open, waiting for 1:1 to move to BE
   STATE_BE       = 3,   // SL at breakeven, waiting for 1:4 to start trailing
   STATE_TRAILING = 4    // Trailing active, TP removed
};

enum ENUM_TSPOT_DIR
{
   TSPOT_NONE    = 0,
   TSPOT_BULLISH = 1,
   TSPOT_BEARISH = -1
};

enum ENUM_HTF_BIAS
{
   BIAS_NEUTRAL  = 0,
   BIAS_BULLISH  = 1,
   BIAS_BEARISH  = -1
};


//+------------------------------------------------------------------+
//| STRUCTURES                                                        |
//+------------------------------------------------------------------+
struct TSpotData
{
   ENUM_TSPOT_DIR direction;
   double         midline;        // LogMidpoint of sweep candle
   double         close_level;    // Close of sweep candle
   double         high;           // High of sweep candle
   double         low;            // Low of sweep candle
   datetime       time_start;     // Start time of T-Spot
   datetime       time_end;       // End time (extended)
   bool           is_active;      // Still valid
   bool           is_hidden;      // Hidden due to trading against
   string         pattern_name;   // Detection pattern name
   bool           touched;        // Price has touched close_level
   datetime       touch_time;     // When touch occurred
   double         pivot_price;    // Pivot price for confirmation
   datetime       pivot_time;     // When pivot formed
   bool           pivot_formed;   // Pivot has formed
   bool           confirmed;      // Sweep confirmation completed
   datetime       confirm_time;   // When confirmation occurred
   int            invalidation_count; // Consecutive closes beyond midline
   int            index;          // Array index for naming
};

struct FVGData
{
   double   high_price;      // Top of FVG
   double   low_price;       // Bottom of FVG
   datetime time_start;      // Start bar time
   datetime time_end;        // End bar time
   bool     is_bullish;      // Direction
   bool     is_active;       // Still valid
   int      index;           // Array index for naming
};

struct PFVGData
{
   double   high_price;
   double   low_price;
   datetime time_start;
   datetime time_end;
   bool     is_bullish;
   bool     is_active;
   int      htf_index;       // Which HTF candle it belongs to
   int      index;
};

struct VIData
{
   double   high_price;
   double   low_price;
   datetime time_start;
   datetime time_end;
   bool     is_bullish;
   bool     is_active;
   int      index;
};

struct CISDData
{
   double   price;           // Break level
   datetime time_start;
   datetime time_end;
   ENUM_TSPOT_DIR direction;
   double   series_range;    // Range of the series
   double   series_high;
   double   series_low;
   bool     is_active;
   int      index;
};

struct ProjectionData
{
   double   levels[5];       // 0.5, 1.0, 1.5, 2.0, 2.5
   double   break_price;
   double   series_range;
   ENUM_TSPOT_DIR direction;
   datetime time_start;
   datetime time_end;
   bool     is_active;
   int      index;
};


//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
double pip, point_size;

// Object prefix for cleanup
string lbl = "GM_";

// State machine
ENUM_TRADE_STATE g_state = STATE_NONE;

// T-Spot arrays
TSpotData g_tspots[];
int g_tspot_count = 0;
int g_tspot_total_idx = 0;

// Active T-Spot for monitoring (latest detected, used for new confirmations)
int g_active_tspot_idx = -1;

// T-Spot that triggered the current trade (locked at entry, used for invalidation)
int g_trade_tspot_idx = -1;

// FVG arrays
FVGData g_fvgs[];
int g_fvg_count = 0;
int g_fvg_total_idx = 0;

// PFVG arrays
PFVGData g_pfvgs[];
int g_pfvg_count = 0;
int g_pfvg_total_idx = 0;

// Volume Imbalance arrays
VIData g_vis[];
int g_vi_count = 0;
int g_vi_total_idx = 0;

// CISD arrays
CISDData g_cisds[];
int g_cisd_count = 0;
int g_cisd_total_idx = 0;

// Projection data
ProjectionData g_projections[];
int g_proj_count = 0;
int g_proj_total_idx = 0;

// HTF tracking
datetime g_last_htf_time = 0;
MqlRates g_htf_rates[];
int g_htf_line_count = 0;
int g_htf_line_total_idx = 0;

// Position tracking
ulong g_position_ticket = 0;
double g_entry_price = 0;
double g_sl_price = 0;
double g_tp_price = 0;
double g_float_pnl = 0;
ENUM_TSPOT_DIR g_trade_dir = TSPOT_NONE;

// Dashboard variables
string g_state_text = "Waiting";
string g_tspot_text = "None";
color  g_tspot_color = clrGray;
string g_bias_text = "Neutral";
color  g_bias_color = clrGray;

// Chart sweep tracking
int g_sweep_line_count = 0;
int g_sweep_total_idx = 0;

// Silver T-Spot tracking
int g_silver_count = 0;
int g_silver_total_idx = 0;
datetime g_last_silver_h4_time = 0;

// Confirmation label tracking
int g_confirm_label_count = 0;
int g_confirm_total_idx = 0;

// Last M1 bar time
datetime g_last_m1_bar = 0;

// Projection level values (parsed)
double g_proj_level_values[5];

// HTF Bias
ENUM_HTF_BIAS g_htf_bias = BIAS_NEUTRAL;

// Pivot tracking for sweep confirmation (per active T-Spot)
double g_pivot_highs[];     // Recent pivot highs
datetime g_pivot_high_times[];
int g_pivot_high_count = 0;
double g_pivot_lows[];      // Recent pivot lows
datetime g_pivot_low_times[];
int g_pivot_low_count = 0;



//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   point_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   pip = (digits == 3 || digits == 5) ? point_size * 10 : point_size;

   // Parse projection levels
   g_proj_level_values[0] = 0.5;
   g_proj_level_values[1] = 1.0;
   g_proj_level_values[2] = 1.5;
   g_proj_level_values[3] = 2.0;
   g_proj_level_values[4] = 2.5;

   // Initialize arrays
   ArrayResize(g_tspots, 0);
   ArrayResize(g_fvgs, 0);
   ArrayResize(g_pfvgs, 0);
   ArrayResize(g_vis, 0);
   ArrayResize(g_cisds, 0);
   ArrayResize(g_projections, 0);
   ArrayResize(g_pivot_highs, 0);
   ArrayResize(g_pivot_high_times, 0);
   ArrayResize(g_pivot_lows, 0);
   ArrayResize(g_pivot_low_times, 0);

   ArraySetAsSeries(g_htf_rates, true);

   // Trade setup
   trade.SetExpertMagicNumber(Magic_Number);
   trade.SetDeviationInPoints(Max_Slippage);

   // Auto-detect fill type (FOK not supported by all brokers, especially for Gold/XAUUSD)
   long fill_type = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((fill_type & SYMBOL_FILLING_IOC) != 0)
      trade.SetTypeFilling(ORDER_FILLING_IOC);
   else if((fill_type & SYMBOL_FILLING_FOK) != 0)
      trade.SetTypeFilling(ORDER_FILLING_FOK);
   else
      trade.SetTypeFilling(ORDER_FILLING_RETURN);

   // Sync from market if position exists
   SyncStateFromMarket();

   // Create dashboard
   if(Show_Dashboard) CreateDashboard();

   // Initial scan for existing structures
   InitialChartScan();

   Print("GFM EA v3.0 initialized | T-Spot Touch Entry | ",
         TFStr(PERIOD_CURRENT), "-", TFStr(HTF_Timeframe), " Model");

   return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, lbl);
   ChartRedraw(0);
}


//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new HTF candle
   bool new_htf = IsNewHTFCandle();

   if(new_htf)
   {
      // Copy HTF rates
      if(CopyRates(_Symbol, HTF_Timeframe, 0, 10, g_htf_rates) < 4) return;

      // Draw HTF start line
      if(Show_HTF) DrawHTFStartLine();

      // Detect T-Spot on closed HTF candle
      DetectTSpot();

      // Detect CISD
      if(Show_CISD) DetectCISD();

      // Calculate HTF Bias
      CalculateHTFBias();

      // Periodic pruning of inactive array entries to prevent memory growth
      PruneInactiveEntries();
   }

   // New M1 bar processing
   datetime cur_m1 = iTime(_Symbol, PERIOD_M1, 0);
   if(cur_m1 != g_last_m1_bar)
   {
      g_last_m1_bar = cur_m1;

      // Detect FVGs on M1
      if(Show_FVG) DetectFVG();

      // Detect PFVGs
      if(Show_PFVG) DetectPFVG();

      // Detect Volume Imbalances
      if(Show_Volume_Imbalance) DetectVolumeImbalance();

      // Update pivot tracking for swing detection
      UpdatePivotTracking();

      // Check invalidation (close beyond midline)
      CheckAllInvalidation();

      // Silver T-Spot detection
      if(Show_Silver_TSpot) DetectSilverTSpot();

      // Hide T-Spots trading against
      if(Hide_TSpots_Against) CheckHideTSpots();
   }

   // Check touch entry every tick (price may touch T-Spot box at any moment)
   CheckTouchEntry();

   // Position management every tick
   ManagePosition();

   // Extend latest T-Spot and projections to current bar
   if(Extend_Latest_TSpot) ExtendLatestTSpot();
   if(Extend_Latest_Projections) ExtendLatestProjections();

   // Update dashboard
   if(Show_Dashboard) UpdateDashboard();
}


//+------------------------------------------------------------------+
//| IS NEW HTF CANDLE                                                 |
//+------------------------------------------------------------------+
bool IsNewHTFCandle()
{
   datetime current_htf_time = iTime(_Symbol, HTF_Timeframe, 0);
   if(current_htf_time == 0) return false;

   if(g_last_htf_time == 0)
   {
      g_last_htf_time = current_htf_time;
      return true;  // First run
   }

   if(current_htf_time != g_last_htf_time)
   {
      g_last_htf_time = current_htf_time;
      return true;
   }

   return false;
}



//+------------------------------------------------------------------+
//| DRAW HTF START LINE                                               |
//| Vertical dotted line at each HTF candle boundary                 |
//+------------------------------------------------------------------+
void DrawHTFStartLine()
{
   datetime htf_time = iTime(_Symbol, HTF_Timeframe, 0);
   if(htf_time == 0) return;

   string name = lbl + "htf_vline_" + IntegerToString(g_htf_line_total_idx);
   g_htf_line_total_idx++;
   g_htf_line_count++;

   ObjectCreate(0, name, OBJ_VLINE, 0, htf_time, 0);
   ObjectSetInteger(0, name, OBJPROP_COLOR, HTF_Line_Color);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, HTF_Line_Width);
   ObjectSetInteger(0, name, OBJPROP_STYLE, HTF_Line_Style);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);

   // Remove old lines if too many
   int max_lines = Max_Display * 20;
   if(g_htf_line_count > max_lines)
   {
      int remove_idx = g_htf_line_total_idx - g_htf_line_count;
      string old_name = lbl + "htf_vline_" + IntegerToString(remove_idx);
      ObjectDelete(0, old_name);
      g_htf_line_count--;
   }
}


//+------------------------------------------------------------------+
//| T-SPOT DETECTION ENGINE                                           |
//| Runs on new HTF candle close. Detects 6 patterns.                |
//| Does NOT place trade - starts monitoring for sweep confirmation  |
//+------------------------------------------------------------------+
void DetectTSpot()
{
   if(ArraySize(g_htf_rates) < 4) return;

   // g_htf_rates[0] = current building candle
   // g_htf_rates[1] = last closed candle (the sweep candle)
   // g_htf_rates[2] = previous closed candle
   // g_htf_rates[3] = two candles back
   MqlRates last_closed = g_htf_rates[1];
   MqlRates prev_closed = g_htf_rates[2];
   MqlRates prev_prev   = g_htf_rates[3];

   double close_price = last_closed.close;

   // Calculate midpoints
   double mid_last = LogMidpoint(last_closed.high, last_closed.low,
                                  last_closed.open, last_closed.close);
   double mid_prev = LogMidpoint(prev_closed.high, prev_closed.low,
                                  prev_closed.open, prev_closed.close);
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

   // If T-Spot detected, add to array and draw
   if(detected != TSPOT_NONE)
   {
      TSpotData ts;
      ts.direction     = detected;
      ts.midline       = mid_last;
      ts.close_level   = last_closed.close;
      ts.high          = last_closed.high;
      ts.low           = last_closed.low;
      ts.time_start    = g_htf_rates[0].time;                          // New HTF candle open time (after sweep closed)
      ts.time_end      = g_htf_rates[0].time + PeriodSeconds(HTF_Timeframe); // End of new HTF candle period
      ts.is_active     = true;
      ts.is_hidden     = false;
      ts.pattern_name  = pattern_name;
      ts.touched       = false;
      ts.touch_time    = 0;
      ts.pivot_price   = 0;
      ts.pivot_time    = 0;
      ts.pivot_formed  = false;
      ts.confirmed     = false;
      ts.confirm_time  = 0;
      ts.invalidation_count = 0;
      ts.index         = g_tspot_total_idx;

      // Add to array
      int size = ArraySize(g_tspots);
      ArrayResize(g_tspots, size + 1);
      g_tspots[size] = ts;
      g_tspot_count++;
      g_tspot_total_idx++;

      // Set as active monitoring target
      g_active_tspot_idx = size;

      // Draw the T-Spot zone on chart
      if(Show_TSpot) DrawTSpotZone(g_tspots[size]);

      // Draw TTFM label
      if(Show_TTFM_Labels) DrawTTFMLabel(g_tspots[size]);

      // Draw chart sweep line (the level that was swept)
      if(Show_Chart_Sweeps) DrawChartSweepLine(detected, last_closed, prev_closed);

      // Reset pivot tracking for new confirmation cycle
      ResetPivotTracking();

      Print("T-Spot Detected: ", pattern_name,
            " | Dir=", (detected == TSPOT_BULLISH ? "BULL" : "BEAR"),
            " | Midline=", DoubleToString(mid_last, _Digits),
            " | Close=", DoubleToString(last_closed.close, _Digits),
            " | MONITORING for touch entry");

      // Update state for monitoring
      if(g_state == STATE_NONE)
      {
         g_state = STATE_MONITORING;
         g_state_text = "Monitoring";
         g_tspot_text = (detected == TSPOT_BULLISH) ? "Bullish" : "Bearish";
         g_tspot_color = (detected == TSPOT_BULLISH) ? clrLime : clrRed;
      }

      // Limit display
      EnforceMaxDisplay();
   }
}



//+------------------------------------------------------------------+
//| DRAW T-SPOT ZONE                                                  |
//| Rectangle from midline to close level + midline + close line     |
//+------------------------------------------------------------------+
void DrawTSpotZone(const TSpotData &ts)
{
   int idx = ts.index;
   string base = lbl + "tspot_" + IntegerToString(idx);

   // Zone rectangle (OBJ_RECTANGLE)
   string box_name = base + "_box";
   double top_price = MathMax(ts.midline, ts.close_level);
   double bot_price = MathMin(ts.midline, ts.close_level);

   ObjectCreate(0, box_name, OBJ_RECTANGLE, 0,
                ts.time_start, top_price,
                ts.time_end, bot_price);

   // Calculate muted color based on transparency (simulates alpha on dark background)
   int intensity = (int)(255.0 * (100 - TSpot_Transparency) / 100.0);
   intensity = MathMax(15, MathMin(intensity, 255));
   
   color zone_color;
   if(ts.direction == TSPOT_BULLISH)
      zone_color = (color)StringToColor(StringFormat("%d,%d,%d", 0, intensity, 0));
   else
      zone_color = (color)StringToColor(StringFormat("%d,%d,%d", intensity, 0, 0));

   ObjectSetInteger(0, box_name, OBJPROP_COLOR, zone_color);
   ObjectSetInteger(0, box_name, OBJPROP_FILL, true);
   ObjectSetInteger(0, box_name, OBJPROP_BACK, true);
   ObjectSetInteger(0, box_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, box_name, OBJPROP_HIDDEN, true);

   // Midline (OBJ_TREND horizontal)
   string mid_name = base + "_mid";
   ObjectCreate(0, mid_name, OBJ_TREND, 0,
                ts.time_start, ts.midline,
                ts.time_end, ts.midline);
   color mid_color = (ts.direction == TSPOT_BULLISH) ? C'0,180,0' : C'180,0,0';
   ObjectSetInteger(0, mid_name, OBJPROP_COLOR, mid_color);
   ObjectSetInteger(0, mid_name, OBJPROP_WIDTH, Midline_Width);
   ObjectSetInteger(0, mid_name, OBJPROP_STYLE, TSpot_Line_Style);
   ObjectSetInteger(0, mid_name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, mid_name, OBJPROP_BACK, false);
   ObjectSetInteger(0, mid_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, mid_name, OBJPROP_HIDDEN, true);

   // Close line (OBJ_TREND horizontal)
   if(Show_TSpot_Close_Line)
   {
      string close_name = base + "_close";
      ObjectCreate(0, close_name, OBJ_TREND, 0,
                   ts.time_start, ts.close_level,
                   ts.time_end, ts.close_level);
      color close_color = (ts.direction == TSPOT_BULLISH) ? C'0,100,0' : C'100,0,0';
      ObjectSetInteger(0, close_name, OBJPROP_COLOR, close_color);
      ObjectSetInteger(0, close_name, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, close_name, OBJPROP_STYLE, TSpot_Line_Style);
      ObjectSetInteger(0, close_name, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, close_name, OBJPROP_BACK, false);
      ObjectSetInteger(0, close_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, close_name, OBJPROP_HIDDEN, true);
   }
}


//+------------------------------------------------------------------+
//| DRAW CHART SWEEP LINE                                             |
//| Horizontal line showing the level that was swept                 |
//+------------------------------------------------------------------+
void DrawChartSweepLine(ENUM_TSPOT_DIR dir, const MqlRates &last_closed, const MqlRates &prev_closed)
{
   double sweep_level = 0;
   if(dir == TSPOT_BEARISH)
      sweep_level = prev_closed.high;  // The high that was swept
   else
      sweep_level = prev_closed.low;   // The low that was swept

   string name = lbl + "sweep_" + IntegerToString(g_sweep_total_idx);
   g_sweep_total_idx++;
   g_sweep_line_count++;

   ObjectCreate(0, name, OBJ_TREND, 0,
                prev_closed.time, sweep_level,
                last_closed.time + PeriodSeconds(HTF_Timeframe), sweep_level);
   color sweep_color = (dir == TSPOT_BULLISH) ? clrDarkGreen : clrDarkRed;
   ObjectSetInteger(0, name, OBJPROP_COLOR, sweep_color);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| TOUCH ENTRY SYSTEM                                                |
//| Checks if price touches the T-Spot box (midline to close_level)  |
//| Immediate market order on touch - no sweep confirmation needed   |
//+------------------------------------------------------------------+
void CheckTouchEntry()
{
   if(g_state != STATE_MONITORING && g_state != STATE_NONE) return;

   // Already in a trade - don't enter
   if(g_state == STATE_FULL || g_state == STATE_TRAILING) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   for(int i = ArraySize(g_tspots) - 1; i >= 0; i--)
   {
      if(!g_tspots[i].is_active) continue;
      if(g_tspots[i].confirmed) continue;
      if(g_tspots[i].is_hidden) continue;

      // The T-Spot box is the zone between midline and close_level
      // For BEARISH: midline is above close_level (box top=midline, bottom=close_level)
      //   Price enters from below: ask >= close_level -> SELL
      // For BULLISH: midline is below close_level (box top=close_level, bottom=midline)
      //   Price enters from above: bid <= close_level -> BUY

      if(g_tspots[i].direction == TSPOT_BEARISH)
      {
         // Bearish T-Spot box: midline (top) to close_level (bottom)
         // Entry when price enters from below: ask >= close_level
         // Wrong direction check: if price is above midline, don't enter (coming from above)
         if(ask >= g_tspots[i].close_level && ask <= g_tspots[i].midline)
         {
            Print("TOUCH ENTRY! | idx=", i,
                  " | Dir=BEAR | Ask=", DoubleToString(ask, _Digits),
                  " | Close_Level=", DoubleToString(g_tspots[i].close_level, _Digits),
                  " | PLACING SELL");

            // Execute trade - only mark confirmed if trade succeeds
            g_active_tspot_idx = i;
            if(ExecuteTradeEntry(g_tspots[i]))
            {
               g_tspots[i].confirmed = true;
               g_tspots[i].confirm_time = TimeCurrent();

               if(Show_Confirmation_Lines)
                  DrawConfirmationLine(g_tspots[i]);
               if(Show_TSpot_Sweeps)
                  DrawSweepConfirmationLabel(g_tspots[i]);

               CalculateProjections(g_tspots[i]);
            }
            // If trade failed, don't mark confirmed - will retry next tick
            break;
         }
      }
      else // TSPOT_BULLISH
      {
         // Bullish T-Spot box: close_level (top) to midline (bottom)
         // Entry when price enters from above: bid <= close_level
         // Wrong direction check: if price is below midline, don't enter (coming from below)
         if(bid <= g_tspots[i].close_level && bid >= g_tspots[i].midline)
         {
            Print("TOUCH ENTRY! | idx=", i,
                  " | Dir=BULL | Bid=", DoubleToString(bid, _Digits),
                  " | Close_Level=", DoubleToString(g_tspots[i].close_level, _Digits),
                  " | PLACING BUY");

            // Execute trade - only mark confirmed if trade succeeds
            g_active_tspot_idx = i;
            if(ExecuteTradeEntry(g_tspots[i]))
            {
               g_tspots[i].confirmed = true;
               g_tspots[i].confirm_time = TimeCurrent();

               if(Show_Confirmation_Lines)
                  DrawConfirmationLine(g_tspots[i]);
               if(Show_TSpot_Sweeps)
                  DrawSweepConfirmationLabel(g_tspots[i]);

               CalculateProjections(g_tspots[i]);
            }
            // If trade failed, don't mark confirmed - will retry next tick
            break;
         }
      }
   }
}



//+------------------------------------------------------------------+
//| UPDATE PIVOT TRACKING                                             |
//| Finds pivot highs and lows using body (Use_Body_Confirmation)    |
//| Lookback of 2 bars (ta.pivothigh/low equivalent)                 |
//+------------------------------------------------------------------+
void UpdatePivotTracking()
{
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 0, 6, m1) < 6) return;

   // Pivot high check on bar[2] (need 2 bars on each side)
   // Using body: max(open, close)
   double bar2_val, bar1_val, bar3_val;

   if(Use_Body_Confirmation)
   {
      bar2_val = MathMax(m1[2].open, m1[2].close);
      bar1_val = MathMax(m1[1].open, m1[1].close);
      bar3_val = MathMax(m1[3].open, m1[3].close);
   }
   else
   {
      bar2_val = m1[2].high;
      bar1_val = m1[1].high;
      bar3_val = m1[3].high;
   }

   // Check if bar[2] is a pivot high (higher than both neighbors)
   if(bar2_val > bar1_val && bar2_val > bar3_val)
   {
      // Check if this pivot already exists
      bool exists = false;
      for(int p = 0; p < g_pivot_high_count; p++)
      {
         if(g_pivot_high_times[p] == m1[2].time) { exists = true; break; }
      }
      if(!exists)
      {
         int sz = g_pivot_high_count;
         ArrayResize(g_pivot_highs, sz + 1);
         ArrayResize(g_pivot_high_times, sz + 1);
         g_pivot_highs[sz] = bar2_val;
         g_pivot_high_times[sz] = m1[2].time;
         g_pivot_high_count++;

         // Keep only last 50 pivots
         if(g_pivot_high_count > 50)
         {
            ArrayRemove(g_pivot_highs, 0, 1);
            ArrayRemove(g_pivot_high_times, 0, 1);
            g_pivot_high_count--;
         }
      }
   }

   // Pivot low check on bar[2]
   double bar2_low_val, bar1_low_val, bar3_low_val;

   if(Use_Body_Confirmation)
   {
      bar2_low_val = MathMin(m1[2].open, m1[2].close);
      bar1_low_val = MathMin(m1[1].open, m1[1].close);
      bar3_low_val = MathMin(m1[3].open, m1[3].close);
   }
   else
   {
      bar2_low_val = m1[2].low;
      bar1_low_val = m1[1].low;
      bar3_low_val = m1[3].low;
   }

   // Check if bar[2] is a pivot low (lower than both neighbors)
   if(bar2_low_val < bar1_low_val && bar2_low_val < bar3_low_val)
   {
      bool exists = false;
      for(int p = 0; p < g_pivot_low_count; p++)
      {
         if(g_pivot_low_times[p] == m1[2].time) { exists = true; break; }
      }
      if(!exists)
      {
         int sz = g_pivot_low_count;
         ArrayResize(g_pivot_lows, sz + 1);
         ArrayResize(g_pivot_low_times, sz + 1);
         g_pivot_lows[sz] = bar2_low_val;
         g_pivot_low_times[sz] = m1[2].time;
         g_pivot_low_count++;

         if(g_pivot_low_count > 50)
         {
            ArrayRemove(g_pivot_lows, 0, 1);
            ArrayRemove(g_pivot_low_times, 0, 1);
            g_pivot_low_count--;
         }
      }
   }
}


//+------------------------------------------------------------------+
//| DRAW CONFIRMATION LINE                                            |
//| Solid line at the pivot level that was swept                     |
//+------------------------------------------------------------------+
void DrawConfirmationLine(const TSpotData &ts)
{
   string name = lbl + "confirm_" + IntegerToString(ts.index);

   datetime end_time = ts.confirm_time + PeriodSeconds(HTF_Timeframe);
   ObjectCreate(0, name, OBJ_TREND, 0,
                ts.pivot_time, ts.pivot_price,
                end_time, ts.pivot_price);

   color conf_color = (ts.direction == TSPOT_BULLISH) ? clrLime : clrRed;
   ObjectSetInteger(0, name, OBJPROP_COLOR, conf_color);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, name, OBJPROP_STYLE, Confirm_Line_Style);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| DRAW SWEEP CONFIRMATION LABEL                                     |
//| Triangle marker with "T-Spot\nSweep" text                       |
//+------------------------------------------------------------------+
void DrawSweepConfirmationLabel(const TSpotData &ts)
{
   string name = lbl + "sweeplbl_" + IntegerToString(g_confirm_total_idx);
   g_confirm_total_idx++;
   g_confirm_label_count++;

   double label_price = 0;
   if(ts.direction == TSPOT_BULLISH)
      label_price = ts.pivot_price + 5 * pip;  // Above for bullish
   else
      label_price = ts.pivot_price - 5 * pip;  // Below for bearish

   ObjectCreate(0, name, OBJ_TEXT, 0, ts.confirm_time, label_price);
   ObjectSetString(0, name, OBJPROP_TEXT, "T-Spot\nSweep");
   ObjectSetInteger(0, name, OBJPROP_COLOR, Sweep_Label_Color);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
   ObjectSetString(0, name, OBJPROP_FONT, "Arial");
   ObjectSetInteger(0, name, OBJPROP_ANCHOR,
      (ts.direction == TSPOT_BULLISH) ? ANCHOR_LOWER : ANCHOR_UPPER);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);

   // Triangle marker
   string tri_name = lbl + "sweeptri_" + IntegerToString(g_confirm_total_idx - 1);
   int arrow_code = (ts.direction == TSPOT_BULLISH) ? 233 : 234;  // Up/Down triangle
   ObjectCreate(0, tri_name, OBJ_ARROW, 0, ts.confirm_time, ts.pivot_price);
   ObjectSetInteger(0, tri_name, OBJPROP_ARROWCODE, arrow_code);
   ObjectSetInteger(0, tri_name, OBJPROP_COLOR, Sweep_Label_Color);
   ObjectSetInteger(0, tri_name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, tri_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, tri_name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| DRAW C2/C3/C4 LABELS                                              |
//| C2 = Swept level bar, C3/C4 = Expansion bars                    |
//+------------------------------------------------------------------+
void DrawC234Labels(const TSpotData &ts)
{
   // C2 label at the swept level (pivot)
   string c2_name = lbl + "c2_" + IntegerToString(ts.index);
   ObjectCreate(0, c2_name, OBJ_TEXT, 0, ts.pivot_time, ts.pivot_price);
   ObjectSetString(0, c2_name, OBJPROP_TEXT, "C2");
   ObjectSetInteger(0, c2_name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, c2_name, OBJPROP_FONTSIZE, 7);
   ObjectSetString(0, c2_name, OBJPROP_FONT, "Arial");
   ObjectSetInteger(0, c2_name, OBJPROP_ANCHOR, ANCHOR_RIGHT);
   ObjectSetInteger(0, c2_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, c2_name, OBJPROP_HIDDEN, true);

   // C3 label at confirmation bar
   string c3_name = lbl + "c3_" + IntegerToString(ts.index);
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 0, 3, m1) >= 2)
   {
      double c3_price = (ts.direction == TSPOT_BULLISH) ? m1[1].high : m1[1].low;
      ObjectCreate(0, c3_name, OBJ_TEXT, 0, m1[1].time, c3_price);
      ObjectSetString(0, c3_name, OBJPROP_TEXT, "C3");
      ObjectSetInteger(0, c3_name, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, c3_name, OBJPROP_FONTSIZE, 7);
      ObjectSetString(0, c3_name, OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, c3_name, OBJPROP_ANCHOR,
         (ts.direction == TSPOT_BULLISH) ? ANCHOR_LOWER : ANCHOR_UPPER);
      ObjectSetInteger(0, c3_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, c3_name, OBJPROP_HIDDEN, true);

      // C4 label one bar further
      string c4_name = lbl + "c4_" + IntegerToString(ts.index);
      double c4_price = (ts.direction == TSPOT_BULLISH) ? m1[0].high : m1[0].low;
      ObjectCreate(0, c4_name, OBJ_TEXT, 0, m1[0].time, c4_price);
      ObjectSetString(0, c4_name, OBJPROP_TEXT, "C4");
      ObjectSetInteger(0, c4_name, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, c4_name, OBJPROP_FONTSIZE, 7);
      ObjectSetString(0, c4_name, OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, c4_name, OBJPROP_ANCHOR,
         (ts.direction == TSPOT_BULLISH) ? ANCHOR_LOWER : ANCHOR_UPPER);
      ObjectSetInteger(0, c4_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, c4_name, OBJPROP_HIDDEN, true);
   }
}



//+------------------------------------------------------------------+
//| FVG DETECTION AND DRAWING                                         |
//| Scans M1 bars for Fair Value Gaps                                |
//+------------------------------------------------------------------+
void DetectFVG()
{
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 0, 5, m1) < 5) return;

   // Check bar[1] for FVG (need bars 0, 1, 2 relative = current, prev, prev_prev)
   // Bullish FVG: bar[0].low > bar[2].high (gap up)
   // Bearish FVG: bar[0].high < bar[2].low (gap down)

   // Check the most recently closed triple (bars 1, 2, 3)
   // Bullish FVG: m1[1].low > m1[3].high
   if(m1[1].low > m1[3].high)
   {
      FVGData fvg;
      fvg.high_price  = m1[1].low;
      fvg.low_price   = m1[3].high;
      fvg.time_start  = m1[3].time;
      fvg.time_end    = m1[1].time;
      fvg.is_bullish  = true;
      fvg.is_active   = true;
      fvg.index       = g_fvg_total_idx;

      int sz = ArraySize(g_fvgs);
      ArrayResize(g_fvgs, sz + 1);
      g_fvgs[sz] = fvg;
      g_fvg_count++;
      g_fvg_total_idx++;

      DrawFVGBox(fvg);
      LimitFVGDisplay();
   }

   // Bearish FVG: m1[1].high < m1[3].low
   if(m1[1].high < m1[3].low)
   {
      FVGData fvg;
      fvg.high_price  = m1[3].low;
      fvg.low_price   = m1[1].high;
      fvg.time_start  = m1[3].time;
      fvg.time_end    = m1[1].time;
      fvg.is_bullish  = false;
      fvg.is_active   = true;
      fvg.index       = g_fvg_total_idx;

      int sz = ArraySize(g_fvgs);
      ArrayResize(g_fvgs, sz + 1);
      g_fvgs[sz] = fvg;
      g_fvg_count++;
      g_fvg_total_idx++;

      DrawFVGBox(fvg);
      LimitFVGDisplay();
   }
}


//+------------------------------------------------------------------+
//| DRAW FVG BOX                                                      |
//| Gray transparent rectangle                                       |
//+------------------------------------------------------------------+
void DrawFVGBox(const FVGData &fvg)
{
   string name = lbl + "fvg_" + IntegerToString(fvg.index);

   ObjectCreate(0, name, OBJ_RECTANGLE, 0,
                fvg.time_start, fvg.high_price,
                fvg.time_end, fvg.low_price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, FVG_Color);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   int alpha = (int)(FVG_Transparency * 255 / 100);
   SetObjectTransparency(name, alpha);
}


//+------------------------------------------------------------------+
//| PFVG DETECTION AND DRAWING                                        |
//| First Presented FVG within each HTF candle                       |
//+------------------------------------------------------------------+
void DetectPFVG()
{
   // Only detect PFVG if we are in a new HTF candle period
   datetime cur_htf = iTime(_Symbol, HTF_Timeframe, 0);
   if(cur_htf == 0) return;

   // Check if we already have a PFVG for this HTF candle
   if(!Show_All_PFVGs)
   {
      for(int i = ArraySize(g_pfvgs) - 1; i >= 0; i--)
      {
         if(g_pfvgs[i].htf_index == (int)(cur_htf / PeriodSeconds(HTF_Timeframe)))
            return;  // Already have one for this HTF period
      }
   }

   // Check PFVG filter
   if(PFVG_Only_With_TSpot)
   {
      bool has_active_tspot = false;
      for(int i = ArraySize(g_tspots) - 1; i >= 0; i--)
      {
         if(g_tspots[i].is_active)
         {
            has_active_tspot = true;
            break;
         }
      }
      if(!has_active_tspot) return;
   }

   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 0, 5, m1) < 5) return;

   // Check for FVG in recent bars within current HTF candle
   // Bullish PFVG
   if(m1[1].low > m1[3].high && m1[2].time >= cur_htf)
   {
      PFVGData pfvg;
      pfvg.high_price  = m1[1].low;
      pfvg.low_price   = m1[3].high;
      pfvg.time_start  = m1[3].time;
      pfvg.time_end    = m1[1].time;
      pfvg.is_bullish  = true;
      pfvg.is_active   = true;
      pfvg.htf_index   = (int)(cur_htf / PeriodSeconds(HTF_Timeframe));
      pfvg.index       = g_pfvg_total_idx;

      int sz = ArraySize(g_pfvgs);
      ArrayResize(g_pfvgs, sz + 1);
      g_pfvgs[sz] = pfvg;
      g_pfvg_count++;
      g_pfvg_total_idx++;

      DrawPFVGBox(pfvg);
      LimitPFVGDisplay();
      return;
   }

   // Bearish PFVG
   if(m1[1].high < m1[3].low && m1[2].time >= cur_htf)
   {
      PFVGData pfvg;
      pfvg.high_price  = m1[3].low;
      pfvg.low_price   = m1[1].high;
      pfvg.time_start  = m1[3].time;
      pfvg.time_end    = m1[1].time;
      pfvg.is_bullish  = false;
      pfvg.is_active   = true;
      pfvg.htf_index   = (int)(cur_htf / PeriodSeconds(HTF_Timeframe));
      pfvg.index       = g_pfvg_total_idx;

      int sz = ArraySize(g_pfvgs);
      ArrayResize(g_pfvgs, sz + 1);
      g_pfvgs[sz] = pfvg;
      g_pfvg_count++;
      g_pfvg_total_idx++;

      DrawPFVGBox(pfvg);
      LimitPFVGDisplay();
   }
}


//+------------------------------------------------------------------+
//| DRAW PFVG BOX                                                     |
//| Blue transparent rectangle                                       |
//+------------------------------------------------------------------+
void DrawPFVGBox(const PFVGData &pfvg)
{
   string name = lbl + "pfvg_" + IntegerToString(pfvg.index);

   ObjectCreate(0, name, OBJ_RECTANGLE, 0,
                pfvg.time_start, pfvg.high_price,
                pfvg.time_end, pfvg.low_price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, PFVG_Color);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   int alpha = (int)(PFVG_Transparency * 255 / 100);
   SetObjectTransparency(name, alpha);
}


//+------------------------------------------------------------------+
//| VOLUME IMBALANCE DETECTION AND DRAWING                            |
//| Gap between consecutive candle bodies                            |
//+------------------------------------------------------------------+
void DetectVolumeImbalance()
{
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 0, 4, m1) < 4) return;

   // Volume Imbalance: gap between bodies of bar[1] and bar[2]
   double body1_high = MathMax(m1[1].open, m1[1].close);
   double body1_low  = MathMin(m1[1].open, m1[1].close);
   double body2_high = MathMax(m1[2].open, m1[2].close);
   double body2_low  = MathMin(m1[2].open, m1[2].close);

   // Bullish VI: body1_low > body2_high (gap up between bodies)
   if(body1_low > body2_high)
   {
      VIData vi;
      vi.high_price  = body1_low;
      vi.low_price   = body2_high;
      vi.time_start  = m1[2].time;
      vi.time_end    = m1[1].time;
      vi.is_bullish  = true;
      vi.is_active   = true;
      vi.index       = g_vi_total_idx;

      int sz = ArraySize(g_vis);
      ArrayResize(g_vis, sz + 1);
      g_vis[sz] = vi;
      g_vi_count++;
      g_vi_total_idx++;

      DrawVIBox(vi);
      LimitVIDisplay();
   }

   // Bearish VI: body2_low > body1_high (gap down between bodies)
   if(body2_low > body1_high)
   {
      VIData vi;
      vi.high_price  = body2_low;
      vi.low_price   = body1_high;
      vi.time_start  = m1[2].time;
      vi.time_end    = m1[1].time;
      vi.is_bullish  = false;
      vi.is_active   = true;
      vi.index       = g_vi_total_idx;

      int sz = ArraySize(g_vis);
      ArrayResize(g_vis, sz + 1);
      g_vis[sz] = vi;
      g_vi_count++;
      g_vi_total_idx++;

      DrawVIBox(vi);
      LimitVIDisplay();
   }
}


//+------------------------------------------------------------------+
//| DRAW VOLUME IMBALANCE BOX                                         |
//| Orange transparent rectangle                                     |
//+------------------------------------------------------------------+
void DrawVIBox(const VIData &vi)
{
   string name = lbl + "vi_" + IntegerToString(vi.index);

   ObjectCreate(0, name, OBJ_RECTANGLE, 0,
                vi.time_start, vi.high_price,
                vi.time_end, vi.low_price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, VI_Color);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   int alpha = (int)(VI_Transparency * 255 / 100);
   SetObjectTransparency(name, alpha);
}



//+------------------------------------------------------------------+
//| CISD DETECTION AND DRAWING                                        |
//| Change In State of Delivery - series break detection             |
//+------------------------------------------------------------------+
void DetectCISD()
{
   if(ArraySize(g_htf_rates) < 8) return;

   // Look for consecutive same-polarity HTF candles (series)
   // then detect when price breaks through the series extreme

   // Check from bar 1 backward (last closed)
   MqlRates last = g_htf_rates[1];

   // Find the series ending at bar 2
   double series_high = 0;
   double series_low  = DBL_MAX;
   int series_len = 0;
   bool series_bullish = (g_htf_rates[2].close > g_htf_rates[2].open);

   for(int i = 2; i < ArraySize(g_htf_rates) && i < 8; i++)
   {
      bool is_bull = (g_htf_rates[i].close > g_htf_rates[i].open);
      if(is_bull != series_bullish && series_len > 0) break;

      if(Use_Body_Confirmation)
      {
         series_high = MathMax(series_high, MathMax(g_htf_rates[i].open, g_htf_rates[i].close));
         series_low  = MathMin(series_low,  MathMin(g_htf_rates[i].open, g_htf_rates[i].close));
      }
      else
      {
         series_high = MathMax(series_high, g_htf_rates[i].high);
         series_low  = MathMin(series_low,  g_htf_rates[i].low);
      }
      series_len++;
   }

   if(series_len < 2) return;  // Need at least 2 candles in series
   if(series_high == 0 || series_low == DBL_MAX) return;

   double series_range = series_high - series_low;
   if(series_range <= 0) return;

   bool cisd_detected = false;
   double break_price = 0;
   ENUM_TSPOT_DIR cisd_dir = TSPOT_NONE;

   // Bullish CISD: bearish series broken upward
   if(!series_bullish && last.close > series_high)
   {
      cisd_detected = true;
      break_price = series_high;
      cisd_dir = TSPOT_BULLISH;
   }

   // Bearish CISD: bullish series broken downward
   if(series_bullish && last.close < series_low)
   {
      cisd_detected = true;
      break_price = series_low;
      cisd_dir = TSPOT_BEARISH;
   }

   if(cisd_detected)
   {
      CISDData cisd;
      cisd.price        = break_price;
      cisd.time_start   = g_htf_rates[series_len + 1].time;
      cisd.time_end     = last.time + PeriodSeconds(HTF_Timeframe) * 3;
      cisd.direction    = cisd_dir;
      cisd.series_range = series_range;
      cisd.series_high  = series_high;
      cisd.series_low   = series_low;
      cisd.is_active    = true;
      cisd.index        = g_cisd_total_idx;

      int sz = ArraySize(g_cisds);
      ArrayResize(g_cisds, sz + 1);
      g_cisds[sz] = cisd;
      g_cisd_count++;
      g_cisd_total_idx++;

      DrawCISDLine(cisd);

      // Draw projections from this CISD
      if(Show_Projections) DrawProjectionLines(cisd);

      // Limit display
      if(!Show_All_CISD) LimitCISDDisplay();

      Print("CISD Detected | Dir=", (cisd_dir == TSPOT_BULLISH ? "BULL" : "BEAR"),
            " | Break=", DoubleToString(break_price, _Digits),
            " | Range=", DoubleToString(series_range, _Digits));
   }
}


//+------------------------------------------------------------------+
//| DRAW CISD LINE                                                    |
//| Yellow horizontal line at break level                            |
//+------------------------------------------------------------------+
void DrawCISDLine(const CISDData &cisd)
{
   string name = lbl + "cisd_" + IntegerToString(cisd.index);

   ObjectCreate(0, name, OBJ_TREND, 0,
                cisd.time_start, cisd.price,
                cisd.time_end, cisd.price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, CISD_Color);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, CISD_Width);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| DRAW PROJECTION LINES                                             |
//| Dotted horizontal lines at 0.5, 1.0, 1.5, 2.0, 2.5 levels      |
//+------------------------------------------------------------------+
void DrawProjectionLines(const CISDData &cisd)
{
   ProjectionData proj;
   proj.break_price   = cisd.price;
   proj.series_range  = cisd.series_range;
   proj.direction     = cisd.direction;
   proj.time_start    = cisd.time_start;
   proj.time_end      = iTime(_Symbol, PERIOD_M1, 0);
   proj.is_active     = true;
   proj.index         = g_proj_total_idx;

   double mult = (cisd.direction == TSPOT_BULLISH) ? 1.0 : -1.0;

   for(int l = 0; l < 5; l++)
   {
      proj.levels[l] = cisd.price + cisd.series_range * g_proj_level_values[l] * mult;
   }

   int sz = ArraySize(g_projections);
   ArrayResize(g_projections, sz + 1);
   g_projections[sz] = proj;
   g_proj_count++;
   g_proj_total_idx++;

   // Draw each projection line
   for(int l = 0; l < 5; l++)
   {
      string name = lbl + "proj_" + IntegerToString(proj.index) + "_" +
                    DoubleToString(g_proj_level_values[l], 1);

      datetime end_time = proj.time_end;
      ObjectCreate(0, name, OBJ_TREND, 0,
                   cisd.time_start, proj.levels[l],
                   end_time, proj.levels[l]);

      color proj_color = C'100,150,200';
      if(g_proj_level_values[l] >= 2.0) proj_color = clrGold;

      ObjectSetInteger(0, name, OBJPROP_COLOR, proj_color);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
      ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, name, OBJPROP_BACK, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);

      // Tiny label for level number
      string lbl_name = lbl + "projlbl_" + IntegerToString(proj.index) + "_" +
                        DoubleToString(g_proj_level_values[l], 1);
      ObjectCreate(0, lbl_name, OBJ_TEXT, 0, end_time, proj.levels[l]);
      ObjectSetString(0, lbl_name, OBJPROP_TEXT,
                      DoubleToString(g_proj_level_values[l], 1));
      ObjectSetInteger(0, lbl_name, OBJPROP_COLOR, proj_color);
      ObjectSetInteger(0, lbl_name, OBJPROP_FONTSIZE, 6);  // Tiny
      ObjectSetString(0, lbl_name, OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, lbl_name, OBJPROP_ANCHOR, ANCHOR_LEFT);
      ObjectSetInteger(0, lbl_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, lbl_name, OBJPROP_HIDDEN, true);
   }
}


//+------------------------------------------------------------------+
//| SILVER T-SPOT DETECTION                                           |
//| Check 4H candle block 4 (hour >= 13 NYC) or 5                   |
//+------------------------------------------------------------------+
void DetectSilverTSpot()
{
   // Get current time in NYC (UTC-5, or UTC-4 during DST)
   // Approximate: use server time minus 5 hours
   datetime server_time = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(server_time, dt);

   // Approximate NYC hour (simple offset - adjust for DST if needed)
   int nyc_hour = dt.hour - 5;
   if(nyc_hour < 0) nyc_hour += 24;

   // 4H block determination
   // Block 0: 00-04, Block 1: 04-08, Block 2: 08-12, Block 3: 12-16, Block 4: 16-20, Block 5: 20-24
   int block = nyc_hour / 4;

   // Only check in block 4 (hour >= 13 NYC within block) or block 5
   bool in_silver_window = false;
   if(block == 4 && nyc_hour >= 13) in_silver_window = true;
   if(block == 5) in_silver_window = true;

   if(!in_silver_window) return;

   // Get H4 rates for comparison
   MqlRates h4[];
   ArraySetAsSeries(h4, true);
   if(CopyRates(_Symbol, PERIOD_H4, 0, 4, h4) < 4) return;

   // Only detect once per H4 candle — prevent spam
   if(h4[0].time == g_last_silver_h4_time) return;

   // Check M1 for silver pattern
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 0, 3, m1) < 3) return;

   double close = m1[1].close;
   bool silver_bull = false;
   bool silver_bear = false;
   bool expansive = false;

   // For bullish: close > prev_prev.high AND close > prev_closed.high (H4 context)
   if(close > h4[2].high && close > h4[1].high)
   {
      silver_bull = true;
      if(close > h4[2].high + (h4[2].high - h4[2].low))
         expansive = true;
   }

   // For bearish: close < prev_prev.low AND close < prev_closed.low
   if(close < h4[2].low && close < h4[1].low)
   {
      silver_bear = true;
      if(close < h4[2].low - (h4[2].high - h4[2].low))
         expansive = true;
   }

   if(silver_bull || silver_bear)
   {
      g_last_silver_h4_time = h4[0].time;

      string label_text = expansive ? "Silver Expansive T-Spot" : "Silver T-Spot";
      string name = lbl + "silver_" + IntegerToString(g_silver_total_idx);
      g_silver_total_idx++;
      g_silver_count++;

      double label_price = silver_bull ? m1[1].high + 3 * pip : m1[1].low - 3 * pip;
      ObjectCreate(0, name, OBJ_TEXT, 0, m1[1].time, label_price);
      ObjectSetString(0, name, OBJPROP_TEXT, label_text);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clrSilver);
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 7);
      ObjectSetString(0, name, OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, name, OBJPROP_ANCHOR,
         silver_bull ? ANCHOR_LOWER : ANCHOR_UPPER);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);

      Print("Silver T-Spot detected: ", label_text,
            " | Dir=", (silver_bull ? "BULL" : "BEAR"));
   }
}



//+------------------------------------------------------------------+
//| CALCULATE PROJECTIONS FOR VISUAL REFERENCE                        |
//| Still draws projection levels on chart but NOT used for TP       |
//+------------------------------------------------------------------+
void CalculateProjections(const TSpotData &ts)
{
   // Find the most recent CISD that matches direction (visual reference only)
   int best_cisd = -1;
   for(int i = ArraySize(g_cisds) - 1; i >= 0; i--)
   {
      if(g_cisds[i].is_active && g_cisds[i].direction == ts.direction)
      {
         best_cisd = i;
         break;
      }
   }

   // Projections are drawn as visual aids by DrawProjectionLines
   // They are NOT used for trade TP (TP is 1:1 RR from entry/SL)
   // This function is kept for compatibility with projection drawing
}


//+------------------------------------------------------------------+
//| EXECUTE TRADE ENTRY                                               |
//| Market order when price touches T-Spot box                       |
//+------------------------------------------------------------------+
bool ExecuteTradeEntry(const TSpotData &ts)
{
   // Don't open if already in a trade
   if(g_state == STATE_FULL || g_state == STATE_BE || g_state == STATE_TRAILING)
   {
      Print("Already in a position, skipping new entry");
      return false;
   }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // Entry at current market price
   double entry_price = (ts.direction == TSPOT_BULLISH) ? ask : bid;

   // Calculate SL from nearest swing high/low
   double sl = CalculateSwingStopLoss(ts.direction, entry_price, ts);

   // Calculate 1:1.5 Risk:Reward TP
   double sl_distance = MathAbs(entry_price - sl);
   if(sl_distance <= 0) sl_distance = CalculateATRDistance();
   double tp = 0;
   if(ts.direction == TSPOT_BULLISH)
      tp = NormalizeDouble(entry_price + sl_distance * 1.5, _Digits);
   else
      tp = NormalizeDouble(entry_price - sl_distance * 1.5, _Digits);

   // Calculate lot size
   double lot = CalculateLotSize(sl_distance);

   bool result = false;

   if(ts.direction == TSPOT_BULLISH)
   {
      result = trade.Buy(lot, _Symbol, entry_price, sl, tp, EA_Comment);
   }
   else
   {
      result = trade.Sell(lot, _Symbol, entry_price, sl, tp, EA_Comment);
   }

   if(result)
   {
      g_position_ticket = trade.ResultOrder();
      g_state = STATE_FULL;
      g_state_text = "Open";
      g_entry_price = entry_price;
      g_sl_price = sl;
      g_tp_price = tp;
      g_trade_dir = ts.direction;
      g_trade_tspot_idx = g_active_tspot_idx;  // Lock the T-Spot index at trade entry

      Print("TRADE EXECUTED | Dir=", (ts.direction == TSPOT_BULLISH ? "BUY" : "SELL"),
            " | Entry=", DoubleToString(entry_price, _Digits),
            " | SL=", DoubleToString(sl, _Digits),
            " | TP=", DoubleToString(tp, _Digits),
            " | Lot=", DoubleToString(lot, 2),
            " | RR=1:1.5");

      // Draw entry marker and SL/TP lines on chart
      DrawEntryMarker(entry_price, TimeCurrent(), ts.direction);
      DrawSLTPLines(sl, tp, TimeCurrent());
      return true;
   }
   else
   {
      Print("TRADE FAILED | Error=", trade.ResultRetcodeDescription(),
            " | Code=", trade.ResultRetcode());
      return false;
   }
}


//+------------------------------------------------------------------+
//| CALCULATE SWING-BASED STOP LOSS                                   |
//| Uses pivot high/low detection (lookback 5 bars on M1)            |
//+------------------------------------------------------------------+
double CalculateSwingStopLoss(ENUM_TSPOT_DIR direction, double entry_price, const TSpotData &ts)
{
   MqlRates m1_rates[];
   ArraySetAsSeries(m1_rates, true);

   int bars_copied = CopyRates(_Symbol, PERIOD_M1, 0, Swing_Lookback_Bars, m1_rates);
   if(bars_copied < 10) return FallbackSL(direction, entry_price, ts);

   double sl = 0;

   if(direction == TSPOT_BEARISH)
   {
      // Find nearest swing HIGH above entry price
      // Pivot high: bar[i] higher than 5 bars on each side (lookback 5)
      double nearest_swing_high = 0;
      double min_distance = DBL_MAX;

      for(int i = 5; i < bars_copied - 5; i++)
      {
         bool is_pivot_high = true;
         for(int j = 1; j <= 5; j++)
         {
            if(m1_rates[i].high <= m1_rates[i-j].high ||
               m1_rates[i].high <= m1_rates[i+j].high)
            {
               is_pivot_high = false;
               break;
            }
         }

         if(is_pivot_high && m1_rates[i].high > entry_price)
         {
            double dist = m1_rates[i].high - entry_price;
            if(dist < min_distance)
            {
               min_distance = dist;
               nearest_swing_high = m1_rates[i].high;
            }
         }
      }

      if(nearest_swing_high > 0)
         sl = nearest_swing_high + SL_Buffer_Pips * pip;
      else
         sl = FallbackSL(direction, entry_price, ts);
   }
   else // TSPOT_BULLISH
   {
      // Find nearest swing LOW below entry price
      double nearest_swing_low = 0;
      double min_distance = DBL_MAX;

      for(int i = 5; i < bars_copied - 5; i++)
      {
         bool is_pivot_low = true;
         for(int j = 1; j <= 5; j++)
         {
            if(m1_rates[i].low >= m1_rates[i-j].low ||
               m1_rates[i].low >= m1_rates[i+j].low)
            {
               is_pivot_low = false;
               break;
            }
         }

         if(is_pivot_low && m1_rates[i].low < entry_price)
         {
            double dist = entry_price - m1_rates[i].low;
            if(dist < min_distance)
            {
               min_distance = dist;
               nearest_swing_low = m1_rates[i].low;
            }
         }
      }

      if(nearest_swing_low > 0)
         sl = nearest_swing_low - SL_Buffer_Pips * pip;
      else
         sl = FallbackSL(direction, entry_price, ts);
   }

   return NormalizeDouble(sl, _Digits);
}


//+------------------------------------------------------------------+
//| FALLBACK SL                                                       |
//| Uses T-Spot candle extremes when no swing found within lookback  |
//+------------------------------------------------------------------+
double FallbackSL(ENUM_TSPOT_DIR direction, double entry_price, const TSpotData &ts)
{
   // Use T-Spot candle extreme (high for sell, low for buy) + buffer
   if(direction == TSPOT_BEARISH)
      return NormalizeDouble(ts.high + SL_Buffer_Pips * pip, _Digits);
   else
      return NormalizeDouble(ts.low - SL_Buffer_Pips * pip, _Digits);
}


//+------------------------------------------------------------------+
//| CALCULATE ATR-BASED SL DISTANCE                                   |
//| Uses 14-period ATR on M15 to scale with instrument volatility    |
//+------------------------------------------------------------------+
double CalculateATRDistance()
{
   // Calculate ATR manually from M15 bars (14 period)
   MqlRates atr_rates[];
   ArraySetAsSeries(atr_rates, true);
   int copied = CopyRates(_Symbol, PERIOD_M15, 0, 15, atr_rates);
   if(copied < 15)
   {
      // Fallback if not enough data: use 2x recent M1 range
      MqlRates m1_fallback[];
      ArraySetAsSeries(m1_fallback, true);
      int m1_copied = CopyRates(_Symbol, PERIOD_M1, 0, 14, m1_fallback);
      if(m1_copied >= 14)
      {
         double sum_range = 0;
         for(int i = 0; i < 14; i++)
            sum_range += (m1_fallback[i].high - m1_fallback[i].low);
         return MathMax((sum_range / 14.0) * 2.0, 10 * pip);
      }
      // Absolute minimum fallback
      return 100 * pip;
   }

   // Manual ATR calculation (14-period average true range on M15)
   double sum_tr = 0;
   for(int i = 0; i < 14; i++)
   {
      double tr = atr_rates[i].high - atr_rates[i].low;
      double tr2 = MathAbs(atr_rates[i].high - atr_rates[i+1].close);
      double tr3 = MathAbs(atr_rates[i].low - atr_rates[i+1].close);
      tr = MathMax(tr, MathMax(tr2, tr3));
      sum_tr += tr;
   }
   double atr = sum_tr / 14.0;

   // Use 1.5x ATR as stop distance (ensures room for volatility)
   double sl_distance = atr * 1.5;

   // Ensure a reasonable minimum (at least 10 pips)
   sl_distance = MathMax(sl_distance, 10 * pip);

   return sl_distance;
}


//+------------------------------------------------------------------+
//| POSITION MANAGEMENT STATE MACHINE                                 |
//+------------------------------------------------------------------+
void ManagePosition()
{
   switch(g_state)
   {
      case STATE_NONE:
      case STATE_MONITORING:
         break;

      case STATE_FULL:
         ManageStateFull();
         break;

      case STATE_BE:
         ManageStateBE();
         break;

      case STATE_TRAILING:
         ManageStateTrailing();
         break;
   }
}


//+------------------------------------------------------------------+
//| STATE: FULL POSITION MANAGEMENT                                   |
//| Monitor for 1:1 RR level to move SL to breakeven                 |
//+------------------------------------------------------------------+
void ManageStateFull()
{
   if(!FindOurPosition())
   {
      ResetState();
      Print("Position closed (SL/TP hit or external)");
      return;
   }

   if(!posInfo.SelectByTicket(g_position_ticket)) { ResetState(); return; }

   bool is_buy = (posInfo.PositionType() == POSITION_TYPE_BUY);
   double cur_price = is_buy ? SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                               SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   g_float_pnl = posInfo.Profit() + posInfo.Swap();

   // Check if price has reached 1:1 RR (SL distance in profit)
   if(g_entry_price != 0 && g_sl_price != 0)
   {
      double sl_distance = MathAbs(g_entry_price - g_sl_price);
      bool hit_1to1 = false;

      if(g_trade_dir == TSPOT_BULLISH)
         hit_1to1 = (cur_price >= g_entry_price + sl_distance);
      else
         hit_1to1 = (cur_price <= g_entry_price - sl_distance);

      if(hit_1to1)
      {
         // Move SL to breakeven + few pips
         double be_sl = 0;
         if(g_trade_dir == TSPOT_BULLISH)
            be_sl = NormalizeDouble(g_entry_price + BE_Plus_Pips * pip, _Digits);
         else
            be_sl = NormalizeDouble(g_entry_price - BE_Plus_Pips * pip, _Digits);

         double cur_tp = posInfo.TakeProfit();
         trade.PositionModify(g_position_ticket, be_sl, cur_tp);
         g_sl_price = be_sl;
         g_state = STATE_BE;
         g_state_text = "BE";
         Print("1:1 reached - SL moved to BE+", BE_Plus_Pips, " pips | New SL=",
               DoubleToString(be_sl, _Digits));
      }
   }
}


//+------------------------------------------------------------------+
//| STATE: BREAKEVEN - WAITING FOR 1:4 TO START TRAILING             |
//| SL is at breakeven, monitoring for 1:4 RR level                  |
//+------------------------------------------------------------------+
void ManageStateBE()
{
   if(!FindOurPosition())
   {
      ResetState();
      Print("Position closed (SL/TP hit or external)");
      return;
   }

   if(!posInfo.SelectByTicket(g_position_ticket)) { ResetState(); return; }

   bool is_buy = (posInfo.PositionType() == POSITION_TYPE_BUY);
   double cur_price = is_buy ? SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                               SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   g_float_pnl = posInfo.Profit() + posInfo.Swap();

   // Check if price has reached 1:4 RR
   if(g_entry_price != 0 && g_tp_price != 0)
   {
      // Original SL distance = TP distance / 1.5 (since TP was set at 1.5x SL)
      double tp_distance = MathAbs(g_tp_price - g_entry_price);
      double orig_sl_distance = tp_distance / 1.5;
      
      bool hit_1to4 = false;
      if(g_trade_dir == TSPOT_BULLISH)
         hit_1to4 = (cur_price >= g_entry_price + orig_sl_distance * 4.0);
      else
         hit_1to4 = (cur_price <= g_entry_price - orig_sl_distance * 4.0);

      if(hit_1to4)
      {
         // Remove TP and start trailing
         double trail_dist = Trail_Distance_Pips * pip;
         double trail_sl = 0;
         if(g_trade_dir == TSPOT_BULLISH)
            trail_sl = NormalizeDouble(cur_price - trail_dist, _Digits);
         else
            trail_sl = NormalizeDouble(cur_price + trail_dist, _Digits);

         trade.PositionModify(g_position_ticket, trail_sl, 0);  // TP = 0 removes it
         g_sl_price = trail_sl;
         g_tp_price = 0;
         g_state = STATE_TRAILING;
         g_state_text = "Trailing";
         Print("1:4 reached - TP removed, trailing active | Trail dist=",
               Trail_Distance_Pips, " pips");
      }
   }
}


//+------------------------------------------------------------------+
//| STATE: TRAILING STOP                                              |
//| Tight trailing stop (Trail_Distance_Pips)                        |
//| Only moves SL in favorable direction (never moves back)          |
//+------------------------------------------------------------------+
void ManageStateTrailing()
{
   if(!FindOurPosition())
   {
      ResetState();
      Print("Position closed by trailing stop");
      return;
   }

   if(!posInfo.SelectByTicket(g_position_ticket)) { ResetState(); return; }

   double cur_sl = posInfo.StopLoss();
   bool is_buy = (posInfo.PositionType() == POSITION_TYPE_BUY);
   g_float_pnl = posInfo.Profit() + posInfo.Swap();

   double trail_dist = Trail_Distance_Pips * pip;

   if(is_buy)
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double new_sl = NormalizeDouble(bid - trail_dist, _Digits);
      if(new_sl > cur_sl + point_size)
      {
         trade.PositionModify(g_position_ticket, new_sl, 0);
         g_sl_price = new_sl;
      }
   }
   else
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double new_sl = NormalizeDouble(ask + trail_dist, _Digits);
      if(cur_sl == 0 || new_sl < cur_sl - point_size)
      {
         trade.PositionModify(g_position_ticket, new_sl, 0);
         g_sl_price = new_sl;
      }
   }
}


//+------------------------------------------------------------------+
//| CHECK TRADE INVALIDATION                                          |
//| Close beyond midline of active T-Spot                            |
//+------------------------------------------------------------------+
bool CheckTradeInvalidation()
{
   if(g_trade_tspot_idx < 0 || g_trade_tspot_idx >= ArraySize(g_tspots))
      return false;

   double midline = g_tspots[g_trade_tspot_idx].midline;
   if(midline == 0) return false;

   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 1, 1, m1) < 1) return false;

   if(g_trade_dir == TSPOT_BEARISH && m1[0].close > midline)
      return true;
   if(g_trade_dir == TSPOT_BULLISH && m1[0].close < midline)
      return true;

   return false;
}


//+------------------------------------------------------------------+
//| CHECK ALL INVALIDATION                                            |
//| Deactivate T-Spots where price has closed beyond midline         |
//+------------------------------------------------------------------+
void CheckAllInvalidation()
{
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 1, 1, m1) < 1) return;

   for(int i = ArraySize(g_tspots) - 1; i >= 0; i--)
   {
      if(!g_tspots[i].is_active) continue;
      if(g_tspots[i].confirmed) continue;  // Don't invalidate confirmed ones

      double midline = g_tspots[i].midline;

      bool beyond_midline = false;
      if(g_tspots[i].direction == TSPOT_BEARISH && m1[0].close > midline)
         beyond_midline = true;
      if(g_tspots[i].direction == TSPOT_BULLISH && m1[0].close < midline)
         beyond_midline = true;

      if(beyond_midline)
         g_tspots[i].invalidation_count++;
      else
         g_tspots[i].invalidation_count = 0;

      if(g_tspots[i].invalidation_count >= Invalidation_Closes)
      {
         g_tspots[i].is_active = false;

         // Force-close position if it was linked to this T-Spot
         if((g_state == STATE_FULL || g_state == STATE_BE) && i == g_trade_tspot_idx)
         {
            if(g_position_ticket > 0)
            {
               trade.PositionClose(g_position_ticket);
               Print("Position force-closed - T-Spot invalidated after ", Invalidation_Closes, " closes beyond midline");
            }
            ResetState();
         }

         if(Delete_TSpots_Against)
         {
            // Delete the objects
            string base = lbl + "tspot_" + IntegerToString(g_tspots[i].index);
            ObjectDelete(0, base + "_box");
            ObjectDelete(0, base + "_mid");
            ObjectDelete(0, base + "_close");
         }

         // If this was the monitoring target, reset
         if(i == g_active_tspot_idx && g_state == STATE_MONITORING)
         {
            g_state = STATE_NONE;
            g_state_text = "Waiting";
            g_active_tspot_idx = -1;
         }
      }
   }
}


//+------------------------------------------------------------------+
//| CHECK HIDE T-SPOTS WHEN TRADING AGAINST                           |
//| Make transparent when price closes beyond midline                |
//+------------------------------------------------------------------+
void CheckHideTSpots()
{
   MqlRates m1[];
   ArraySetAsSeries(m1, true);
   if(CopyRates(_Symbol, PERIOD_M1, 1, 1, m1) < 1) return;

   for(int i = ArraySize(g_tspots) - 1; i >= 0; i--)
   {
      if(!g_tspots[i].is_active) continue;

      double midline = g_tspots[i].midline;
      bool beyond_midline = false;

      if(g_tspots[i].direction == TSPOT_BEARISH && m1[0].close > midline)
         beyond_midline = true;
      if(g_tspots[i].direction == TSPOT_BULLISH && m1[0].close < midline)
         beyond_midline = true;

      bool should_hide = (g_tspots[i].invalidation_count >= Invalidation_Closes);

      if(should_hide && !g_tspots[i].is_hidden)
      {
         g_tspots[i].is_hidden = true;
         // Make objects transparent/hidden
         string base = lbl + "tspot_" + IntegerToString(g_tspots[i].index);
         SetObjectTransparency(base + "_box", 250);  // Nearly invisible
         ObjectSetInteger(0, base + "_mid", OBJPROP_COLOR, clrNONE);
         ObjectSetInteger(0, base + "_close", OBJPROP_COLOR, clrNONE);
      }
      else if(!should_hide && g_tspots[i].is_hidden)
      {
         // Restore visibility
         g_tspots[i].is_hidden = false;
         string base = lbl + "tspot_" + IntegerToString(g_tspots[i].index);
         int intensity = (int)(255.0 * (100 - TSpot_Transparency) / 100.0);
         intensity = MathMax(15, MathMin(intensity, 255));
         color zone_color;
         if(g_tspots[i].direction == TSPOT_BULLISH)
            zone_color = (color)StringToColor(StringFormat("%d,%d,%d", 0, intensity, 0));
         else
            zone_color = (color)StringToColor(StringFormat("%d,%d,%d", intensity, 0, 0));
         ObjectSetInteger(0, base + "_box", OBJPROP_COLOR, zone_color);
         color mid_color = (g_tspots[i].direction == TSPOT_BULLISH) ? C'0,180,0' : C'180,0,0';
         ObjectSetInteger(0, base + "_mid", OBJPROP_COLOR, mid_color);
         color close_color = (g_tspots[i].direction == TSPOT_BULLISH) ? C'0,100,0' : C'100,0,0';
         ObjectSetInteger(0, base + "_close", OBJPROP_COLOR, close_color);
      }
   }
}



//+------------------------------------------------------------------+
//| EXTEND LATEST T-SPOT TO CURRENT BAR                               |
//+------------------------------------------------------------------+
void ExtendLatestTSpot()
{
   if(ArraySize(g_tspots) == 0) return;

   // Find the latest active T-Spot
   int latest = -1;
   if(Show_Only_Latest_TSpot)
   {
      for(int i = ArraySize(g_tspots) - 1; i >= 0; i--)
      {
         if(g_tspots[i].is_active && !g_tspots[i].is_hidden)
         {
            latest = i;
            break;
         }
      }
   }
   else
   {
      latest = ArraySize(g_tspots) - 1;
   }

   if(latest < 0) return;

   datetime cur_time = iTime(_Symbol, PERIOD_M1, 0);
   string base = lbl + "tspot_" + IntegerToString(g_tspots[latest].index);

   // Update rectangle end time
   ObjectSetInteger(0, base + "_box", OBJPROP_TIME, 1, cur_time);

   // Update midline end time
   ObjectSetInteger(0, base + "_mid", OBJPROP_TIME, 1, cur_time);

   // Update close line end time
   if(Show_TSpot_Close_Line)
      ObjectSetInteger(0, base + "_close", OBJPROP_TIME, 1, cur_time);

   g_tspots[latest].time_end = cur_time;
}


//+------------------------------------------------------------------+
//| EXTEND LATEST PROJECTIONS TO CURRENT BAR                          |
//+------------------------------------------------------------------+
void ExtendLatestProjections()
{
   if(ArraySize(g_projections) == 0) return;

   int latest = ArraySize(g_projections) - 1;
   if(!g_projections[latest].is_active) return;

   datetime cur_time = iTime(_Symbol, PERIOD_M1, 0);
   int idx = g_projections[latest].index;

   for(int l = 0; l < 5; l++)
   {
      string name = lbl + "proj_" + IntegerToString(idx) + "_" +
                    DoubleToString(g_proj_level_values[l], 1);
      ObjectSetInteger(0, name, OBJPROP_TIME, 1, cur_time);

      // Move label too
      string lbl_name = lbl + "projlbl_" + IntegerToString(idx) + "_" +
                        DoubleToString(g_proj_level_values[l], 1);
      ObjectSetInteger(0, lbl_name, OBJPROP_TIME, 0, cur_time);
   }

   g_projections[latest].time_end = cur_time;
}


//+------------------------------------------------------------------+
//| HTF BIAS CALCULATION                                              |
//| Uses last 2 closed HTF candles to determine bias                 |
//+------------------------------------------------------------------+
void CalculateHTFBias()
{
   if(ArraySize(g_htf_rates) < 4) return;

   // GetHTFBias logic from Pine Script:
   // Check if last closed HTF candle swept previous and closed in direction
   MqlRates last = g_htf_rates[1];     // Last closed
   MqlRates prev = g_htf_rates[2];     // Previous closed

   // Bullish bias: last candle swept prev low and closed above prev high
   // or closed above prev midpoint after sweeping low
   bool bull_sweep = (last.low < prev.low && last.close > prev.close);
   bool bear_sweep = (last.high > prev.high && last.close < prev.close);

   if(bull_sweep && !bear_sweep)
   {
      g_htf_bias = BIAS_BULLISH;
      g_bias_text = "Bullish";
      g_bias_color = clrLime;
   }
   else if(bear_sweep && !bull_sweep)
   {
      g_htf_bias = BIAS_BEARISH;
      g_bias_text = "Bearish";
      g_bias_color = clrRed;
   }
   else
   {
      g_htf_bias = BIAS_NEUTRAL;
      g_bias_text = "Neutral";
      g_bias_color = clrGray;
   }
}


//+------------------------------------------------------------------+
//| INITIAL CHART SCAN                                                |
//| Scans recent history on startup to populate structures            |
//+------------------------------------------------------------------+
void InitialChartScan()
{
   // Copy HTF rates for initial T-Spot detection
   MqlRates htf[];
   ArraySetAsSeries(htf, true);
   int copied = CopyRates(_Symbol, HTF_Timeframe, 0, Max_Display + 5, htf);
   if(copied < 5) return;

   ArrayResize(g_htf_rates, copied);
   for(int i = 0; i < copied; i++)
      g_htf_rates[i] = htf[i];

   // Draw initial HTF start lines
   if(Show_HTF)
   {
      int lines_to_draw = MathMin(copied, Max_Display * 4);
      for(int i = 1; i < lines_to_draw; i++)
      {
         string name = lbl + "htf_vline_" + IntegerToString(g_htf_line_total_idx);
         g_htf_line_total_idx++;
         g_htf_line_count++;

         ObjectCreate(0, name, OBJ_VLINE, 0, htf[i].time, 0);
         ObjectSetInteger(0, name, OBJPROP_COLOR, HTF_Line_Color);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, HTF_Line_Width);
         ObjectSetInteger(0, name, OBJPROP_STYLE, HTF_Line_Style);
         ObjectSetInteger(0, name, OBJPROP_BACK, true);
         ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      }
   }

   // Initial T-Spot detection for recent candles
   for(int shift = MathMin(copied - 3, Max_Display); shift >= 1; shift--)
   {
      MqlRates last_closed = htf[shift];
      MqlRates prev_closed = htf[shift + 1];
      MqlRates prev_prev   = htf[shift + 2];

      double close_price = last_closed.close;
      double mid_last = LogMidpoint(last_closed.high, last_closed.low,
                                     last_closed.open, last_closed.close);
      double mid_prev = LogMidpoint(prev_closed.high, prev_closed.low,
                                     prev_closed.open, prev_closed.close);
      double mid_level_prev = MidLevel(prev_closed.high, prev_closed.low);

      bool swept_both = IsSweptBoth(last_closed, prev_closed);
      ENUM_TSPOT_DIR detected = TSPOT_NONE;
      string pattern_name = "";

      // Pattern 1 - Normal Bearish
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
      // Pattern 2 - Normal Bullish
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
      // Pattern 3 - Expansive Bearish
      if(detected == TSPOT_NONE && !swept_both)
      {
         if(prev_closed.high > prev_prev.high &&
            last_closed.close < MathMax(prev_closed.open, prev_closed.close) &&
            close_price < mid_last)
         {
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
      // Pattern 4 - Expansive Bullish
      if(detected == TSPOT_NONE && !swept_both)
      {
         if(prev_closed.low < prev_prev.low &&
            last_closed.close > MathMin(prev_closed.open, prev_closed.close) &&
            close_price > mid_last)
         {
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
      // Pattern 5 - Pro-Trend Bullish
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
      // Pattern 6 - Pro-Trend Bearish
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

      if(detected != TSPOT_NONE && Show_TSpot)
      {
         TSpotData ts;
         ts.direction     = detected;
         ts.midline       = mid_last;
         ts.close_level   = last_closed.close;
         ts.high          = last_closed.high;
         ts.low           = last_closed.low;
         ts.time_start    = htf[shift-1].time;                             // Candle after sweep (matches DetectTSpot logic)
         ts.time_end      = htf[shift-1].time + PeriodSeconds(HTF_Timeframe); // End of that candle period
         ts.is_active     = true;
         ts.is_hidden     = false;
         ts.pattern_name  = pattern_name;
         ts.touched       = false;
         ts.touch_time    = 0;
         ts.pivot_price   = 0;
         ts.pivot_time    = 0;
         ts.pivot_formed  = false;
         ts.confirmed     = false;
         ts.confirm_time  = 0;
         ts.invalidation_count = 0;
         ts.index         = g_tspot_total_idx;

         int size = ArraySize(g_tspots);
         ArrayResize(g_tspots, size + 1);
         g_tspots[size] = ts;
         g_tspot_count++;
         g_tspot_total_idx++;
         g_active_tspot_idx = size;

         DrawTSpotZone(ts);
      }
   }

   // Calculate initial bias
   CalculateHTFBias();
}



//+------------------------------------------------------------------+
//| DASHBOARD - Create info table (matching indicator)                |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   ObjectsDeleteAll(0, lbl + "dash_");
   int x = 20, y = 50, row = 18;
   color bg = C'20,28,42', border = C'40,55,85';

   ObjRect(lbl+"dash_bg", x-10, y-10, 260, 420, bg, border, 1);

   // Title
   ObjLbl(lbl+"dash_title",  "GFM EA v3", x, y, clrWhite, 9, true);
   ObjLbl(lbl+"dash_sub",    "T-Spot Touch Entry Model", x, y+14, C'150,150,150', 7, false);
   ObjLine(lbl+"dash_d0", x, y+28, 240);

   int r = y + 38;
   // Model info
   string model_name = TFStr(PERIOD_CURRENT) + "-" + TFStr(HTF_Timeframe) + " Model";
   ObjLbl(lbl+"dash_l_model", "Model",      x, r,       clrSilver, 8);
   ObjLbl(lbl+"dash_v_model", model_name,   x+120, r,   clrWhite,  8);
   ObjLbl(lbl+"dash_l_htf",   "HTF Bias",   x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_htf",   "Neutral",    x+120, r+row, clrGray, 8);
   ObjLbl(lbl+"dash_l_time",  "HTF Close",  x, r+row*2, clrSilver, 8);
   ObjLbl(lbl+"dash_v_time",  "---",        x+120, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"dash_l_state", "State",      x, r+row*3, clrSilver, 8);
   ObjLbl(lbl+"dash_v_state", "Waiting",    x+120, r+row*3, clrYellow, 8);
   ObjLine(lbl+"dash_d1", x, r+row*4+4, 240);

   r = r + row*4 + 14;
   ObjLbl(lbl+"dash_l_tspot", "T-Spot",     x, r,         clrSilver, 8);
   ObjLbl(lbl+"dash_v_tspot", "None",       x+120, r,      clrGray,   8);
   ObjLbl(lbl+"dash_l_pat",   "Pattern",    x, r+row,     clrSilver, 8);
   ObjLbl(lbl+"dash_v_pat",   "---",        x+120, r+row,  clrWhite,  8);
   ObjLbl(lbl+"dash_l_mid",   "Midline",    x, r+row*2,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_mid",   "---",        x+120, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"dash_l_cls",   "Close Lvl",  x, r+row*3,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_cls",   "---",        x+120, r+row*3, clrWhite, 8);
   ObjLine(lbl+"dash_d2", x, r+row*4+4, 240);

   r = r + row*4 + 14;
   ObjLbl(lbl+"dash_l_conf",  "Confirm",    x, r,         clrSilver, 8);
   ObjLbl(lbl+"dash_v_conf",  "---",        x+120, r,      clrGray,   8);
   ObjLbl(lbl+"dash_l_entry", "Entry",      x, r+row,     clrSilver, 8);
   ObjLbl(lbl+"dash_v_entry", "---",        x+120, r+row,  clrWhite,  8);
   ObjLbl(lbl+"dash_l_sl",    "SL",         x, r+row*2,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_sl",    "---",        x+120, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"dash_l_tp",    "TP (1:1.5)", x, r+row*3,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_tp",    "---",        x+120, r+row*3, clrWhite, 8);
   ObjLbl(lbl+"dash_l_pnl",   "P/L",        x, r+row*4,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_pnl",   "---",        x+120, r+row*4, clrWhite, 8);
   ObjLine(lbl+"dash_d3", x, r+row*5+4, 240);

   r = r + row*5 + 14;
   ObjLbl(lbl+"dash_l_fvg",   "FVGs",       x, r,         clrSilver, 8);
   ObjLbl(lbl+"dash_v_fvg",   "0",          x+120, r,      clrGray,   8);
   ObjLbl(lbl+"dash_l_pfvg",  "PFVGs",      x, r+row,     clrSilver, 8);
   ObjLbl(lbl+"dash_v_pfvg",  "0",          x+120, r+row,  clrDodgerBlue, 8);
   ObjLbl(lbl+"dash_l_vi",    "Vol Imb",    x, r+row*2,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_vi",    "0",          x+120, r+row*2, clrOrange, 8);
   ObjLbl(lbl+"dash_l_cisd",  "CISDs",      x, r+row*3,   clrSilver, 8);
   ObjLbl(lbl+"dash_v_cisd",  "0",          x+120, r+row*3, clrYellow, 8);

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
   if(g_state == STATE_FULL) state_col = clrLime;
   if(g_state == STATE_BE) state_col = clrDodgerBlue;
   if(g_state == STATE_TRAILING) state_col = clrAqua;
   if(g_state == STATE_MONITORING) state_col = clrOrange;
   ObjSet(lbl+"dash_v_state", g_state_text, state_col);

   // HTF Bias
   ObjSet(lbl+"dash_v_htf", g_bias_text, g_bias_color);

   // HTF Countdown timer
   string countdown = GetHTFCountdown();
   ObjSet(lbl+"dash_v_time", countdown, clrWhite);

   // T-Spot info
   ObjSet(lbl+"dash_v_tspot", g_tspot_text, g_tspot_color);

   // Pattern and levels
   if(g_active_tspot_idx >= 0 && g_active_tspot_idx < ArraySize(g_tspots))
   {
      ObjSet(lbl+"dash_v_pat", g_tspots[g_active_tspot_idx].pattern_name, clrWhite);
      ObjSet(lbl+"dash_v_mid",
             DoubleToString(g_tspots[g_active_tspot_idx].midline, _Digits), clrWhite);
      ObjSet(lbl+"dash_v_cls",
             DoubleToString(g_tspots[g_active_tspot_idx].close_level, _Digits), clrWhite);

      // Confirmation status
      string conf_txt = "Waiting Touch";
      color conf_col = clrGray;
      if(g_tspots[g_active_tspot_idx].confirmed)
      {
         conf_txt = "TOUCHED/ENTERED";
         conf_col = clrLime;
      }
      ObjSet(lbl+"dash_v_conf", conf_txt, conf_col);
   }
   else
   {
      ObjSet(lbl+"dash_v_pat", "---", clrGray);
      ObjSet(lbl+"dash_v_mid", "---", clrGray);
      ObjSet(lbl+"dash_v_cls", "---", clrGray);
      ObjSet(lbl+"dash_v_conf", "---", clrGray);
   }

   // Entry/SL/TP/PnL
   if(g_entry_price > 0)
      ObjSet(lbl+"dash_v_entry", DoubleToString(g_entry_price, _Digits), clrWhite);
   else
      ObjSet(lbl+"dash_v_entry", "---", clrGray);

   if(g_sl_price > 0)
      ObjSet(lbl+"dash_v_sl", DoubleToString(g_sl_price, _Digits), clrTomato);
   else
      ObjSet(lbl+"dash_v_sl", "---", clrGray);

   if(g_tp_price > 0)
      ObjSet(lbl+"dash_v_tp", DoubleToString(g_tp_price, _Digits), clrGold);
   else
      ObjSet(lbl+"dash_v_tp", "---", clrGray);

   if(g_state >= STATE_FULL)
      ObjSet(lbl+"dash_v_pnl", StringFormat("%.2f", g_float_pnl),
             g_float_pnl >= 0 ? clrLime : clrTomato);
   else
      ObjSet(lbl+"dash_v_pnl", "---", clrGray);

   // Counts
   ObjSet(lbl+"dash_v_fvg", IntegerToString(g_fvg_count), clrGray);
   ObjSet(lbl+"dash_v_pfvg", IntegerToString(g_pfvg_count), clrDodgerBlue);
   ObjSet(lbl+"dash_v_vi", IntegerToString(g_vi_count), clrOrange);
   ObjSet(lbl+"dash_v_cisd", IntegerToString(g_cisd_count), clrYellow);

   ChartRedraw(0);
}


//+------------------------------------------------------------------+
//| GET HTF COUNTDOWN STRING                                          |
//| Time remaining until next HTF candle close                       |
//+------------------------------------------------------------------+
string GetHTFCountdown()
{
   int htf_seconds = PeriodSeconds(HTF_Timeframe);
   if(htf_seconds <= 0) return "---";

   datetime htf_start = iTime(_Symbol, HTF_Timeframe, 0);
   if(htf_start == 0) return "---";

   datetime htf_end = htf_start + htf_seconds;
   int remaining = (int)(htf_end - TimeCurrent());

   if(remaining <= 0) return "00:00";

   int mins = remaining / 60;
   int secs = remaining % 60;

   return StringFormat("%02d:%02d", mins, secs);
}



//+------------------------------------------------------------------+
//| HELPER: LOGARITHMIC MIDPOINT                                      |
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
//| HELPER: ARITHMETIC MIDPOINT                                       |
//+------------------------------------------------------------------+
double MidLevel(double high, double low)
{
   return (high + low) / 2.0;
}


//+------------------------------------------------------------------+
//| HELPER: SWEPT-BOTH INVALIDATION                                   |
//+------------------------------------------------------------------+
bool IsSweptBoth(const MqlRates &last_closed, const MqlRates &prev_closed)
{
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
//| HELPER: NORMALIZE LOT SIZE                                        |
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
//| HELPER: CALCULATE LOT SIZE (Risk-Based)                           |
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
//| HELPER: FIND OUR POSITION                                         |
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
//| HELPER: RESET STATE                                               |
//+------------------------------------------------------------------+
void ResetState()
{
   g_state = STATE_NONE;
   g_state_text = "Waiting";
   g_trade_dir = TSPOT_NONE;
   g_entry_price = 0;
   g_sl_price = 0;
   g_tp_price = 0;
   g_position_ticket = 0;
   g_float_pnl = 0;
   g_trade_tspot_idx = -1;
}


//+------------------------------------------------------------------+
//| HELPER: RESET PIVOT TRACKING                                      |
//+------------------------------------------------------------------+
void ResetPivotTracking()
{
   ArrayResize(g_pivot_highs, 0);
   ArrayResize(g_pivot_high_times, 0);
   ArrayResize(g_pivot_lows, 0);
   ArrayResize(g_pivot_low_times, 0);
   g_pivot_high_count = 0;
   g_pivot_low_count = 0;
}


//+------------------------------------------------------------------+
//| HELPER: PRUNE INACTIVE ENTRIES                                    |
//| Removes inactive entries from backing arrays to prevent          |
//| unbounded memory growth during multi-day sessions                |
//+------------------------------------------------------------------+
void PruneInactiveEntries()
{
   // Prune T-Spots (keep active ones and the trade T-Spot)
   int tspot_size = ArraySize(g_tspots);
   if(tspot_size > Max_Display * 2)
   {
      for(int i = tspot_size - 1; i >= 0; i--)
      {
         if(!g_tspots[i].is_active && i != g_active_tspot_idx && i != g_trade_tspot_idx)
         {
            // Adjust indices that reference positions after removed element
            if(g_active_tspot_idx > i) g_active_tspot_idx--;
            if(g_trade_tspot_idx > i) g_trade_tspot_idx--;
            ArrayRemove(g_tspots, i, 1);
         }
      }
   }

   // Prune FVGs
   int fvg_size = ArraySize(g_fvgs);
   if(fvg_size > Max_Display * 4)
   {
      for(int i = fvg_size - 1; i >= 0; i--)
      {
         if(!g_fvgs[i].is_active)
            ArrayRemove(g_fvgs, i, 1);
      }
   }

   // Prune PFVGs
   int pfvg_size = ArraySize(g_pfvgs);
   if(pfvg_size > Max_PFVG_Display * 4)
   {
      for(int i = pfvg_size - 1; i >= 0; i--)
      {
         if(!g_pfvgs[i].is_active)
            ArrayRemove(g_pfvgs, i, 1);
      }
   }

   // Prune Volume Imbalances
   int vi_size = ArraySize(g_vis);
   if(vi_size > Max_Display * 4)
   {
      for(int i = vi_size - 1; i >= 0; i--)
      {
         if(!g_vis[i].is_active)
            ArrayRemove(g_vis, i, 1);
      }
   }

   // Prune CISDs
   int cisd_size = ArraySize(g_cisds);
   if(cisd_size > Max_Display * 4)
   {
      for(int i = cisd_size - 1; i >= 0; i--)
      {
         if(!g_cisds[i].is_active)
            ArrayRemove(g_cisds, i, 1);
      }
   }

   // Prune Projections
   int proj_size = ArraySize(g_projections);
   if(proj_size > Max_Display * 4)
   {
      for(int i = proj_size - 1; i >= 0; i--)
      {
         if(!g_projections[i].is_active)
            ArrayRemove(g_projections, i, 1);
      }
   }
}


//+------------------------------------------------------------------+
//| HELPER: SYNC STATE FROM MARKET                                    |
//+------------------------------------------------------------------+
void SyncStateFromMarket()
{
   if(FindOurPosition())
   {
      posInfo.SelectByTicket(g_position_ticket);
      g_state = STATE_FULL;
      g_state_text = "Open";
      g_trade_dir = (posInfo.PositionType() == POSITION_TYPE_BUY) ? TSPOT_BULLISH : TSPOT_BEARISH;
      g_entry_price = posInfo.PriceOpen();
      g_sl_price = posInfo.StopLoss();
      g_tp_price = posInfo.TakeProfit();
      g_tspot_text = (g_trade_dir == TSPOT_BULLISH) ? "Bullish" : "Bearish";
      g_tspot_color = (g_trade_dir == TSPOT_BULLISH) ? clrLime : clrRed;
      Print("Synced position from market | Ticket=", g_position_ticket,
            " | Dir=", (g_trade_dir == TSPOT_BULLISH ? "BUY" : "SELL"));
   }
}


//+------------------------------------------------------------------+
//| HELPER: SET OBJECT TRANSPARENCY                                   |
//| Uses ColorToARGB() for actual transparency (MetaTrader build 2361+)|
//+------------------------------------------------------------------+
void SetObjectTransparency(string name, int alpha)
{
   // alpha: 0 = fully opaque, 255 = fully transparent
   if(ObjectFind(0, name) < 0) return;

   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);

   // Get current color and apply alpha channel using ColorToARGB
   // ColorToARGB takes (color, opacity) where opacity 0=transparent, 255=opaque
   // Our alpha parameter is 0=opaque, 255=transparent, so we invert it
   color cur_color = (color)ObjectGetInteger(0, name, OBJPROP_COLOR);
   if(cur_color == clrNONE) return;

   int opacity = 255 - alpha;  // Convert transparency to opacity
   if(opacity < 0) opacity = 0;
   if(opacity > 255) opacity = 255;

   ObjectSetInteger(0, name, OBJPROP_COLOR, ColorToARGB(cur_color, (uchar)opacity));
}


//+------------------------------------------------------------------+
//| HELPER: ENFORCE MAX DISPLAY                                       |
//+------------------------------------------------------------------+
void EnforceMaxDisplay()
{
   if(Show_Only_Latest_TSpot)
   {
      // Hide all except the latest
      for(int i = 0; i < ArraySize(g_tspots) - 1; i++)
      {
         if(g_tspots[i].is_active)
         {
            string base = lbl + "tspot_" + IntegerToString(g_tspots[i].index);
            ObjectSetInteger(0, base + "_box", OBJPROP_COLOR, clrNONE);
            ObjectSetInteger(0, base + "_mid", OBJPROP_COLOR, clrNONE);
            if(Show_TSpot_Close_Line)
               ObjectSetInteger(0, base + "_close", OBJPROP_COLOR, clrNONE);
         }
      }
   }
   else
   {
      // Remove oldest if exceeding Max_Display
      while(g_tspot_count > Max_Display)
      {
         // Find and remove oldest
         for(int i = 0; i < ArraySize(g_tspots); i++)
         {
            if(g_tspots[i].is_active)
            {
               g_tspots[i].is_active = false;
               string base = lbl + "tspot_" + IntegerToString(g_tspots[i].index);
               ObjectDelete(0, base + "_box");
               ObjectDelete(0, base + "_mid");
               ObjectDelete(0, base + "_close");
               g_tspot_count--;
               break;
            }
         }
      }
   }
}



//+------------------------------------------------------------------+
//| HELPER: LIMIT FVG DISPLAY                                         |
//+------------------------------------------------------------------+
void LimitFVGDisplay()
{
   int max_fvg = Max_Display * 10;
   while(g_fvg_count > max_fvg)
   {
      for(int i = 0; i < ArraySize(g_fvgs); i++)
      {
         if(g_fvgs[i].is_active)
         {
            g_fvgs[i].is_active = false;
            string name = lbl + "fvg_" + IntegerToString(g_fvgs[i].index);
            ObjectDelete(0, name);
            g_fvg_count--;
            break;
         }
      }
   }
}


//+------------------------------------------------------------------+
//| HELPER: LIMIT PFVG DISPLAY                                        |
//+------------------------------------------------------------------+
void LimitPFVGDisplay()
{
   while(g_pfvg_count > Max_PFVG_Display)
   {
      for(int i = 0; i < ArraySize(g_pfvgs); i++)
      {
         if(g_pfvgs[i].is_active)
         {
            g_pfvgs[i].is_active = false;
            string name = lbl + "pfvg_" + IntegerToString(g_pfvgs[i].index);
            ObjectDelete(0, name);
            g_pfvg_count--;
            break;
         }
      }
   }
}


//+------------------------------------------------------------------+
//| HELPER: LIMIT VI DISPLAY                                          |
//+------------------------------------------------------------------+
void LimitVIDisplay()
{
   int max_vi = Max_Display * 10;
   while(g_vi_count > max_vi)
   {
      for(int i = 0; i < ArraySize(g_vis); i++)
      {
         if(g_vis[i].is_active)
         {
            g_vis[i].is_active = false;
            string name = lbl + "vi_" + IntegerToString(g_vis[i].index);
            ObjectDelete(0, name);
            g_vi_count--;
            break;
         }
      }
   }
}


//+------------------------------------------------------------------+
//| HELPER: LIMIT CISD DISPLAY                                        |
//+------------------------------------------------------------------+
void LimitCISDDisplay()
{
   int max_cisd = Max_Display * 3;
   while(g_cisd_count > max_cisd)
   {
      for(int i = 0; i < ArraySize(g_cisds); i++)
      {
         if(g_cisds[i].is_active)
         {
            g_cisds[i].is_active = false;
            string name = lbl + "cisd_" + IntegerToString(g_cisds[i].index);
            ObjectDelete(0, name);
            g_cisd_count--;
            break;
         }
      }
   }
}


//+------------------------------------------------------------------+
//| DRAW TTFM LABELS (Time To First Move)                             |
//| Small labels showing T-Spot formation timing                     |
//+------------------------------------------------------------------+
void DrawTTFMLabel(const TSpotData &ts)
{
   if(!Show_TTFM_Labels) return;

   string name = lbl + "ttfm_" + IntegerToString(ts.index);
   double label_price = (ts.direction == TSPOT_BULLISH) ?
                        ts.low - 2 * pip : ts.high + 2 * pip;

   // OBJ_TEXT for TTFM marker on chart
   ObjectCreate(0, name, OBJ_TEXT, 0, ts.time_start, label_price);
   ObjectSetString(0, name, OBJPROP_TEXT, "TTFM");
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 6);
   ObjectSetString(0, name, OBJPROP_FONT, "Arial");
   ObjectSetInteger(0, name, OBJPROP_ANCHOR,
      (ts.direction == TSPOT_BULLISH) ? ANCHOR_UPPER : ANCHOR_LOWER);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| DRAW INVALIDATION MARKER                                          |
//| Shows where T-Spot was invalidated                               |
//+------------------------------------------------------------------+
void DrawInvalidationMarker(const TSpotData &ts)
{
   string name = lbl + "inv_" + IntegerToString(ts.index);
   datetime cur_time = iTime(_Symbol, PERIOD_M1, 0);

   // OBJ_TEXT for invalidation cross marker
   ObjectCreate(0, name, OBJ_TEXT, 0, cur_time, ts.midline);
   ObjectSetString(0, name, OBJPROP_TEXT, "X");
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrRed);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 10);
   ObjectSetString(0, name, OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_CENTER);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| DRAW ENTRY MARKER ON CHART                                        |
//| Arrow at entry price when trade executes                         |
//+------------------------------------------------------------------+
void DrawEntryMarker(double price, datetime time, ENUM_TSPOT_DIR dir)
{
   string name = lbl + "entry_" + IntegerToString((int)time);

   // OBJ_ARROW for entry marker on chart
   int arrow_code = (dir == TSPOT_BULLISH) ? 241 : 242;  // Buy/Sell arrows
   ObjectCreate(0, name, OBJ_ARROW, 0, time, price);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, arrow_code);
   ObjectSetInteger(0, name, OBJPROP_COLOR, (dir == TSPOT_BULLISH) ? clrLime : clrRed);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 3);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| DRAW SL/TP LINES ON CHART                                         |
//| Horizontal lines at SL and TP levels                             |
//+------------------------------------------------------------------+
void DrawSLTPLines(double sl, double tp, datetime time)
{
   datetime end_time = time + PeriodSeconds(HTF_Timeframe) * 3;

   // OBJ_TREND for SL line (horizontal)
   string sl_name = lbl + "sl_line_" + IntegerToString((int)time);
   ObjectCreate(0, sl_name, OBJ_TREND, 0, time, sl, end_time, sl);
   ObjectSetInteger(0, sl_name, OBJPROP_COLOR, clrTomato);
   ObjectSetInteger(0, sl_name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, sl_name, OBJPROP_STYLE, STYLE_DASH);
   ObjectSetInteger(0, sl_name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, sl_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, sl_name, OBJPROP_HIDDEN, true);

   // OBJ_TREND for TP line (horizontal)
   string tp_name = lbl + "tp_line_" + IntegerToString((int)time);
   ObjectCreate(0, tp_name, OBJ_TREND, 0, time, tp, end_time, tp);
   ObjectSetInteger(0, tp_name, OBJPROP_COLOR, clrGold);
   ObjectSetInteger(0, tp_name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, tp_name, OBJPROP_STYLE, STYLE_DASH);
   ObjectSetInteger(0, tp_name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, tp_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, tp_name, OBJPROP_HIDDEN, true);
}


//+------------------------------------------------------------------+
//| DASHBOARD HELPER - Create Label Object (OBJ_LABEL)                |
//+------------------------------------------------------------------+
void ObjLbl(string name, string txt, int x, int y, color c, int fs=8, bool bold=false)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0,  name, OBJPROP_TEXT,      txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR,     c);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,  fs);
   ObjectSetString(0,  name, OBJPROP_FONT,      bold ? "Consolas" : "Consolas");
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
   int n = (int)(w / 6.0);
   for(int i = 0; i < n; i++) d += "_";
   ObjLbl(name, d, x, y, C'40,55,85', 7);
}


//+------------------------------------------------------------------+
//| DASHBOARD HELPER - Create Background Rectangle (OBJ_RECTANGLE_LABEL)|
//+------------------------------------------------------------------+
void ObjRect(string name, int x, int y, int w, int h, color bg, color border, int bw)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER,      CORNER_LEFT_UPPER);
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
//| HELPER: TIMEFRAME STRING                                          |
//+------------------------------------------------------------------+
string TFStr(ENUM_TIMEFRAMES tf)
{
   if(tf == PERIOD_CURRENT) tf = (ENUM_TIMEFRAMES)Period();
   switch(tf)
   {
      case PERIOD_M1:  return "1M";
      case PERIOD_M5:  return "5M";
      case PERIOD_M15: return "15M";
      case PERIOD_M30: return "30M";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return "?";
   }
}
//+------------------------------------------------------------------+

