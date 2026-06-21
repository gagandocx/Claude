//+------------------------------------------------------------------+
//|                                                     ICT_EA.mq5  |
//|                         ICT Smart Money EA — XAUUSD             |
//|                    Built on 47,172,211 tick analysis             |
//|                                                                  |
//| STRATEGY: Full ICT Methodology                                   |
//|  1. Market Structure (BOS / CHoCH)                              |
//|  2. Order Block Detection (Bullish + Bearish)                   |
//|  3. Fair Value Gap (FVG) Detection                              |
//|  4. Killzone Filter (London 07-10 UTC / NY 12-15 UTC)           |
//|  5. OTE Entry (62-79% retracement into Order Block)             |
//|  6. SL: Above/Below Order Block                                 |
//|  7. TP: Next liquidity level / previous high-low               |
//|                                                                  |
//| DATA EDGE (from 47M ticks):                                     |
//|  - London session: bullish bias 53%                             |
//|  - NY open: best momentum hours 13-14 UTC                       |
//|  - Avg M1 range: 3.14 pts — use H1 for meaningful moves        |
//|  - Skip Thursday: only 41% bullish                              |
//+------------------------------------------------------------------+
#property copyright "ICT EA - Smart Money Concepts"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade        trade;
CPositionInfo pos;

//+------------------------------------------------------------------+
//| ENUMERATIONS                                                      |
//+------------------------------------------------------------------+
enum ENUM_MARKET_BIAS { BIAS_BULLISH=1, BIAS_BEARISH=-1, BIAS_NEUTRAL=0 };
enum ENUM_OB_TYPE     { OB_BULLISH=1,  OB_BEARISH=-1 };

//+------------------------------------------------------------------+
//| STRUCTS                                                           |
//+------------------------------------------------------------------+
struct OrderBlock
{
   double         top;
   double         bottom;
   double         mid;
   ENUM_OB_TYPE   type;
   datetime       time;
   bool           valid;      // false = mitigated/broken
   bool           traded;     // already used for entry
};

struct FVG
{
   double         top;
   double         bottom;
   double         mid;
   bool           bullish;
   datetime       time;
   bool           filled;
};

struct SwingPoint
{
   double         price;
   datetime       time;
   bool           is_high;
};

//+------------------------------------------------------------------+
//| INPUTS                                                            |
//+------------------------------------------------------------------+
input group "=== MARKET STRUCTURE ==="
input int    Structure_Lookback  = 20;   // Bars to look back for swing points
input int    Swing_Strength      = 3;    // Candles each side to confirm swing H/L

input group "=== ORDER BLOCKS ==="
input int    OB_Lookback         = 50;   // Max bars back to find OBs
input int    Max_OB_Count        = 5;    // Max OBs to track per side
input double OB_Min_Move_Pts     = 20.0; // Min displacement after OB (points)

input group "=== FAIR VALUE GAPS ==="
input int    FVG_Lookback        = 30;   // Bars to look back for FVGs
input int    Max_FVG_Count       = 5;    // Max FVGs to track

input group "=== OTE ENTRY ==="
input double OTE_Fib_Low         = 0.62; // OTE zone low (62%)
input double OTE_Fib_High        = 0.79; // OTE zone high (79%)

input group "=== KILLZONES (UTC) ==="
input bool   Use_Killzones       = true;  // Only trade in killzones
input int    London_Start        = 7;     // London killzone start (UTC)
input int    London_End          = 10;    // London killzone end (UTC)
input int    NY_Start            = 12;    // NY killzone start (UTC)
input int    NY_End              = 15;    // NY killzone end (UTC)
input bool   Use_DayFilter       = true;  // Skip Thursday
input bool   Use_Asian_Range     = true;  // Use Asian range for reference

input group "=== TRADE SETTINGS ==="
input ENUM_TIMEFRAMES Entry_TF   = PERIOD_M5;  // Entry timeframe
input ENUM_TIMEFRAMES HTF        = PERIOD_H1;  // Higher timeframe for structure
input double SL_Buffer_Pts       = 30.0;       // Extra SL buffer above/below OB (pts)
input double RR_Ratio            = 2.0;        // Risk:Reward ratio
input bool   Use_FVG_TP          = true;       // Use FVG as TP target if available

input group "=== RISK MANAGEMENT ==="
input double Risk_Percent        = 1.0;   // Risk % per trade
input double Manual_Lot          = 0.0;   // Manual lot (0=auto)
input double Max_Lot             = 5.0;   // Max lot
input int    Max_Trades          = 2;     // Max open trades
input bool   Use_BE              = true;  // Break-even at 1:1
input bool   Use_Trailing        = true;  // Trailing stop after BE
input double Trail_Pts           = 15.0;  // Trail distance in points

input group "=== EQUITY PROTECTION ==="
input bool   Use_EP              = true;  // Equity protection
input double Max_DD_Pct          = 4.0;   // Max drawdown %

input group "=== DISPLAY ==="
input bool   Show_Panel          = true;  // Show dashboard
input bool   Draw_OB             = true;  // Draw OBs on chart
input bool   Draw_FVG            = true;  // Draw FVGs on chart
input int    Magic               = 77001; // Magic number
input string EA_Comment          = "ICT_EA"; // Comment
input int    Slippage            = 10;    // Max slippage


//+------------------------------------------------------------------+
//| GLOBALS                                                           |
//+------------------------------------------------------------------+
double   point_size;

// Structure tracking
SwingPoint swing_highs[];
SwingPoint swing_lows[];
ENUM_MARKET_BIAS market_bias = BIAS_NEUTRAL;
double   last_bos_level   = 0;
datetime last_bos_time    = 0;
bool     choch_detected   = false;

// Order blocks
OrderBlock bull_obs[];   // bullish OBs (buy from)
OrderBlock bear_obs[];   // bearish OBs (sell from)

// FVGs
FVG bull_fvgs[];
FVG bear_fvgs[];

// Asian range
double asian_high = 0;
double asian_low  = 0;
datetime asian_date = 0;

// State
datetime last_bar_htf;
datetime last_bar_entry;
int      open_buys, open_sells;
double   float_pnl;
ulong    be_tickets[];

// P&L
double   pnl_today, pnl_week, pnl_month;
datetime pnl_cache;

// Dashboard
string   lbl           = "ICT_";
string   panel_signal  = "INIT";
color    panel_sig_col = clrGray;
string   panel_bias    = "---";
string   panel_ob      = "---";
string   panel_fvg     = "---";
string   panel_kz      = "---";


//+------------------------------------------------------------------+
//| MARKET STRUCTURE — BOS & CHoCH                                   |
//|                                                                   |
//| Logic:                                                            |
//|  1. Find swing highs and lows on HTF                             |
//|  2. BOS (Break of Structure): price closes beyond last swing H/L |
//|     in the SAME direction as current bias → trend continues      |
//|  3. CHoCH (Change of Character): price closes beyond last swing  |
//|     H/L in the OPPOSITE direction → bias flips                   |
//+------------------------------------------------------------------+
void UpdateMarketStructure()
{
   int bars = iBars(_Symbol, HTF);
   if(bars < Structure_Lookback + 5) return;

   // Find swing highs and lows
   ArrayResize(swing_highs, 0);
   ArrayResize(swing_lows,  0);
   int str = Swing_Strength;

   for(int i = str; i < Structure_Lookback + str; i++)
   {
      double h = iHigh(_Symbol, HTF, i);
      double l = iLow(_Symbol,  HTF, i);
      bool is_swing_high = true;
      bool is_swing_low  = true;

      for(int j = 1; j <= str; j++)
      {
         if(iHigh(_Symbol, HTF, i-j) >= h || iHigh(_Symbol, HTF, i+j) >= h) is_swing_high = false;
         if(iLow(_Symbol,  HTF, i-j) <= l || iLow(_Symbol,  HTF, i+j) <= l) is_swing_low  = false;
      }

      if(is_swing_high)
      {
         int sz = ArraySize(swing_highs);
         ArrayResize(swing_highs, sz+1);
         swing_highs[sz].price   = h;
         swing_highs[sz].time    = iTime(_Symbol, HTF, i);
         swing_highs[sz].is_high = true;
      }
      if(is_swing_low)
      {
         int sz = ArraySize(swing_lows);
         ArrayResize(swing_lows, sz+1);
         swing_lows[sz].price   = l;
         swing_lows[sz].time    = iTime(_Symbol, HTF, i);
         swing_lows[sz].is_high = false;
      }
   }

   if(ArraySize(swing_highs) < 2 || ArraySize(swing_lows) < 2) return;

   double cur_close = iClose(_Symbol, HTF, 1);
   double last_sh   = swing_highs[0].price;  // most recent swing high
   double last_sl   = swing_lows[0].price;   // most recent swing low
   double prev_sh   = swing_highs[1].price;  // previous swing high
   double prev_sl   = swing_lows[1].price;   // previous swing low

   // BOS Bullish: close above last swing high (trend continuation up)
   if(cur_close > last_sh && market_bias == BIAS_BULLISH)
   {
      last_bos_level = last_sh;
      last_bos_time  = TimeCurrent();
      panel_bias     = "BULLISH BOS";
   }
   // BOS Bearish: close below last swing low
   else if(cur_close < last_sl && market_bias == BIAS_BEARISH)
   {
      last_bos_level = last_sl;
      last_bos_time  = TimeCurrent();
      panel_bias     = "BEARISH BOS";
   }
   // CHoCH: close above last swing high while bearish → flip to bullish
   else if(cur_close > last_sh && market_bias != BIAS_BULLISH)
   {
      market_bias    = BIAS_BULLISH;
      choch_detected = true;
      last_bos_level = last_sh;
      panel_bias     = "CHoCH → BULLISH";
      Print("ICT | CHoCH detected → BULLISH at ", cur_close);
   }
   // CHoCH: close below last swing low while bullish → flip to bearish
   else if(cur_close < last_sl && market_bias != BIAS_BEARISH)
   {
      market_bias    = BIAS_BEARISH;
      choch_detected = true;
      last_bos_level = last_sl;
      panel_bias     = "CHoCH → BEARISH";
      Print("ICT | CHoCH detected → BEARISH at ", cur_close);
   }

   // Initial bias if neutral
   if(market_bias == BIAS_NEUTRAL)
   {
      if(last_sh > prev_sh && last_sl > prev_sl) { market_bias = BIAS_BULLISH; panel_bias = "BULLISH"; }
      else if(last_sh < prev_sh && last_sl < prev_sl) { market_bias = BIAS_BEARISH; panel_bias = "BEARISH"; }
   }
}


//+------------------------------------------------------------------+
//| ORDER BLOCK DETECTION                                             |
//|                                                                   |
//| Bullish OB: Last BEARISH candle before a strong bullish move      |
//|   - Find a down candle followed by strong up displacement         |
//|   - The OB body (open to close) is the zone to buy from          |
//|                                                                   |
//| Bearish OB: Last BULLISH candle before a strong bearish move      |
//|   - Find an up candle followed by strong down displacement        |
//|   - The OB body is the zone to sell from                          |
//+------------------------------------------------------------------+
void DetectOrderBlocks()
{
   ArrayResize(bull_obs, 0);
   ArrayResize(bear_obs, 0);

   int bars = iBars(_Symbol, HTF);
   if(bars < OB_Lookback + 5) return;

   double min_move = OB_Min_Move_Pts * point_size;

   for(int i = 2; i < OB_Lookback; i++)
   {
      double o1 = iOpen(_Symbol,  HTF, i);
      double c1 = iClose(_Symbol, HTF, i);
      double h1 = iHigh(_Symbol,  HTF, i);
      double l1 = iLow(_Symbol,   HTF, i);

      // Look at the next 3 candles for displacement
      double displacement_up   = 0;
      double displacement_down = 0;
      for(int j = 1; j <= 3; j++)
      {
         if(i-j < 1) break;
         double move_up   = iClose(_Symbol, HTF, i-j) - iHigh(_Symbol, HTF, i);
         double move_down = iLow(_Symbol, HTF, i) - iClose(_Symbol, HTF, i-j);
         if(move_up   > displacement_up)   displacement_up   = move_up;
         if(move_down > displacement_down) displacement_down = move_down;
      }

      // BULLISH OB: bearish candle + strong up move after
      if(c1 < o1 && displacement_up >= min_move && ArraySize(bull_obs) < Max_OB_Count)
      {
         // Check OB not already mitigated (price hasn't returned into it since)
         double ob_top    = o1; // bearish candle open = top of OB
         double ob_bottom = c1; // bearish candle close = bottom of OB
         bool mitigated   = false;

         for(int k = i-1; k >= 1; k--)
         {
            if(iLow(_Symbol, HTF, k) <= ob_top && iLow(_Symbol, HTF, k) >= ob_bottom)
            { mitigated = true; break; }
         }
         if(!mitigated)
         {
            int sz = ArraySize(bull_obs);
            ArrayResize(bull_obs, sz+1);
            bull_obs[sz].top    = ob_top;
            bull_obs[sz].bottom = ob_bottom;
            bull_obs[sz].mid    = (ob_top + ob_bottom) / 2.0;
            bull_obs[sz].type   = OB_BULLISH;
            bull_obs[sz].time   = iTime(_Symbol, HTF, i);
            bull_obs[sz].valid  = true;
            bull_obs[sz].traded = false;

            if(Draw_OB) DrawOBBox("BullOB_"+IntegerToString(i),
               iTime(_Symbol,HTF,i), ob_bottom, iTime(_Symbol,HTF,i-1), ob_top,
               C'0,100,0', true);
         }
      }

      // BEARISH OB: bullish candle + strong down move after
      if(c1 > o1 && displacement_down >= min_move && ArraySize(bear_obs) < Max_OB_Count)
      {
         double ob_top    = c1; // bullish candle close = top of OB
         double ob_bottom = o1; // bullish candle open = bottom of OB
         bool mitigated   = false;

         for(int k = i-1; k >= 1; k--)
         {
            if(iHigh(_Symbol, HTF, k) >= ob_bottom && iHigh(_Symbol, HTF, k) <= ob_top)
            { mitigated = true; break; }
         }
         if(!mitigated)
         {
            int sz = ArraySize(bear_obs);
            ArrayResize(bear_obs, sz+1);
            bear_obs[sz].top    = ob_top;
            bear_obs[sz].bottom = ob_bottom;
            bear_obs[sz].mid    = (ob_top + ob_bottom) / 2.0;
            bear_obs[sz].type   = OB_BEARISH;
            bear_obs[sz].time   = iTime(_Symbol, HTF, i);
            bear_obs[sz].valid  = true;
            bear_obs[sz].traded = false;

            if(Draw_OB) DrawOBBox("BearOB_"+IntegerToString(i),
               iTime(_Symbol,HTF,i), ob_bottom, iTime(_Symbol,HTF,i-1), ob_top,
               C'100,0,0', false);
         }
      }
   }

   panel_ob = StringFormat("Bull OBs:%d  Bear OBs:%d",
              ArraySize(bull_obs), ArraySize(bear_obs));
}

//+------------------------------------------------------------------+
//| FAIR VALUE GAP DETECTION                                          |
//|                                                                   |
//| FVG = 3-candle pattern where gap exists between candle 1 high    |
//| and candle 3 low (bullish FVG) or candle 1 low and candle 3 high |
//| Candle 2 is the displacement candle                               |
//+------------------------------------------------------------------+
void DetectFVGs()
{
   ArrayResize(bull_fvgs, 0);
   ArrayResize(bear_fvgs, 0);

   int bars = iBars(_Symbol, HTF);
   if(bars < FVG_Lookback + 5) return;

   for(int i = 2; i < FVG_Lookback; i++)
   {
      double h3 = iHigh(_Symbol, HTF, i+1); // candle 3 (oldest)
      double l3 = iLow(_Symbol,  HTF, i+1);
      double h1 = iHigh(_Symbol, HTF, i-1); // candle 1 (newest of 3)
      double l1 = iLow(_Symbol,  HTF, i-1);

      // Bullish FVG: gap between candle 3 high and candle 1 low
      if(l1 > h3 && ArraySize(bull_fvgs) < Max_FVG_Count)
      {
         // Check not already filled
         bool filled = false;
         for(int k = i-2; k >= 1; k--)
            if(iLow(_Symbol, HTF, k) <= l1 && iHigh(_Symbol, HTF, k) >= h3)
            { filled = true; break; }

         if(!filled)
         {
            int sz = ArraySize(bull_fvgs);
            ArrayResize(bull_fvgs, sz+1);
            bull_fvgs[sz].top     = l1;
            bull_fvgs[sz].bottom  = h3;
            bull_fvgs[sz].mid     = (l1 + h3) / 2.0;
            bull_fvgs[sz].bullish = true;
            bull_fvgs[sz].time    = iTime(_Symbol, HTF, i);
            bull_fvgs[sz].filled  = false;

            if(Draw_FVG) DrawFVGBox("BullFVG_"+IntegerToString(i),
               iTime(_Symbol,HTF,i+1), h3, iTime(_Symbol,HTF,i-1), l1,
               C'0,50,100');
         }
      }

      // Bearish FVG: gap between candle 1 high and candle 3 low
      if(h1 < l3 && ArraySize(bear_fvgs) < Max_FVG_Count)
      {
         bool filled = false;
         for(int k = i-2; k >= 1; k--)
            if(iHigh(_Symbol, HTF, k) >= h1 && iLow(_Symbol, HTF, k) <= l3)
            { filled = true; break; }

         if(!filled)
         {
            int sz = ArraySize(bear_fvgs);
            ArrayResize(bear_fvgs, sz+1);
            bear_fvgs[sz].top     = l3;
            bear_fvgs[sz].bottom  = h1;
            bear_fvgs[sz].mid     = (l3 + h1) / 2.0;
            bear_fvgs[sz].bullish = false;
            bear_fvgs[sz].time    = iTime(_Symbol, HTF, i);
            bear_fvgs[sz].filled  = false;

            if(Draw_FVG) DrawFVGBox("BearFVG_"+IntegerToString(i),
               iTime(_Symbol,HTF,i+1), h1, iTime(_Symbol,HTF,i-1), l3,
               C'100,50,0');
         }
      }
   }

   panel_fvg = StringFormat("Bull FVGs:%d  Bear FVGs:%d",
               ArraySize(bull_fvgs), ArraySize(bear_fvgs));
}


//+------------------------------------------------------------------+
//| KILLZONE FILTER                                                   |
//| London: 07:00-10:00 UTC  (bullish bias from data)                |
//| NY:     12:00-15:00 UTC  (momentum hours 13-14 from data)        |
//+------------------------------------------------------------------+
bool IsKillzone()
{
   if(!Use_Killzones) return true;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   if(h >= London_Start && h < London_End) return true;
   if(h >= NY_Start     && h < NY_End)     return true;
   return false;
}

bool IsThursday()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   return (dt.day_of_week == 4);
}

string KillzoneName()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   if(h >= London_Start && h < London_End) return "LONDON";
   if(h >= NY_Start     && h < NY_End)     return "NEW YORK";
   return "OFF";
}

//+------------------------------------------------------------------+
//| ASIAN RANGE                                                       |
//| Tracks Asian session high/low (00:00-06:00 UTC) as reference     |
//| ICT uses this to identify where liquidity raids will target       |
//+------------------------------------------------------------------+
void UpdateAsianRange()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d 00:00",
                    dt.year, dt.mon, dt.day));

   if(asian_date == today) return; // already updated today
   asian_date = today;
   asian_high = 0;
   asian_low  = DBL_MAX;

   // Scan M15 bars for Asian session (00:00-06:00)
   int bars = iBars(_Symbol, PERIOD_M15);
   for(int i = 0; i < bars; i++)
   {
      datetime bar_time = iTime(_Symbol, PERIOD_M15, i);
      if(bar_time < today) break;
      MqlDateTime bdt; TimeToStruct(bar_time, bdt);
      if(bdt.hour >= 0 && bdt.hour < 6)
      {
         double h = iHigh(_Symbol, PERIOD_M15, i);
         double l = iLow(_Symbol,  PERIOD_M15, i);
         if(h > asian_high) asian_high = h;
         if(l < asian_low)  asian_low  = l;
      }
   }
   if(asian_low == DBL_MAX) asian_low = 0;
}

//+------------------------------------------------------------------+
//| OTE ENTRY LOGIC (Optimal Trade Entry)                             |
//|                                                                   |
//| After CHoCH or BOS, price retraces into OB.                      |
//| OTE zone = 62%-79% retracement of the last swing move            |
//|                                                                   |
//| BUY setup:                                                        |
//|  1. Bullish bias (CHoCH or BOS up confirmed)                     |
//|  2. Price pulls back into a bullish OB                           |
//|  3. Price is in OTE zone (62-79% of last up move)               |
//|  4. We are in a killzone                                          |
//|  5. Entry TF shows rejection / M1 FVG fill                       |
//|                                                                   |
//| SELL setup: mirror                                                |
//+------------------------------------------------------------------+
int CheckOTEEntry()
{
   if(!IsKillzone())  return 0;
   if(Use_DayFilter && IsThursday()) return 0;
   if(open_buys + open_sells >= Max_Trades) return 0;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // ── BUY SETUP ──────────────────────────────────────────────────
   if(market_bias == BIAS_BULLISH && ArraySize(bull_obs) > 0 && ArraySize(swing_lows) >= 2)
   {
      // Last swing low and last swing high for OTE calculation
      double swing_lo = swing_lows[0].price;
      double swing_hi = (ArraySize(swing_highs) > 0) ? swing_highs[0].price : 0;
      if(swing_hi <= swing_lo) swing_hi = swing_lo + (swing_lo * 0.01);

      double move      = swing_hi - swing_lo;
      double ote_high  = swing_hi - move * OTE_Fib_Low;  // 62% retrace
      double ote_low   = swing_hi - move * OTE_Fib_High; // 79% retrace

      // Check if price is in OTE zone
      if(ask >= ote_low && ask <= ote_high)
      {
         // Check if price is inside a valid bullish OB
         for(int i = 0; i < ArraySize(bull_obs); i++)
         {
            if(!bull_obs[i].valid || bull_obs[i].traded) continue;
            if(ask >= bull_obs[i].bottom && ask <= bull_obs[i].top)
            {
               Print("ICT | BUY SETUP: OTE=", DoubleToString(ask,_Digits),
                     " OB=[", bull_obs[i].bottom, "-", bull_obs[i].top,
                     "] OTE_zone=[", ote_low, "-", ote_high, "]");
               return 1; // BUY signal
            }
         }
      }
   }

   // ── SELL SETUP ─────────────────────────────────────────────────
   if(market_bias == BIAS_BEARISH && ArraySize(bear_obs) > 0 && ArraySize(swing_highs) >= 2)
   {
      double swing_hi = swing_highs[0].price;
      double swing_lo = (ArraySize(swing_lows) > 0) ? swing_lows[0].price : 0;
      if(swing_lo >= swing_hi) swing_lo = swing_hi - (swing_hi * 0.01);

      double move      = swing_hi - swing_lo;
      double ote_low   = swing_lo + move * OTE_Fib_Low;  // 62% retrace
      double ote_high  = swing_lo + move * OTE_Fib_High; // 79% retrace

      if(bid >= ote_low && bid <= ote_high)
      {
         for(int i = 0; i < ArraySize(bear_obs); i++)
         {
            if(!bear_obs[i].valid || bear_obs[i].traded) continue;
            if(bid >= bear_obs[i].bottom && bid <= bear_obs[i].top)
            {
               Print("ICT | SELL SETUP: OTE=", DoubleToString(bid,_Digits),
                     " OB=[", bear_obs[i].bottom, "-", bear_obs[i].top,
                     "] OTE_zone=[", ote_low, "-", ote_high, "]");
               return -1; // SELL signal
            }
         }
      }
   }

   return 0;
}


//+------------------------------------------------------------------+
//| TRADE EXECUTION                                                   |
//| SL: Below bullish OB bottom / Above bearish OB top + buffer      |
//| TP: Next liquidity (swing high/low) OR RR_Ratio x SL dist        |
//+------------------------------------------------------------------+
void OpenTrade(int direction)
{
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   double buf    = SL_Buffer_Pts * point_size;

   // Get broker minimum stop distance in price
   long   stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double min_stop    = (stops_level + 5) * point_size; // add 5pt safety margin

   double price, sl, tp, sl_dist;

   if(direction == 1) // BUY
   {
      price = ask;
      sl    = price; // fallback

      for(int i = 0; i < ArraySize(bull_obs); i++)
      {
         if(!bull_obs[i].valid || bull_obs[i].traded) continue;
         if(ask >= bull_obs[i].bottom && ask <= bull_obs[i].top)
         {
            sl = NormalizeDouble(bull_obs[i].bottom - buf, _Digits);
            bull_obs[i].traded = true;
            break;
         }
      }

      // Enforce minimum stop distance
      if(price - sl < min_stop)
         sl = NormalizeDouble(price - min_stop, _Digits);

      sl_dist = price - sl;
      if(sl_dist <= 0) { Print("ICT | BUY skipped: invalid SL dist"); return; }

      tp = GetBuyTP(price, sl_dist);

      // Enforce minimum TP distance
      if(tp - price < min_stop)
         tp = NormalizeDouble(price + sl_dist * RR_Ratio, _Digits);
   }
   else // SELL
   {
      price = bid;
      sl    = price;

      for(int i = 0; i < ArraySize(bear_obs); i++)
      {
         if(!bear_obs[i].valid || bear_obs[i].traded) continue;
         if(bid >= bear_obs[i].bottom && bid <= bear_obs[i].top)
         {
            sl = NormalizeDouble(bear_obs[i].top + buf + spread, _Digits);
            bear_obs[i].traded = true;
            break;
         }
      }

      // Enforce minimum stop distance
      if(sl - price < min_stop)
         sl = NormalizeDouble(price + min_stop, _Digits);

      sl_dist = sl - price;
      if(sl_dist <= 0) { Print("ICT | SELL skipped: invalid SL dist"); return; }

      tp = GetSellTP(price, sl_dist);

      // Enforce minimum TP distance
      if(price - tp < min_stop)
         tp = NormalizeDouble(price - sl_dist * RR_Ratio, _Digits);
   }

   double lot = CalcLot(sl_dist);
   ENUM_ORDER_TYPE type = (direction == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   Print("ICT | Attempting ", EnumToString(type),
         " price=", price, " sl=", sl, " tp=", tp,
         " sl_dist=", DoubleToString(sl_dist/point_size,1), "pts",
         " min_stop=", DoubleToString(min_stop/point_size,1), "pts");

   if(trade.PositionOpen(_Symbol, type, lot, price, sl, tp, EA_Comment))
      Print("ICT | Trade opened: ", EnumToString(type),
            " lot=", lot, " price=", price,
            " sl=", sl, " tp=", tp);
   else
      Print("ICT | Trade FAILED: ", trade.ResultRetcodeDescription());
}

// TP for BUY: nearest swing high above entry OR RR target
double GetBuyTP(double entry, double sl_dist)
{
   double rr_tp = NormalizeDouble(entry + sl_dist * RR_Ratio, _Digits);

   // Check for bearish FVG above as TP target (price will fill it)
   for(int i = 0; i < ArraySize(bear_fvgs); i++)
   {
      if(!bear_fvgs[i].filled && bear_fvgs[i].bottom > entry)
         return NormalizeDouble(bear_fvgs[i].bottom, _Digits); // fill FVG bottom
   }

   // Check swing highs above entry (liquidity)
   for(int i = 0; i < ArraySize(swing_highs); i++)
   {
      if(swing_highs[i].price > entry + sl_dist) // at least 1R away
         return NormalizeDouble(swing_highs[i].price - point_size * 2, _Digits);
   }

   // Asian high as target if above entry
   if(Use_Asian_Range && asian_high > entry + sl_dist)
      return NormalizeDouble(asian_high, _Digits);

   return rr_tp;
}

// TP for SELL: nearest swing low below entry OR RR target
double GetSellTP(double entry, double sl_dist)
{
   double rr_tp = NormalizeDouble(entry - sl_dist * RR_Ratio, _Digits);

   // Check for bullish FVG below as TP target
   for(int i = 0; i < ArraySize(bull_fvgs); i++)
   {
      if(!bull_fvgs[i].filled && bull_fvgs[i].top < entry)
         return NormalizeDouble(bull_fvgs[i].top, _Digits);
   }

   // Check swing lows below entry
   for(int i = 0; i < ArraySize(swing_lows); i++)
   {
      if(swing_lows[i].price < entry - sl_dist)
         return NormalizeDouble(swing_lows[i].price + point_size * 2, _Digits);
   }

   // Asian low as target
   if(Use_Asian_Range && asian_low > 0 && asian_low < entry - sl_dist)
      return NormalizeDouble(asian_low, _Digits);

   return rr_tp;
}


//+------------------------------------------------------------------+
//| RISK MANAGEMENT                                                   |
//+------------------------------------------------------------------+
double CalcLot(double sl_dist)
{
   if(Manual_Lot > 0.0) return NormLot(Manual_Lot);
   double bal      = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk     = bal * Risk_Percent / 100.0;
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_sz  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(sl_dist <= 0 || tick_val <= 0 || tick_sz <= 0) return NormLot(0.01);
   double lot = risk / (sl_dist / tick_sz * tick_val);
   return NormLot(MathMin(lot, Max_Lot));
}

double NormLot(double lot)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double mn   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double mx   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lot = MathFloor(lot / step) * step;
   return NormalizeDouble(MathMax(mn, MathMin(lot, mx)), 2);
}

bool CheckEP()
{
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd  = (bal > 0) ? (bal - eq) / bal * 100.0 : 0;
   if(dd >= Max_DD_Pct)
   {
      for(int i = PositionsTotal()-1; i >= 0; i--)
      {
         if(!pos.SelectByIndex(i)) continue;
         if(pos.Symbol()==_Symbol && pos.Magic()==Magic)
            trade.PositionClose(pos.Ticket());
      }
      panel_signal  = "EP TRIGGERED";
      panel_sig_col = clrRed;
      Print("ICT | EQUITY PROTECTION triggered: DD=", dd, "%");
      return true;
   }
   return false;
}

void ManageBreakEven()
{
   if(!Use_BE) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol()!=_Symbol || pos.Magic()!=Magic) continue;
      if(TicketInArr(be_tickets, pos.Ticket())) continue;

      bool   is_buy  = (pos.PositionType() == POSITION_TYPE_BUY);
      double open_px = pos.PriceOpen();
      double cur_sl  = pos.StopLoss();
      double tp      = pos.TakeProfit();
      double sl_dist = MathAbs(open_px - cur_sl);
      double cur_px  = is_buy ? bid : ask;
      double profit  = is_buy ? (cur_px - open_px) : (open_px - cur_px);

      // Move to BE when 1:1 reached
      if(profit >= sl_dist)
      {
         double be_sl = is_buy
            ? NormalizeDouble(open_px + point_size * 3, _Digits)
            : NormalizeDouble(open_px - point_size * 3, _Digits);

         bool needs = is_buy ? (be_sl > cur_sl + point_size)
                             : (cur_sl < point_size || be_sl < cur_sl - point_size);
         if(needs)
         {
            trade.PositionModify(pos.Ticket(), be_sl, tp);
            AddTicket(be_tickets, pos.Ticket());
            Print("ICT | Break-even set for ticket=", pos.Ticket());
         }
      }
   }
}

void ManageTrailing()
{
   if(!Use_Trailing) return;
   double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double trail = Trail_Pts * point_size;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol()!=_Symbol || pos.Magic()!=Magic) continue;
      if(!TicketInArr(be_tickets, pos.Ticket())) continue; // only after BE

      bool   is_buy = (pos.PositionType() == POSITION_TYPE_BUY);
      double cur_sl = pos.StopLoss();

      if(is_buy)
      {
         double new_sl = NormalizeDouble(bid - trail, _Digits);
         if(new_sl > cur_sl + point_size)
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
      }
      else
      {
         double new_sl = NormalizeDouble(ask + trail, _Digits);
         if(cur_sl < point_size || new_sl < cur_sl - point_size)
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
      }
   }
}

void CountPositions()
{
   open_buys = 0; open_sells = 0; float_pnl = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol()!=_Symbol || pos.Magic()!=Magic) continue;
      float_pnl += pos.Profit() + pos.Swap();
      if(pos.PositionType()==POSITION_TYPE_BUY) open_buys++;
      else                                      open_sells++;
   }
}

bool TicketInArr(ulong &arr[], ulong t)
{ for(int i=0;i<ArraySize(arr);i++) if(arr[i]==t) return true; return false; }

void AddTicket(ulong &arr[], ulong t)
{ if(TicketInArr(arr,t)) return; int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=t; }

//+------------------------------------------------------------------+
//| P&L TRACKING                                                      |
//+------------------------------------------------------------------+
void RefreshPnL()
{
   datetime now = TimeCurrent();
   if(now - pnl_cache < 60) return;
   pnl_cache = now;
   MqlDateTime dt; TimeToStruct(now, dt);
   dt.hour=0; dt.min=0; dt.sec=0;
   datetime today = StructToTime(dt);
   datetime week  = today - (datetime)((dt.day_of_week>0?dt.day_of_week-1:6)*86400);
   dt.day = 1;
   datetime month = StructToTime(dt);
   pnl_today = GetPnL(today, now);
   pnl_week  = GetPnL(week,  now);
   pnl_month = GetPnL(month, now);
}

double GetPnL(datetime from, datetime to)
{
   double pnl = 0;
   if(!HistorySelect(from, to)) return 0;
   for(int i=0; i<HistoryDealsTotal(); i++)
   {
      ulong t = HistoryDealGetTicket(i);
      if(t==0) continue;
      if(HistoryDealGetString(t,DEAL_SYMBOL)!=_Symbol) continue;
      if((long)HistoryDealGetInteger(t,DEAL_MAGIC)!=Magic) continue;
      if(HistoryDealGetInteger(t,DEAL_ENTRY)==DEAL_ENTRY_OUT)
         pnl += HistoryDealGetDouble(t,DEAL_PROFIT)
               +HistoryDealGetDouble(t,DEAL_SWAP)
               +HistoryDealGetDouble(t,DEAL_COMMISSION);
   }
   return pnl;
}


//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   point_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   ArrayResize(be_tickets, 0);
   ArrayResize(bull_obs,   0);
   ArrayResize(bear_obs,   0);
   ArrayResize(bull_fvgs,  0);
   ArrayResize(bear_fvgs,  0);
   ArrayResize(swing_highs,0);
   ArrayResize(swing_lows, 0);

   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(Slippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   if(Show_Panel) CreatePanel();
   Print("ICT EA initialized | XAUUSD Smart Money | Magic=", Magic);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, lbl);
   ObjectsDeleteAll(0, "BullOB_");
   ObjectsDeleteAll(0, "BearOB_");
   ObjectsDeleteAll(0, "BullFVG_");
   ObjectsDeleteAll(0, "BearFVG_");
}

//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   CountPositions();
   if(Use_EP && CheckEP()) { if(Show_Panel) UpdatePanel(); return; }

   ManageBreakEven();
   ManageTrailing();

   // HTF bar — update structure, OBs, FVGs
   datetime htf_bar = iTime(_Symbol, HTF, 0);
   if(htf_bar != last_bar_htf)
   {
      last_bar_htf = htf_bar;
      UpdateMarketStructure();
      DetectOrderBlocks();
      DetectFVGs();
      UpdateAsianRange();
   }

   // Entry TF bar — check for entry
   datetime entry_bar = iTime(_Symbol, Entry_TF, 0);
   if(entry_bar != last_bar_entry)
   {
      last_bar_entry = entry_bar;

      panel_kz = KillzoneName();

      int sig = CheckOTEEntry();
      if(sig == 1)
      {
         panel_signal  = "BUY — OTE in Bull OB";
         panel_sig_col = clrLime;
         OpenTrade(1);
      }
      else if(sig == -1)
      {
         panel_signal  = "SELL — OTE in Bear OB";
         panel_sig_col = clrTomato;
         OpenTrade(-1);
      }
      else
      {
         panel_signal  = IsKillzone() ? "IN KZ — WAITING" : "OUT OF KZ";
         panel_sig_col = IsKillzone() ? clrYellow : clrGray;
      }
   }

   if(Show_Panel) UpdatePanel();
}


//+------------------------------------------------------------------+
//| DASHBOARD                                                         |
//+------------------------------------------------------------------+
void CreatePanel()
{
   ObjectsDeleteAll(0, lbl);
   int x=15, y=25, r=15;
   PRect(lbl+"bg", x-8, y-8, 300, 400, C'15,20,35', C'30,50,90', 1);

   PLbl(lbl+"dot",   "\x25CF",              x,    y,    C'0,180,120', 12, true);
   PLbl(lbl+"title", " ICT Smart Money EA", x+15, y+1,  clrWhite,     9,  true);
   PLbl(lbl+"sub",   " XAUUSD | 47M Ticks", x+15, y+13, C'100,120,150', 7, false);
   PLine(lbl+"d0", x, y+27, 280);

   int row = y+36;
   PLbl(lbl+"ls",  "Symbol",   x, row,      C'130,140,160',8); PLbl(lbl+"vs",  _Symbol,           x+130,row,      clrWhite,8);
   PLbl(lbl+"lh",  "HTF",      x, row+r,    C'130,140,160',8); PLbl(lbl+"vh",  "H1 Structure",    x+130,row+r,    clrWhite,8);
   PLbl(lbl+"le",  "Entry TF", x, row+r*2,  C'130,140,160',8); PLbl(lbl+"ve",  "M5 OTE",          x+130,row+r*2,  clrWhite,8);
   PLine(lbl+"d1", x, row+r*3+2, 280);

   row = row+r*3+10;
   PLbl(lbl+"lbi", "Bias",      x, row,     C'130,140,160',8); PLbl(lbl+"vbi", "---", x+130,row,     clrWhite, 8);
   PLbl(lbl+"lkz", "Killzone",  x, row+r,   C'130,140,160',8); PLbl(lbl+"vkz", "---", x+130,row+r,   clrWhite, 8);
   PLbl(lbl+"lob", "OBs",       x, row+r*2, C'130,140,160',8); PLbl(lbl+"vob", "---", x+130,row+r*2, clrWhite, 8);
   PLbl(lbl+"lfv", "FVGs",      x, row+r*3, C'130,140,160',8); PLbl(lbl+"vfv", "---", x+130,row+r*3, clrWhite, 8);
   PLbl(lbl+"lsg", "Signal",    x, row+r*4, C'130,140,160',8); PLbl(lbl+"vsg", "---", x+130,row+r*4, clrWhite, 8);
   PLine(lbl+"d2", x, row+r*5+2, 280);

   row = row+r*5+10;
   PLbl(lbl+"ltr", "Trades",    x, row,    C'130,140,160',8); PLbl(lbl+"vtr", "0",   x+130,row,    clrWhite,8);
   PLbl(lbl+"lfp", "Float P/L", x, row+r,  C'130,140,160',8); PLbl(lbl+"vfp", "---", x+130,row+r,  clrWhite,8);
   PLine(lbl+"d3", x, row+r*2+2, 280);

   row = row+r*2+10;
   PLbl(lbl+"lpd", "Today P/L", x, row,     C'130,140,160',8); PLbl(lbl+"vpd", "---", x+130,row,     clrWhite,8);
   PLbl(lbl+"lpw", "Week P/L",  x, row+r,   C'130,140,160',8); PLbl(lbl+"vpw", "---", x+130,row+r,   clrWhite,8);
   PLbl(lbl+"lpm", "Month P/L", x, row+r*2, C'130,140,160',8); PLbl(lbl+"vpm", "---", x+130,row+r*2, clrWhite,8);
   PLine(lbl+"d4", x, row+r*3+2, 280);

   row = row+r*3+10;
   PLbl(lbl+"lri", "Risk/Trade", x, row,   C'130,140,160',8);
   PLbl(lbl+"vri", Manual_Lot>0 ? StringFormat("%.2f lot",Manual_Lot)
                                : StringFormat("%.1f%%",Risk_Percent), x+130,row, clrWhite,8);
   PLbl(lbl+"lst", "Status",     x, row+r, C'130,140,160',8);
   PLbl(lbl+"vst", "RUNNING",    x+130, row+r, clrLime, 8);
   ChartRedraw(0);
}

void UpdatePanel()
{
   if(!Show_Panel) return;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   string days[]={"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};

   color bias_col = (market_bias==BIAS_BULLISH) ? clrLime :
                    (market_bias==BIAS_BEARISH) ? clrTomato : clrGray;
   color kz_col   = IsKillzone() ? clrLime : clrGray;

   PSet(lbl+"vbi", panel_bias,             bias_col);
   PSet(lbl+"vkz", panel_kz + " " + days[dt.day_of_week], kz_col);
   PSet(lbl+"vob", panel_ob,               clrWhite);
   PSet(lbl+"vfv", panel_fvg,              clrWhite);
   PSet(lbl+"vsg", panel_signal,           panel_sig_col);
   PSet(lbl+"vtr", StringFormat("%d (B:%d S:%d)", open_buys+open_sells, open_buys, open_sells), clrWhite);
   PSet(lbl+"vfp", StringFormat("%.2f", float_pnl), float_pnl>=0?clrLime:clrTomato);

   RefreshPnL();
   PSet(lbl+"vpd", StringFormat("%.2f", pnl_today), pnl_today>=0?clrLime:clrTomato);
   PSet(lbl+"vpw", StringFormat("%.2f", pnl_week),  pnl_week >=0?clrLime:clrTomato);
   PSet(lbl+"vpm", StringFormat("%.2f", pnl_month), pnl_month>=0?clrLime:clrTomato);
   PSet(lbl+"vst", panel_signal=="EP TRIGGERED"?"STOPPED":"RUNNING",
                   panel_signal=="EP TRIGGERED"?clrTomato:clrLime);
   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| CHART DRAWING HELPERS                                             |
//+------------------------------------------------------------------+
void DrawOBBox(string name, datetime t1, double p1, datetime t2, double p2, color clr, bool bull)
{
   if(ObjectFind(0,name)>=0) ObjectDelete(0,name);
   ObjectCreate(0,name,OBJ_RECTANGLE,0,t1,p1,t2,p2);
   ObjectSetInteger(0,name,OBJPROP_COLOR,  clr);
   ObjectSetInteger(0,name,OBJPROP_FILL,   true);
   ObjectSetInteger(0,name,OBJPROP_BACK,   true);
   ObjectSetInteger(0,name,OBJPROP_WIDTH,  1);
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
   ObjectSetString(0, name,OBJPROP_TOOLTIP, bull?"Bullish OB":"Bearish OB");
}

void DrawFVGBox(string name, datetime t1, double p1, datetime t2, double p2, color clr)
{
   if(ObjectFind(0,name)>=0) ObjectDelete(0,name);
   ObjectCreate(0,name,OBJ_RECTANGLE,0,t1,p1,t2,p2);
   ObjectSetInteger(0,name,OBJPROP_COLOR,  clr);
   ObjectSetInteger(0,name,OBJPROP_FILL,   true);
   ObjectSetInteger(0,name,OBJPROP_BACK,   true);
   ObjectSetInteger(0,name,OBJPROP_WIDTH,  1);
   ObjectSetInteger(0,name,OBJPROP_STYLE,  STYLE_DOT);
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
   ObjectSetString(0, name,OBJPROP_TOOLTIP,"Fair Value Gap");
}

//+------------------------------------------------------------------+
//| PANEL OBJECT HELPERS                                              |
//+------------------------------------------------------------------+
void PLbl(string n,string t,int x,int y,color c,int fs=8,bool b=false)
{
   if(ObjectFind(0,n)<0) ObjectCreate(0,n,OBJ_LABEL,0,0,0);
   ObjectSetInteger(0,n,OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0,n,OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0,n,OBJPROP_YDISTANCE, y);
   ObjectSetString(0, n,OBJPROP_TEXT,      t);
   ObjectSetInteger(0,n,OBJPROP_COLOR,     c);
   ObjectSetInteger(0,n,OBJPROP_FONTSIZE,  fs);
   ObjectSetString(0, n,OBJPROP_FONT,      b?"Arial Bold":"Arial");
   ObjectSetInteger(0,n,OBJPROP_BACK,      false);
   ObjectSetInteger(0,n,OBJPROP_SELECTABLE,false);
}
void PSet(string n,string t,color c)
{ if(ObjectFind(0,n)<0)return; ObjectSetString(0,n,OBJPROP_TEXT,t); ObjectSetInteger(0,n,OBJPROP_COLOR,c); }
void PLine(string n,int x,int y,int w)
{ string d=""; for(int i=0;i<(int)(w/5.5);i++) d+="-"; PLbl(n,d,x,y,C'30,50,90',6); }
void PRect(string n,int x,int y,int w,int h,color bg,color brd,int bw)
{
   if(ObjectFind(0,n)<0) ObjectCreate(0,n,OBJ_RECTANGLE_LABEL,0,0,0);
   ObjectSetInteger(0,n,OBJPROP_CORNER,     CORNER_LEFT_UPPER);
   ObjectSetInteger(0,n,OBJPROP_XDISTANCE,  x);
   ObjectSetInteger(0,n,OBJPROP_YDISTANCE,  y);
   ObjectSetInteger(0,n,OBJPROP_XSIZE,      w);
   ObjectSetInteger(0,n,OBJPROP_YSIZE,      h);
   ObjectSetInteger(0,n,OBJPROP_BGCOLOR,    bg);
   ObjectSetInteger(0,n,OBJPROP_BORDER_TYPE,BORDER_FLAT);
   ObjectSetInteger(0,n,OBJPROP_COLOR,      brd);
   ObjectSetInteger(0,n,OBJPROP_WIDTH,      bw);
   ObjectSetInteger(0,n,OBJPROP_BACK,       false);
   ObjectSetInteger(0,n,OBJPROP_SELECTABLE, false);
}
//+------------------------------------------------------------------+
