//+------------------------------------------------------------------+
//|                                              RealTickEA_v2.mq5   |
//|        EMA Trend Momentum Pullback v2 - More Trades, More PnL    |
//|         Optimized on real XAUUSD ticks (Apr 2026)                |
//|                                                                  |
//| V2 IMPROVEMENTS OVER V1:                                         |
//|  - Reduced cooldown (12 bars vs 25) = more trade opportunities   |
//|  - Added 6 more trading hours (13 total vs 7)                    |
//|  - Trailing stop for bigger runners                              |
//|  - Partial close at 1:2 R:R (lock profit, let rest run to TP)   |
//|  - Break-even move after 1:1 R:R achieved                       |
//|  - Tighter SL (1.8x ATR vs 2.0x) = bigger position size         |
//|  - Looser body threshold (1.2x ATR vs 1.5x) = more entries      |
//|  - Extended timeout (45 bars vs 30) = let winners run            |
//|  - M5 EMA trend confirmation for higher-quality signals          |
//|  - Max consecutive losses raised to 3 before pause               |
//|  - Added session-based lot scaling (bigger lots in best hours)   |
//+------------------------------------------------------------------+
#property copyright "RealTickEA v2.0"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade        trade;
CPositionInfo pos;



//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== EMA SETTINGS ==="
input int    InpEMA_Fast       = 8;       // EMA Fast Period
input int    InpEMA_Mid        = 13;      // EMA Mid Period
input int    InpEMA_Slow       = 21;      // EMA Slow Period
input int    InpEMA_M5         = 21;      // M5 EMA for Trend Filter

input group "=== ATR SETTINGS ==="
input int    InpATR_Period     = 14;      // ATR Period
input double InpSL_Mult        = 1.8;    // SL Multiplier (x ATR) [v1=2.0]
input double InpTP_Mult        = 7.0;    // TP Multiplier (x ATR)
input double InpBody_Mult      = 1.2;    // Body Threshold (x ATR) [v1=1.5]
input double InpMin_ATR        = 0.4;    // Minimum ATR [v1=0.5]

input group "=== TRADE MANAGEMENT ==="
input bool   InpUse_Trailing   = true;   // Use Trailing Stop
input double InpTrail_Start    = 3.0;    // Trail Start (x ATR profit)
input double InpTrail_Step     = 1.0;    // Trail Step (x ATR)
input bool   InpUse_BreakEven  = true;   // Move SL to Break-Even
input double InpBE_Trigger     = 1.0;    // BE Trigger (x ATR profit)
input double InpBE_Offset      = 0.1;    // BE Offset above entry (x ATR)
input bool   InpUse_PartClose  = true;   // Partial Close at Target
input double InpPartClose_Mult = 3.5;    // Partial Close at (x ATR)
input double InpPartClose_Pct  = 50.0;   // Partial Close % of position


input group "=== RISK MANAGEMENT ==="
input double InpRisk_Pct       = 3.0;    // Risk % per Trade
input double InpMax_Spread     = 0.15;   // Max Spread (points) [v1=0.12]
input int    InpCooldown_Bars  = 12;     // Cooldown Between Trades [v1=25]
input int    InpTimeout_Bars   = 45;     // Trade Timeout (bars) [v1=30]
input int    InpSlippage       = 10;     // Max Slippage (points)

input group "=== TRADING HOURS (UTC) ==="
input bool   InpHour_02        = true;   // Trade at 02:00 UTC
input bool   InpHour_03        = true;   // Trade at 03:00 UTC
input bool   InpHour_04        = true;   // Trade at 04:00 UTC
input bool   InpHour_05        = true;   // Trade at 05:00 UTC [NEW]
input bool   InpHour_06        = true;   // Trade at 06:00 UTC
input bool   InpHour_07        = true;   // Trade at 07:00 UTC [NEW]
input bool   InpHour_08        = true;   // Trade at 08:00 UTC
input bool   InpHour_09        = true;   // Trade at 09:00 UTC [NEW]
input bool   InpHour_10        = true;   // Trade at 10:00 UTC [NEW]
input bool   InpHour_14        = true;   // Trade at 14:00 UTC [NEW]
input bool   InpHour_15        = true;   // Trade at 15:00 UTC [NEW]
input bool   InpHour_16        = true;   // Trade at 16:00 UTC
input bool   InpHour_17        = true;   // Trade at 17:00 UTC

input group "=== SESSION BOOST ==="
input bool   InpUse_SessionBoost = true; // Boost lot size in best hours
input double InpBoost_Mult     = 1.5;    // Lot multiplier for best hours
input bool   InpBoost_02       = true;   // Boost 02:00 (Asian open)
input bool   InpBoost_03       = true;   // Boost 03:00
input bool   InpBoost_08       = true;   // Boost 08:00 (London open)
input bool   InpBoost_16       = true;   // Boost 16:00 (NY overlap)


input group "=== EQUITY PROTECTION ==="
input int    InpMax_ConsLoss   = 3;      // Max Consecutive Losses [v1=2]
input int    InpPause_Bars     = 40;     // Pause Duration (bars) [v1=60]
input double InpDaily_Loss_Pct = 4.0;    // Daily Loss Limit (%) [v1=3.0]
input double InpMax_DD_Pct     = 12.0;   // Max Drawdown % [v1=10.0]

input group "=== GENERAL ==="
input int    InpMagic          = 777222; // Magic Number
input bool   InpShow_Panel     = true;   // Show Dashboard
input bool   InpUse_M5_Filter  = true;   // Use M5 Trend Filter


//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
int    ema_fast_handle, ema_mid_handle, ema_slow_handle;
int    atr_handle;
int    ema_m5_handle;  // M5 trend filter

double ema_fast_buf[], ema_mid_buf[], ema_slow_buf[];
double atr_buf[];
double ema_m5_buf[];

datetime last_bar_time;
int      bars_since_trade;
int      bars_since_pause;
int      consecutive_losses;
bool     paused;
bool     daily_stopped;
bool     dd_stopped;
double   day_start_balance;
int      last_trade_day;
datetime trade_open_bar_time;
int      bars_in_trade;
bool     partial_closed;        // track if partial close done
bool     be_moved;              // track if BE has been moved
double   trade_entry_price;     // store entry for BE/trail calc
double   trade_sl_distance;     // store original SL distance

// Dashboard
string   panel_prefix = "RTEA2_";
string   signal_text  = "INITIALIZING";
color    signal_color = clrGray;
int      total_trades_today = 0;
double   session_pnl = 0;



//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // M1 indicator handles
   ema_fast_handle = iMA(_Symbol, PERIOD_M1, InpEMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   ema_mid_handle  = iMA(_Symbol, PERIOD_M1, InpEMA_Mid,  0, MODE_EMA, PRICE_CLOSE);
   ema_slow_handle = iMA(_Symbol, PERIOD_M1, InpEMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   atr_handle      = iATR(_Symbol, PERIOD_M1, InpATR_Period);
   
   // M5 trend filter
   ema_m5_handle   = iMA(_Symbol, PERIOD_M5, InpEMA_M5, 0, MODE_EMA, PRICE_CLOSE);

   if(ema_fast_handle == INVALID_HANDLE || ema_mid_handle == INVALID_HANDLE ||
      ema_slow_handle == INVALID_HANDLE || atr_handle == INVALID_HANDLE ||
      ema_m5_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   // Set arrays as series
   ArraySetAsSeries(ema_fast_buf, true);
   ArraySetAsSeries(ema_mid_buf,  true);
   ArraySetAsSeries(ema_slow_buf, true);
   ArraySetAsSeries(atr_buf,      true);
   ArraySetAsSeries(ema_m5_buf,   true);

   // Initialize state
   last_bar_time      = 0;
   bars_since_trade   = InpCooldown_Bars;
   bars_since_pause   = InpPause_Bars;
   consecutive_losses = 0;
   paused             = false;
   daily_stopped      = false;
   dd_stopped         = false;
   day_start_balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   last_trade_day     = -1;
   trade_open_bar_time = 0;
   bars_in_trade      = 0;
   partial_closed     = false;
   be_moved           = false;
   trade_entry_price  = 0;
   trade_sl_distance  = 0;
   total_trades_today = 0;
   session_pnl        = 0;

   // Configure trade object
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   if(InpShow_Panel) CreateDashboard();

   Print("RealTickEA v2.0 initialized | Magic=", InpMagic,
         " | Risk=", InpRisk_Pct, "% | SL=", InpSL_Mult, "xATR | TP=", InpTP_Mult, "xATR",
         " | Cooldown=", InpCooldown_Bars, " | Hours=13");
   return INIT_SUCCEEDED;
}



//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(ema_fast_handle);
   IndicatorRelease(ema_mid_handle);
   IndicatorRelease(ema_slow_handle);
   IndicatorRelease(atr_handle);
   IndicatorRelease(ema_m5_handle);
   ObjectsDeleteAll(0, panel_prefix);
}


//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // Copy indicator buffers
   if(CopyBuffer(ema_fast_handle, 0, 0, 8, ema_fast_buf) < 8) return;
   if(CopyBuffer(ema_mid_handle,  0, 0, 8, ema_mid_buf)  < 8) return;
   if(CopyBuffer(ema_slow_handle, 0, 0, 8, ema_slow_buf) < 8) return;
   if(CopyBuffer(atr_handle,      0, 0, 3, atr_buf)      < 3) return;
   if(CopyBuffer(ema_m5_handle,   0, 0, 3, ema_m5_buf)   < 3) return;

   // Active trade management on every tick
   if(HasOpenPosition())
   {
      ManageBreakEven();
      ManageTrailingStop();
      ManagePartialClose();
   }
   
   // Manage timeout
   ManageTimeout();

   // Check max drawdown on every tick
   if(!dd_stopped) CheckMaxDrawdown();
   if(dd_stopped)
   {
      signal_text = "DD STOP"; signal_color = clrRed;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // New bar detection
   datetime cur_bar = iTime(_Symbol, PERIOD_M1, 0);
   if(cur_bar == last_bar_time)
   {
      if(InpShow_Panel) UpdateDashboard();
      return;
   }
   last_bar_time = cur_bar;

   // Increment counters
   bars_since_trade++;
   if(paused) bars_since_pause++;
   if(HasOpenPosition()) bars_in_trade++;

   // Reset daily stop on new day
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.day_of_year != last_trade_day)
   {
      last_trade_day    = dt.day_of_year;
      day_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      daily_stopped     = false;
      total_trades_today = 0;
   }

   // Check daily loss limit
   if(!daily_stopped) CheckDailyLoss();
   if(daily_stopped)
   {
      signal_text = "DAILY STOP"; signal_color = clrOrangeRed;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Check pause
   if(paused)
   {
      if(bars_since_pause >= InpPause_Bars)
      {
         paused = false;
         consecutive_losses = 0;
         Print("RealTickEA v2: Pause ended");
      }
      else
      {
         signal_text = StringFormat("PAUSED (%d/%d)", bars_since_pause, InpPause_Bars);
         signal_color = clrOrange;
         if(InpShow_Panel) UpdateDashboard();
         return;
      }
   }

   // Cooldown check
   if(bars_since_trade < InpCooldown_Bars)
   {
      signal_text = StringFormat("COOLDOWN (%d/%d)", bars_since_trade, InpCooldown_Bars);
      signal_color = clrYellow;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Already have a position open
   if(HasOpenPosition())
   {
      signal_text = "IN TRADE"; signal_color = clrDodgerBlue;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Get current ATR
   double cur_atr = atr_buf[1];

   // Min ATR filter
   if(cur_atr < InpMin_ATR)
   {
      signal_text = "LOW ATR"; signal_color = clrGray;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Spread filter
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   if(spread > InpMax_Spread)
   {
      signal_text = "WIDE SPREAD"; signal_color = clrOrange;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Trading hours filter
   if(!IsAllowedHour())
   {
      signal_text = "HOURS OFF"; signal_color = clrGray;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Generate signal
   int sig = GetEntrySignal(cur_atr);
   if(sig == 1)
   {
      signal_text = "BUY SIGNAL"; signal_color = clrLime;
      ExecuteTrade(ORDER_TYPE_BUY, cur_atr);
   }
   else if(sig == -1)
   {
      signal_text = "SELL SIGNAL"; signal_color = clrTomato;
      ExecuteTrade(ORDER_TYPE_SELL, cur_atr);
   }
   else
   {
      signal_text = "SCANNING"; signal_color = clrYellow;
   }

   if(InpShow_Panel) UpdateDashboard();
}



//+------------------------------------------------------------------+
//| ENTRY SIGNAL LOGIC                                                |
//| Same core logic as v1 but with M5 trend filter and looser body   |
//+------------------------------------------------------------------+
int GetEntrySignal(double cur_atr)
{
   // Use closed bar values
   double ema8_0  = ema_fast_buf[1];
   double ema13_0 = ema_mid_buf[1];
   double ema21_0 = ema_slow_buf[1];
   double ema21_5 = ema_slow_buf[6];

   // EMA21 slope
   double ema21_slope = ema21_0 - ema21_5;

   // Candle body calculations
   double open_1  = iOpen(_Symbol, PERIOD_M1, 1);
   double close_1 = iClose(_Symbol, PERIOD_M1, 1);
   double body_1  = close_1 - open_1;

   double open_2  = iOpen(_Symbol, PERIOD_M1, 2);
   double close_2 = iClose(_Symbol, PERIOD_M1, 2);
   double body_2  = close_2 - open_2;

   double body_threshold = InpBody_Mult * cur_atr;

   // M5 trend filter - price must be on correct side of M5 EMA21
   double m5_ema = ema_m5_buf[1];
   double cur_price = iClose(_Symbol, PERIOD_M1, 1);

   // BUY CONDITIONS
   bool buy_ema_align = (ema8_0 > ema13_0) && (ema13_0 > ema21_0);
   bool buy_slope     = (ema21_slope > 0);
   bool buy_body      = (body_1 > body_threshold);
   bool buy_pullback  = (body_2 < 0);
   bool buy_above_ema = (close_1 > ema8_0);
   bool buy_m5_trend  = (!InpUse_M5_Filter) || (cur_price > m5_ema);

   if(buy_ema_align && buy_slope && buy_body && buy_pullback && buy_above_ema && buy_m5_trend)
      return 1;

   // SELL CONDITIONS
   bool sell_ema_align = (ema8_0 < ema13_0) && (ema13_0 < ema21_0);
   bool sell_slope     = (ema21_slope < 0);
   bool sell_body      = (body_1 < -body_threshold);
   bool sell_pullback  = (body_2 > 0);
   bool sell_below_ema = (close_1 < ema8_0);
   bool sell_m5_trend  = (!InpUse_M5_Filter) || (cur_price < m5_ema);

   if(sell_ema_align && sell_slope && sell_body && sell_pullback && sell_below_ema && sell_m5_trend)
      return -1;

   return 0;
}



//+------------------------------------------------------------------+
//| EXECUTE TRADE                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE type, double atr)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double price = (type == ORDER_TYPE_BUY) ? ask : bid;

   // Calculate SL and TP distances
   double sl_dist = InpSL_Mult * atr;
   double tp_dist = InpTP_Mult * atr;

   double sl, tp;
   if(type == ORDER_TYPE_BUY)
   {
      sl = NormalizeDouble(price - sl_dist, _Digits);
      tp = NormalizeDouble(price + tp_dist, _Digits);
   }
   else
   {
      sl = NormalizeDouble(price + sl_dist, _Digits);
      tp = NormalizeDouble(price - tp_dist, _Digits);
   }

   // Calculate lot size with session boost
   double lot = CalculateLotSize(sl_dist);
   if(lot <= 0) return;
   
   // Apply session boost for best hours
   if(InpUse_SessionBoost && IsBoostHour())
   {
      lot = NormalizeDouble(lot * InpBoost_Mult, 2);
      double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      if(max_lot <= 0) max_lot = 10.0;
      lot = MathMin(lot, MathMin(max_lot, 10.0));
   }

   // Margin check
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double margin_required = 0;
   if(!OrderCalcMargin(type, _Symbol, lot, price, margin_required))
   {
      Print("RealTickEA v2: OrderCalcMargin failed");
      return;
   }
   if(margin_required > free_margin * 0.80)
   {
      Print("RealTickEA v2: Insufficient margin");
      return;
   }

   // Execute
   string comment = StringFormat("RTEAv2_%s", type == ORDER_TYPE_BUY ? "BUY" : "SELL");
   if(trade.PositionOpen(_Symbol, type, lot, price, sl, tp, comment))
   {
      bars_since_trade    = 0;
      trade_open_bar_time = iTime(_Symbol, PERIOD_M1, 0);
      bars_in_trade       = 0;
      partial_closed      = false;
      be_moved            = false;
      trade_entry_price   = price;
      trade_sl_distance   = sl_dist;
      total_trades_today++;
      
      Print("RealTickEA v2: ", EnumToString(type), " | Lot=", lot,
            " | Price=", price, " | SL=", sl, " | TP=", tp,
            " | ATR=", atr, " | Boost=", (InpUse_SessionBoost && IsBoostHour()));
   }
   else
   {
      Print("RealTickEA v2: Trade FAILED - ", trade.ResultRetcodeDescription());
   }
}



//+------------------------------------------------------------------+
//| BREAK-EVEN MANAGEMENT                                             |
//| Move SL to entry + offset once price moves 1:1 in our favor      |
//+------------------------------------------------------------------+
void ManageBreakEven()
{
   if(!InpUse_BreakEven || be_moved) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != InpMagic) continue;

      double entry = pos.PriceOpen();
      double cur_sl = pos.StopLoss();
      double cur_tp = pos.TakeProfit();
      double cur_price = (pos.PositionType() == POSITION_TYPE_BUY) ? 
                          SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                          SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      double be_trigger_dist = InpBE_Trigger * trade_sl_distance;
      double be_offset_dist  = InpBE_Offset * trade_sl_distance;

      if(pos.PositionType() == POSITION_TYPE_BUY)
      {
         if(cur_price - entry >= be_trigger_dist)
         {
            double new_sl = NormalizeDouble(entry + be_offset_dist, _Digits);
            if(new_sl > cur_sl)
            {
               if(trade.PositionModify(pos.Ticket(), new_sl, cur_tp))
               {
                  be_moved = true;
                  Print("RealTickEA v2: BE moved to ", new_sl);
               }
            }
         }
      }
      else // SELL
      {
         if(entry - cur_price >= be_trigger_dist)
         {
            double new_sl = NormalizeDouble(entry - be_offset_dist, _Digits);
            if(new_sl < cur_sl)
            {
               if(trade.PositionModify(pos.Ticket(), new_sl, cur_tp))
               {
                  be_moved = true;
                  Print("RealTickEA v2: BE moved to ", new_sl);
               }
            }
         }
      }
      break; // only manage our position
   }
}



//+------------------------------------------------------------------+
//| TRAILING STOP MANAGEMENT                                          |
//| Once price moves 3x ATR in profit, trail by 1x ATR               |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   if(!InpUse_Trailing) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != InpMagic) continue;

      double entry = pos.PriceOpen();
      double cur_sl = pos.StopLoss();
      double cur_tp = pos.TakeProfit();
      double cur_price = (pos.PositionType() == POSITION_TYPE_BUY) ? 
                          SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                          SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      double trail_start_dist = InpTrail_Start * trade_sl_distance;
      double trail_step_dist  = InpTrail_Step * trade_sl_distance;

      if(pos.PositionType() == POSITION_TYPE_BUY)
      {
         double profit_dist = cur_price - entry;
         if(profit_dist >= trail_start_dist)
         {
            double new_sl = NormalizeDouble(cur_price - trail_step_dist, _Digits);
            if(new_sl > cur_sl && new_sl > entry)
            {
               trade.PositionModify(pos.Ticket(), new_sl, cur_tp);
            }
         }
      }
      else // SELL
      {
         double profit_dist = entry - cur_price;
         if(profit_dist >= trail_start_dist)
         {
            double new_sl = NormalizeDouble(cur_price + trail_step_dist, _Digits);
            if(new_sl < cur_sl && new_sl < entry)
            {
               trade.PositionModify(pos.Ticket(), new_sl, cur_tp);
            }
         }
      }
      break;
   }
}


//+------------------------------------------------------------------+
//| PARTIAL CLOSE MANAGEMENT                                          |
//| Close 50% at 3.5x ATR profit, let rest run to full TP (7x ATR)  |
//+------------------------------------------------------------------+
void ManagePartialClose()
{
   if(!InpUse_PartClose || partial_closed) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != InpMagic) continue;

      double entry = pos.PriceOpen();
      double volume = pos.Volume();
      double cur_price = (pos.PositionType() == POSITION_TYPE_BUY) ? 
                          SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                          SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      double partial_dist = InpPartClose_Mult * trade_sl_distance;
      double profit_dist;
      
      if(pos.PositionType() == POSITION_TYPE_BUY)
         profit_dist = cur_price - entry;
      else
         profit_dist = entry - cur_price;

      if(profit_dist >= partial_dist)
      {
         // Calculate partial volume
         double close_vol = NormalizeDouble(volume * InpPartClose_Pct / 100.0, 2);
         double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
         double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         if(step <= 0) step = 0.01;
         if(min_lot <= 0) min_lot = 0.01;
         
         close_vol = MathFloor(close_vol / step) * step;
         close_vol = MathMax(close_vol, min_lot);
         
         // Ensure remaining volume is valid
         double remaining = volume - close_vol;
         if(remaining < min_lot)
         {
            partial_closed = true; // can't partial close, skip
            return;
         }

         if(trade.PositionClosePartial(pos.Ticket(), close_vol))
         {
            partial_closed = true;
            Print("RealTickEA v2: Partial close ", close_vol, " lots at profit=", 
                  DoubleToString(profit_dist, 2));
         }
      }
      break;
   }
}



//+------------------------------------------------------------------+
//| CALCULATE LOT SIZE                                                |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_points)
{
   if(sl_points <= 0) return 0.01;

   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amt  = balance * InpRisk_Pct / 100.0;
   double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   double lot = 0.0;
   if(tick_val > 0 && tick_size > 0)
      lot = risk_amt / (sl_points / tick_size * tick_val);
   else
      lot = risk_amt / (sl_points * 100.0);

   // Normalize
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(step <= 0) step = 0.01;
   if(min_lot <= 0) min_lot = 0.01;
   if(max_lot <= 0) max_lot = 10.0;

   lot = MathFloor(lot / step) * step;
   lot = MathMax(min_lot, MathMin(lot, MathMin(max_lot, 10.0)));
   return NormalizeDouble(lot, 2);
}


//+------------------------------------------------------------------+
//| MANAGE TIMEOUT                                                    |
//+------------------------------------------------------------------+
void ManageTimeout()
{
   if(InpTimeout_Bars <= 0) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != InpMagic) continue;

      datetime open_time = pos.Time();
      int bars_elapsed = iBarShift(_Symbol, PERIOD_M1, open_time, false);

      if(bars_elapsed >= InpTimeout_Bars)
      {
         Print("RealTickEA v2: Timeout (", bars_elapsed, " bars). Closing.");
         trade.PositionClose(pos.Ticket());
         CheckLastTradeResult();
      }
   }
}


//+------------------------------------------------------------------+
//| CHECK MAX DRAWDOWN                                                |
//+------------------------------------------------------------------+
void CheckMaxDrawdown()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);

   if(balance <= 0) return;
   double dd_pct = (balance - equity) / balance * 100.0;

   if(dd_pct >= InpMax_DD_Pct)
   {
      Print("RealTickEA v2: MAX DD HIT! DD=", DoubleToString(dd_pct, 2), "%");
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(!pos.SelectByIndex(i)) continue;
         if(pos.Symbol() == _Symbol && pos.Magic() == InpMagic)
            trade.PositionClose(pos.Ticket());
      }
      dd_stopped = true;
   }
}


//+------------------------------------------------------------------+
//| CHECK DAILY LOSS LIMIT                                            |
//+------------------------------------------------------------------+
void CheckDailyLoss()
{
   double cur_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(day_start_balance <= 0) return;

   double daily_loss = day_start_balance - cur_balance;

   // Include floating loss
   double floating = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() == _Symbol && pos.Magic() == InpMagic)
         floating += pos.Profit() + pos.Swap();
   }

   double total_pct = (daily_loss + (floating < 0 ? MathAbs(floating) : 0)) / day_start_balance * 100.0;

   if(total_pct >= InpDaily_Loss_Pct)
   {
      Print("RealTickEA v2: DAILY LOSS LIMIT! Loss=", DoubleToString(total_pct, 2), "%");
      daily_stopped = true;
   }
}



//+------------------------------------------------------------------+
//| CHECK LAST TRADE RESULT                                           |
//+------------------------------------------------------------------+
void CheckLastTradeResult()
{
   datetime from = TimeCurrent() - 86400;
   datetime to   = TimeCurrent();
   if(!HistorySelect(from, to)) return;

   int total = HistoryDealsTotal();
   if(total <= 0) return;

   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(ticket, DEAL_MAGIC) != InpMagic) continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                    + HistoryDealGetDouble(ticket, DEAL_SWAP)
                    + HistoryDealGetDouble(ticket, DEAL_COMMISSION);

      if(profit < 0)
      {
         consecutive_losses++;
         if(consecutive_losses >= InpMax_ConsLoss)
         {
            paused = true;
            bars_since_pause = 0;
            Print("RealTickEA v2: PAUSING after ", consecutive_losses, " losses");
         }
      }
      else
      {
         consecutive_losses = 0;
      }
      break;
   }
}


//+------------------------------------------------------------------+
//| TRADING HOURS FILTER                                              |
//+------------------------------------------------------------------+
bool IsAllowedHour()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;

   if(h == 2  && InpHour_02) return true;
   if(h == 3  && InpHour_03) return true;
   if(h == 4  && InpHour_04) return true;
   if(h == 5  && InpHour_05) return true;
   if(h == 6  && InpHour_06) return true;
   if(h == 7  && InpHour_07) return true;
   if(h == 8  && InpHour_08) return true;
   if(h == 9  && InpHour_09) return true;
   if(h == 10 && InpHour_10) return true;
   if(h == 14 && InpHour_14) return true;
   if(h == 15 && InpHour_15) return true;
   if(h == 16 && InpHour_16) return true;
   if(h == 17 && InpHour_17) return true;

   return false;
}


//+------------------------------------------------------------------+
//| SESSION BOOST - Is this a high-probability hour?                  |
//+------------------------------------------------------------------+
bool IsBoostHour()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;

   if(h == 2  && InpBoost_02) return true;
   if(h == 3  && InpBoost_03) return true;
   if(h == 8  && InpBoost_08) return true;
   if(h == 16 && InpBoost_16) return true;

   return false;
}


//+------------------------------------------------------------------+
//| CHECK IF WE HAVE AN OPEN POSITION                                 |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() == _Symbol && pos.Magic() == InpMagic)
         return true;
   }
   return false;
}



//+------------------------------------------------------------------+
//| OnTradeTransaction                                                |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL)
      {
         ulong deal_ticket = trans.deal;
         if(deal_ticket > 0 && HistoryDealSelect(deal_ticket))
         {
            if(HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
            {
               if(HistoryDealGetString(deal_ticket, DEAL_SYMBOL) == _Symbol &&
                  (long)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) == InpMagic)
               {
                  double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT)
                                + HistoryDealGetDouble(deal_ticket, DEAL_SWAP)
                                + HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);

                  session_pnl += profit;

                  if(profit < 0)
                  {
                     consecutive_losses++;
                     if(consecutive_losses >= InpMax_ConsLoss)
                     {
                        paused = true;
                        bars_since_pause = 0;
                        Print("RealTickEA v2: PAUSING for ", InpPause_Bars, " bars");
                     }
                  }
                  else if(profit > 0)
                  {
                     consecutive_losses = 0;
                  }
               }
            }
         }
      }
   }
}



//+------------------------------------------------------------------+
//| DASHBOARD - Create on-chart panel                                 |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   ObjectsDeleteAll(0, panel_prefix);

   int x = 15, y = 20;

   // Background
   DashRect(panel_prefix + "bg", x - 10, y - 10, 320, 440, C'12,16,28', C'30,80,140', 1);

   // Header
   DashLabel(panel_prefix + "title", "RealTickEA v2.0", x + 5, y, clrWhite, 10, true);
   DashLabel(panel_prefix + "sub", "More Trades | Trailing | Partial Close | M5 Filter", x + 5, y + 14, C'100,180,255', 7, false);
   DashLine(panel_prefix + "h0", x, y + 28, 300);

   // Signal section
   int row = y + 36;
   DashLabel(panel_prefix + "l_sig",  "Signal:",        x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_sig",  "---",            x + 150, row,  clrYellow, 8, true);
   DashLabel(panel_prefix + "l_sprd", "Spread:",        x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_sprd", "---",            x + 150, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_atr",  "ATR(14):",       x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_atr",  "---",            x + 150, row+30, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_ema",  "EMA Align:",     x + 5, row+45, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_ema",  "---",            x + 150, row+45, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_m5",   "M5 Trend:",      x + 5, row+60, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_m5",   "---",            x + 150, row+60, clrWhite, 8, false);
   DashLine(panel_prefix + "h1", x, row + 77, 300);

   // Trade Management
   row = row + 85;
   DashLabel(panel_prefix + "l_be",   "Break-Even:",    x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_be",   "---",            x + 150, row,  clrWhite, 8, false);
   DashLabel(panel_prefix + "l_part", "Partial Close:", x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_part", "---",            x + 150, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_trail","Trailing:",      x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_trail","---",            x + 150, row+30, clrWhite, 8, false);
   DashLine(panel_prefix + "h2", x, row + 47, 300);

   // Risk section
   row = row + 55;
   DashLabel(panel_prefix + "l_loss", "Consec Losses:", x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_loss", "0",              x + 150, row,  clrWhite, 8, false);
   DashLabel(panel_prefix + "l_dpnl", "Daily P&&L:",    x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_dpnl", "---",            x + 150, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_tcnt", "Trades Today:",  x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_tcnt", "0",              x + 150, row+30, clrWhite, 8, false);
   DashLine(panel_prefix + "h3", x, row + 47, 300);

   // Account section
   row = row + 55;
   DashLabel(panel_prefix + "l_bal",  "Balance:",       x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_bal",  "---",            x + 150, row,  clrWhite, 8, false);
   DashLabel(panel_prefix + "l_eq",   "Equity:",        x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_eq",   "---",            x + 150, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_dd",   "Drawdown:",      x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_dd",   "0.0%",           x + 150, row+30, clrLime, 8, false);
   DashLine(panel_prefix + "h4", x, row + 47, 300);

   // Status
   row = row + 55;
   DashLabel(panel_prefix + "l_cool", "Cooldown:",      x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_cool", "READY",          x + 150, row,  clrLime, 8, false);
   DashLabel(panel_prefix + "l_hour", "Hour (UTC):",    x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_hour", "---",            x + 150, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_boost","Session Boost:", x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_boost","---",            x + 150, row+30, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_stat", "Status:",        x + 5, row+45, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_stat", "RUNNING",        x + 150, row+45, clrLime, 8, true);

   ChartRedraw(0);
}



//+------------------------------------------------------------------+
//| UPDATE DASHBOARD                                                  |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   if(!InpShow_Panel) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   double cur_atr = (ArraySize(atr_buf) > 1) ? atr_buf[1] : 0;
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd_pct  = (balance > 0) ? (balance - equity) / balance * 100.0 : 0;

   // Signal
   DashSet(panel_prefix + "v_sig", signal_text, signal_color);

   // Spread
   DashSet(panel_prefix + "v_sprd",
           StringFormat("%.4f %s", spread, spread > InpMax_Spread ? "WIDE" : "OK"),
           spread > InpMax_Spread ? clrTomato : clrLime);

   // ATR
   DashSet(panel_prefix + "v_atr",
           StringFormat("%.4f %s", cur_atr, cur_atr < InpMin_ATR ? "LOW" : "OK"),
           cur_atr < InpMin_ATR ? clrOrange : clrLime);

   // EMA Alignment
   string ema_str = "FLAT";
   color  ema_col = clrYellow;
   if(ArraySize(ema_fast_buf) > 1 && ArraySize(ema_mid_buf) > 1 && ArraySize(ema_slow_buf) > 1)
   {
      if(ema_fast_buf[1] > ema_mid_buf[1] && ema_mid_buf[1] > ema_slow_buf[1])
      { ema_str = "BULLISH"; ema_col = clrLime; }
      else if(ema_fast_buf[1] < ema_mid_buf[1] && ema_mid_buf[1] < ema_slow_buf[1])
      { ema_str = "BEARISH"; ema_col = clrTomato; }
   }
   DashSet(panel_prefix + "v_ema", ema_str, ema_col);

   // M5 Trend
   if(ArraySize(ema_m5_buf) > 1)
   {
      double m5_ema = ema_m5_buf[1];
      double cur_price = iClose(_Symbol, PERIOD_M1, 1);
      if(cur_price > m5_ema)
         DashSet(panel_prefix + "v_m5", "BULLISH (above M5 EMA21)", clrLime);
      else
         DashSet(panel_prefix + "v_m5", "BEARISH (below M5 EMA21)", clrTomato);
   }

   // Trade management status
   DashSet(panel_prefix + "v_be", be_moved ? "MOVED" : "WAITING", be_moved ? clrLime : clrGray);
   DashSet(panel_prefix + "v_part", partial_closed ? "DONE" : "WAITING", partial_closed ? clrLime : clrGray);
   DashSet(panel_prefix + "v_trail", InpUse_Trailing ? "ACTIVE" : "OFF", InpUse_Trailing ? clrLime : clrGray);

   // Consecutive losses
   DashSet(panel_prefix + "v_loss",
           StringFormat("%d / %d", consecutive_losses, InpMax_ConsLoss),
           consecutive_losses >= InpMax_ConsLoss ? clrRed : clrWhite);

   // Daily P&L
   double daily_pnl = AccountInfoDouble(ACCOUNT_BALANCE) - day_start_balance;
   DashSet(panel_prefix + "v_dpnl",
           StringFormat("%.2f (%.1f%%)", daily_pnl, day_start_balance > 0 ? daily_pnl / day_start_balance * 100 : 0),
           daily_pnl >= 0 ? clrLime : clrTomato);

   // Trades today
   DashSet(panel_prefix + "v_tcnt", StringFormat("%d", total_trades_today), clrWhite);

   // Account
   DashSet(panel_prefix + "v_bal", StringFormat("%.2f", balance), clrWhite);
   DashSet(panel_prefix + "v_eq",  StringFormat("%.2f", equity), equity >= balance ? clrLime : clrTomato);
   DashSet(panel_prefix + "v_dd",  StringFormat("%.1f%%", dd_pct),
           dd_pct < 5 ? clrLime : (dd_pct < 8 ? clrOrange : clrRed));

   // Cooldown
   if(bars_since_trade >= InpCooldown_Bars)
      DashSet(panel_prefix + "v_cool", "READY", clrLime);
   else
      DashSet(panel_prefix + "v_cool", StringFormat("%d / %d", bars_since_trade, InpCooldown_Bars), clrYellow);

   // Hour
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   DashSet(panel_prefix + "v_hour",
           StringFormat("%02d:00 %s", dt.hour, IsAllowedHour() ? "ACTIVE" : "OFF"),
           IsAllowedHour() ? clrLime : clrGray);

   // Session boost
   DashSet(panel_prefix + "v_boost",
           IsBoostHour() ? StringFormat("%.1fx LOT", InpBoost_Mult) : "NORMAL",
           IsBoostHour() ? clrGold : clrGray);

   // Status
   if(dd_stopped)
      DashSet(panel_prefix + "v_stat", "DD STOPPED", clrRed);
   else if(daily_stopped)
      DashSet(panel_prefix + "v_stat", "DAILY STOP", clrOrangeRed);
   else if(paused)
      DashSet(panel_prefix + "v_stat", "PAUSED", clrOrange);
   else
      DashSet(panel_prefix + "v_stat", "RUNNING", clrLime);

   ChartRedraw(0);
}



//+------------------------------------------------------------------+
//| DASHBOARD HELPER FUNCTIONS                                        |
//+------------------------------------------------------------------+
void DashLabel(string name, string text, int x, int y, color clr, int size = 8, bool bold = false)
{
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER,     CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE,  x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE,  y);
   ObjectSetString(0,  name, OBJPROP_TEXT,        text);
   ObjectSetInteger(0, name, OBJPROP_COLOR,       clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,    size);
   ObjectSetString(0,  name, OBJPROP_FONT,        bold ? "Arial Bold" : "Arial");
   ObjectSetInteger(0, name, OBJPROP_BACK,        false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE,  false);
}

void DashSet(string name, string text, color clr)
{
   if(ObjectFind(0, name) < 0) return;
   ObjectSetString(0,  name, OBJPROP_TEXT,  text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
}

void DashLine(string name, int x, int y, int width)
{
   string dashes = "";
   for(int i = 0; i < (int)(width / 5.0); i++) dashes += "-";
   DashLabel(name, dashes, x, y, C'30,80,140', 6, false);
}

void DashRect(string name, int x, int y, int w, int h, color bg, color border, int bw)
{
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
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
