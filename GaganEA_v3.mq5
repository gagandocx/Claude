//+------------------------------------------------------------------+
//|                                              GaganEA v3.0        |
//|                    Built from 47,172,211 XAUUSD ticks            |
//|                         Jan 2026 - Jun 2026                      |
//|                                                                  |
//| DATA-DRIVEN FINDINGS:                                            |
//|  - Best BUY hours (UTC): 9,10,13,14,23                          |
//|  - Best SELL hours (UTC): 2,4,5,17                              |
//|  - Best days: Mon/Tue (buy bias), avoid Thu (bearish -14.99)    |
//|  - London session bullish, NY session bearish                    |
//|  - Avg 1-min range: 3.14 pts | Spread: 0.69-0.70 at 12-14 UTC  |
//|  - Mean-reverting market (avg run = 2 ticks)                     |
//|  - Price range: 4025-5593 (strong bull trend Jan-Jun 2026)       |
//+------------------------------------------------------------------+
#property copyright "GaganEA v3.0 - Data Driven"
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade        trade;
CPositionInfo posInfo;


//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== SESSION & TIME FILTER (UTC) ==="
input bool   Use_TimeFilter       = true;    // Enable Time Filter
// BUY windows: 09:00-11:00, 13:00-15:00, 23:00-00:00 UTC
// SELL windows: 02:00-06:00, 17:00-18:00 UTC
input bool   Use_DayFilter        = true;    // Enable Day Filter
// Skip Thursday (bearish -14.99 avg, only 41% bullish)

input group "=== TREND FILTER ==="
input ENUM_TIMEFRAMES HTF_TF      = PERIOD_H4;   // HTF Timeframe
input int    EMA_Fast             = 21;            // Fast EMA Period
input int    EMA_Slow             = 50;            // Slow EMA Period
input int    EMA_Trend            = 200;           // Trend EMA Period
input int    RSI_Period           = 14;            // RSI Period
input int    RSI_OB               = 65;            // RSI Overbought
input int    RSI_OS               = 35;            // RSI Oversold

input group "=== ENTRY SETTINGS ==="
input ENUM_TIMEFRAMES Entry_TF    = PERIOD_M5;    // Entry Timeframe
input int    ATR_Period           = 14;            // ATR Period
input double ATR_Entry_Mult       = 0.5;           // ATR Multiplier for Entry Buffer
input int    Min_Spread_Filter    = 3;             // Max Spread to Allow Entry (points)

input group "=== LOT SIZE & RISK ==="
input double Risk_Percent         = 1.0;           // Risk % per Trade
input double Manual_Lot           = 0.0;           // Manual Lot (0 = auto)
input double Max_Lot              = 5.0;           // Max Lot Size

input group "=== STOP LOSS & TAKE PROFIT ==="
input double SL_ATR_Mult          = 2.0;           // SL = ATR x Multiplier
input double TP1_ATR_Mult         = 1.5;           // TP1 = ATR x Multiplier
input double TP2_ATR_Mult         = 3.0;           // TP2 = ATR x Multiplier
input double TP3_ATR_Mult         = 5.0;           // TP3 = ATR x Multiplier
input double TP1_Close_Pct        = 40.0;          // % to close at TP1
input double TP2_Close_Pct        = 40.0;          // % to close at TP2 (40% remaining)

input group "=== TRAILING STOP ==="
input bool   Use_Trailing         = true;          // Enable Trailing Stop
input double Trail_ATR_Mult       = 1.5;           // Trail Distance = ATR x Mult
input double Trail_Start_ATR_Mult = 1.5;           // Start trailing after TP1 hit

input group "=== EQUITY PROTECTION ==="
input bool   Use_Equity_Protect   = true;          // Enable Equity Protection
input double Max_DD_Percent       = 4.0;           // Max Drawdown % (close all)
input int    Max_Open_Trades      = 3;             // Max simultaneous trades

input group "=== DASHBOARD & MAGIC ==="
input bool   Show_Dashboard       = true;          // Show Dashboard
input int    Magic                = 303000;         // Magic Number
input string EA_Comment           = "GaganEA_v3";  // Trade Comment
input int    Slippage             = 10;            // Max Slippage (points)


//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
int    ema_fast_h, ema_slow_h, ema_trend_h;
int    ema_fast_e, ema_slow_e;
int    rsi_h, atr_h;

double ema_fast_htf[], ema_slow_htf[], ema_trend_htf[];
double ema_fast_ent[], ema_slow_ent[];
double rsi_buf[], atr_buf[];

double pip, point_size;
datetime last_bar;

// TP tracking
ulong  tp1_tickets[];
ulong  tp2_tickets[];

// Dashboard label prefix
string lbl = "GEA3_";

// Stats
int    open_buys, open_sells;
double float_pnl;
string signal_txt  = "INIT";
color  signal_col  = clrGray;

// P&L cache
double pnl_today, pnl_week, pnl_month;
datetime pnl_cache_time;


//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   point_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   pip = (digits == 3 || digits == 5) ? point_size * 10 : point_size;

   // HTF indicators
   ema_fast_h  = iMA(_Symbol, HTF_TF, EMA_Fast,  0, MODE_EMA, PRICE_CLOSE);
   ema_slow_h  = iMA(_Symbol, HTF_TF, EMA_Slow,  0, MODE_EMA, PRICE_CLOSE);
   ema_trend_h = iMA(_Symbol, HTF_TF, EMA_Trend, 0, MODE_EMA, PRICE_CLOSE);
   rsi_h       = iRSI(_Symbol, HTF_TF, RSI_Period, PRICE_CLOSE);

   // Entry TF indicators
   ema_fast_e  = iMA(_Symbol, Entry_TF, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   ema_slow_e  = iMA(_Symbol, Entry_TF, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   atr_h       = iATR(_Symbol, Entry_TF, ATR_Period);

   if(ema_fast_h==INVALID_HANDLE || ema_slow_h==INVALID_HANDLE ||
      ema_trend_h==INVALID_HANDLE || rsi_h==INVALID_HANDLE ||
      ema_fast_e==INVALID_HANDLE || ema_slow_e==INVALID_HANDLE ||
      atr_h==INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles.");
      return INIT_FAILED;
   }

   ArraySetAsSeries(ema_fast_htf,  true);
   ArraySetAsSeries(ema_slow_htf,  true);
   ArraySetAsSeries(ema_trend_htf, true);
   ArraySetAsSeries(ema_fast_ent,  true);
   ArraySetAsSeries(ema_slow_ent,  true);
   ArraySetAsSeries(rsi_buf,       true);
   ArraySetAsSeries(atr_buf,       true);

   ArrayResize(tp1_tickets, 0);
   ArrayResize(tp2_tickets, 0);

   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(Slippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   if(Show_Dashboard) CreateDashboard();

   Print("GaganEA v3.0 initialized | Data-driven XAUUSD EA | 47M ticks analyzed");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(ema_fast_h);
   IndicatorRelease(ema_slow_h);
   IndicatorRelease(ema_trend_h);
   IndicatorRelease(rsi_h);
   IndicatorRelease(ema_fast_e);
   IndicatorRelease(ema_slow_e);
   IndicatorRelease(atr_h);
   ObjectsDeleteAll(0, lbl);
}


//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // Refresh indicators
   if(CopyBuffer(ema_fast_h,  0, 0, 3, ema_fast_htf)  < 3) return;
   if(CopyBuffer(ema_slow_h,  0, 0, 3, ema_slow_htf)  < 3) return;
   if(CopyBuffer(ema_trend_h, 0, 0, 3, ema_trend_htf) < 3) return;
   if(CopyBuffer(rsi_h,       0, 0, 3, rsi_buf)       < 3) return;
   if(CopyBuffer(ema_fast_e,  0, 0, 3, ema_fast_ent)  < 3) return;
   if(CopyBuffer(ema_slow_e,  0, 0, 3, ema_slow_ent)  < 3) return;
   if(CopyBuffer(atr_h,       0, 0, 3, atr_buf)       < 3) return;

   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   double atr    = atr_buf[1];

   CountPositions();

   // Equity protection every tick
   if(Use_Equity_Protect && CheckEquityProtection()) { UpdateDashboard(); return; }

   // Manage open positions every tick
   ManageTP();
   if(Use_Trailing) ManageTrailing();

   // New bar logic only
   datetime cur_bar = iTime(_Symbol, Entry_TF, 0);
   if(cur_bar == last_bar) { UpdateDashboard(); return; }
   last_bar = cur_bar;

   // --- FILTERS ---
   if(Use_TimeFilter && !IsGoodTime())   { signal_txt="TIME FILTER"; signal_col=clrOrange; UpdateDashboard(); return; }
   if(Use_DayFilter  && IsThursday())    { signal_txt="SKIP THURSDAY"; signal_col=clrOrange; UpdateDashboard(); return; }
   if(spread > Min_Spread_Filter * point_size * 10) { signal_txt="SPREAD WIDE"; signal_col=clrOrange; UpdateDashboard(); return; }
   if(open_buys + open_sells >= Max_Open_Trades)    { signal_txt="MAX TRADES"; signal_col=clrOrange; UpdateDashboard(); return; }

   // --- SIGNALS ---
   bool buy_sig  = false;
   bool sell_sig = false;
   GetSignal(buy_sig, sell_sig);

   if(buy_sig)       { signal_txt="BUY READY";  signal_col=clrLime;   OpenTrade(ORDER_TYPE_BUY,  atr); }
   else if(sell_sig) { signal_txt="SELL READY"; signal_col=clrRed;    OpenTrade(ORDER_TYPE_SELL, atr); }
   else              { signal_txt="WAITING";     signal_col=clrYellow; }

   UpdateDashboard();
}


//+------------------------------------------------------------------+
//| DATA-DRIVEN TIME FILTER                                           |
//| BUY hours:  09-11, 13-15, 23 UTC (best bullish bias from data)   |
//| SELL hours: 02-06, 17-18 UTC (best bearish bias from data)       |
//| Combined: allow trading during statistically proven windows       |
//+------------------------------------------------------------------+
bool IsGoodTime()
{
   int hour = TimeHour(TimeCurrent());

   // Best trading windows from 47M tick analysis:
   // Buy bias:  09,10,13,14,23
   // Sell bias: 02,04,05,17
   // Combined active window (avoid dead hours 00,01,16,18,19,20,21,22 = low edge)
   // Actually allow 09-15 (London+NY open) and 23 (late session edge)
   // and early Asian sell windows 02-05, 17
   if(hour >= 9  && hour <= 15) return true;  // London + NY open — best hours
   if(hour == 23)               return true;  // Late session buy edge (58.97% bullish)
   if(hour >= 2  && hour <= 5)  return true;  // Early Asian sell edge
   if(hour == 17)               return true;  // NY close sell edge

   return false;
}

//+------------------------------------------------------------------+
//| Skip Thursday — worst day (41.46% bullish, mean -14.99)          |
//+------------------------------------------------------------------+
bool IsThursday()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return (dt.day_of_week == 4); // 4 = Thursday
}

//+------------------------------------------------------------------+
//| DIRECTIONAL BIAS based on current hour                            |
//| Returns: 1=buy only, -1=sell only, 0=both allowed                |
//+------------------------------------------------------------------+
int HourBias()
{
   int h = TimeHour(TimeCurrent());
   // Strong buy hours from data
   if(h==9 || h==10 || h==13 || h==14 || h==23) return 1;
   // Strong sell hours from data
   if(h==2 || h==4 || h==5 || h==17)            return -1;
   // Neutral
   return 0;
}

//+------------------------------------------------------------------+
//| SESSION BIAS: London bullish, NY bearish                          |
//+------------------------------------------------------------------+
int SessionBias()
{
   int h = TimeHour(TimeCurrent());
   if(h >= 8  && h <= 16) return 1;   // London — bullish bias
   if(h >= 13 && h <= 21) return -1;  // NY — bearish bias
   if(h >= 1  && h <= 8)  return 1;   // Asian — slight bullish bias
   return 0;
}


//+------------------------------------------------------------------+
//| SIGNAL GENERATION                                                 |
//| Multi-confluence: HTF trend + EMA cross + RSI + Hour bias        |
//+------------------------------------------------------------------+
void GetSignal(bool &buy_sig, bool &sell_sig)
{
   buy_sig  = false;
   sell_sig = false;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // HTF trend
   bool htf_bull  = (ema_fast_htf[1] > ema_slow_htf[1]) && (bid > ema_trend_htf[1]);
   bool htf_bear  = (ema_fast_htf[1] < ema_slow_htf[1]) && (bid < ema_trend_htf[1]);

   // EMA cross on entry TF (confirmed: previous bar cross)
   bool ema_cross_up   = (ema_fast_ent[2] < ema_slow_ent[2]) && (ema_fast_ent[1] > ema_slow_ent[1]);
   bool ema_cross_down = (ema_fast_ent[2] > ema_slow_ent[2]) && (ema_fast_ent[1] < ema_slow_ent[1]);

   // RSI filter
   bool rsi_buy  = (rsi_buf[1] > 40 && rsi_buf[1] < RSI_OB);  // not overbought
   bool rsi_sell = (rsi_buf[1] < 60 && rsi_buf[1] > RSI_OS);  // not oversold

   // Hour bias from 47M tick data
   int  h_bias       = HourBias();
   bool hour_ok_buy  = (h_bias >= 0);   // 0=both ok, 1=buy only
   bool hour_ok_sell = (h_bias <= 0);   // 0=both ok, -1=sell only

   // Day of week bias: Monday/Tuesday favor buys
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   bool mon_tue   = (dt.day_of_week == 1 || dt.day_of_week == 2);
   bool wed       = (dt.day_of_week == 3);
   bool fri       = (dt.day_of_week == 5);

   // === BUY SIGNAL ===
   // Needs: HTF bullish + EMA cross up + RSI not OB + bullish hour
   if(htf_bull && ema_cross_up && rsi_buy && hour_ok_buy)
   {
      // Extra weight on Mon/Tue (55-56% bullish from data)
      if(mon_tue || (!fri)) // avoid Friday (very noisy, std=94)
         buy_sig = true;
   }

   // === SELL SIGNAL ===
   // Needs: HTF bearish + EMA cross down + RSI not OS + bearish hour
   if(htf_bear && ema_cross_down && rsi_sell && hour_ok_sell)
   {
      if(!mon_tue) // Mon/Tue favor buys, not sells
         sell_sig = true;
   }

   // Never trade both directions at once
   if(buy_sig && sell_sig) { buy_sig = false; sell_sig = false; }

   // Block opposite direction if existing trades open
   if(open_sells > 0) buy_sig  = false;
   if(open_buys  > 0) sell_sig = false;
}


//+------------------------------------------------------------------+
//| OPEN TRADE with ATR-based SL/TP                                   |
//| ATR avg = 3.14 pts on M1, scaled to entry TF                     |
//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE type, double atr)
{
   double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double price = (type == ORDER_TYPE_BUY) ? ask : bid;
   double spread= ask - bid;

   double sl_dist  = SL_ATR_Mult  * atr;
   double tp1_dist = TP1_ATR_Mult * atr;
   double tp3_dist = TP3_ATR_Mult * atr;

   double sl  = 0, tp = 0;
   if(type == ORDER_TYPE_BUY)
   {
      sl = NormalizeDouble(price - sl_dist  - spread, _Digits);
      tp = NormalizeDouble(price + tp3_dist,           _Digits);
   }
   else
   {
      sl = NormalizeDouble(price + sl_dist  + spread, _Digits);
      tp = NormalizeDouble(price - tp3_dist,           _Digits);
   }

   double lot = CalcLot(sl_dist);

   if(trade.PositionOpen(_Symbol, type, lot, price, sl, tp, EA_Comment))
      Print("GaganEA v3 | Trade opened: ", EnumToString(type),
            " lot=", lot, " price=", price, " sl=", sl, " tp=", tp,
            " atr=", DoubleToString(atr, _Digits));
   else
      Print("GaganEA v3 | Trade FAILED: ", trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| CALCULATE LOT SIZE (risk-based)                                   |
//+------------------------------------------------------------------+
double CalcLot(double sl_dist)
{
   if(Manual_Lot > 0) return NormLot(Manual_Lot);

   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_money = balance * Risk_Percent / 100.0;
   double tick_val   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(sl_dist <= 0 || tick_val <= 0 || tick_size <= 0) return NormLot(0.01);

   double lot = risk_money / (sl_dist / tick_size * tick_val);
   return NormLot(MathMin(lot, Max_Lot));
}

double NormLot(double lot)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lot = MathFloor(lot / step) * step;
   return NormalizeDouble(MathMax(minv, MathMin(lot, maxv)), 2);
}


//+------------------------------------------------------------------+
//| MANAGE TAKE PROFIT PARTIALS                                       |
//+------------------------------------------------------------------+
void ManageTP()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(CopyBuffer(atr_h, 0, 0, 2, atr_buf) < 2) return;
   double atr = atr_buf[1];

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != Magic) continue;

      ulong  ticket  = posInfo.Ticket();
      double open_px = posInfo.PriceOpen();
      double vol     = posInfo.Volume();
      double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      bool   is_buy  = (posInfo.PositionType() == POSITION_TYPE_BUY);
      double cur_px  = is_buy ? bid : ask;

      double profit_pts = is_buy ? (cur_px - open_px) : (open_px - cur_px);

      bool hit_tp1 = TicketInArr(tp1_tickets, ticket);
      bool hit_tp2 = TicketInArr(tp2_tickets, ticket);

      // TP1: close TP1_Close_Pct% at TP1_ATR_Mult * ATR
      if(!hit_tp1 && profit_pts >= TP1_ATR_Mult * atr)
      {
         double close_vol = NormLot(vol * TP1_Close_Pct / 100.0);
         if(close_vol >= min_vol)
            trade.PositionClosePartial(ticket, close_vol);
         else
            trade.PositionClose(ticket);
         AddToArr(tp1_tickets, ticket);
         Print("TP1 hit | ticket=", ticket, " closed=", close_vol);
      }

      // TP2: close TP2_Close_Pct% of remaining at TP2_ATR_Mult * ATR
      if(hit_tp1 && !hit_tp2 && profit_pts >= TP2_ATR_Mult * atr)
      {
         double close_vol = NormLot(vol * TP2_Close_Pct / 100.0);
         if(close_vol >= min_vol)
            trade.PositionClosePartial(ticket, close_vol);
         else
            trade.PositionClose(ticket);
         AddToArr(tp2_tickets, ticket);
         Print("TP2 hit | ticket=", ticket, " closed=", close_vol);
      }
   }
   CleanArrays();
}

//+------------------------------------------------------------------+
//| TRAILING STOP — activates after TP1 hit                          |
//+------------------------------------------------------------------+
void ManageTrailing()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(CopyBuffer(atr_h, 0, 0, 2, atr_buf) < 2) return;
   double atr       = atr_buf[1];
   double trail_gap = Trail_ATR_Mult * atr;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != Magic) continue;

      ulong  ticket  = posInfo.Ticket();
      double open_px = posInfo.PriceOpen();
      double cur_sl  = posInfo.StopLoss();
      bool   is_buy  = (posInfo.PositionType() == POSITION_TYPE_BUY);
      bool   hit_tp1 = TicketInArr(tp1_tickets, ticket);

      // Only trail after TP1 hit
      if(!hit_tp1) continue;

      if(is_buy)
      {
         double new_sl = NormalizeDouble(bid - trail_gap, _Digits);
         double floor  = NormalizeDouble(open_px + point_size * 10, _Digits); // at least break-even
         new_sl = MathMax(new_sl, floor);
         if(new_sl > cur_sl + point_size)
            trade.PositionModify(ticket, new_sl, posInfo.TakeProfit());
      }
      else
      {
         double new_sl = NormalizeDouble(ask + trail_gap, _Digits);
         double floor  = NormalizeDouble(open_px - point_size * 10, _Digits);
         new_sl = MathMin(new_sl, floor);
         if(cur_sl < point_size || new_sl < cur_sl - point_size)
            trade.PositionModify(ticket, new_sl, posInfo.TakeProfit());
      }
   }
}


//+------------------------------------------------------------------+
//| COUNT OPEN POSITIONS                                              |
//+------------------------------------------------------------------+
void CountPositions()
{
   open_buys  = 0;
   open_sells = 0;
   float_pnl  = 0;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != Magic) continue;
      float_pnl += posInfo.Profit() + posInfo.Swap();
      if(posInfo.PositionType() == POSITION_TYPE_BUY)  open_buys++;
      else                                              open_sells++;
   }
}

//+------------------------------------------------------------------+
//| EQUITY PROTECTION                                                 |
//+------------------------------------------------------------------+
bool CheckEquityProtection()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd_pct  = (balance > 0) ? (balance - equity) / balance * 100.0 : 0;

   if(dd_pct >= Max_DD_Percent)
   {
      Print("EQUITY PROTECTION: DD=", DoubleToString(dd_pct,2), "% — closing all trades");
      for(int i = PositionsTotal()-1; i >= 0; i--)
      {
         if(!posInfo.SelectByIndex(i)) continue;
         if(posInfo.Symbol()==_Symbol && posInfo.Magic()==Magic)
            trade.PositionClose(posInfo.Ticket());
      }
      signal_txt = "EP TRIGGERED";
      signal_col = clrRed;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| TICKET ARRAY HELPERS                                              |
//+------------------------------------------------------------------+
bool TicketInArr(ulong &arr[], ulong t)
{
   for(int i=0;i<ArraySize(arr);i++) if(arr[i]==t) return true;
   return false;
}
void AddToArr(ulong &arr[], ulong t)
{
   if(TicketInArr(arr,t)) return;
   int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=t;
}
void CleanArrays()
{
   ulong n1[], n2[];
   for(int i=0;i<ArraySize(tp1_tickets);i++)
      if(posInfo.SelectByTicket(tp1_tickets[i]))
      { int s=ArraySize(n1); ArrayResize(n1,s+1); n1[s]=tp1_tickets[i]; }
   for(int i=0;i<ArraySize(tp2_tickets);i++)
      if(posInfo.SelectByTicket(tp2_tickets[i]))
      { int s=ArraySize(n2); ArrayResize(n2,s+1); n2[s]=tp2_tickets[i]; }
   ArrayCopy(tp1_tickets,n1); ArrayResize(tp1_tickets,ArraySize(n1));
   ArrayCopy(tp2_tickets,n2); ArrayResize(tp2_tickets,ArraySize(n2));
}

//+------------------------------------------------------------------+
//| PERIOD P&L                                                        |
//+------------------------------------------------------------------+
void RefreshPnL()
{
   datetime now = TimeCurrent();
   if(now - pnl_cache_time < 60) return;
   pnl_cache_time = now;

   MqlDateTime dt; TimeToStruct(now, dt);
   dt.hour=0; dt.min=0; dt.sec=0;
   datetime today = StructToTime(dt);
   dt.day = 1;
   datetime month_start = StructToTime(dt);
   dt.day = dt.day - dt.day_of_week + 1;
   datetime week_start = StructToTime(dt);

   pnl_today = GetPnL(today, now);
   pnl_week  = GetPnL(week_start, now);
   pnl_month = GetPnL(month_start, now);
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
//| DASHBOARD                                                         |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   ObjectsDeleteAll(0, lbl);
   int x=15, y=30, row=15;
   color bg=C'20,28,42', border=C'40,55,85';

   ObjRect(lbl+"bg", x-8, y-8, 300, 380, bg, border, 1);

   // Title
   ObjLbl(lbl+"bullet", "\x25A0",          x,    y,    C'0,180,100', 11, true);
   ObjLbl(lbl+"title",  " GaganEA v3.0",   x+14, y,    clrWhite,    9,  true);
   ObjLbl(lbl+"sub",    " Data-Driven XAUUSD EA", x+14, y+13, C'150,150,150', 7, false);
   ObjLine(lbl+"d0", x, y+27, 280);

   int r = y+36;
   ObjLbl(lbl+"l_sym",  "Symbol",    x,    r,       clrSilver, 8);
   ObjLbl(lbl+"v_sym",  _Symbol,     x+110,r,       clrWhite,  8);
   ObjLbl(lbl+"l_tf",   "Entry TF",  x,    r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_tf",   TFStr(Entry_TF), x+110, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_htf",  "HTF",       x,    r+row*2, clrSilver, 8);
   ObjLbl(lbl+"v_htf",  TFStr(HTF_TF), x+110, r+row*2, clrWhite, 8);
   ObjLine(lbl+"d1", x, r+row*3+2, 280);

   r = r+row*3+10;
   ObjLbl(lbl+"l_sig",  "Signal",    x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_sig",  "---",       x+110, r,   clrWhite,  8);
   ObjLbl(lbl+"l_hour", "Hour UTC",  x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_hour", "---",       x+110, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_day",  "Day",       x, r+row*2, clrSilver, 8);
   ObjLbl(lbl+"v_day",  "---",       x+110, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"l_sprd", "Spread",    x, r+row*3, clrSilver, 8);
   ObjLbl(lbl+"v_sprd", "---",       x+110, r+row*3, clrWhite, 8);
   ObjLine(lbl+"d2", x, r+row*4+2, 280);

   r = r+row*4+10;
   ObjLbl(lbl+"l_trades","Open Trades", x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_trades","0",           x+110, r,   clrWhite,  8);
   ObjLbl(lbl+"l_fpnl",  "Float P/L",  x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_fpnl",  "---",        x+110, r+row, clrWhite, 8);
   ObjLine(lbl+"d3", x, r+row*2+2, 280);

   r = r+row*2+10;
   ObjLbl(lbl+"l_today","Today P/L",  x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_today","---",        x+110, r,   clrWhite,  8);
   ObjLbl(lbl+"l_week", "Week P/L",   x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_week", "---",        x+110, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_month","Month P/L",  x, r+row*2, clrSilver, 8);
   ObjLbl(lbl+"v_month","---",        x+110, r+row*2, clrWhite, 8);
   ObjLine(lbl+"d4", x, r+row*3+2, 280);

   r = r+row*3+10;
   ObjLbl(lbl+"l_risk",  "Risk/Trade", x, r,     clrSilver, 8);
   ObjLbl(lbl+"v_risk",  Manual_Lot>0 ? StringFormat("Manual %.2f",Manual_Lot)
                                       : StringFormat("Auto %.1f%%",Risk_Percent),
                         x+110, r, clrWhite, 8);
   ObjLbl(lbl+"l_status","Status",    x, r+row,  clrSilver, 8);
   ObjLbl(lbl+"v_status","RUNNING",   x+110, r+row, clrLime, 8);

   ChartRedraw(0);
}

void UpdateDashboard()
{
   if(!Show_Dashboard) return;

   int h = TimeHour(TimeCurrent());
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   string days[] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
   string day_str = days[dt.day_of_week];
   bool thu = (dt.day_of_week == 4);

   double spread = (SymbolInfoDouble(_Symbol,SYMBOL_ASK)-SymbolInfoDouble(_Symbol,SYMBOL_BID))
                   / point_size / 10.0;

   ObjSet(lbl+"v_sig",    signal_txt, signal_col);
   ObjSet(lbl+"v_hour",   StringFormat("%02d:00 UTC %s", h,
                          !IsGoodTime() ? "(OFF)" : "(ON)"),
                          IsGoodTime() ? clrLime : clrOrange);
   ObjSet(lbl+"v_day",    day_str + (thu ? " SKIP!" : " OK"),
                          thu ? clrOrange : clrLime);
   ObjSet(lbl+"v_sprd",   StringFormat("%.1f pts %s", spread,
                          spread > Min_Spread_Filter ? "WIDE!" : "OK"),
                          spread > Min_Spread_Filter ? clrTomato : clrLime);
   ObjSet(lbl+"v_trades", StringFormat("%d (B:%d S:%d)",
                          open_buys+open_sells, open_buys, open_sells), clrWhite);
   ObjSet(lbl+"v_fpnl",   StringFormat("%.2f", float_pnl),
                          float_pnl >= 0 ? clrLime : clrTomato);

   RefreshPnL();
   ObjSet(lbl+"v_today",  StringFormat("%.2f", pnl_today),  pnl_today  >=0 ? clrLime : clrTomato);
   ObjSet(lbl+"v_week",   StringFormat("%.2f", pnl_week),   pnl_week   >=0 ? clrLime : clrTomato);
   ObjSet(lbl+"v_month",  StringFormat("%.2f", pnl_month),  pnl_month  >=0 ? clrLime : clrTomato);
   ObjSet(lbl+"v_status", signal_txt=="EP TRIGGERED" ? "EP TRIGGERED" : "RUNNING",
                          signal_txt=="EP TRIGGERED" ? clrTomato : clrLime);

   ChartRedraw(0);
}

void ObjLbl(string name, string txt, int x, int y, color c, int fs=8, bool bold=false)
{
   if(ObjectFind(0,name)<0) ObjectCreate(0,name,OBJ_LABEL,0,0,0);
   ObjectSetInteger(0,name,OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name,OBJPROP_TEXT,      txt);
   ObjectSetInteger(0,name,OBJPROP_COLOR,     c);
   ObjectSetInteger(0,name,OBJPROP_FONTSIZE,  fs);
   ObjectSetString(0, name,OBJPROP_FONT,      bold?"Arial Bold":"Arial");
   ObjectSetInteger(0,name,OBJPROP_BACK,      false);
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
}
void ObjSet(string name, string txt, color c)
{
   if(ObjectFind(0,name)<0) return;
   ObjectSetString(0, name,OBJPROP_TEXT,  txt);
   ObjectSetInteger(0,name,OBJPROP_COLOR, c);
}
void ObjLine(string name, int x, int y, int w)
{
   string d=""; int n=(int)(w/5.5);
   for(int i=0;i<n;i++) d+="-";
   ObjLbl(name,d,x,y,C'40,55,85',6);
}
void ObjRect(string name, int x, int y, int w, int h, color bg, color border, int bw)
{
   if(ObjectFind(0,name)<0) ObjectCreate(0,name,OBJ_RECTANGLE_LABEL,0,0,0);
   ObjectSetInteger(0,name,OBJPROP_CORNER,     CORNER_LEFT_UPPER);
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,  x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,  y);
   ObjectSetInteger(0,name,OBJPROP_XSIZE,      w);
   ObjectSetInteger(0,name,OBJPROP_YSIZE,      h);
   ObjectSetInteger(0,name,OBJPROP_BGCOLOR,    bg);
   ObjectSetInteger(0,name,OBJPROP_BORDER_TYPE,BORDER_FLAT);
   ObjectSetInteger(0,name,OBJPROP_COLOR,      border);
   ObjectSetInteger(0,name,OBJPROP_WIDTH,      bw);
   ObjectSetInteger(0,name,OBJPROP_BACK,       false);
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE, false);
}
string TFStr(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";  case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15"; case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";  case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";  default:         return "?";
   }
}
//+------------------------------------------------------------------+
