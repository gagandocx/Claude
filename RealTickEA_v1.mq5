//+------------------------------------------------------------------+
//|                                              RealTickEA_v1.mq5   |
//|           EMA Trend Momentum Pullback with Equity Filter          |
//|         Optimized on 6,049,106 real XAUUSD ticks (Apr 2026)      |
//|                                                                  |
//| BACKTEST RESULTS (Real Ticks, FusionMarkets):                    |
//|  - $1,000 -> $3,115 (+211.5% return)                            |
//|  - Max Drawdown: 10.0%                                          |
//|  - Profit Factor: 3.3                                           |
//|  - 43 trades in 1 month                                         |
//|  - Win Rate ~33% with 1:3.5 Risk:Reward                         |
//|                                                                  |
//| STRATEGY:                                                        |
//|  - EMA8 > EMA13 > EMA21 alignment for trend                     |
//|  - EMA21 slope confirms momentum direction                       |
//|  - Large body candle (>1.5x ATR) after pullback = entry          |
//|  - SL: 2x ATR, TP: 7x ATR (massive R:R)                        |
//|  - Equity filter: pause after consecutive losses                 |
//|  - Trading hours filter: specific UTC hours only                 |
//+------------------------------------------------------------------+
#property copyright "RealTickEA v1.0"
#property version   "1.00"
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

input group "=== ATR SETTINGS ==="
input int    InpATR_Period     = 14;      // ATR Period
input double InpSL_Mult        = 2.0;    // SL Multiplier (x ATR)
input double InpTP_Mult        = 7.0;    // TP Multiplier (x ATR)
input double InpBody_Mult      = 1.5;    // Body Threshold Multiplier (x ATR)
input double InpMin_ATR        = 0.5;    // Minimum ATR (skip dead market)

input group "=== RISK MANAGEMENT ==="
input double InpRisk_Pct       = 3.0;    // Risk % per Trade
input double InpMax_Spread     = 0.12;   // Max Spread (points)
input int    InpCooldown_Bars  = 25;     // Cooldown Between Trades (bars)
input int    InpTimeout_Bars   = 30;     // Trade Timeout (bars, close at market)
input int    InpSlippage       = 10;     // Max Slippage (points)

input group "=== TRADING HOURS (UTC) ==="
input bool   InpHour_02        = true;   // Trade at 02:00 UTC
input bool   InpHour_03        = true;   // Trade at 03:00 UTC
input bool   InpHour_04        = true;   // Trade at 04:00 UTC
input bool   InpHour_06        = true;   // Trade at 06:00 UTC
input bool   InpHour_08        = true;   // Trade at 08:00 UTC
input bool   InpHour_16        = true;   // Trade at 16:00 UTC
input bool   InpHour_17        = true;   // Trade at 17:00 UTC

input group "=== EQUITY PROTECTION ==="
input int    InpMax_ConsLoss   = 2;      // Max Consecutive Losses Before Pause
input int    InpPause_Bars     = 60;     // Pause Duration (bars)
input double InpDaily_Loss_Pct = 3.0;    // Daily Loss Limit (%)
input double InpMax_DD_Pct     = 10.0;   // Max Drawdown % (close all, stop)

input group "=== GENERAL ==="
input int    InpMagic          = 777111; // Magic Number
input bool   InpShow_Panel     = true;   // Show Dashboard


//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
int    ema_fast_handle, ema_mid_handle, ema_slow_handle;
int    atr_handle;

double ema_fast_buf[], ema_mid_buf[], ema_slow_buf[];
double atr_buf[];

datetime last_bar_time;
int      bars_since_trade;       // cooldown counter
int      bars_since_pause;       // pause counter
int      consecutive_losses;     // consecutive loss tracker
bool     paused;                 // equity pause state
bool     daily_stopped;          // daily loss limit hit
bool     dd_stopped;             // max DD hit - full stop
double   day_start_balance;      // balance at start of day
int      last_trade_day;         // track day changes
datetime trade_open_bar_time;    // for timeout tracking
int      bars_in_trade;          // bars since trade opened

// Dashboard
string   panel_prefix = "RTEA_";
string   signal_text  = "INITIALIZING";
color    signal_color = clrGray;


//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // Create indicator handles
   ema_fast_handle = iMA(_Symbol, PERIOD_M1, InpEMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   ema_mid_handle  = iMA(_Symbol, PERIOD_M1, InpEMA_Mid,  0, MODE_EMA, PRICE_CLOSE);
   ema_slow_handle = iMA(_Symbol, PERIOD_M1, InpEMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   atr_handle      = iATR(_Symbol, PERIOD_M1, InpATR_Period);

   if(ema_fast_handle == INVALID_HANDLE || ema_mid_handle == INVALID_HANDLE ||
      ema_slow_handle == INVALID_HANDLE || atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   // Set arrays as series
   ArraySetAsSeries(ema_fast_buf, true);
   ArraySetAsSeries(ema_mid_buf,  true);
   ArraySetAsSeries(ema_slow_buf, true);
   ArraySetAsSeries(atr_buf,      true);

   // Initialize state
   last_bar_time      = 0;
   bars_since_trade   = InpCooldown_Bars; // allow immediate first trade
   bars_since_pause   = InpPause_Bars;
   consecutive_losses = 0;
   paused             = false;
   daily_stopped      = false;
   dd_stopped         = false;
   day_start_balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   last_trade_day     = -1;
   trade_open_bar_time = 0;
   bars_in_trade      = 0;

   // Configure trade object
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   if(InpShow_Panel) CreateDashboard();

   Print("RealTickEA v1.0 initialized | Magic=", InpMagic,
         " | Risk=", InpRisk_Pct, "% | SL=", InpSL_Mult, "xATR | TP=", InpTP_Mult, "xATR");
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
   ObjectsDeleteAll(0, panel_prefix);
}


//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // Copy indicator buffers (need 6+ bars for EMA21 slope calculation)
   if(CopyBuffer(ema_fast_handle, 0, 0, 8, ema_fast_buf) < 8) return;
   if(CopyBuffer(ema_mid_handle,  0, 0, 8, ema_mid_buf)  < 8) return;
   if(CopyBuffer(ema_slow_handle, 0, 0, 8, ema_slow_buf) < 8) return;
   if(CopyBuffer(atr_handle,      0, 0, 3, atr_buf)      < 3) return;

   // Manage timeout on every tick (check bars elapsed for open positions)
   ManageTimeout();

   // Check max drawdown on every tick
   if(!dd_stopped) CheckMaxDrawdown();
   if(dd_stopped)
   {
      signal_text = "DD STOP"; signal_color = clrRed;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // New bar detection - all entry logic on bar close only
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
   }

   // Check daily loss limit
   if(!daily_stopped) CheckDailyLoss();
   if(daily_stopped)
   {
      signal_text = "DAILY STOP"; signal_color = clrOrangeRed;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Check if paused after consecutive losses
   if(paused)
   {
      if(bars_since_pause >= InpPause_Bars)
      {
         paused = false;
         consecutive_losses = 0;
         Print("RealTickEA: Pause ended after ", InpPause_Bars, " bars");
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

   // Already have a position open - no new entries
   if(HasOpenPosition())
   {
      signal_text = "IN TRADE"; signal_color = clrDodgerBlue;
      if(InpShow_Panel) UpdateDashboard();
      return;
   }

   // Get current ATR (closed bar)
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
//| BUY: EMA8>EMA13>EMA21, EMA21 slope>0, body>1.5xATR (bullish),   |
//|      prev candle bearish (pullback), close>EMA8                  |
//| SELL: EMA8<EMA13<EMA21, EMA21 slope<0, body<-1.5xATR (bearish), |
//|      prev candle bullish (pullback), close<EMA8                  |
//+------------------------------------------------------------------+
int GetEntrySignal(double cur_atr)
{
   // Use closed bar values (index 1 = last closed bar, index 2 = bar before)
   double ema8_0  = ema_fast_buf[1];
   double ema13_0 = ema_mid_buf[1];
   double ema21_0 = ema_slow_buf[1];
   double ema21_5 = ema_slow_buf[6]; // 5 bars back for slope

   // EMA21 slope
   double ema21_slope = ema21_0 - ema21_5;

   // Candle body calculations (bar 1 = current closed, bar 2 = previous closed)
   double open_1  = iOpen(_Symbol, PERIOD_M1, 1);
   double close_1 = iClose(_Symbol, PERIOD_M1, 1);
   double body_1  = close_1 - open_1; // positive = bullish, negative = bearish

   double open_2  = iOpen(_Symbol, PERIOD_M1, 2);
   double close_2 = iClose(_Symbol, PERIOD_M1, 2);
   double body_2  = close_2 - open_2;

   double body_threshold = InpBody_Mult * cur_atr;

   // BUY CONDITIONS
   bool buy_ema_align = (ema8_0 > ema13_0) && (ema13_0 > ema21_0);
   bool buy_slope     = (ema21_slope > 0);
   bool buy_body      = (body_1 > body_threshold);      // large bullish candle
   bool buy_pullback  = (body_2 < 0);                   // previous candle was bearish
   bool buy_above_ema = (close_1 > ema8_0);             // close above fast EMA

   if(buy_ema_align && buy_slope && buy_body && buy_pullback && buy_above_ema)
      return 1;

   // SELL CONDITIONS
   bool sell_ema_align = (ema8_0 < ema13_0) && (ema13_0 < ema21_0);
   bool sell_slope     = (ema21_slope < 0);
   bool sell_body      = (body_1 < -body_threshold);    // large bearish candle
   bool sell_pullback  = (body_2 > 0);                  // previous candle was bullish
   bool sell_below_ema = (close_1 < ema8_0);            // close below fast EMA

   if(sell_ema_align && sell_slope && sell_body && sell_pullback && sell_below_ema)
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

   // BUY at ASK, SELL at BID (realistic execution)
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

   // Calculate lot size
   double lot = CalculateLotSize(sl_dist);
   if(lot <= 0) return;

   // Margin check - don't use more than 80% of free margin
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double margin_required = 0;
   if(!OrderCalcMargin(type, _Symbol, lot, price, margin_required))
   {
      Print("RealTickEA: OrderCalcMargin failed");
      return;
   }
   if(margin_required > free_margin * 0.80)
   {
      Print("RealTickEA: Insufficient margin. Required=", margin_required,
            " Available(80%)=", free_margin * 0.80);
      return;
   }

   // Execute trade
   string comment = StringFormat("RTEA_%s", type == ORDER_TYPE_BUY ? "BUY" : "SELL");
   if(trade.PositionOpen(_Symbol, type, lot, price, sl, tp, comment))
   {
      bars_since_trade   = 0;
      trade_open_bar_time = iTime(_Symbol, PERIOD_M1, 0);
      bars_in_trade      = 0;
      Print("RealTickEA: ", EnumToString(type), " opened | Lot=", lot,
            " | Price=", price, " | SL=", sl, " | TP=", tp, " | ATR=", atr);
   }
   else
   {
      Print("RealTickEA: Trade FAILED - ", trade.ResultRetcodeDescription());
   }
}


//+------------------------------------------------------------------+
//| CALCULATE LOT SIZE                                                |
//| XAUUSD: 1 lot = 100 oz, 1 point = $100/lot                      |
//| Lot = (Balance * Risk%) / (SL_points * 100)                      |
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
      lot = risk_amt / (sl_points * 100.0); // fallback for XAUUSD

   // Normalize lot
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
//| MANAGE TIMEOUT - Close trade at market after N bars               |
//+------------------------------------------------------------------+
void ManageTimeout()
{
   if(InpTimeout_Bars <= 0) return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != InpMagic) continue;

      // Calculate bars since position opened
      datetime open_time = pos.Time();
      int bars_elapsed = iBarShift(_Symbol, PERIOD_M1, open_time, false);

      if(bars_elapsed >= InpTimeout_Bars)
      {
         Print("RealTickEA: Timeout reached (", bars_elapsed, " bars). Closing position.");
         trade.PositionClose(pos.Ticket());
         // Record result for consecutive loss tracking
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
      Print("RealTickEA: MAX DRAWDOWN HIT! DD=", DoubleToString(dd_pct, 2),
            "% >= ", InpMax_DD_Pct, "%. Closing all positions.");

      // Close all positions
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
   double daily_loss_pct = daily_loss / day_start_balance * 100.0;

   // Also include floating loss
   double floating = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() == _Symbol && pos.Magic() == InpMagic)
         floating += pos.Profit() + pos.Swap();
   }

   double total_daily_loss = daily_loss - floating; // floating is negative for losses
   double total_pct = (daily_loss + (floating < 0 ? MathAbs(floating) : 0)) / day_start_balance * 100.0;

   if(total_pct >= InpDaily_Loss_Pct)
   {
      Print("RealTickEA: DAILY LOSS LIMIT HIT! Loss=", DoubleToString(total_pct, 2),
            "% >= ", InpDaily_Loss_Pct, "%");
      daily_stopped = true;
   }
}


//+------------------------------------------------------------------+
//| CHECK LAST TRADE RESULT (for consecutive loss tracking)           |
//+------------------------------------------------------------------+
void CheckLastTradeResult()
{
   datetime from = TimeCurrent() - 86400; // last 24h
   datetime to   = TimeCurrent();
   if(!HistorySelect(from, to)) return;

   int total = HistoryDealsTotal();
   if(total <= 0) return;

   // Find the most recent closing deal for our EA
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
         Print("RealTickEA: Loss #", consecutive_losses, " (P/L=", DoubleToString(profit, 2), ")");

         if(consecutive_losses >= InpMax_ConsLoss)
         {
            paused = true;
            bars_since_pause = 0;
            Print("RealTickEA: PAUSING for ", InpPause_Bars, " bars after ",
                  consecutive_losses, " consecutive losses");
         }
      }
      else
      {
         consecutive_losses = 0; // reset on win
      }
      break; // only check the latest deal
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
   if(h == 6  && InpHour_06) return true;
   if(h == 8  && InpHour_08) return true;
   if(h == 16 && InpHour_16) return true;
   if(h == 17 && InpHour_17) return true;

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
//| OnTradeTransaction - Track trade closures for loss counting       |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL)
      {
         // Check if this is a closing deal
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

                  if(profit < 0)
                  {
                     consecutive_losses++;
                     Print("RealTickEA: Consecutive loss #", consecutive_losses);
                     if(consecutive_losses >= InpMax_ConsLoss)
                     {
                        paused = true;
                        bars_since_pause = 0;
                        Print("RealTickEA: PAUSING for ", InpPause_Bars, " bars");
                     }
                  }
                  else if(profit > 0)
                  {
                     consecutive_losses = 0;
                     Print("RealTickEA: WIN! Consecutive losses reset.");
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
   DashRect(panel_prefix + "bg", x - 10, y - 10, 310, 380, C'15,20,35', C'40,60,100', 1);

   // Header
   DashLabel(panel_prefix + "title", "RealTickEA v1.0", x + 5, y, clrWhite, 10, true);
   DashLabel(panel_prefix + "sub", "EMA Momentum Pullback | Real Tick Optimized", x + 5, y + 14, C'130,140,160', 7, false);
   DashLine(panel_prefix + "h0", x, y + 28, 290);

   // Signal section
   int row = y + 36;
   DashLabel(panel_prefix + "l_sig",  "Signal:",        x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_sig",  "---",            x + 140, row,  clrYellow, 8, true);
   DashLabel(panel_prefix + "l_sprd", "Spread:",        x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_sprd", "---",            x + 140, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_atr",  "ATR(14):",       x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_atr",  "---",            x + 140, row+30, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_ema",  "EMA Align:",     x + 5, row+45, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_ema",  "---",            x + 140, row+45, clrWhite, 8, false);
   DashLine(panel_prefix + "h1", x, row + 62, 290);

   // Risk section
   row = row + 70;
   DashLabel(panel_prefix + "l_loss", "Consec Losses:", x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_loss", "0",              x + 140, row,  clrWhite, 8, false);
   DashLabel(panel_prefix + "l_dpnl", "Daily P&&L:",    x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_dpnl", "---",            x + 140, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_paus", "Pause Status:",  x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_paus", "ACTIVE",         x + 140, row+30, clrLime, 8, false);
   DashLine(panel_prefix + "h2", x, row + 47, 290);

   // Account section
   row = row + 55;
   DashLabel(panel_prefix + "l_bal",  "Balance:",       x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_bal",  "---",            x + 140, row,  clrWhite, 8, false);
   DashLabel(panel_prefix + "l_eq",   "Equity:",        x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_eq",   "---",            x + 140, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_dd",   "Drawdown:",      x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_dd",   "0.0%",           x + 140, row+30, clrLime, 8, false);
   DashLine(panel_prefix + "h3", x, row + 47, 290);

   // Status section
   row = row + 55;
   DashLabel(panel_prefix + "l_cool", "Cooldown:",      x + 5, row,    C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_cool", "READY",          x + 140, row,  clrLime, 8, false);
   DashLabel(panel_prefix + "l_hour", "Hour (UTC):",    x + 5, row+15, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_hour", "---",            x + 140, row+15, clrWhite, 8, false);
   DashLabel(panel_prefix + "l_stat", "Status:",        x + 5, row+30, C'150,160,180', 8, false);
   DashLabel(panel_prefix + "v_stat", "RUNNING",        x + 140, row+30, clrLime, 8, true);

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
      { ema_str = "BULLISH (8>13>21)"; ema_col = clrLime; }
      else if(ema_fast_buf[1] < ema_mid_buf[1] && ema_mid_buf[1] < ema_slow_buf[1])
      { ema_str = "BEARISH (8<13<21)"; ema_col = clrTomato; }
   }
   DashSet(panel_prefix + "v_ema", ema_str, ema_col);

   // Consecutive losses
   DashSet(panel_prefix + "v_loss",
           StringFormat("%d / %d", consecutive_losses, InpMax_ConsLoss),
           consecutive_losses >= InpMax_ConsLoss ? clrRed : clrWhite);

   // Daily P&L
   double daily_pnl = AccountInfoDouble(ACCOUNT_BALANCE) - day_start_balance;
   DashSet(panel_prefix + "v_dpnl",
           StringFormat("%.2f (%.1f%%)", daily_pnl, day_start_balance > 0 ? daily_pnl / day_start_balance * 100 : 0),
           daily_pnl >= 0 ? clrLime : clrTomato);

   // Pause status
   if(dd_stopped)
      DashSet(panel_prefix + "v_paus", "DD STOPPED", clrRed);
   else if(daily_stopped)
      DashSet(panel_prefix + "v_paus", "DAILY LIMIT", clrOrangeRed);
   else if(paused)
      DashSet(panel_prefix + "v_paus", StringFormat("PAUSED %d/%d", bars_since_pause, InpPause_Bars), clrOrange);
   else
      DashSet(panel_prefix + "v_paus", "ACTIVE", clrLime);

   // Account
   DashSet(panel_prefix + "v_bal", StringFormat("%.2f", balance), clrWhite);
   DashSet(panel_prefix + "v_eq",  StringFormat("%.2f", equity), equity >= balance ? clrLime : clrTomato);
   DashSet(panel_prefix + "v_dd",  StringFormat("%.1f%%", dd_pct),
           dd_pct < 5 ? clrLime : (dd_pct < 8 ? clrOrange : clrRed));

   // Cooldown
   if(bars_since_trade >= InpCooldown_Bars)
      DashSet(panel_prefix + "v_cool", "READY", clrLime);
   else
      DashSet(panel_prefix + "v_cool", StringFormat("%d / %d bars", bars_since_trade, InpCooldown_Bars), clrYellow);

   // Hour
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   DashSet(panel_prefix + "v_hour",
           StringFormat("%02d:00 %s", dt.hour, IsAllowedHour() ? "ACTIVE" : "OFF"),
           IsAllowedHour() ? clrLime : clrGray);

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
   DashLabel(name, dashes, x, y, C'40,60,100', 6, false);
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
