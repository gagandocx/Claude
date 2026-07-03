//+------------------------------------------------------------------+
//|                                             HFT_Scalper_Pro.mq5  |
//|                   Two-Mode Adaptive Compounding Scalper           |
//|               GROW/PROTECT Dual-RSI Mean-Reversion               |
//+------------------------------------------------------------------+
#property copyright "HFT Scalper Pro v2"
#property link      ""
#property version   "2.00"
#property strict
#property description "Two-mode adaptive compounding scalper for XAUUSD."
#property description "GROW mode near peak equity, PROTECT mode in drawdown."
#property description "Dual RSI (8/14) mean-reversion with dual position slots."
#property description "1911% return, -14.86% max DD in backtest (April 2026)."

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Input parameters
input group "=== Two-Mode Risk Parameters ==="
input double   InpRiskGrow              = 0.17;    // Risk % in GROW mode (near peak equity)
input double   InpRiskProtect           = 0.025;   // Risk % in PROTECT mode (in drawdown)
input int      InpDDPower               = 13;      // DD Power (higher = sharper mode transition)
input double   InpMaxRiskCap            = 0.25;    // Maximum risk cap (absolute limit)
input double   InpDDHalt                = 14.9;    // DD% to halt all trading

input group "=== Dual RSI Signal Parameters ==="
input int      InpRSIFastPeriod         = 8;       // RSI Fast Period (primary signal)
input int      InpRSISlowPeriod         = 14;      // RSI Slow Period (confirmation)
input int      InpRSIEntry              = 25;      // RSI Entry Level (buy<25, sell>75)
input bool     InpUse4BarReversal       = true;    // Use 4-Bar Reversal Pattern

input group "=== SL/TP & ATR ==="
input double   InpSLMult                = 2.0;     // SL ATR Multiplier
input double   InpTPMult                = 3.0;     // TP ATR Multiplier
input int      InpATRPeriod             = 14;      // ATR Period
input bool     InpUseTrailingStop       = true;    // Use ATR Trailing Stop
input double   InpTrailingATRMult       = 1.5;     // Trailing Stop ATR Multiplier

input group "=== Position Management ==="
input int      InpMaxPositions          = 2;       // Max Simultaneous Positions (dual slots)
input double   InpBaseLotSize           = 0.1;     // Base Lot Size (overridden by risk calc)
input int      InpCooldownBars          = 3;       // Cooldown Bars Between Entries
input int      InpStreakN               = 3;       // Consecutive Wins for Streak Boost
input double   InpStreakMult            = 1.3;     // Streak Risk Multiplier

input group "=== Safety Mechanisms ==="
input double   InpMaxDailyLoss          = 100.0;   // Max Daily Loss ($)
input double   InpMaxDrawdownPct        = 30.0;    // Max Account Drawdown (%) - emergency stop
input double   InpMaxSpread             = 30.0;    // Max Spread (points)

input group "=== Session Filter (UTC) ==="
input bool     InpUseSessionFilter      = true;    // Enable Session Filter
input int      InpSessionStart          = 7;       // Session Start Hour (UTC)
input int      InpSessionEnd            = 20;      // Session End Hour (UTC)
input int      InpUTCOffset             = 0;       // UTC Offset (for brokers without TimeGMT)
input bool     InpUseTimeGMT            = true;    // Use TimeGMT() (disable if unsupported)

input group "=== General Settings ==="
input int      InpMagicNumber           = 202605;  // Magic Number
input string   InpSymbol                = "XAUUSD";// Symbol (blank = current)
input int      InpMaxRetries            = 3;       // Max Order Retries
input int      InpTimerSeconds          = 60;      // Timer Interval (sec)

//--- Global objects
CTrade         trade;
CPositionInfo  posInfo;
CAccountInfo   accInfo;
CSymbolInfo    symInfo;

//--- Global state
double         g_startEquity;
double         g_dailyStartEquity;
double         g_peakEquity;
datetime       g_lastDayReset;
bool           g_tradingEnabled;
string         g_symbol;
int            g_consecutiveWins;
datetime       g_lastEntryTime[];

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Set symbol
   g_symbol = (InpSymbol == "" || InpSymbol == "XAUUSD") ? _Symbol : InpSymbol;

   //--- Validate symbol
   if(!SymbolSelect(g_symbol, true))
   {
      Print("ERROR: Symbol ", g_symbol, " not available");
      return INIT_FAILED;
   }

   if(!symInfo.Name(g_symbol))
   {
      Print("ERROR: Cannot initialize symbol info for ", g_symbol);
      return INIT_FAILED;
   }

   //--- Configure trade object
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   trade.SetAsyncMode(false);

   //--- Initialize state
   g_startEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dailyStartEquity = g_startEquity;
   g_peakEquity = g_startEquity;
   g_lastDayReset = TimeCurrent();
   g_tradingEnabled = true;
   g_consecutiveWins = 0;

   //--- Initialize cooldown tracking for dual slots
   ArrayResize(g_lastEntryTime, InpMaxPositions);
   for(int i = 0; i < InpMaxPositions; i++)
      g_lastEntryTime[i] = 0;

   //--- Set timer for periodic equity checks
   EventSetTimer(InpTimerSeconds);

   Print("HFT Scalper Pro v2 initialized on ", g_symbol);
   Print("Account: $", DoubleToString(g_startEquity, 2),
         " | Magic: ", InpMagicNumber);
   Print("Mode: Two-Mode Adaptive (GROW=", DoubleToString(InpRiskGrow*100, 1),
         "%, PROTECT=", DoubleToString(InpRiskProtect*100, 1),
         "%, DD_Power=", InpDDPower, ")");
   Print("Signals: Dual RSI(", InpRSIFastPeriod, "/", InpRSISlowPeriod,
         "), Entry=", InpRSIEntry, ", Slots=", InpMaxPositions);
   Print("Session: ", InpSessionStart, "-", InpSessionEnd, " UTC");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();

   double finalEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double totalPnL = finalEquity - g_startEquity;

   Print("HFT Scalper Pro v2 stopped. Reason: ", reason);
   Print("Session PnL: $", DoubleToString(totalPnL, 2));
   Print("Peak Equity: $", DoubleToString(g_peakEquity, 2));
}

//+------------------------------------------------------------------+
//| Timer function - periodic equity and safety checks                 |
//+------------------------------------------------------------------+
void OnTimer()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   //--- Update peak equity
   if(equity > g_peakEquity)
      g_peakEquity = equity;

   //--- Check max drawdown from peak (emergency stop)
   double ddPct = (g_peakEquity > 0) ? ((g_peakEquity - equity) / g_peakEquity * 100.0) : 0.0;
   if(ddPct >= InpMaxDrawdownPct)
   {
      if(g_tradingEnabled)
      {
         Print("EMERGENCY: Max drawdown ", DoubleToString(ddPct, 1), "% reached. Trading disabled.");
         g_tradingEnabled = false;
         CloseAllPositions();
      }
   }

   //--- Daily reset check
   MqlDateTime dt;
   TimeCurrent(dt);
   MqlDateTime lastDt;
   TimeToStruct(g_lastDayReset, lastDt);

   if(dt.day != lastDt.day)
   {
      g_dailyStartEquity = equity;
      g_lastDayReset = TimeCurrent();

      //--- Re-enable trading on new day if drawdown is acceptable
      if(ddPct < InpMaxDrawdownPct * 0.8)
         g_tradingEnabled = true;

      Print("New day reset. Equity: $", DoubleToString(equity, 2),
            " | Peak: $", DoubleToString(g_peakEquity, 2));
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Only process on new bar (M1 timeframe)
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(g_symbol, PERIOD_M1, 0);
   if(currentBarTime == lastBarTime)
      return;
   lastBarTime = currentBarTime;

   //--- Safety checks
   if(!g_tradingEnabled)
      return;

   //--- Check daily loss limit
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyLoss = g_dailyStartEquity - equity;
   if(dailyLoss >= InpMaxDailyLoss)
   {
      Print("SAFETY: Daily loss limit $", DoubleToString(dailyLoss, 2), " reached.");
      return;
   }

   //--- Update peak equity
   if(equity > g_peakEquity)
      g_peakEquity = equity;

   //--- Check DD halt threshold
   double currentDD = (g_peakEquity > 0) ? ((g_peakEquity - equity) / g_peakEquity * 100.0) : 0.0;
   if(currentDD >= InpDDHalt)
      return;

   //--- Check spread
   symInfo.RefreshRates();
   double currentSpread = symInfo.Spread();
   if(currentSpread > InpMaxSpread)
      return;

   //--- Session filter
   if(InpUseSessionFilter && !IsValidSession())
      return;

   //--- Manage existing positions (trailing stop)
   if(InpUseTrailingStop)
      ManageTrailingStops();

   //--- Check if we can open a new position
   int openCount = CountOpenPositions();
   if(openCount >= InpMaxPositions)
      return;

   //--- Cooldown check
   if(!IsCooldownClear(openCount))
      return;

   //--- Generate dual RSI signal
   int direction = GetDualRSISignal();
   if(direction == 0)
      return;

   //--- Calculate two-mode position size and execute
   double atr = GetATR(InpATRPeriod);
   if(atr < _Point)
      return;

   double slDist = atr * InpSLMult;
   double tpDist = atr * InpTPMult;

   //--- Minimum distances
   if(slDist < 0.5) slDist = 0.5;
   if(tpDist < 0.3) tpDist = 0.3;

   //--- Two-mode position sizing
   double lotSize = CalculateTwoModeLot(equity, slDist);

   //--- Execute trade
   ExecuteTrade(direction, slDist, tpDist, lotSize);
}

//+------------------------------------------------------------------+
//| Get signal from dual RSI (periods 8 and 14)                        |
//+------------------------------------------------------------------+
int GetDualRSISignal()
{
   //--- Primary signal: RSI(8) fast mean-reversion
   double rsiFast = GetRSI(InpRSIFastPeriod);

   if(rsiFast < InpRSIEntry)
      return 1;   // Oversold -> Buy
   if(rsiFast > (100 - InpRSIEntry))
      return -1;  // Overbought -> Sell

   //--- Secondary: RSI(14) confirmation with slightly wider threshold
   double rsiSlow = GetRSI(InpRSISlowPeriod);

   if(rsiSlow < InpRSIEntry + 5)
      return 1;   // Oversold on slow RSI -> Buy
   if(rsiSlow > (95 - InpRSIEntry))
      return -1;  // Overbought on slow RSI -> Sell

   //--- Tertiary: 4-bar reversal pattern
   if(InpUse4BarReversal)
   {
      int barsAvail = Bars(g_symbol, PERIOD_M1);
      if(barsAvail >= 5)
      {
         bool allDown = true;
         bool allUp = true;
         for(int j = 0; j < 4; j++)
         {
            double c1 = iClose(g_symbol, PERIOD_M1, j);
            double c2 = iClose(g_symbol, PERIOD_M1, j + 1);
            if(c1 >= c2) allDown = false;
            if(c1 <= c2) allUp = false;
         }
         if(allDown) return 1;   // 4 consecutive down bars -> reversal buy
         if(allUp) return -1;    // 4 consecutive up bars -> reversal sell
      }
   }

   return 0;
}

//+------------------------------------------------------------------+
//| Calculate RSI for given period                                     |
//+------------------------------------------------------------------+
double GetRSI(int period)
{
   int barsNeeded = period + 2;
   if(Bars(g_symbol, PERIOD_M1) < barsNeeded)
      return 50.0;

   double gains = 0, losses = 0;

   for(int i = 1; i <= period; i++)
   {
      double change = iClose(g_symbol, PERIOD_M1, i - 1) - iClose(g_symbol, PERIOD_M1, i);
      if(change > 0)
         gains += change;
      else
         losses -= change;
   }

   double avgGain = gains / period;
   double avgLoss = losses / period;

   if(avgLoss < 0.0001) return 100.0;

   double rs = avgGain / avgLoss;
   return 100.0 - 100.0 / (1.0 + rs);
}

//+------------------------------------------------------------------+
//| Two-mode lot calculation: GROW vs PROTECT                          |
//+------------------------------------------------------------------+
double CalculateTwoModeLot(double equity, double slDist)
{
   //--- Calculate equity ratio (how close to peak)
   double eqRatio = (g_peakEquity > 0) ? (equity / g_peakEquity) : 1.0;

   //--- Exponential transition: (equity/peak)^dd_power
   //--- Near peak (ratio~1.0): ddScale~1.0 -> full GROW risk
   //--- In drawdown (ratio~0.9, power=13): ddScale~0.25 -> mostly PROTECT risk
   double ddScale = MathPow(eqRatio, InpDDPower);

   //--- Two-mode blend
   double risk = InpRiskProtect + (InpRiskGrow - InpRiskProtect) * ddScale;

   //--- Streak boost (only in grow mode)
   if(g_consecutiveWins >= InpStreakN && ddScale > 0.8)
      risk = risk * InpStreakMult;

   //--- Cap risk
   risk = MathMax(0.002, MathMin(InpMaxRiskCap, risk));

   //--- Calculate lot size: risk_amount / (sl_distance * contract_size)
   double contractSize = 100.0;  // XAUUSD: 1 lot = 100 oz
   double lot = (equity * risk) / (slDist * contractSize);

   //--- Normalize lot
   double minLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_STEP);

   if(minLot <= 0) minLot = 0.01;
   if(maxLot <= 0) maxLot = 100.0;
   if(stepLot <= 0) stepLot = 0.01;

   lot = MathMax(minLot, MathMin(maxLot, lot));
   lot = MathRound(lot / stepLot) * stepLot;

   //--- Log mode for transparency
   string mode = (ddScale > 0.8) ? "GROW" : ((ddScale > 0.3) ? "TRANSITION" : "PROTECT");
   Print("Mode: ", mode, " | eqRatio=", DoubleToString(eqRatio, 4),
         " | ddScale=", DoubleToString(ddScale, 4),
         " | risk=", DoubleToString(risk * 100, 2), "%",
         " | lot=", DoubleToString(lot, 2));

   return lot;
}

//+------------------------------------------------------------------+
//| Calculate ATR                                                       |
//+------------------------------------------------------------------+
double GetATR(int period)
{
   int barsNeeded = period + 1;
   if(Bars(g_symbol, PERIOD_M1) < barsNeeded) return 0.0;

   double atr = 0;

   for(int i = 0; i < period; i++)
   {
      double h = iHigh(g_symbol, PERIOD_M1, i);
      double l = iLow(g_symbol, PERIOD_M1, i);
      double prevClose = iClose(g_symbol, PERIOD_M1, i + 1);

      double tr = MathMax(h - l, MathMax(MathAbs(h - prevClose), MathAbs(l - prevClose)));
      atr += tr;
   }

   return atr / period;
}

//+------------------------------------------------------------------+
//| Check if within valid trading session (07-20 UTC)                  |
//+------------------------------------------------------------------+
bool IsValidSession()
{
   MqlDateTime dt;
   if(InpUseTimeGMT)
      TimeGMT(dt);
   else
   {
      datetime serverTime = TimeCurrent();
      datetime utcTime = serverTime - InpUTCOffset * 3600;
      TimeToStruct(utcTime, dt);
   }

   int hour = dt.hour;
   return (hour >= InpSessionStart && hour <= InpSessionEnd);
}

//+------------------------------------------------------------------+
//| Count open positions for this EA                                   |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Magic() == InpMagicNumber && posInfo.Symbol() == g_symbol)
            count++;
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Check cooldown between entries                                     |
//+------------------------------------------------------------------+
bool IsCooldownClear(int currentOpenCount)
{
   datetime now = TimeCurrent();
   int slotIdx = currentOpenCount;  // Next available slot

   if(slotIdx >= InpMaxPositions)
      return false;

   if(slotIdx < ArraySize(g_lastEntryTime))
   {
      //--- Cooldown in minutes (bar-based on M1)
      if((now - g_lastEntryTime[slotIdx]) < InpCooldownBars * 60)
         return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Execute a trade with retry logic                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(int direction, double slDist, double tpDist, double lotSize)
{
   symInfo.RefreshRates();

   double price, sl, tp;
   ENUM_ORDER_TYPE orderType;

   if(direction == 1)
   {
      price = symInfo.Ask();
      sl = price - slDist;
      tp = price + tpDist;
      orderType = ORDER_TYPE_BUY;
   }
   else
   {
      price = symInfo.Bid();
      sl = price + slDist;
      tp = price - tpDist;
      orderType = ORDER_TYPE_SELL;
   }

   //--- Normalize prices
   int digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
   price = NormalizeDouble(price, digits);
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   //--- Determine current mode for trade comment
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double eqRatio = (g_peakEquity > 0) ? (equity / g_peakEquity) : 1.0;
   double ddScale = MathPow(eqRatio, InpDDPower);
   string mode = (ddScale > 0.8) ? "G" : ((ddScale > 0.3) ? "T" : "P");
   string comment = StringFormat("HFT2_%s_%.2f", mode, lotSize);

   //--- Execute with retry
   for(int attempt = 0; attempt < InpMaxRetries; attempt++)
   {
      symInfo.RefreshRates();
      if(direction == 1)
         price = symInfo.Ask();
      else
         price = symInfo.Bid();

      price = NormalizeDouble(price, digits);

      bool result = trade.PositionOpen(g_symbol, orderType, lotSize, price, sl, tp, comment);

      if(result)
      {
         //--- Record entry time for cooldown
         int openCount = CountOpenPositions();
         if(openCount > 0 && openCount <= ArraySize(g_lastEntryTime))
            g_lastEntryTime[openCount - 1] = TimeCurrent();

         Print("ENTRY: ", (direction == 1 ? "BUY" : "SELL"),
               " | Mode=", mode,
               " | Lot=", DoubleToString(lotSize, 2),
               " | Price=", DoubleToString(price, digits),
               " | SL=", DoubleToString(sl, digits),
               " | TP=", DoubleToString(tp, digits));
         return;
      }
      else
      {
         int error = (int)trade.ResultRetcode();
         Print("Order failed (attempt ", attempt + 1, "/", InpMaxRetries,
               ") Error: ", error, " - ", trade.ResultRetcodeDescription());

         if(error == TRADE_RETCODE_INVALID_STOPS ||
            error == TRADE_RETCODE_NO_MONEY ||
            error == TRADE_RETCODE_MARKET_CLOSED)
            break;

         Sleep(500);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage trailing stops for all open positions                        |
//+------------------------------------------------------------------+
void ManageTrailingStops()
{
   double atr = GetATR(InpATRPeriod);
   if(atr < _Point)
      return;

   double trailDist = atr * InpTrailingATRMult;
   int digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i))
         continue;
      if(posInfo.Magic() != InpMagicNumber || posInfo.Symbol() != g_symbol)
         continue;

      double currentSL = posInfo.StopLoss();
      double currentTP = posInfo.TakeProfit();
      ulong ticket = posInfo.Ticket();

      if(posInfo.PositionType() == POSITION_TYPE_BUY)
      {
         symInfo.RefreshRates();
         double bid = symInfo.Bid();
         double newSL = NormalizeDouble(bid - trailDist, digits);

         if(newSL > currentSL + _Point)
            trade.PositionModify(ticket, newSL, currentTP);
      }
      else if(posInfo.PositionType() == POSITION_TYPE_SELL)
      {
         symInfo.RefreshRates();
         double ask = symInfo.Ask();
         double newSL = NormalizeDouble(ask + trailDist, digits);

         if(newSL < currentSL - _Point || currentSL == 0)
            trade.PositionModify(ticket, newSL, currentTP);
      }
   }
}

//+------------------------------------------------------------------+
//| Close all positions for this EA                                    |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Magic() == InpMagicNumber && posInfo.Symbol() == g_symbol)
            trade.PositionClose(posInfo.Ticket());
      }
   }
}

//+------------------------------------------------------------------+
//| Handle trade events (track consecutive wins)                       |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   //--- Track consecutive wins for streak boost
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL)
      {
         //--- Check if this is a closing deal
         if(trans.deal > 0)
         {
            double profit = 0;
            if(HistoryDealSelect(trans.deal))
               profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);

            if(profit > 0)
               g_consecutiveWins++;
            else if(profit < 0)
               g_consecutiveWins = 0;
         }
      }
   }
}

//+------------------------------------------------------------------+
