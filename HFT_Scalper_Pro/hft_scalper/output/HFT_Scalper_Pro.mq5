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

//--- Input parameters (defaults match Python best_params exactly)
input group "=== Two-Mode Risk Parameters ==="
input double   InpRiskGrow              = 0.17;    // Risk % in GROW mode (near peak equity)
input double   InpRiskProtect           = 0.025;   // Risk % in PROTECT mode (in drawdown)
input int      InpDDPower               = 13;      // DD Power (higher = sharper mode transition)
input double   InpMaxRiskCap            = 0.25;    // Maximum risk cap (absolute limit)
input double   InpDDHalt                = 14.9;    // DD% to halt all trading

input group "=== Dual RSI Signal Parameters ==="
input int      InpRSIFastPeriod         = 8;       // RSI Fast Period (primary signal)
input int      InpRSISlowPeriod         = 14;      // RSI Slow Period (secondary signal)
input int      InpRSIEntry              = 25;      // RSI Entry Level (fast: buy<25, sell>75)
input bool     InpUse4BarReversal       = true;    // Use 4-Bar Reversal Pattern

input group "=== SL/TP & ATR ==="
input double   InpSLMult                = 2.0;     // SL ATR Multiplier
input double   InpTPMult                = 3.0;     // TP ATR Multiplier
input int      InpATRPeriod             = 14;      // ATR Period

input group "=== Position Management ==="
input int      InpMaxPositions          = 2;       // Max Simultaneous Positions (dual slots)
input int      InpCooldownBars          = 3;       // Cooldown Bars Between Entries
input int      InpStreakN               = 3;       // Consecutive Wins for Streak Boost
input double   InpStreakMult            = 1.3;     // Streak Risk Multiplier

input group "=== Session Filter (UTC) ==="
input int      InpSessionStart          = 7;       // Session Start Hour (UTC)
input int      InpSessionEnd            = 20;      // Session End Hour (UTC)
input int      InpUTCOffset             = 0;       // Broker UTC Offset (hours, e.g. +2 or -5)

input group "=== General Settings ==="
input int      InpMagicNumber           = 202605;  // Magic Number
input int      InpMaxRetries            = 3;       // Max Order Retries

//--- Global objects
CTrade         trade;
CPositionInfo  posInfo;
CSymbolInfo    symInfo;

//--- Indicator handles (Wilder smoothing via MT5 built-in)
int            g_hRSIFast;       // iRSI handle for period 8
int            g_hRSISlow;       // iRSI handle for period 14
int            g_hATR;           // iATR handle for period 14

//--- Global state
string         g_symbol;
double         g_peakEquity;
int            g_consecutiveWins;
int            g_lastEntryBar[];  // Bar count at last entry per slot
int            g_barCount;        // Total bars processed since start
datetime       g_lastBarTime;     // For new-bar detection

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Set symbol (always use chart symbol)
   g_symbol = _Symbol;

   //--- Validate symbol
   if(!symInfo.Name(g_symbol))
   {
      Print("ERROR: Cannot initialize symbol info for ", g_symbol);
      return INIT_FAILED;
   }

   //--- Create indicator handles (MT5 iRSI uses Wilder smoothing internally)
   g_hRSIFast = iRSI(g_symbol, PERIOD_M1, InpRSIFastPeriod, PRICE_CLOSE);
   if(g_hRSIFast == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create RSI(", InpRSIFastPeriod, ") handle");
      return INIT_FAILED;
   }

   g_hRSISlow = iRSI(g_symbol, PERIOD_M1, InpRSISlowPeriod, PRICE_CLOSE);
   if(g_hRSISlow == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create RSI(", InpRSISlowPeriod, ") handle");
      return INIT_FAILED;
   }

   g_hATR = iATR(g_symbol, PERIOD_M1, InpATRPeriod);
   if(g_hATR == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create ATR(", InpATRPeriod, ") handle");
      return INIT_FAILED;
   }

   //--- Configure trade object
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   trade.SetAsyncMode(false);

   //--- Initialize state
   g_peakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_consecutiveWins = 0;
   g_barCount = 0;
   g_lastBarTime = 0;

   //--- Initialize cooldown tracking for dual slots
   ArrayResize(g_lastEntryBar, InpMaxPositions);
   for(int i = 0; i < InpMaxPositions; i++)
      g_lastEntryBar[i] = -InpCooldownBars - 1;  // Allow immediate first entry

   Print("HFT Scalper Pro v2 initialized on ", g_symbol);
   Print("Account: $", DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2),
         " | Magic: ", InpMagicNumber);
   Print("Mode: Two-Mode Adaptive (GROW=", DoubleToString(InpRiskGrow*100, 1),
         "%, PROTECT=", DoubleToString(InpRiskProtect*100, 1),
         "%, DD_Power=", InpDDPower, ")");
   Print("Signals: Dual RSI(", InpRSIFastPeriod, "/", InpRSISlowPeriod,
         "), Entry=", InpRSIEntry, ", 4Bar=", InpUse4BarReversal,
         ", Slots=", InpMaxPositions);
   Print("Session: ", InpSessionStart, "-", InpSessionEnd,
         " UTC (offset=", InpUTCOffset, ")");
   Print("SL=", DoubleToString(InpSLMult, 1), "xATR, TP=",
         DoubleToString(InpTPMult, 1), "xATR, Cooldown=", InpCooldownBars, " bars");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Release indicator handles
   if(g_hRSIFast != INVALID_HANDLE) IndicatorRelease(g_hRSIFast);
   if(g_hRSISlow != INVALID_HANDLE) IndicatorRelease(g_hRSISlow);
   if(g_hATR != INVALID_HANDLE)     IndicatorRelease(g_hATR);

   Print("HFT Scalper Pro v2 stopped. Reason: ", reason);
   Print("Peak Equity: $", DoubleToString(g_peakEquity, 2));
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Only process on new M1 bar (matches Python bar-by-bar processing)
   datetime currentBarTime = iTime(g_symbol, PERIOD_M1, 0);
   if(currentBarTime == g_lastBarTime)
      return;
   g_lastBarTime = currentBarTime;
   g_barCount++;

   //--- Get current equity
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   //--- Update peak equity
   if(equity > g_peakEquity)
      g_peakEquity = equity;

   //--- DD halt check (matches Python: if current_dd >= dd_halt: continue)
   double currentDD = 0.0;
   if(g_peakEquity > 0)
      currentDD = (g_peakEquity - equity) / g_peakEquity * 100.0;
   if(currentDD >= InpDDHalt)
      return;

   //--- Session filter (matches Python: hours[i] >= 7 and hours[i] <= 20)
   if(!IsValidSession())
      return;

   //--- Check ATR validity (matches Python: if atr[i] < 0.5: continue)
   double atrValue[1];
   if(CopyBuffer(g_hATR, 0, 1, 1, atrValue) != 1)
      return;
   double atr = atrValue[0];
   if(atr < 0.5)
      return;

   //--- Check if we can open a new position
   int openCount = CountOpenPositions();
   if(openCount >= InpMaxPositions)
      return;

   //--- Cooldown check (matches Python: (i - last_entry_bars[s]) >= cooldown)
   if(!IsCooldownClear(openCount))
      return;

   //--- Generate signal (matches Python: 3 independent sources in priority order)
   int direction = GetSignal();
   if(direction == 0)
      return;

   //--- Calculate SL/TP distances (matches Python exactly)
   double slDist = atr * InpSLMult;
   double tpDist = atr * InpTPMult;

   //--- Minimum distances (matches Python: if sl_dist < 0.5: sl_dist = 0.5)
   if(slDist < 0.5) slDist = 0.5;
   if(tpDist < 0.3) tpDist = 0.3;

   //--- Two-mode position sizing (matches Python formula exactly)
   double lotSize = CalculateTwoModeLot(equity, slDist);

   //--- Execute trade
   ExecuteTrade(direction, slDist, tpDist, lotSize, openCount);
}

//+------------------------------------------------------------------+
//| Generate signal from 3 independent sources (priority order)        |
//| Matches Python exactly:                                            |
//|   1. RSI(8) < 25 -> buy; RSI(8) > 75 -> sell                     |
//|   2. RSI(14) < 30 -> buy; RSI(14) > 70 -> sell                   |
//|   3. 4-bar reversal pattern                                        |
//+------------------------------------------------------------------+
int GetSignal()
{
   //--- Copy RSI values (bar index 1 = last completed bar)
   double rsiFastBuf[1], rsiSlowBuf[1];

   if(CopyBuffer(g_hRSIFast, 0, 1, 1, rsiFastBuf) != 1)
      return 0;
   if(CopyBuffer(g_hRSISlow, 0, 1, 1, rsiSlowBuf) != 1)
      return 0;

   double rsiFast = rsiFastBuf[0];
   double rsiSlow = rsiSlowBuf[0];

   //--- Primary signal: RSI(8) fast mean-reversion
   //--- Python: if rsi_fast[i] < rsi_entry: signal = 1
   //---         elif rsi_fast[i] > (100 - rsi_entry): signal = -1
   if(rsiFast < InpRSIEntry)
      return 1;   // Oversold -> Buy
   if(rsiFast > (100 - InpRSIEntry))
      return -1;  // Overbought -> Sell

   //--- Secondary: RSI(14) with shifted thresholds
   //--- Python: if rsi_slow[i] < rsi_entry + 5: signal = 1
   //---         elif rsi_slow[i] > (95 - rsi_entry): signal = -1
   if(rsiSlow < InpRSIEntry + 5)
      return 1;   // Oversold on slow RSI -> Buy
   if(rsiSlow > (95 - InpRSIEntry))
      return -1;  // Overbought on slow RSI -> Sell

   //--- Tertiary: 4-bar reversal pattern
   //--- Python: all_down = all(close[i-j] < close[i-j-1] for j in range(4))
   //--- In MQL5 bar indexing (0=current, 1=last completed):
   //---   Check bars 1,2,3,4,5 (completed bars)
   //---   all_down: bar1 < bar2, bar2 < bar3, bar3 < bar4, bar4 < bar5
   //---   all_up:   bar1 > bar2, bar2 > bar3, bar3 > bar4, bar4 > bar5
   if(InpUse4BarReversal)
   {
      if(Bars(g_symbol, PERIOD_M1) >= 6)
      {
         double c1 = iClose(g_symbol, PERIOD_M1, 1);
         double c2 = iClose(g_symbol, PERIOD_M1, 2);
         double c3 = iClose(g_symbol, PERIOD_M1, 3);
         double c4 = iClose(g_symbol, PERIOD_M1, 4);
         double c5 = iClose(g_symbol, PERIOD_M1, 5);

         bool allDown = (c1 < c2) && (c2 < c3) && (c3 < c4) && (c4 < c5);
         bool allUp   = (c1 > c2) && (c2 > c3) && (c3 > c4) && (c4 > c5);

         if(allDown) return 1;   // 4 consecutive down closes -> reversal buy
         if(allUp)   return -1;  // 4 consecutive up closes -> reversal sell
      }
   }

   return 0;
}

//+------------------------------------------------------------------+
//| Two-mode lot calculation matching Python exactly                    |
//| Python formula:                                                     |
//|   eq_ratio = equity / peak_equity                                  |
//|   dd_scale = eq_ratio ^ dd_power                                   |
//|   risk = risk_protect + (risk_grow - risk_protect) * dd_scale      |
//|   if consec_wins >= streak_n and dd_scale > 0.8:                   |
//|       risk = risk * streak_mult                                    |
//|   risk = max(0.002, min(max_risk_cap, risk))                       |
//|   lot = (equity * risk) / (sl_dist * CONTRACT_SIZE)                |
//|   lot = max(0.01, min(200.0, round(lot, 2)))                       |
//+------------------------------------------------------------------+
double CalculateTwoModeLot(double equity, double slDist)
{
   //--- Calculate equity ratio
   double eqRatio = (g_peakEquity > 0) ? (equity / g_peakEquity) : 1.0;

   //--- Exponential transition: (equity/peak)^dd_power
   double ddScale = MathPow(eqRatio, (double)InpDDPower);

   //--- Two-mode blend (matches Python exactly)
   double risk = InpRiskProtect + (InpRiskGrow - InpRiskProtect) * ddScale;

   //--- Streak boost (only in grow mode, matches Python)
   if(g_consecutiveWins >= InpStreakN && ddScale > 0.8)
      risk = risk * InpStreakMult;

   //--- Cap risk (matches Python: max(0.002, min(max_risk_cap, risk)))
   risk = MathMax(0.002, MathMin(InpMaxRiskCap, risk));

   //--- Calculate lot size (matches Python: lot = (equity * risk) / (sl_dist * CONTRACT_SIZE))
   //--- CONTRACT_SIZE for XAUUSD = 100 (1 lot = 100 oz)
   double contractSize = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   if(contractSize <= 0) contractSize = 100.0;  // Fallback for XAUUSD

   double lot = (equity * risk) / (slDist * contractSize);

   //--- Match Python: lot = max(0.01, min(200.0, round(lot, 2)))
   lot = NormalizeDouble(lot, 2);
   lot = MathMax(0.01, MathMin(200.0, lot));

   //--- Also respect broker limits
   double minLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_STEP);

   if(minLot > 0) lot = MathMax(minLot, lot);
   if(maxLot > 0) lot = MathMin(maxLot, lot);
   if(stepLot > 0) lot = MathFloor(lot / stepLot) * stepLot;

   return lot;
}

//+------------------------------------------------------------------+
//| Check if within valid trading session                               |
//| Matches Python: hours[i] >= 7 and hours[i] <= 20                  |
//| Uses bar time with UTC offset for broker compatibility             |
//+------------------------------------------------------------------+
bool IsValidSession()
{
   //--- Get the time of the last completed bar (bar 1)
   datetime barTime = iTime(g_symbol, PERIOD_M1, 1);

   //--- Apply UTC offset to convert broker time to UTC
   //--- If broker is UTC+2, offset = 2, so UTC = broker_time - 2h
   datetime utcTime = barTime - InpUTCOffset * 3600;

   MqlDateTime dt;
   TimeToStruct(utcTime, dt);

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
//| Check cooldown between entries (bar-count based)                   |
//| Matches Python: (i - last_entry_bars[s]) >= cooldown               |
//+------------------------------------------------------------------+
bool IsCooldownClear(int currentOpenCount)
{
   //--- Find next available slot
   for(int s = 0; s < InpMaxPositions; s++)
   {
      if(s >= currentOpenCount)
      {
         //--- This slot is free, check its cooldown
         if((g_barCount - g_lastEntryBar[s]) >= InpCooldownBars)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Execute a trade with retry logic                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(int direction, double slDist, double tpDist, double lotSize, int slotIdx)
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

   //--- Trade comment showing current mode
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double eqRatio = (g_peakEquity > 0) ? (equity / g_peakEquity) : 1.0;
   double ddScale = MathPow(eqRatio, (double)InpDDPower);
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

      //--- Recalculate SL/TP from current price
      if(direction == 1)
      {
         sl = NormalizeDouble(price - slDist, digits);
         tp = NormalizeDouble(price + tpDist, digits);
      }
      else
      {
         sl = NormalizeDouble(price + slDist, digits);
         tp = NormalizeDouble(price - tpDist, digits);
      }

      bool result = trade.PositionOpen(g_symbol, orderType, lotSize, price, sl, tp, comment);

      if(result)
      {
         //--- Record entry bar for cooldown (matches Python: last_entry_bars[s] = i)
         if(slotIdx < ArraySize(g_lastEntryBar))
            g_lastEntryBar[slotIdx] = g_barCount;

         Print("ENTRY: ", (direction == 1 ? "BUY" : "SELL"),
               " | Mode=", mode,
               " | Lot=", DoubleToString(lotSize, 2),
               " | Price=", DoubleToString(price, digits),
               " | SL=", DoubleToString(sl, digits),
               " | TP=", DoubleToString(tp, digits),
               " | Bar=", g_barCount);
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
//| Handle trade events (track consecutive wins for streak boost)      |
//| Matches Python: if pnl_d > 0: consec_wins += 1 else: consec_wins=0|
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   //--- Track consecutive wins for streak boost
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      //--- Only process deals for our magic number
      if(trans.deal > 0 && HistoryDealSelect(trans.deal))
      {
         long dealMagic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
         if(dealMagic != InpMagicNumber)
            return;

         //--- Check if this is a closing deal (entry=IN, exit=OUT or INOUT)
         long dealEntry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         if(dealEntry == DEAL_ENTRY_OUT || dealEntry == DEAL_ENTRY_INOUT)
         {
            double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                          + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION)
                          + HistoryDealGetDouble(trans.deal, DEAL_SWAP);

            if(profit > 0)
               g_consecutiveWins++;
            else
               g_consecutiveWins = 0;
         }
      }
   }
}

//+------------------------------------------------------------------+
