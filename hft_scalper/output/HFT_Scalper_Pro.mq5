//+------------------------------------------------------------------+
//|                                             HFT_Scalper_Pro.mq5  |
//|                        Ensemble HFT Scalping EA for XAUUSD       |
//|                     OrderFlow + MomentumMTF + SpreadFade         |
//+------------------------------------------------------------------+
#property copyright "HFT Scalper Pro"
#property link      ""
#property version   "1.00"
#property strict
#property description "Ensemble HFT scalping EA combining OrderFlow contrarian,"
#property description "Multi-Timeframe Momentum, and Spread Fade strategies."
#property description "Trades XAUUSD on M1 timeframe with consensus-based entries."

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Input parameters
input group "=== Ensemble Strategy Parameters ==="
input int      InpOFIPeriod             = 30;      // OFI Lookback Period
input double   InpOFIThreshold          = 2.0;     // OFI Z-Score Threshold
input int      InpSlowPeriod            = 40;      // Momentum Slow EMA Period
input int      InpFastRSIPeriod         = 7;       // Fast RSI Period (pullback)
input double   InpFastRSIOB             = 75.0;    // Fast RSI Overbought Level
input double   InpFastRSIOS             = 20.0;    // Fast RSI Oversold Level
input double   InpTrendThreshold        = 0.1;     // Trend Strength Threshold
input int      InpSpreadLookback        = 30;      // Spread Lookback Period
input double   InpWideThreshold         = 2.5;     // Spread Wide Threshold (x median)
input double   InpContractThreshold     = 1.5;     // Spread Contract Threshold (x median)
input int      InpMinScore              = 2;       // Minimum Consensus Score (2-3)

input group "=== Risk Management ==="
input double   InpLotSize               = 0.1;     // Lot Size
input double   InpSLMultHigh            = 1.5;     // SL ATR Multiplier (Score=3, High Conf)
input double   InpSLMultLow             = 2.5;     // SL ATR Multiplier (Score=2, Mod Conf)
input double   InpTPMultHigh            = 3.0;     // TP ATR Multiplier (Score=3)
input double   InpTPMultLow             = 2.0;     // TP ATR Multiplier (Score=2)
input int      InpATRPeriod             = 14;      // ATR Period
input double   InpMaxDailyLoss          = 50.0;    // Max Daily Loss ($)
input double   InpMaxDrawdownPct        = 30.0;    // Max Account Drawdown (%)
input double   InpMaxSpread             = 30.0;    // Max Spread (points)
input bool     InpUseTrailingStop       = true;    // Use ATR Trailing Stop
input double   InpTrailingATRMult       = 1.0;     // Trailing Stop ATR Multiplier

input group "=== Session Filter ==="
input bool     InpUseSessionFilter      = true;    // Enable Session Filter
input int      InpSessionStart1         = 4;       // Session 1 Start Hour (UTC)
input int      InpSessionEnd1           = 4;       // Session 1 End Hour (UTC)
input int      InpSessionStart2         = 8;       // Session 2 Start Hour (UTC)
input int      InpSessionEnd2           = 21;      // Session 2 End Hour (UTC)
input int      InpUTCOffset             = 0;       // UTC Offset (hours, for brokers without TimeGMT)
input bool     InpUseTimeGMT            = true;    // Use TimeGMT() (disable if broker unsupported)

input group "=== General Settings ==="
input int      InpMagicNumber           = 202604;  // Magic Number
input int      InpCooldownBars          = 3;       // Cooldown Bars Between Trades
input string   InpSymbol                = "XAUUSD";// Symbol (blank = current)
input int      InpMaxRetries            = 3;       // Max Order Retries
input int      InpTimerSeconds          = 60;      // Timer Interval (sec)

//--- Global objects
CTrade         trade;
CPositionInfo  posInfo;
CAccountInfo   accInfo;
CSymbolInfo    symInfo;

//--- Global variables
double         g_startEquity;
double         g_dailyStartEquity;
datetime       g_lastDayReset;
int            g_lastSignalBar;
bool           g_tradingEnabled;
double         g_peakEquity;
string         g_symbol;

//--- OFI calculation arrays
double         g_ofiRaw[];
double         g_ofiZscore[];
double         g_slowEMA[];
double         g_fastRSI[];
double         g_atr[];
double         g_spreadHistory[];

//--- Spread fade state
bool           g_wasWide;
double         g_wideStartPrice;

//--- Spread ring buffer for historical per-bar tracking
double         g_spreadRingBuffer[];
int            g_spreadRingIndex;
int            g_spreadRingCount;

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
   g_lastSignalBar = -InpCooldownBars - 1;
   g_tradingEnabled = true;
   g_wasWide = false;
   g_wideStartPrice = 0.0;
   
   //--- Allocate arrays
   ArrayResize(g_ofiRaw, InpOFIPeriod + 1);
   ArrayResize(g_ofiZscore, 1);
   ArrayResize(g_spreadHistory, InpSpreadLookback + 1);
   ArrayInitialize(g_ofiRaw, 0.0);
   ArrayInitialize(g_spreadHistory, 0.0);
   
   //--- Initialize spread ring buffer for per-bar spread tracking
   ArrayResize(g_spreadRingBuffer, InpSpreadLookback + 1);
   ArrayInitialize(g_spreadRingBuffer, 0.0);
   g_spreadRingIndex = 0;
   g_spreadRingCount = 0;
   
   //--- Set timer for periodic equity checks
   EventSetTimer(InpTimerSeconds);
   
   Print("HFT Scalper Pro initialized on ", g_symbol);
   Print("Account: $", DoubleToString(g_startEquity, 2),
         " | Lot: ", DoubleToString(InpLotSize, 2),
         " | Magic: ", InpMagicNumber);
   Print("Strategy: Ensemble (OrderFlow+MomentumMTF+SpreadFade), Min Score: ", InpMinScore);
   
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
   
   Print("HFT Scalper Pro stopped. Reason: ", reason);
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
   
   //--- Check max drawdown from peak
   double ddPct = (g_peakEquity > 0) ? ((g_peakEquity - equity) / g_peakEquity * 100.0) : 0.0;
   if(ddPct >= InpMaxDrawdownPct)
   {
      if(g_tradingEnabled)
      {
         Print("SAFETY: Max drawdown reached (", DoubleToString(ddPct, 1), "%). Trading disabled.");
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
         
      Print("New day reset. Equity: $", DoubleToString(equity, 2));
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Only process on new bar
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(g_symbol, PERIOD_M1, 0);
   if(currentBarTime == lastBarTime)
      return;
   lastBarTime = currentBarTime;
   
   //--- Record per-bar spread into ring buffer (capture spread at bar close)
   double barSpread = (double)SymbolInfoInteger(g_symbol, SYMBOL_SPREAD) * _Point;
   if(barSpread < _Point)
   {
      //--- Fallback: use previous bar's high-low range as spread proxy
      double prevHigh = iHigh(g_symbol, PERIOD_M1, 1);
      double prevLow = iLow(g_symbol, PERIOD_M1, 1);
      barSpread = (prevHigh - prevLow) * 0.1;
   }
   int bufSize = InpSpreadLookback + 1;
   g_spreadRingBuffer[g_spreadRingIndex] = barSpread;
   g_spreadRingIndex = (g_spreadRingIndex + 1) % bufSize;
   if(g_spreadRingCount < bufSize)
      g_spreadRingCount++;
   
   //--- Safety checks
   if(!g_tradingEnabled)
      return;
   
   //--- Check daily loss limit
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyLoss = g_dailyStartEquity - equity;
   if(dailyLoss >= InpMaxDailyLoss)
   {
      Print("SAFETY: Daily loss limit reached ($", DoubleToString(dailyLoss, 2), "). No new trades today.");
      return;
   }
   
   //--- Check spread
   symInfo.RefreshRates();
   double currentSpread = symInfo.Spread();
   if(currentSpread > InpMaxSpread)
      return;
   
   //--- Session filter
   if(InpUseSessionFilter && !IsValidSession())
      return;
   
   //--- Manage existing position (trailing stop)
   if(HasPosition())
   {
      if(InpUseTrailingStop)
         ManageTrailingStop();
      return;  // Only one position at a time
   }
   
   //--- Cooldown check
   int currentBar = Bars(g_symbol, PERIOD_M1);
   if((currentBar - g_lastSignalBar) < InpCooldownBars)
      return;
   
   //--- Generate ensemble signal
   int direction = 0;
   int score = 0;
   GetEnsembleSignal(direction, score);
   
   if(direction == 0 || score < InpMinScore)
      return;
   
   //--- Calculate SL/TP based on score
   double atr = GetATR(InpATRPeriod);
   if(atr < _Point)
      return;
   
   double slDist, tpDist;
   if(score >= 3)
   {
      slDist = atr * InpSLMultHigh;
      tpDist = atr * InpTPMultHigh;
   }
   else
   {
      slDist = atr * InpSLMultLow;
      tpDist = atr * InpTPMultLow;
   }
   
   //--- Execute trade
   ExecuteTrade(direction, slDist, tpDist, score);
   g_lastSignalBar = currentBar;
}

//+------------------------------------------------------------------+
//| Get ensemble signal from combined sub-strategies                    |
//+------------------------------------------------------------------+
void GetEnsembleSignal(int &direction, int &score)
{
   direction = 0;
   score = 0;
   
   int buyScore = 0;
   int sellScore = 0;
   
   //--- Sub-strategy 1: OrderFlow (contrarian)
   double ofiZ = GetOFIZscore();
   if(ofiZ > InpOFIThreshold)
      sellScore++;   // Extreme buying -> fade -> sell
   else if(ofiZ < -InpOFIThreshold)
      buyScore++;    // Extreme selling -> fade -> buy
   
   //--- Sub-strategy 2: MomentumMTF (trend + pullback)
   double trendSlope = GetTrendSlope();
   double rsi = GetFastRSI();
   
   if(trendSlope > InpTrendThreshold && rsi < InpFastRSIOS)
      buyScore++;    // Uptrend with oversold pullback -> buy
   else if(trendSlope < -InpTrendThreshold && rsi > InpFastRSIOB)
      sellScore++;   // Downtrend with overbought pullback -> sell
   
   //--- Sub-strategy 3: SpreadFade
   int spreadDir = GetSpreadFadeSignal();
   if(spreadDir == 1)
      buyScore++;
   else if(spreadDir == -1)
      sellScore++;
   
   //--- Determine consensus
   if(buyScore >= InpMinScore)
   {
      direction = 1;
      score = buyScore;
   }
   else if(sellScore >= InpMinScore)
   {
      direction = -1;
      score = sellScore;
   }
}

//+------------------------------------------------------------------+
//| Calculate Order Flow Imbalance Z-Score                             |
//+------------------------------------------------------------------+
double GetOFIZscore()
{
   int barsNeeded = InpOFIPeriod + 1;
   
   double ofiValues[];
   ArrayResize(ofiValues, barsNeeded);
   
   for(int i = 0; i < barsNeeded; i++)
   {
      double high = iHigh(g_symbol, PERIOD_M1, i);
      double low = iLow(g_symbol, PERIOD_M1, i);
      double close = iClose(g_symbol, PERIOD_M1, i);
      long tickVol = iTickVolume(g_symbol, PERIOD_M1, i);
      
      double range = high - low;
      if(range < _Point) range = _Point;
      
      double closePos = (close - low) / range;
      ofiValues[i] = (closePos - 0.5) * 2.0 * (double)tickVol;
   }
   
   //--- Calculate z-score of current bar vs lookback
   double sum = 0, sumSq = 0;
   for(int i = 1; i <= InpOFIPeriod; i++)
   {
      sum += ofiValues[i];
      sumSq += ofiValues[i] * ofiValues[i];
   }
   
   double mean = sum / InpOFIPeriod;
   double variance = (sumSq / InpOFIPeriod) - (mean * mean);
   double std = (variance > 0) ? MathSqrt(variance) : 0.0;
   
   if(std < 0.0001) return 0.0;
   
   return (ofiValues[0] - mean) / std;
}

//+------------------------------------------------------------------+
//| Calculate Trend Slope (normalized by ATR)                          |
//+------------------------------------------------------------------+
double GetTrendSlope()
{
   double atr = GetATR(InpATRPeriod);
   if(atr < _Point) return 0.0;
   
   //--- Compute slow EMA
   int barsNeeded = InpSlowPeriod + InpSlowPeriod / 4 + 1;
   if(Bars(g_symbol, PERIOD_M1) < barsNeeded) return 0.0;
   
   double emaFull[];
   ArrayResize(emaFull, barsNeeded);
   
   double multiplier = 2.0 / (InpSlowPeriod + 1.0);
   
   //--- Build EMA from oldest to newest (reverse index)
   emaFull[barsNeeded - 1] = iClose(g_symbol, PERIOD_M1, barsNeeded - 1);
   for(int i = barsNeeded - 2; i >= 0; i--)
   {
      double closePrice = iClose(g_symbol, PERIOD_M1, i);
      emaFull[i] = closePrice * multiplier + emaFull[i + 1] * (1.0 - multiplier);
   }
   
   //--- Slope: current EMA vs EMA from slow_period/4 bars ago
   int lookback = InpSlowPeriod / 4;
   double slopeRaw = emaFull[0] - emaFull[lookback];
   
   return slopeRaw / atr;
}

//+------------------------------------------------------------------+
//| Calculate Fast RSI                                                 |
//+------------------------------------------------------------------+
double GetFastRSI()
{
   int barsNeeded = InpFastRSIPeriod + 2;
   if(Bars(g_symbol, PERIOD_M1) < barsNeeded) return 50.0;
   
   double gains = 0, losses = 0;
   
   //--- Initial average gain/loss
   for(int i = 1; i <= InpFastRSIPeriod; i++)
   {
      double change = iClose(g_symbol, PERIOD_M1, i - 1) - iClose(g_symbol, PERIOD_M1, i);
      if(change > 0)
         gains += change;
      else
         losses -= change;
   }
   
   double avgGain = gains / InpFastRSIPeriod;
   double avgLoss = losses / InpFastRSIPeriod;
   
   if(avgLoss < 0.0001) return 100.0;
   
   double rs = avgGain / avgLoss;
   return 100.0 - 100.0 / (1.0 + rs);
}

//+------------------------------------------------------------------+
//| Get Spread Fade Signal                                             |
//+------------------------------------------------------------------+
int GetSpreadFadeSignal()
{
   int barsNeeded = InpSpreadLookback + 1;
   if(Bars(g_symbol, PERIOD_M1) < barsNeeded) return 0;
   
   //--- Need enough historical spread data in the ring buffer
   if(g_spreadRingCount < barsNeeded)
      return 0;
   
   //--- Get current bar spread (most recently stored value)
   int bufSize = InpSpreadLookback + 1;
   int currentIdx = (g_spreadRingIndex - 1 + bufSize) % bufSize;
   double currentSpread = g_spreadRingBuffer[currentIdx];
   
   //--- Compute median spread from lookback (excluding current bar)
   double sortedSpreads[];
   ArrayResize(sortedSpreads, InpSpreadLookback);
   for(int i = 0; i < InpSpreadLookback; i++)
   {
      //--- Walk backwards from the bar before current
      int idx = (currentIdx - 1 - i + bufSize) % bufSize;
      sortedSpreads[i] = g_spreadRingBuffer[idx];
   }
   ArraySort(sortedSpreads);
   
   double medianSpread = sortedSpreads[InpSpreadLookback / 2];
   if(medianSpread < _Point) return 0;
   
   double spreadRatio = currentSpread / medianSpread;
   
   //--- Detect wide spread -> contraction pattern
   if(spreadRatio >= InpWideThreshold && !g_wasWide)
   {
      g_wasWide = true;
      g_wideStartPrice = iClose(g_symbol, PERIOD_M1, 0);
   }
   
   if(g_wasWide && spreadRatio <= InpContractThreshold)
   {
      g_wasWide = false;
      double priceChange = iClose(g_symbol, PERIOD_M1, 0) - g_wideStartPrice;
      if(priceChange > 0)
         return 1;   // Buy
      else if(priceChange < 0)
         return -1;  // Sell
   }
   
   //--- Reset if spread normalizes without proper contraction
   if(g_wasWide && spreadRatio < InpWideThreshold * 0.7)
      g_wasWide = false;
   
   return 0;
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
      double high = iHigh(g_symbol, PERIOD_M1, i);
      double low = iLow(g_symbol, PERIOD_M1, i);
      double prevClose = iClose(g_symbol, PERIOD_M1, i + 1);
      
      double tr = MathMax(high - low, MathMax(MathAbs(high - prevClose), MathAbs(low - prevClose)));
      atr += tr;
   }
   
   return atr / period;
}

//+------------------------------------------------------------------+
//| Check if currently within valid trading session                     |
//+------------------------------------------------------------------+
bool IsValidSession()
{
   MqlDateTime dt;
   //--- Use TimeGMT for accurate UTC time; fall back to TimeCurrent with offset
   if(InpUseTimeGMT)
      TimeGMT(dt);
   else
   {
      datetime serverTime = TimeCurrent();
      //--- Apply UTC offset: subtract broker offset to get UTC
      datetime utcTime = serverTime - InpUTCOffset * 3600;
      TimeToStruct(utcTime, dt);
   }
   int hour = dt.hour;
   
   //--- Session 1: single hour
   if(hour == InpSessionStart1)
      return true;
   
   //--- Session 2: range
   if(hour >= InpSessionStart2 && hour <= InpSessionEnd2)
      return true;
   
   return false;
}

//+------------------------------------------------------------------+
//| Check if we have an open position                                  |
//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Magic() == InpMagicNumber && posInfo.Symbol() == g_symbol)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Execute a trade with retry logic                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(int direction, double slDist, double tpDist, int signalScore)
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
   
   //--- Build trade comment with signal score
   string comment = StringFormat("HFT_Ens_S%d", signalScore);
   
   //--- Execute with retry
   for(int attempt = 0; attempt < InpMaxRetries; attempt++)
   {
      //--- Refresh prices before each attempt
      symInfo.RefreshRates();
      if(direction == 1)
         price = symInfo.Ask();
      else
         price = symInfo.Bid();
      
      price = NormalizeDouble(price, digits);
      
      bool result = trade.PositionOpen(g_symbol, orderType, InpLotSize, price, sl, tp, comment);
      
      if(result)
      {
         Print("Trade opened: ", (direction == 1 ? "BUY" : "SELL"),
               " Score=", signalScore,
               " Price=", DoubleToString(price, digits),
               " SL=", DoubleToString(sl, digits),
               " TP=", DoubleToString(tp, digits));
         return;
      }
      else
      {
         int error = (int)trade.ResultRetcode();
         Print("Order failed (attempt ", attempt + 1, "/", InpMaxRetries,
               ") Error: ", error, " - ", trade.ResultRetcodeDescription());
         
         //--- Don't retry on permanent errors
         if(error == TRADE_RETCODE_INVALID_STOPS ||
            error == TRADE_RETCODE_NO_MONEY ||
            error == TRADE_RETCODE_MARKET_CLOSED)
            break;
         
         Sleep(500);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage ATR-based trailing stop                                     |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i))
         continue;
      if(posInfo.Magic() != InpMagicNumber || posInfo.Symbol() != g_symbol)
         continue;
      
      double atr = GetATR(InpATRPeriod);
      if(atr < _Point)
         return;
      
      double trailDist = atr * InpTrailingATRMult;
      int digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
      
      double currentSL = posInfo.StopLoss();
      double currentTP = posInfo.TakeProfit();
      ulong ticket = posInfo.Ticket();
      
      if(posInfo.PositionType() == POSITION_TYPE_BUY)
      {
         double bid = symInfo.Bid();
         double newSL = NormalizeDouble(bid - trailDist, digits);
         
         //--- Only move SL up, never down
         if(newSL > currentSL + _Point)
         {
            trade.PositionModify(ticket, newSL, currentTP);
         }
      }
      else if(posInfo.PositionType() == POSITION_TYPE_SELL)
      {
         double ask = symInfo.Ask();
         double newSL = NormalizeDouble(ask + trailDist, digits);
         
         //--- Only move SL down, never up
         if(newSL < currentSL - _Point || currentSL == 0)
         {
            trade.PositionModify(ticket, newSL, currentTP);
         }
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
         {
            trade.PositionClose(posInfo.Ticket());
         }
      }
   }
}

//+------------------------------------------------------------------+
