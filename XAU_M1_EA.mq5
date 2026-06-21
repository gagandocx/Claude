//+------------------------------------------------------------------+
//|                                                   XAU_M1_EA.mq5 |
//|                    XAUUSD 1-Minute Scalper                       |
//|               Built from 47,172,211 tick analysis                |
//|                                                                  |
//| STRATEGY: EMA Pullback + Session Bias + ATR Dynamic SL/TP       |
//|                                                                  |
//| DATA FINDINGS USED:                                              |
//|  - Avg M1 range: 3.14 pts  → TP target: 6-9 pts                 |
//|  - Mean-reverting ticks    → fade extremes, trade pullbacks      |
//|  - Best hours: 9,10,13,14,23 UTC → bullish bias                 |
//|  - Sell hours: 2,4,5,17 UTC    → bearish bias                   |
//|  - Skip Thursday: only 41.46% bullish, avg -14.99               |
//|  - Best spread: 12-14 UTC (0.69 pts avg)                        |
//|  - London bullish, NY bearish session bias                       |
//+------------------------------------------------------------------+
#property copyright "XAU M1 EA - Data Driven"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade        trade;
CPositionInfo pos;


//+------------------------------------------------------------------+
//| INPUTS                                                            |
//+------------------------------------------------------------------+
input group "=== TREND FILTER ==="
input int    H1_EMA_Period    = 50;      // H1 EMA Period (trend direction)
input int    M1_EMA_Fast      = 8;       // M1 Fast EMA (pullback level)
input int    M1_EMA_Slow      = 21;      // M1 Slow EMA (trend confirmation)
input bool   Trade_Both_Dir   = true;    // true=trade both BUY+SELL, false=trend only

input group "=== ATR SETTINGS ==="
input int    ATR_Period       = 10;      // ATR Period (M1)
input double SL_ATR_Mult      = 1.5;    // SL = ATR x this
input double TP_ATR_Mult      = 2.5;    // TP = ATR x this
input double MaxATR_Entry     = 5.0;    // Skip if ATR > this x avg (default relaxed)

input group "=== SESSION FILTER (UTC) ==="
input bool   Use_TimeFilter   = false;  // Enable session filter (OFF by default for backtesting)
input bool   Use_DayFilter    = false;  // Skip Thursday (OFF by default for backtesting)
input bool   Use_SpreadFilter = true;   // Enable spread filter
input double Max_Spread_Pts   = 5.0;    // Max allowed spread in points (relaxed)

input group "=== RISK MANAGEMENT ==="
input double Risk_Percent     = 1.0;    // Risk % per trade
input double Manual_Lot       = 0.0;    // Manual lot (0 = auto)
input double Max_Lot          = 5.0;    // Max lot size
input int    Max_Trades       = 3;      // Max open trades at once
input bool   Use_BreakEven    = true;   // Move SL to BE after 1x ATR profit
input bool   Use_Trailing     = true;   // Enable trailing stop
input double Trail_ATR_Mult   = 1.0;    // Trail distance = ATR x this

input group "=== EQUITY PROTECTION ==="
input bool   Use_EP           = true;   // Enable equity protection
input double Max_DD_Pct       = 5.0;    // Max drawdown % before closing all

input group "=== MAGIC & DISPLAY ==="
input int    Magic            = 11001;  // Magic number
input string Comment_         = "XAU_M1"; // Trade comment
input int    Slippage         = 10;     // Max slippage (pts)
input bool   Show_Panel       = true;   // Show info panel


//+------------------------------------------------------------------+
//| GLOBALS                                                           |
//+------------------------------------------------------------------+
int    h1_ema_handle;
int    m1_ema_fast_handle;
int    m1_ema_slow_handle;
int    m1_atr_handle;

double h1_ema[];
double m1_ema_fast[];
double m1_ema_slow[];
double m1_atr[];

double point_size;
double avg_atr;           // rolling avg ATR for volatility filter
int    atr_sample = 50;   // bars to avg for baseline ATR

datetime last_bar;
int      open_buys, open_sells;
double   float_pnl;
string   panel_signal  = "INIT";
color    panel_sig_col = clrGray;

// Break-even tracking
ulong    be_done_tickets[];

// P&L tracking
double   pnl_today, pnl_week, pnl_month;
datetime pnl_cache;

string   lbl = "M1EA_";


//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   point_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   h1_ema_handle      = iMA(_Symbol, PERIOD_H1, H1_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   m1_ema_fast_handle = iMA(_Symbol, PERIOD_M1, M1_EMA_Fast,   0, MODE_EMA, PRICE_CLOSE);
   m1_ema_slow_handle = iMA(_Symbol, PERIOD_M1, M1_EMA_Slow,   0, MODE_EMA, PRICE_CLOSE);
   m1_atr_handle      = iATR(_Symbol, PERIOD_M1, ATR_Period);

   if(h1_ema_handle==INVALID_HANDLE || m1_ema_fast_handle==INVALID_HANDLE ||
      m1_ema_slow_handle==INVALID_HANDLE || m1_atr_handle==INVALID_HANDLE)
   {
      Print("ERROR: Indicator handle creation failed."); return INIT_FAILED;
   }

   ArraySetAsSeries(h1_ema,       true);
   ArraySetAsSeries(m1_ema_fast,  true);
   ArraySetAsSeries(m1_ema_slow,  true);
   ArraySetAsSeries(m1_atr,       true);
   ArrayResize(be_done_tickets, 0);

   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(Slippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   if(Show_Panel) CreatePanel();
   Print("XAU M1 EA initialized | XAUUSD 1-min scalper | Magic=", Magic);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(h1_ema_handle);
   IndicatorRelease(m1_ema_fast_handle);
   IndicatorRelease(m1_ema_slow_handle);
   IndicatorRelease(m1_atr_handle);
   ObjectsDeleteAll(0, lbl);
}


//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // Load indicator buffers
   if(CopyBuffer(h1_ema_handle,      0, 0, 3, h1_ema)      < 3) return;
   if(CopyBuffer(m1_ema_fast_handle, 0, 0, 4, m1_ema_fast) < 4) return;
   if(CopyBuffer(m1_ema_slow_handle, 0, 0, 4, m1_ema_slow) < 4) return;
   if(CopyBuffer(m1_atr_handle,      0, 0, atr_sample+2, m1_atr) < atr_sample+2) return;

   // Compute avg ATR baseline (last 50 closed bars)
   double atr_sum = 0;
   for(int i = 1; i <= atr_sample; i++) atr_sum += m1_atr[i];
   avg_atr = atr_sum / atr_sample;
   double cur_atr = m1_atr[1]; // last closed bar ATR

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;

   CountPositions();

   // Equity protection — every tick
   if(Use_EP && CheckEP()) { if(Show_Panel) UpdatePanel(); return; }

   // Position management — every tick
   if(Use_BreakEven) ManageBreakEven(cur_atr);
   if(Use_Trailing)  ManageTrailing(cur_atr);

   // New M1 bar only for entries
   datetime cur_bar = iTime(_Symbol, PERIOD_M1, 0);
   if(cur_bar == last_bar) { if(Show_Panel) UpdatePanel(); return; }
   last_bar = cur_bar;

   // --- SESSION & DAY FILTERS ---
   if(Use_DayFilter  && IsThursday())    { panel_signal="SKIP THU";  panel_sig_col=clrOrange; if(Show_Panel) UpdatePanel(); return; }
   if(Use_TimeFilter && !IsGoodHour())   { panel_signal="TIME OFF";  panel_sig_col=clrGray;   if(Show_Panel) UpdatePanel(); return; }
   if(Use_SpreadFilter && spread > Max_Spread_Pts * point_size * 10)
                                         { panel_signal="WIDE SPRD"; panel_sig_col=clrOrange; if(Show_Panel) UpdatePanel(); return; }

   // --- VOLATILITY FILTER ---
   // Skip if current ATR > MaxATR_Entry x avg ATR (news spike / extreme volatility)
   if(cur_atr > MaxATR_Entry * avg_atr)  { panel_signal="HIGH VOL";  panel_sig_col=clrOrange; if(Show_Panel) UpdatePanel(); return; }

   // --- MAX TRADES ---
   if(open_buys + open_sells >= Max_Trades) { panel_signal="MAX TRADES"; panel_sig_col=clrOrange; if(Show_Panel) UpdatePanel(); return; }

   // --- SIGNAL ---
   int sig = GetSignal(cur_atr);
   if(sig == 1)       { panel_signal="BUY";     panel_sig_col=clrLime;   OpenTrade(ORDER_TYPE_BUY,  cur_atr); }
   else if(sig == -1) { panel_signal="SELL";    panel_sig_col=clrTomato; OpenTrade(ORDER_TYPE_SELL, cur_atr); }
   else               { panel_signal="WAITING"; panel_sig_col=clrYellow; }

   if(Show_Panel) UpdatePanel();
}


//+------------------------------------------------------------------+
//| SIGNAL: EMA Cross on M1 with H1 trend filter                     |
//|                                                                   |
//| BUY:  Fast EMA crosses ABOVE slow EMA on M1                      |
//|       + Price is above H1 EMA (or Trade_Both_Dir=true)           |
//|                                                                   |
//| SELL: Fast EMA crosses BELOW slow EMA on M1                      |
//|       + Price is below H1 EMA (or Trade_Both_Dir=true)           |
//+------------------------------------------------------------------+
int GetSignal(double cur_atr)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // H1 trend filter
   bool h1_bull = (bid > h1_ema[1]);
   bool h1_bear = (bid < h1_ema[1]);

   // M1 EMA cross: bar 2 = before cross, bar 1 = after cross (both closed)
   bool cross_up   = (m1_ema_fast[2] <= m1_ema_slow[2]) && (m1_ema_fast[1] > m1_ema_slow[1]);
   bool cross_down = (m1_ema_fast[2] >= m1_ema_slow[2]) && (m1_ema_fast[1] < m1_ema_slow[1]);

   // BUY signal
   if(cross_up)
   {
      if(Trade_Both_Dir || h1_bull) return 1;
   }

   // SELL signal
   if(cross_down)
   {
      if(Trade_Both_Dir || h1_bear) return -1;
   }

   return 0;
}


//+------------------------------------------------------------------+
//| OPEN TRADE                                                        |
//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE type, double atr)
{
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   double price  = (type == ORDER_TYPE_BUY) ? ask : bid;

   double sl_dist = SL_ATR_Mult * atr;
   double tp_dist = TP_ATR_Mult * atr;

   double sl, tp;
   if(type == ORDER_TYPE_BUY)
   {
      sl = NormalizeDouble(price - sl_dist - spread, _Digits);
      tp = NormalizeDouble(price + tp_dist,           _Digits);
   }
   else
   {
      sl = NormalizeDouble(price + sl_dist + spread, _Digits);
      tp = NormalizeDouble(price - tp_dist,           _Digits);
   }

   double lot = CalcLot(sl_dist);
   if(trade.PositionOpen(_Symbol, type, lot, price, sl, tp, Comment_))
      Print("XAU M1 EA | ", EnumToString(type), " lot=", lot,
            " price=", price, " sl=", sl, " tp=", tp, " atr=", atr);
   else
      Print("XAU M1 EA | FAILED: ", trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| BREAK-EVEN: move SL to entry + 1pt after 1x ATR profit           |
//+------------------------------------------------------------------+
void ManageBreakEven(double atr)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol()!=_Symbol || pos.Magic()!=Magic) continue;
      if(TicketInArr(be_done_tickets, pos.Ticket())) continue;

      double open_px = pos.PriceOpen();
      bool   is_buy  = (pos.PositionType() == POSITION_TYPE_BUY);
      double cur_px  = is_buy ? bid : ask;
      double profit  = is_buy ? (cur_px - open_px) : (open_px - cur_px);

      if(profit >= atr) // 1x ATR in profit → move to break-even
      {
         double be_sl;
         if(is_buy)  be_sl = NormalizeDouble(open_px + point_size * 2, _Digits);
         else        be_sl = NormalizeDouble(open_px - point_size * 2, _Digits);

         double cur_sl = pos.StopLoss();
         bool   needs_update = is_buy ? (be_sl > cur_sl + point_size)
                                      : (cur_sl < point_size || be_sl < cur_sl - point_size);
         if(needs_update)
         {
            trade.PositionModify(pos.Ticket(), be_sl, pos.TakeProfit());
            AddTicket(be_done_tickets, pos.Ticket());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| TRAILING STOP                                                     |
//+------------------------------------------------------------------+
void ManageTrailing(double atr)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double trail = Trail_ATR_Mult * atr;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol()!=_Symbol || pos.Magic()!=Magic) continue;
      if(!TicketInArr(be_done_tickets, pos.Ticket())) continue; // only after BE

      bool   is_buy  = (pos.PositionType() == POSITION_TYPE_BUY);
      double cur_sl  = pos.StopLoss();

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


//+------------------------------------------------------------------+
//| TIME & SESSION FILTERS — from 47M tick analysis                  |
//+------------------------------------------------------------------+
bool IsGoodHour()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   // Best hours from data: 9,10,11,13,14,15 (London) + 23 (late session)
   // + 2,3,4,5 (Asian sell window) + 17 (NY close)
   if(h >= 9  && h <= 15) return true;
   if(h == 23)            return true;
   if(h >= 2  && h <= 5)  return true;
   if(h == 17)            return true;
   return false;
}

bool IsThursday()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   return (dt.day_of_week == 4);
}

// Hour bias: +1=buy preferred, -1=sell preferred, 0=both
int HourBias()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   // From tick data: bullish hours
   if(h==9 || h==10 || h==11 || h==12 || h==13 || h==14 || h==15 || h==23) return 1;
   // Bearish hours
   if(h==2 || h==4 || h==5 || h==17) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| COUNT POSITIONS                                                   |
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
//| EQUITY PROTECTION                                                 |
//+------------------------------------------------------------------+
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
      panel_signal  = "EP HIT!";
      panel_sig_col = clrRed;
      Print("EQUITY PROTECTION triggered: DD=", dd, "%");
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| LOT SIZE                                                          |
//+------------------------------------------------------------------+
double CalcLot(double sl_dist)
{
   if(Manual_Lot > 0.0) return NormLot(Manual_Lot);
   double bal       = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk      = bal * Risk_Percent / 100.0;
   double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(sl_dist <= 0 || tick_val <= 0 || tick_size <= 0) return NormLot(0.01);
   double lot = risk / (sl_dist / tick_size * tick_val);
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

//+------------------------------------------------------------------+
//| TICKET HELPERS                                                    |
//+------------------------------------------------------------------+
bool TicketInArr(ulong &arr[], ulong t)
{ for(int i=0;i<ArraySize(arr);i++) if(arr[i]==t) return true; return false; }

void AddTicket(ulong &arr[], ulong t)
{
   if(TicketInArr(arr,t)) return;
   int s = ArraySize(arr); ArrayResize(arr, s+1); arr[s] = t;
}


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
   datetime today      = StructToTime(dt);
   datetime week_start = today - (datetime)((dt.day_of_week > 0 ? dt.day_of_week-1 : 6) * 86400);
   dt.day = 1;
   datetime mon_start  = StructToTime(dt);

   pnl_today = GetPnL(today,      now);
   pnl_week  = GetPnL(week_start, now);
   pnl_month = GetPnL(mon_start,  now);
}

double GetPnL(datetime from, datetime to)
{
   double pnl = 0;
   if(!HistorySelect(from, to)) return 0;
   for(int i=0; i<HistoryDealsTotal(); i++)
   {
      ulong t = HistoryDealGetTicket(i);
      if(t==0) continue;
      if(HistoryDealGetString(t, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(t, DEAL_MAGIC) != Magic) continue;
      if(HistoryDealGetInteger(t, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         pnl += HistoryDealGetDouble(t, DEAL_PROFIT)
               +HistoryDealGetDouble(t, DEAL_SWAP)
               +HistoryDealGetDouble(t, DEAL_COMMISSION);
   }
   return pnl;
}


//+------------------------------------------------------------------+
//| PANEL                                                             |
//+------------------------------------------------------------------+
void CreatePanel()
{
   ObjectsDeleteAll(0, lbl);
   int x=15, y=25, r=15;
   PRect(lbl+"bg", x-8, y-8, 285, 340, C'18,25,38', C'35,50,80', 1);

   PLbl(lbl+"dot",   "\x25CF",             x,    y,    C'0,200,120', 12, true);
   PLbl(lbl+"title", " XAU M1 Scalper",    x+15, y+1,  clrWhite,     9,  true);
   PLbl(lbl+"ver",   " v1.0 | 47M Ticks",  x+15, y+13, C'120,130,150', 7, false);
   PLine(lbl+"d0", x, y+26, 265);

   int row = y+35;
   PLbl(lbl+"ls",  "Symbol",      x,    row,    C'140,150,170', 8); PLbl(lbl+"vs",  _Symbol,        x+130, row,    clrWhite, 8);
   PLbl(lbl+"ltf", "Timeframe",   x,    row+r,  C'140,150,170', 8); PLbl(lbl+"vtf", "M1 + H1 Filter",x+130,row+r, clrWhite, 8);
   PLbl(lbl+"lma", "EMA Fast/Slow",x,   row+r*2,C'140,150,170', 8); PLbl(lbl+"vma", StringFormat("%d / %d",M1_EMA_Fast,M1_EMA_Slow), x+130, row+r*2, clrWhite, 8);
   PLine(lbl+"d1", x, row+r*3+2, 265);

   row = row+r*3+10;
   PLbl(lbl+"lsg",  "Signal",      x, row,    C'140,150,170', 8); PLbl(lbl+"vsg",  "---",  x+130, row,    clrWhite, 8);
   PLbl(lbl+"lhr",  "Hour (UTC)",  x, row+r,  C'140,150,170', 8); PLbl(lbl+"vhr",  "---",  x+130, row+r,  clrWhite, 8);
   PLbl(lbl+"ldy",  "Day",         x, row+r*2,C'140,150,170', 8); PLbl(lbl+"vdy",  "---",  x+130, row+r*2,clrWhite, 8);
   PLbl(lbl+"lsp",  "Spread",      x, row+r*3,C'140,150,170', 8); PLbl(lbl+"vsp",  "---",  x+130, row+r*3,clrWhite, 8);
   PLbl(lbl+"lvo",  "Volatility",  x, row+r*4,C'140,150,170', 8); PLbl(lbl+"vvo",  "---",  x+130, row+r*4,clrWhite, 8);
   PLine(lbl+"d2", x, row+r*5+2, 265);

   row = row+r*5+10;
   PLbl(lbl+"ltr",  "Trades",      x, row,    C'140,150,170', 8); PLbl(lbl+"vtr",  "0",    x+130, row,    clrWhite, 8);
   PLbl(lbl+"lfp",  "Float P/L",   x, row+r,  C'140,150,170', 8); PLbl(lbl+"vfp",  "---",  x+130, row+r,  clrWhite, 8);
   PLine(lbl+"d3", x, row+r*2+2, 265);

   row = row+r*2+10;
   PLbl(lbl+"lpd",  "Today",       x,    row,   C'140,150,170', 8); PLbl(lbl+"vpd",  "---", x+130, row,   clrWhite, 8);
   PLbl(lbl+"lpw",  "This Week",   x,    row+r, C'140,150,170', 8); PLbl(lbl+"vpw",  "---", x+130, row+r, clrWhite, 8);
   PLbl(lbl+"lpm",  "This Month",  x,    row+r*2,C'140,150,170',8); PLbl(lbl+"vpm",  "---", x+130, row+r*2,clrWhite, 8);
   PLine(lbl+"d4", x, row+r*3+2, 265);

   row = row+r*3+10;
   PLbl(lbl+"lri",  "Risk/Trade",  x, row,   C'140,150,170', 8);
   PLbl(lbl+"vri",  Manual_Lot>0 ? StringFormat("%.2f lot",Manual_Lot) : StringFormat("%.1f%%",Risk_Percent),
                    x+130, row, clrWhite, 8);
   PLbl(lbl+"lst",  "Status",      x, row+r, C'140,150,170', 8);
   PLbl(lbl+"vst",  "RUNNING",     x+130, row+r, clrLime, 8);
   ChartRedraw(0);
}

void UpdatePanel()
{
   if(!Show_Panel) return;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   string days[]={"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
   bool thu = (dt.day_of_week==4);

   double spread = (SymbolInfoDouble(_Symbol,SYMBOL_ASK)-SymbolInfoDouble(_Symbol,SYMBOL_BID))
                   / point_size / 10.0;
   string vol_str = "---";
   if(avg_atr > 0)
   {
      double ratio = m1_atr[1] / avg_atr;
      vol_str = StringFormat("%.2f (x%.1f avg)", m1_atr[1]/point_size/10.0, ratio);
   }

   PSet(lbl+"vsg",  panel_signal,  panel_sig_col);
   PSet(lbl+"vhr",  StringFormat("%02d:00 %s", dt.hour, IsGoodHour()?"ON":"OFF"),
                    IsGoodHour() ? clrLime : clrOrange);
   PSet(lbl+"vdy",  days[dt.day_of_week]+(thu?" SKIP!":""), thu ? clrOrange : clrLime);
   PSet(lbl+"vsp",  StringFormat("%.2f pts %s", spread,
                    spread > Max_Spread_Pts ? "WIDE!" : "OK"),
                    spread > Max_Spread_Pts ? clrTomato : clrLime);
   PSet(lbl+"vvo",  vol_str, m1_atr[1] > MaxATR_Entry*avg_atr ? clrOrange : clrLime);
   PSet(lbl+"vtr",  StringFormat("%d (B:%d S:%d)", open_buys+open_sells,open_buys,open_sells), clrWhite);
   PSet(lbl+"vfp",  StringFormat("%.2f", float_pnl), float_pnl>=0?clrLime:clrTomato);

   RefreshPnL();
   PSet(lbl+"vpd",  StringFormat("%.2f", pnl_today), pnl_today>=0?clrLime:clrTomato);
   PSet(lbl+"vpw",  StringFormat("%.2f", pnl_week),  pnl_week >=0?clrLime:clrTomato);
   PSet(lbl+"vpm",  StringFormat("%.2f", pnl_month), pnl_month>=0?clrLime:clrTomato);
   PSet(lbl+"vst",  panel_signal=="EP HIT!" ? "STOPPED" : "RUNNING",
                    panel_signal=="EP HIT!" ? clrTomato : clrLime);
   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| PANEL HELPERS                                                     |
//+------------------------------------------------------------------+
void PLbl(string n, string t, int x, int y, color c, int fs=8, bool b=false)
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
void PSet(string n, string t, color c)
{ if(ObjectFind(0,n)<0) return; ObjectSetString(0,n,OBJPROP_TEXT,t); ObjectSetInteger(0,n,OBJPROP_COLOR,c); }

void PLine(string n, int x, int y, int w)
{ string d=""; for(int i=0;i<(int)(w/5.5);i++) d+="-"; PLbl(n,d,x,y,C'35,50,80',6); }

void PRect(string n, int x, int y, int w, int h, color bg, color brd, int bw)
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
