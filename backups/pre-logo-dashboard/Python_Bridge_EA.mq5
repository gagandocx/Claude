//+------------------------------------------------------------------+
//|                                            Python_Bridge_EA.mq5   |
//|                              NeuroX - Signal Executor              |
//|                                                                    |
//|  v7.1 - Ultimate: $0.6 SL, $0.05 Trail, 61.6% WR, PF 4.02      |
//|                                                                    |
//|  Reads trade signals from the NeuroX CSV file and                  |
//|  executes trades with proper risk management. Writes execution     |
//|  confirmations back for the Python system to read.                 |
//|                                                                    |
//|  Communication Protocol:                                           |
//|    Python -> MT5: python_bridge_signal.csv (signals)               |
//|    MT5 -> Python: python_bridge_confirm.csv (confirmations)        |
//+------------------------------------------------------------------+
#property copyright "NeuroX"
#property version   "7.10"
// v7.1 - Ultimate: $0.6 SL, $0.05 Trail, 61.6% WR, PF 4.02
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                   |
//+------------------------------------------------------------------+
input string   InpSignalFile       = "python_bridge_signal.csv";   // Signal file name
input string   InpConfirmFile      = "python_bridge_confirm.csv";  // Confirmation file name
input string   InpExitFile         = "python_bridge_exit.csv";     // Exit signal file name
input string   InpHeartbeatFile    = "python_bridge_heartbeat.txt"; // Heartbeat file name
input int      InpMaxSignalAge     = 300;       // Max signal age (seconds)
input int      InpMaxHeartbeatAge  = 60;        // Max heartbeat age (seconds)
input double   InpMaxLotSize       = 1.0;       // Maximum lot size
input double   InpMinConfidence    = 0.10;      // Minimum confidence to trade
input int      InpMagicNumber      = 20240115;  // Magic number for orders
input int      InpSlippage         = 30;        // Slippage in points
input int      InpMaxOpenTrades    = 5;        // Max open trades (HF scalping)
input bool     InpShowDashboard    = true;      // Show dashboard panel
input string   InpStatusFile       = "python_bridge_status.txt"; // Status file name

// --- Dynamic Trailing Stop Parameters ---
input double   InpBreakevenProfit  = 0.70;     // Profit $ to move SL to breakeven
input double   InpBEProfitBuffer   = 0.02;     // Extra $ above entry for BE SL (covers spread + small profit)
input double   InpTrailStart       = 0.70;     // Profit $ to start trailing
input double   InpTrailTight       = 0.20;     // Profit $ for tight trail ($0.05 trail)
input double   InpTrailVeryTight   = 0.40;     // Profit $ for very tight trail ($0.02 trail)
input int      InpMomentumLookback = 30;       // Momentum lookback (seconds)
input int      InpMaxHoldNoProfit  = 1200;     // Max hold without profit (seconds = 20 min)
input double   InpMinProfitTarget  = 1.00;     // Min profit target $ to keep position open
input double   InpMomentumReverse  = 0.50;     // $ reversal threshold to close on momentum fade

// --- Trade Execution Settings (editable for live experimentation) ---
input double   InpMinLotSize       = 0.01;     // Minimum lot size
input double   InpDefaultLotSize   = 0.01;     // Default lot size (if signal lot is 0)
input double   InpFixedSL          = 2.00;     // Fixed SL in $ (0=use signal's SL)
input double   InpTrailDist1       = 0.50;     // Trail distance at Tier 2 ($)
input double   InpTrailDist2       = 0.30;     // Trail distance at Tier 3 ($)
input double   InpTrailDist3       = 0.20;     // Trail distance at Tier 4 ($)

//+------------------------------------------------------------------+
//| Global Variables                                                    |
//+------------------------------------------------------------------+
CTrade         g_trade;
CPositionInfo  g_position;
CAccountInfo   g_account;

// Last signal data
string         g_lastAction     = "HOLD";
double         g_lastConfidence = 0.0;
double         g_lastSLPips     = 0.0;
double         g_lastTPPips     = 0.0;
double         g_lastLotSize    = 0.0;
string         g_lastModel      = "";
string         g_lastRegime     = "";
datetime       g_lastSignalTime = 0;
int            g_signalsRead    = 0;
int            g_tradesExecuted = 0;
string         g_status         = "Initializing...";

// Python bridge status (news warnings, errors, running state)
string         g_statusType     = "OK";
string         g_newsWarning    = "";

// Duplicate signal execution guard: prevents OnTimer and OnTick from
// both executing the same signal within one bar.
datetime       g_lastExecutedSignalTime = 0;

// Emergency close-all cooldown (5 seconds between triggers)
datetime       g_lastEmergencyClose = 0;

// ── Brain settings (overrides input parameters dynamically) ──────────────────
// Updated every cycle by TradingBrain via python_bridge_brain_settings.csv.
// When brain_x > 0, it REPLACES the corresponding input parameter.
// When brain_x == 0, the original input parameter is used as fallback.
double   g_brain_sl           = 0;     // replaces InpFixedSL (0=use input)
double   g_brain_min_conf     = 0;     // replaces InpMinConfidence (0=use input)
double   g_brain_lot_mult     = 1.0;   // multiplied onto signal lot size
double   g_brain_be_profit    = 0;     // replaces InpBreakevenProfit (0=use input)
double   g_brain_trail_start  = 0;     // replaces InpTrailStart (0=use input)
int      g_brain_session_active = 1;   // 0 = brain says pause trading this session
datetime g_brain_last_read    = 0;     // last time brain settings were loaded

// ── Python bridge connection tracking ────────────────────────────────────────
// Updated every second in OnTimer so OnTick/UpdateDashboard never touch a file.
// -1 = heartbeat file never seen, 0+ = seconds since last Python heartbeat write.
int      g_pyHeartbeatAge    = -1;


// --- Dynamic Trailing SL: Position tracking ---
// Store entry time for each position (indexed by ticket)
#define MAX_TRACKED_POSITIONS 20
ulong          g_trackedTickets[MAX_TRACKED_POSITIONS];
datetime       g_trackedEntryTimes[MAX_TRACKED_POSITIONS];
double         g_trackedEntryPrices[MAX_TRACKED_POSITIONS];
double         g_trackedVolumes[MAX_TRACKED_POSITIONS];
int            g_trackedCount = 0;

// Momentum price snapshot for direction detection
double         g_momentumPrice = 0.0;
datetime       g_momentumTime  = 0;

// Trailing status for dashboard display
string         g_trailStatus   = "No positions";

//+------------------------------------------------------------------+
//| Read Brain settings from Python TradingBrain                      |
//| File: python_bridge_brain_settings.csv                            |
//| Called every second — ultra-low overhead (tiny file, no allocs)  |
//+------------------------------------------------------------------+
void ReadBrainSettings()
{
    int fh = FileOpen("python_bridge_brain_settings.csv",
                      FILE_READ | FILE_CSV | FILE_COMMON | FILE_ANSI, ',');
    if(fh == INVALID_HANDLE)
        return;   // Brain hasn't written settings yet — keep defaults

    // Skip header row: "parameter,value"
    if(!FileIsEnding(fh))
    {
        FileReadString(fh);  // "parameter"
        FileReadString(fh);  // "value"
    }

    while(!FileIsEnding(fh))
    {
        string param = FileReadString(fh);
        string val   = FileReadString(fh);
        double v     = StringToDouble(val);

        if     (param == "sl_dollars")      g_brain_sl             = v;
        else if(param == "min_confidence")  g_brain_min_conf       = v;
        else if(param == "lot_multiplier")  g_brain_lot_mult       = v;
        else if(param == "be_profit")       g_brain_be_profit      = v;
        else if(param == "trail_start")     g_brain_trail_start    = v;
        else if(param == "session_active")  g_brain_session_active = (int)v;
    }
    FileClose(fh);
    g_brain_last_read = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    // Configure trade object
    g_trade.SetExpertMagicNumber(InpMagicNumber);
    g_trade.SetDeviationInPoints(InpSlippage);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);

    // Set up a 1-second timer as backup signal reader.
    // Primary signal reading happens on every tick (with 1s throttle).
    // The timer provides a safety net in case tick flow stalls.
    EventSetTimer(1);

    // Dashboard panel uses OBJ_RECTANGLE_LABEL which renders on top by default
    // No CHART_FOREGROUND manipulation needed - trades display normally on chart

    g_status = "Ready - Waiting for signals";
    Print("[PythonBridge] EA initialized. Magic=", InpMagicNumber);
    Print("[PythonBridge] Signal file: ", InpSignalFile);
    Print("[PythonBridge] Min confidence: ", InpMinConfidence);
    Print("[PythonBridge] Timer: 1s backup interval for signal reading");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Kill the 1-second timer
    EventKillTimer();

    // Remove all dashboard graphical objects
    ObjectsDeleteAll(0, "PB_");
    Comment("");

    ChartRedraw(0);
    Print("[PythonBridge] EA removed. Trades executed: ", g_tradesExecuted);
}

//+------------------------------------------------------------------+
//| Timer function - 1-second backup signal reading                    |
//| Provides a safety net when tick flow stalls                        |
//+------------------------------------------------------------------+
void OnTimer()
{
    // Read signal from Python bridge (mid-bar check)
    if(ReadSignalFile())
    {
        g_signalsRead++;

        // Validate and execute signal
        if(ValidateSignal())
        {
            ExecuteSignal();
        }
    }

    // Load latest brain-computed settings (tiny file read, negligible overhead)
    ReadBrainSettings();

    // Process exit signals from Smart Exit Manager (RL agent)
    ProcessExitSignals();

    // Read Python bridge status for dashboard news/warning display
    ReadStatusFile();

    // ── MT5 heartbeat for Python connection detection ──────────────────
    // Python reads this file every cycle to show [MT5 CONNECTED/DISCONNECTED].
    // Written every 1s for instant connection status updates.
    static datetime g_lastMT5HB = 0;
    if(TimeCurrent() - g_lastMT5HB >= 1)
    {
        g_lastMT5HB = TimeCurrent();
        int hbFile = FileOpen("mt5_bridge_heartbeat.txt",
                              FILE_WRITE | FILE_TXT | FILE_COMMON);
        if(hbFile != INVALID_HANDLE)
        {
            FileWriteString(hbFile,
                TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\n");
            FileClose(hbFile);
        }
    }

    // ── Cache Python heartbeat age for tick-safe dashboard reads ───────
    // OnTimer runs every 1 s — we do the file open here once per second
    // so UpdateDashboard (called every tick) reads g_pyHeartbeatAge with
    // zero file I/O and therefore no lag or stutter on fast markets.
    {
        int pyHB = FileOpen(InpHeartbeatFile, FILE_READ | FILE_TXT | FILE_COMMON);
        if(pyHB == INVALID_HANDLE)
        {
            g_pyHeartbeatAge = -1;   // file never written — Python not started
        }
        else
        {
            datetime modTime = (datetime)FileGetInteger(pyHB, FILE_MODIFY_DATE);
            FileClose(pyHB);
            // FILE_MODIFY_DATE reflects local filesystem clock; TimeLocal() is the
            // same clock — using TimeCurrent() (broker time) caused a multi-hour offset.
            int age = (int)(TimeLocal() - modTime);
            g_pyHeartbeatAge = (age < 0) ? 0 : age;
        }
    }

    // Update dashboard
    if(InpShowDashboard)
        UpdateDashboard();
}

//+------------------------------------------------------------------+
//| Expert tick function                                                |
//+------------------------------------------------------------------+
void OnTick()
{
    // --- Dynamic position management runs on EVERY tick ---
    ManageOpenPositions();

    // --- Signal reading every 1 second for instant execution ---
    static datetime lastSignalCheck = 0;
    datetime now = TimeCurrent();
    if(now > lastSignalCheck)
    {
        lastSignalCheck = now;

        if(ReadSignalFile())
        {
            g_signalsRead++;
            if(ValidateSignal())
            {
                ExecuteSignal();
            }
        }

        ProcessExitSignals();
        CheckEmergencyCloseAll();
        ReadStatusFile();
    }

    if(InpShowDashboard)
        UpdateDashboard();
}

//+------------------------------------------------------------------+
//| Read signal from CSV file                                          |
//+------------------------------------------------------------------+
bool ReadSignalFile()
{
    // Open signal file from Common Files folder
    int fileHandle = FileOpen(InpSignalFile, FILE_READ | FILE_CSV | FILE_COMMON | FILE_ANSI,
                              ',');
    if(fileHandle == INVALID_HANDLE)
    {
        // File not found - no signal available
        return false;
    }

    // Skip header row - In FILE_CSV mode, FileReadString() reads ONE FIELD at a time
    // Header has 9 fields: timestamp,symbol,action,confidence,sl_pips,tp_pips,lot_size,model_name,regime
    // We must read all 9 fields to fully skip the header row
    if(!FileIsEnding(fileHandle))
    {
        string h1 = FileReadString(fileHandle);  // timestamp
        string h2 = FileReadString(fileHandle);  // symbol
        string h3 = FileReadString(fileHandle);  // action
        string h4 = FileReadString(fileHandle);  // confidence
        string h5 = FileReadString(fileHandle);  // sl_pips
        string h6 = FileReadString(fileHandle);  // tp_pips
        string h7 = FileReadString(fileHandle);  // lot_size
        string h8 = FileReadString(fileHandle);  // model_name
        string h9 = FileReadString(fileHandle);  // regime
        Print("[PythonBridge] CSV header: ", h1, ",", h2, ",", h3, ",", h4, ",",
              h5, ",", h6, ",", h7, ",", h8, ",", h9);
    }

    // Read data row - each FileReadString() call reads one comma-separated field
    if(!FileIsEnding(fileHandle))
    {
        string timestamp   = FileReadString(fileHandle);
        string symbol      = FileReadString(fileHandle);
        string action      = FileReadString(fileHandle);
        string confidence  = FileReadString(fileHandle);
        string slPips      = FileReadString(fileHandle);
        string tpPips      = FileReadString(fileHandle);
        string lotSize     = FileReadString(fileHandle);
        string modelName   = FileReadString(fileHandle);
        string regime      = FileReadString(fileHandle);

        // Debug: Print all raw field values read from CSV
        Print("[PythonBridge] CSV raw fields - timestamp=", timestamp,
              " symbol=", symbol, " action=", action,
              " confidence=", confidence, " slPips=", slPips,
              " tpPips=", tpPips, " lotSize=", lotSize,
              " model=", modelName, " regime=", regime);

        // Parse values
        g_lastAction     = action;
        g_lastConfidence = StringToDouble(confidence);
        g_lastSLPips     = StringToDouble(slPips);
        g_lastTPPips     = StringToDouble(tpPips);
        g_lastLotSize    = StringToDouble(lotSize);
        g_lastModel      = modelName;
        g_lastRegime     = regime;
        g_lastSignalTime = StringToTime(timestamp);

        // Debug: Print parsed values
        Print("[PythonBridge] Parsed signal - Action=", g_lastAction,
              " Confidence=", DoubleToString(g_lastConfidence, 4),
              " SL=", DoubleToString(g_lastSLPips, 1),
              " TP=", DoubleToString(g_lastTPPips, 1),
              " Lots=", DoubleToString(g_lastLotSize, 2),
              " Model=", g_lastModel,
              " Regime=", g_lastRegime,
              " Time=", TimeToString(g_lastSignalTime, TIME_DATE | TIME_SECONDS));
    }
    else
    {
        Print("[PythonBridge] WARNING: CSV file has no data row after header");
        FileClose(fileHandle);
        return false;
    }

    FileClose(fileHandle);
    return true;
}

//+------------------------------------------------------------------+
//| Check if the Python bridge is alive via heartbeat file             |
//+------------------------------------------------------------------+
bool IsBridgeAlive()
{
    // Open heartbeat file to check its modification time
    int fileHandle = FileOpen(InpHeartbeatFile, FILE_READ | FILE_TXT | FILE_COMMON);
    if(fileHandle == INVALID_HANDLE)
    {
        // No heartbeat file means bridge has never run or file was deleted
        return false;
    }

    // Get the file's properties to determine age
    // FileGetInteger with FILE_MODIFY_DATE returns the last modification datetime
    datetime modTime = (datetime)FileGetInteger(fileHandle, FILE_MODIFY_DATE);
    FileClose(fileHandle);

    // Check age of heartbeat file
    datetime currentTime = TimeCurrent();
    if(currentTime - modTime > InpMaxHeartbeatAge)
    {
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Read status file from Python bridge for dashboard display          |
//| Format: "type|message\r\n"                                         |
//| type: OK, NEWS, WARNING, ERROR                                     |
//+------------------------------------------------------------------+
void ReadStatusFile()
{
    int fileHandle = FileOpen(InpStatusFile, FILE_READ | FILE_TXT | FILE_COMMON | FILE_ANSI);
    if(fileHandle == INVALID_HANDLE)
    {
        // No status file - bridge may not have written one yet
        return;
    }

    // Read the single line: "type|message"
    if(!FileIsEnding(fileHandle))
    {
        string line = FileReadString(fileHandle);
        FileClose(fileHandle);

        if(StringLen(line) == 0)
            return;

        // Parse "type|message" format
        int separatorPos = StringFind(line, "|");
        if(separatorPos > 0)
        {
            g_statusType = StringSubstr(line, 0, separatorPos);
            g_newsWarning = StringSubstr(line, separatorPos + 1);
        }
        else
        {
            // No separator found - treat entire line as message with OK type
            g_statusType = "OK";
            g_newsWarning = line;
        }
    }
    else
    {
        FileClose(fileHandle);
    }
}

//+------------------------------------------------------------------+
//| Validate signal before execution                                   |
//+------------------------------------------------------------------+
bool ValidateSignal()
{
    // Dedup guard: skip if this signal was already executed.
    // Prevents OnTimer and OnTick from both executing the same signal
    // within one bar window.
    if(g_lastSignalTime == g_lastExecutedSignalTime)
    {
        g_status = "Signal already executed at " + TimeToString(g_lastSignalTime, TIME_MINUTES);
        return false;
    }

    // Check action is BUY or SELL
    if(g_lastAction != "BUY" && g_lastAction != "SELL")
    {
        g_status = "Signal: HOLD - No action needed";
        return false;
    }

    // NOTE: Heartbeat check DISABLED - IsBridgeAlive() uses TimeCurrent() (broker server time)
    // but FileGetInteger(FILE_MODIFY_DATE) returns LOCAL system time. When broker timezone
    // differs from local timezone, the difference is always thousands of seconds, causing
    // the bridge to always appear "stale". The signal file itself validates freshness below.
    // if(!IsBridgeAlive())
    // {
    //     g_status = "Bridge offline (heartbeat stale > " + IntegerToString(InpMaxHeartbeatAge) + "s)";
    //     return false;
    // }

    // Check signal freshness using TimeLocal() since Python writes timestamps in LOCAL time
    datetime currentTime = TimeLocal();
    if(currentTime - g_lastSignalTime > InpMaxSignalAge)
    {
        g_status = "Signal expired (age > " + IntegerToString(InpMaxSignalAge) + "s)";
        return false;
    }

    // Check minimum confidence — brain dynamic threshold overrides static input
    double activeMinConf = (g_brain_min_conf > 0) ? g_brain_min_conf : InpMinConfidence;
    if(g_lastConfidence < activeMinConf)
    {
        g_status = "Low confidence: " + DoubleToString(g_lastConfidence, 4) +
                   " (min=" + DoubleToString(activeMinConf, 4) + ")";
        return false;
    }

    // Brain session pause — brain detected poor conditions, pausing all entries
    if(g_brain_session_active == 0)
    {
        g_status = "Brain: session paused (drawdown/poor conditions)";
        return false;
    }

    // Check lot size
    if(g_lastLotSize < 0 || g_lastLotSize > InpMaxLotSize)
    {
        // Allow 0 lot size - ExecuteSignal() will use InpDefaultLotSize
        g_status = "Invalid lot size: " + DoubleToString(g_lastLotSize, 2);
        return false;
    }

    // Check symbol matches
    if(StringFind(_Symbol, "XAU") < 0 && StringFind(_Symbol, "GOLD") < 0)
    {
        g_status = "Symbol mismatch - attach to XAUUSD chart";
        return false;
    }

    // Check if we already have a position from this EA
    int posCount = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(g_position.SelectByIndex(i))
        {
            if(g_position.Magic() == InpMagicNumber &&
               g_position.Symbol() == _Symbol)
            {
                posCount++;
            }
        }
    }
    if(posCount >= InpMaxOpenTrades)  // Max positions configurable for HF scalping
    {
        g_status = "Max positions reached (" + IntegerToString(posCount) + ")";
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Execute the validated signal                                        |
//+------------------------------------------------------------------+
void ExecuteSignal()
{
    double price = 0;
    double sl = 0;
    double tp = 0;
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    // Normalize lot size - apply InpDefaultLotSize if signal lot is 0 or invalid
    double rawLotSize = g_lastLotSize;
    if(rawLotSize <= 0)
        rawLotSize = InpDefaultLotSize;

    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double lotSize = MathFloor(rawLotSize / lotStep) * lotStep;
    lotSize = MathMax(lotSize, InpMinLotSize);
    lotSize = MathMax(lotSize, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));
    lotSize = MathMin(lotSize, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));
    lotSize = MathMin(lotSize, InpMaxLotSize);

    // Apply brain lot multiplier (brain sizes up/down based on edge+regime+session)
    if(g_brain_lot_mult > 0 && g_brain_lot_mult != 1.0)
    {
        lotSize = lotSize * g_brain_lot_mult;
        lotSize = MathFloor(lotSize / lotStep) * lotStep;
        lotSize = MathMax(lotSize, InpMinLotSize);
        lotSize = MathMin(lotSize, InpMaxLotSize);
    }

    // Calculate SL/TP from pips
    // For gold: 1 pip = 0.1 (10 points if 2 digits, 100 if 3)
    double pipValue = point * 10;

    // Dynamic trailing mode: if tp_pips >= 9990, set TP=0 (no fixed TP)
    // The EA ManageOpenPositions() will manage exits dynamically
    bool dynamicTPMode = (g_lastTPPips >= 9990);

    if(g_lastAction == "BUY")
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        sl = NormalizeDouble(price - g_lastSLPips * pipValue, digits);
        // Brain SL override: brain computes regime-optimal stop loss
        double activeSL = (g_brain_sl > 0) ? g_brain_sl : InpFixedSL;
        if(activeSL > 0)
            sl = NormalizeDouble(price - activeSL, digits);
        if(dynamicTPMode)
            tp = 0;  // No fixed TP - EA manages exit via trailing
        else
            tp = NormalizeDouble(price + g_lastTPPips * pipValue, digits);

        if(g_trade.Buy(lotSize, _Symbol, price, sl, tp,
                       "PythonBridge|" + g_lastModel + "|" + g_lastRegime))
        {
            g_tradesExecuted++;
            g_lastExecutedSignalTime = g_lastSignalTime;
            g_status = "BUY executed @ " + DoubleToString(price, digits);
            Print("[PythonBridge] BUY ", lotSize, " lots @ ", price,
                  " SL=", sl, " TP=", (dynamicTPMode ? "DYNAMIC" : DoubleToString(tp, digits)),
                  " Model=", g_lastModel, " Regime=", g_lastRegime);
            WriteConfirmation("BUY", lotSize, price, sl, tp, "FILLED");
            // Track position for dynamic trailing (use ResultDeal as position ticket)
            ulong posTicket = g_trade.ResultDeal();
            if(posTicket == 0) posTicket = g_trade.ResultOrder();
            TrackNewPosition(posTicket, price, lotSize);
        }
        else
        {
            g_status = "BUY FAILED: " + IntegerToString(g_trade.ResultRetcode());
            Print("[PythonBridge] BUY FAILED: ", g_trade.ResultRetcode(),
                  " - ", g_trade.ResultRetcodeDescription());
            WriteConfirmation("BUY", lotSize, price, sl, tp, "FAILED");
        }
    }
    else if(g_lastAction == "SELL")
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        sl = NormalizeDouble(price + g_lastSLPips * pipValue, digits);
        // Brain SL override: brain computes regime-optimal stop loss
        double activeSL_sell = (g_brain_sl > 0) ? g_brain_sl : InpFixedSL;
        if(activeSL_sell > 0)
            sl = NormalizeDouble(price + activeSL_sell, digits);
        if(dynamicTPMode)
            tp = 0;  // No fixed TP - EA manages exit via trailing
        else
            tp = NormalizeDouble(price - g_lastTPPips * pipValue, digits);

        if(g_trade.Sell(lotSize, _Symbol, price, sl, tp,
                        "PythonBridge|" + g_lastModel + "|" + g_lastRegime))
        {
            g_tradesExecuted++;
            g_lastExecutedSignalTime = g_lastSignalTime;
            g_status = "SELL executed @ " + DoubleToString(price, digits);
            Print("[PythonBridge] SELL ", lotSize, " lots @ ", price,
                  " SL=", sl, " TP=", (dynamicTPMode ? "DYNAMIC" : DoubleToString(tp, digits)),
                  " Model=", g_lastModel, " Regime=", g_lastRegime);
            WriteConfirmation("SELL", lotSize, price, sl, tp, "FILLED");
            // Track position for dynamic trailing (use ResultDeal as position ticket)
            ulong posTicket = g_trade.ResultDeal();
            if(posTicket == 0) posTicket = g_trade.ResultOrder();
            TrackNewPosition(posTicket, price, lotSize);
        }
        else
        {
            g_status = "SELL FAILED: " + IntegerToString(g_trade.ResultRetcode());
            Print("[PythonBridge] SELL FAILED: ", g_trade.ResultRetcode(),
                  " - ", g_trade.ResultRetcodeDescription());
            WriteConfirmation("SELL", lotSize, price, sl, tp, "FAILED");
        }
    }
}

//+------------------------------------------------------------------+
//| Track a new position for dynamic trailing management               |
//+------------------------------------------------------------------+
void TrackNewPosition(ulong ticket, double entryPrice, double volume = 0.01)
{
    if(g_trackedCount < MAX_TRACKED_POSITIONS)
    {
        g_trackedTickets[g_trackedCount] = ticket;
        g_trackedEntryTimes[g_trackedCount] = TimeCurrent();
        g_trackedEntryPrices[g_trackedCount] = entryPrice;
        g_trackedVolumes[g_trackedCount] = volume;
        g_trackedCount++;
        Print("[PythonBridge] Tracking position ticket=", ticket,
              " entry=", DoubleToString(entryPrice, 2),
              " vol=", DoubleToString(volume, 2));
    }
}

//+------------------------------------------------------------------+
//| Remove a tracked position (after close)                            |
//+------------------------------------------------------------------+
void UntrackPosition(ulong ticket)
{
    for(int i = 0; i < g_trackedCount; i++)
    {
        if(g_trackedTickets[i] == ticket)
        {
            // Shift remaining entries down
            for(int j = i; j < g_trackedCount - 1; j++)
            {
                g_trackedTickets[j] = g_trackedTickets[j + 1];
                g_trackedEntryTimes[j] = g_trackedEntryTimes[j + 1];
                g_trackedEntryPrices[j] = g_trackedEntryPrices[j + 1];
                g_trackedVolumes[j] = g_trackedVolumes[j + 1];
            }
            g_trackedCount--;
            return;
        }
    }
}

//+------------------------------------------------------------------+
//| Get entry time for a tracked position                              |
//+------------------------------------------------------------------+
datetime GetTrackedEntryTime(ulong ticket)
{
    for(int i = 0; i < g_trackedCount; i++)
    {
        if(g_trackedTickets[i] == ticket)
            return g_trackedEntryTimes[i];
    }
    return 0;
}

//+------------------------------------------------------------------+
//| Get entry price for a tracked position                             |
//+------------------------------------------------------------------+
double GetTrackedEntryPrice(ulong ticket)
{
    for(int i = 0; i < g_trackedCount; i++)
    {
        if(g_trackedTickets[i] == ticket)
            return g_trackedEntryPrices[i];
    }
    return 0;
}

//+------------------------------------------------------------------+
//| Manage open positions with dynamic trailing SL                     |
//| Called on EVERY TICK for real-time position management              |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double pipValue = point * 10;  // For gold: 1 pip = 0.1
    datetime currentTime = TimeCurrent();
    int positionsManaged = 0;
    string trailInfo = "";

    // --- 0. DETECT BROKER-CLOSED POSITIONS (SL/TP hit) ---
    // Loop through tracked tickets and check if they still exist as open positions.
    // If a tracked ticket is no longer open, the broker closed it (SL/TP hit).
    for(int t = g_trackedCount - 1; t >= 0; t--)
    {
        ulong trackedTicket = g_trackedTickets[t];
        if(!PositionSelectByTicket(trackedTicket))
        {
            // Position no longer exists - broker closed it (SL/TP hit)
            double entryPriceTracked = g_trackedEntryPrices[t];
            double trackedVol = g_trackedVolumes[t];

            // Look up the actual close deal from history for accurate price/profit
            double closePrice = 0.0;
            double closedProfit = 0.0;
            double closedVolume = trackedVol;
            string closedAction = "BUY";  // default, will be overridden

            // Search deal history for the close deal of this position
            if(HistorySelect(TimeCurrent() - 300, TimeCurrent()))  // last 5 minutes
            {
                for(int d = HistoryDealsTotal() - 1; d >= 0; d--)
                {
                    ulong dealTicket = HistoryDealGetTicket(d);
                    if(dealTicket == 0) continue;
                    // Match by position ID (the position ticket links to deals)
                    long dealPosId = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
                    if((ulong)dealPosId == trackedTicket &&
                       HistoryDealGetInteger(dealTicket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
                    {
                        closePrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
                        closedProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT)
                                     + HistoryDealGetDouble(dealTicket, DEAL_SWAP)
                                     + HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
                        closedVolume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
                        // Determine direction from the original position type
                        long dealType = HistoryDealGetInteger(dealTicket, DEAL_TYPE);
                        // DEAL_TYPE_SELL means closing a BUY position, DEAL_TYPE_BUY means closing a SELL
                        closedAction = (dealType == DEAL_TYPE_SELL) ? "BUY" : "SELL";
                        break;
                    }
                }
            }

            // Fallback if deal history lookup failed
            if(closePrice == 0.0)
            {
                closePrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                closedAction = (closePrice >= entryPriceTracked) ? "BUY" : "SELL";
            }

            Print("[PythonBridge] SL/TP HIT DETECTED: Ticket ", trackedTicket,
                  " Entry=", DoubleToString(entryPriceTracked, digits),
                  " Close=", DoubleToString(closePrice, digits),
                  " P&L=$", DoubleToString(closedProfit, 2),
                  " Vol=", DoubleToString(closedVolume, 2));

            // Write CLOSED confirmation with actual data from deal history
            WriteConfirmation(closedAction, closedVolume, closePrice, 0, 0, "CLOSED",
                              trackedTicket, closedProfit);
            UntrackPosition(trackedTicket);
        }
    }

    // Update momentum price snapshot every InpMomentumLookback seconds
    if(g_momentumTime == 0 || (currentTime - g_momentumTime) >= InpMomentumLookback)
    {
        g_momentumPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        g_momentumTime = currentTime;
    }

    // Loop through all positions with this EA's magic number
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!g_position.SelectByIndex(i))
            continue;

        if(g_position.Magic() != InpMagicNumber || g_position.Symbol() != _Symbol)
            continue;

        ulong ticket = g_position.Ticket();
        double entryPrice = g_position.PriceOpen();
        double currentSL = g_position.StopLoss();
        double currentTP = g_position.TakeProfit();
        double volume = g_position.Volume();
        double profit = g_position.Profit() + g_position.Swap() + g_position.Commission();
        ENUM_POSITION_TYPE posType = g_position.PositionType();

        // Get current market price for this position direction
        double currentPrice = (posType == POSITION_TYPE_BUY)
            ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
            : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        positionsManaged++;

        // --- 1. TIME-BASED EXIT: Close if held too long without profit ---
        datetime entryTime = GetTrackedEntryTime(ticket);
        if(entryTime == 0)
        {
            // Position opened before tracking started - register it now
            TrackNewPosition(ticket, entryPrice);
            entryTime = currentTime;
        }

        int holdSeconds = (int)(currentTime - entryTime);
        if(holdSeconds > InpMaxHoldNoProfit && profit < InpMinProfitTarget)
        {
            Print("[PythonBridge] TIME EXIT: Ticket ", ticket,
                  " held ", holdSeconds, "s with profit $",
                  DoubleToString(profit, 2), " < $", DoubleToString(InpMinProfitTarget, 2));
            g_trade.PositionClose(ticket);
            WriteConfirmation(posType == POSITION_TYPE_BUY ? "BUY" : "SELL", volume, currentPrice, currentSL, 0, "CLOSED", ticket, profit);
            UntrackPosition(ticket);
            g_trailStatus = "Time exit: " + IntegerToString(holdSeconds) + "s";
            continue;
        }

        // --- 2. MOMENTUM CHECK: Close if price stalling/reversing ---
        if(g_momentumPrice > 0 && holdSeconds > InpMomentumLookback)
        {
            double momentumDiff = currentPrice - g_momentumPrice;

            bool momentumFading = false;
            if(posType == POSITION_TYPE_BUY && momentumDiff < -InpMomentumReverse)
                momentumFading = true;
            else if(posType == POSITION_TYPE_SELL && momentumDiff > InpMomentumReverse)
                momentumFading = true;

            // Only close on momentum fade if position is in profit (preserve capital)
            if(momentumFading && profit > 0)
            {
                Print("[PythonBridge] MOMENTUM EXIT: Ticket ", ticket,
                      " momentum reversed $", DoubleToString(MathAbs(momentumDiff), 2),
                      " against position. Profit: $", DoubleToString(profit, 2));
                g_trade.PositionClose(ticket);
                WriteConfirmation(posType == POSITION_TYPE_BUY ? "BUY" : "SELL", volume, currentPrice, currentSL, 0, "CLOSED", ticket, profit);
                UntrackPosition(ticket);
                g_trailStatus = "Momentum exit: $" + DoubleToString(profit, 2);
                continue;
            }
        }

        // --- 3. PROGRESSIVE TRAILING STOP LOSS ---
        double newSL = currentSL;
        string tierLabel = "";

        // Brain-controlled thresholds — override input params when set
        double activeBeProfit   = (g_brain_be_profit   > 0) ? g_brain_be_profit   : InpBreakevenProfit;
        double activeTrailStart = (g_brain_trail_start > 0) ? g_brain_trail_start : InpTrailStart;

        if(profit >= InpTrailVeryTight)
        {
            // Tier 4: Very tight trail (InpTrailDist3 behind current price)
            double trailDist = InpTrailDist3;
            if(posType == POSITION_TYPE_BUY)
                newSL = NormalizeDouble(currentPrice - trailDist, digits);
            else
                newSL = NormalizeDouble(currentPrice + trailDist, digits);
            tierLabel = "T4($" + DoubleToString(InpTrailDist3, 2) + ")";
        }
        else if(profit >= InpTrailTight)
        {
            // Tier 3: Tight trail (InpTrailDist2 behind current price)
            double trailDist = InpTrailDist2;
            if(posType == POSITION_TYPE_BUY)
                newSL = NormalizeDouble(currentPrice - trailDist, digits);
            else
                newSL = NormalizeDouble(currentPrice + trailDist, digits);
            tierLabel = "T3($" + DoubleToString(InpTrailDist2, 2) + ")";
        }
        else if(profit >= activeTrailStart)
        {
            // Tier 2: Standard trail (InpTrailDist1 behind current price)
            double trailDist = InpTrailDist1;
            if(posType == POSITION_TYPE_BUY)
                newSL = NormalizeDouble(currentPrice - trailDist, digits);
            else
                newSL = NormalizeDouble(currentPrice + trailDist, digits);
            tierLabel = "T2($" + DoubleToString(InpTrailDist1, 2) + ")";
        }
        else if(profit >= activeBeProfit)
        {
            // Tier 1: Breakeven+ (move SL to entry price + spread + buffer)
            // This ensures a breakeven trade does not lose money to spread.
            // SL is placed a few pips in profit beyond entry so if it triggers,
            // the trader still nets a small gain after spread costs.
            // NOTE: SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) fetches the REAL-TIME
            // spread from the broker at the moment of execution, ensuring the
            // breakeven calculation accounts for current market conditions
            // (not a fixed/historical spread value).
            double spreadPoints = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
            double beBuffer = spreadPoints + InpBEProfitBuffer;
            if(posType == POSITION_TYPE_BUY)
                newSL = NormalizeDouble(entryPrice + beBuffer, digits);
            else
                newSL = NormalizeDouble(entryPrice - beBuffer, digits);
            tierLabel = "BE+";
        }
        else
        {
            tierLabel = "Init";
        }

        // Only move SL in favorable direction (never widen the stop)
        bool shouldModify = false;
        if(posType == POSITION_TYPE_BUY)
        {
            // For BUY: new SL must be HIGHER than current SL (tighter)
            if(newSL > currentSL && newSL != currentSL)
                shouldModify = true;
        }
        else
        {
            // For SELL: new SL must be LOWER than current SL (tighter)
            if(currentSL == 0 || (newSL < currentSL && newSL != currentSL))
                shouldModify = true;
        }

        if(shouldModify)
        {
            if(g_trade.PositionModify(ticket, newSL, currentTP))
            {
                Print("[PythonBridge] TRAIL ", tierLabel, ": Ticket ", ticket,
                      " SL moved to ", DoubleToString(newSL, digits),
                      " (profit $", DoubleToString(profit, 2), ")");
            }
        }

        // Build trail info for dashboard
        trailInfo = tierLabel + " P:" + DoubleToString(profit, 2);
    }

    // Update trail status for dashboard
    if(positionsManaged == 0)
        g_trailStatus = "No positions";
    else if(trailInfo != "")
        g_trailStatus = trailInfo;
    else
        g_trailStatus = IntegerToString(positionsManaged) + " pos managed";
}

//+------------------------------------------------------------------+
//| Write execution confirmation for Python to read                    |
//+------------------------------------------------------------------+
void WriteConfirmation(string action, double lots, double price,
                       double sl, double tp, string status,
                       ulong ticketNum = 0, double profit = 0.0)
{
    int fileHandle = FileOpen(InpConfirmFile, FILE_WRITE | FILE_CSV | FILE_COMMON,
                              ',', CP_UTF8);
    if(fileHandle == INVALID_HANDLE)
    {
        Print("[PythonBridge] ERROR: Cannot write confirmation file");
        return;
    }

    // Write header (includes profit field for accurate PnL reporting)
    FileWrite(fileHandle, "timestamp", "ticket", "symbol", "action",
              "lot_size", "open_price", "sl", "tp", "status", "profit");

    // Use provided ticket, or fall back to g_trade.ResultOrder() for FILLED
    string ticketStr;
    if(ticketNum > 0)
        ticketStr = IntegerToString(ticketNum);
    else
        ticketStr = IntegerToString(g_trade.ResultDeal());

    // Write data
    FileWrite(fileHandle,
              TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
              ticketStr, _Symbol, action,
              DoubleToString(lots, 2),
              DoubleToString(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              DoubleToString(sl, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              DoubleToString(tp, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              status,
              DoubleToString(profit, 2));

    FileClose(fileHandle);
}

//+------------------------------------------------------------------+
//| Process exit signals from Python Smart Exit Manager               |
//+------------------------------------------------------------------+
void ProcessExitSignals()
{
    int fileHandle = FileOpen(InpExitFile, FILE_READ | FILE_CSV | FILE_COMMON | FILE_ANSI, ',');
    if(fileHandle == INVALID_HANDLE)
        return;  // No exit signal file

    // Skip header row (6 fields: timestamp,ticket,action,lot_pct,new_sl,reason)
    if(!FileIsEnding(fileHandle))
    {
        for(int i = 0; i < 6; i++)
            FileReadString(fileHandle);
    }

    // Read exit signal rows
    while(!FileIsEnding(fileHandle))
    {
        string timestamp = FileReadString(fileHandle);
        string ticket    = FileReadString(fileHandle);
        string action    = FileReadString(fileHandle);
        string lotPct    = FileReadString(fileHandle);
        string newSL     = FileReadString(fileHandle);
        string reason    = FileReadString(fileHandle);

        if(StringLen(ticket) == 0)
            break;

        long ticketNum = StringToInteger(ticket);
        double lotPercent = StringToDouble(lotPct);
        double newStopLoss = StringToDouble(newSL);

        Print("[PythonBridge] Exit signal: ticket=", ticket,
              " action=", action, " lot_pct=", lotPct,
              " new_sl=", newSL, " reason=", reason);

        // Execute exit action
        if(action == "CLOSE_FULL")
        {
            ClosePosition(ticketNum, 1.0);
        }
        else if(action == "CLOSE_PARTIAL")
        {
            ClosePosition(ticketNum, lotPercent);
        }
        else if(action == "MODIFY_SL")
        {
            ModifyPositionSL(ticketNum, newStopLoss);
        }
    }

    FileClose(fileHandle);

    // Delete the exit file after processing
    // NOTE: Race condition exists here - between FileClose and FileDelete,
    // the Python bridge may write a new exit signal file that gets deleted
    // unread. A safer approach would be a read-then-rename pattern:
    //   1. Rename exit file to .processing
    //   2. Read from .processing
    //   3. Delete .processing
    // This ensures new signals written by Python are never lost.
    // Low priority: at M1 frequency this window is <1ms.
    FileDelete(InpExitFile, FILE_COMMON);
}

//+------------------------------------------------------------------+
//| Close a position (full or partial)                                 |
//+------------------------------------------------------------------+
void ClosePosition(long ticket, double lotPercent)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(g_position.SelectByIndex(i))
        {
            if(g_position.Magic() == InpMagicNumber &&
               g_position.Ticket() == (ulong)ticket)
            {
                double volume = g_position.Volume();
                double closeVolume = NormalizeDouble(volume * lotPercent,
                                     (int)MathLog10(1.0 / SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP)));

                // Ensure minimum lot size
                double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                closeVolume = MathMax(closeVolume, minLot);
                closeVolume = MathMin(closeVolume, volume);

                if(g_position.PositionType() == POSITION_TYPE_BUY)
                {
                    double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                    if(g_trade.Sell(closeVolume, _Symbol, price, 0, 0,
                                    "SmartExit|Close"))
                    {
                        Print("[PythonBridge] SmartExit: Closed ", closeVolume,
                              " lots of ticket ", ticket);
                    }
                }
                else
                {
                    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                    if(g_trade.Buy(closeVolume, _Symbol, price, 0, 0,
                                   "SmartExit|Close"))
                    {
                        Print("[PythonBridge] SmartExit: Closed ", closeVolume,
                              " lots of ticket ", ticket);
                    }
                }
                return;
            }
        }
    }
    Print("[PythonBridge] SmartExit: Ticket ", ticket, " not found");
}

//+------------------------------------------------------------------+
//| Modify stop loss for a position                                    |
//+------------------------------------------------------------------+
void ModifyPositionSL(long ticket, double newSL)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(g_position.SelectByIndex(i))
        {
            if(g_position.Magic() == InpMagicNumber &&
               g_position.Ticket() == (ulong)ticket)
            {
                int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
                double sl = NormalizeDouble(newSL, digits);
                double tp = g_position.TakeProfit();

                if(g_trade.PositionModify((ulong)ticket, sl, tp))
                {
                    Print("[PythonBridge] SmartExit: Modified SL to ",
                          DoubleToString(sl, digits), " for ticket ", ticket);
                }
                else
                {
                    Print("[PythonBridge] SmartExit: Failed to modify SL for ticket ",
                          ticket, " error=", g_trade.ResultRetcode());
                }
                return;
            }
        }
    }
    Print("[PythonBridge] SmartExit: Ticket ", ticket, " not found for SL modify");
}

//+------------------------------------------------------------------+
//| Emergency close all positions if floating loss exceeds $50        |
//+------------------------------------------------------------------+
void CheckEmergencyCloseAll()
{
    // Cooldown: skip if emergency close fired within the last 5 seconds
    if(TimeCurrent() - g_lastEmergencyClose < 5)
        return;

    double totalFloatingLoss = 0.0;

    // Calculate total floating P&L for this EA's positions
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(g_position.SelectByIndex(i))
        {
            if(g_position.Magic() == InpMagicNumber &&
               g_position.Symbol() == _Symbol)
            {
                totalFloatingLoss += g_position.Profit() + g_position.Swap() + g_position.Commission();
            }
        }
    }

    // If total floating loss exceeds $50, close all positions immediately
    if(totalFloatingLoss < -50.0)
    {
        Print("[PythonBridge] EMERGENCY: Floating loss $", DoubleToString(MathAbs(totalFloatingLoss), 2),
              " exceeds $50 limit. Closing ALL positions!");
        g_status = "EMERGENCY CLOSE: Loss > $50";
        g_lastEmergencyClose = TimeCurrent();

        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            if(g_position.SelectByIndex(i))
            {
                if(g_position.Magic() == InpMagicNumber &&
                   g_position.Symbol() == _Symbol)
                {
                    ulong ticket = g_position.Ticket();
                    double volume = g_position.Volume();

                    if(g_position.PositionType() == POSITION_TYPE_BUY)
                    {
                        double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                        g_trade.Sell(volume, _Symbol, price, 0, 0, "Emergency|CloseAll");
                    }
                    else
                    {
                        double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                        g_trade.Buy(volume, _Symbol, price, 0, 0, "Emergency|CloseAll");
                    }
                    Print("[PythonBridge] Emergency closed ticket ", ticket);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate floating P/L for this EA's positions                     |
//+------------------------------------------------------------------+
double CalculateFloatingPL()
{
    double totalPL = 0.0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(g_position.SelectByIndex(i))
        {
            if(g_position.Magic() == InpMagicNumber &&
               g_position.Symbol() == _Symbol)
            {
                totalPL += g_position.Profit() + g_position.Swap() + g_position.Commission();
            }
        }
    }
    return totalPL;
}

//+------------------------------------------------------------------+
//| Count open positions for this EA                                   |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(g_position.SelectByIndex(i))
        {
            if(g_position.Magic() == InpMagicNumber &&
               g_position.Symbol() == _Symbol)
            {
                count++;
            }
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Create or update a label object on the chart                       |
//+------------------------------------------------------------------+
void DashboardLabel(string name, int x, int y, string text, color clr, int fontSize = 9, string font = "Consolas")
{
    string objName = "PB_" + name;
    if(ObjectFind(0, objName) < 0)
    {
        ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
        ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        ObjectSetInteger(0, objName, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
        ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(0, objName, OBJPROP_HIDDEN, true);
    }
    ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y);
    ObjectSetString(0, objName, OBJPROP_TEXT, text);
    ObjectSetInteger(0, objName, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontSize);
    ObjectSetString(0, objName, OBJPROP_FONT, font);
}

//+------------------------------------------------------------------+
//| Create or update the background panel                              |
//+------------------------------------------------------------------+
void DashboardBackground(string name, int x, int y, int width, int height, color bgColor, int transparency)
{
    string objName = "PB_" + name;
    if(ObjectFind(0, objName) < 0)
    {
        ObjectCreate(0, objName, OBJ_RECTANGLE_LABEL, 0, 0, 0);
        ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(0, objName, OBJPROP_HIDDEN, true);
        ObjectSetInteger(0, objName, OBJPROP_BORDER_TYPE, BORDER_FLAT);
    }
    ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, y);
    ObjectSetInteger(0, objName, OBJPROP_XSIZE, width);
    ObjectSetInteger(0, objName, OBJPROP_YSIZE, height);
    ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, bgColor);
    ObjectSetInteger(0, objName, OBJPROP_COLOR, clrDimGray);
    ObjectSetInteger(0, objName, OBJPROP_WIDTH, 1);
    // Ensure rectangle renders in foreground layer (on top of chart elements)
    ObjectSetInteger(0, objName, OBJPROP_BACK, false);
    // Force to top z-order for full opacity
    ObjectSetInteger(0, objName, OBJPROP_ZORDER, 1000);
}

//+------------------------------------------------------------------+
//| Update on-chart dashboard with professional graphical panel        |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
    int panelX      = 10;
    int panelY      = 30;
    int panelWidth  = 330;
    int panelHeight = 550;   // expanded for CONNECTION section (auto-resizes at end)
    int lineHeight  = 18;
    int leftMargin  = 20;
    int valueCol    = 155;
    int y           = panelY + 12;

    // Colors
    color clrTitle      = clrGold;
    color clrHeader     = clrDeepSkyBlue;
    color clrLabel      = clrSilver;
    color clrValue      = clrWhite;
    color clrRunning    = clrLime;
    color clrWarning    = clrRed;
    color clrBgPanel    = C'20,20,30';
    color clrBgHeader   = C'30,35,50';

    // --- Background panels: multiple stacked layers for guaranteed full opacity ---
    // Layer 1 (bottom): base background
    DashboardBackground("bg_main", panelX, panelY, panelWidth, panelHeight, clrBgPanel, 255);
    // Layer 2 (middle): identical rectangle stacked on top for double opacity
    DashboardBackground("bg_main2", panelX, panelY, panelWidth, panelHeight, clrBgPanel, 255);
    // Layer 3 (top): third layer ensures absolutely no bleed-through
    DashboardBackground("bg_main3", panelX, panelY, panelWidth, panelHeight, clrBgPanel, 255);
    // Title bar background
    DashboardBackground("bg_title", panelX, panelY, panelWidth, 28, clrBgHeader, 255);

    // --- Title ---
    DashboardLabel("title", panelX + leftMargin, y, "NEUROX - HF SCALPER", clrTitle, 10, "Consolas Bold");
    y += 28;

    // --- Symbol & Timeframe ---
    DashboardLabel("sym_lbl", panelX + leftMargin, y, "Symbol:", clrLabel);
    DashboardLabel("sym_val", panelX + valueCol, y, _Symbol, clrValue);
    y += lineHeight;
    DashboardLabel("tf_lbl", panelX + leftMargin, y, "Timeframe:", clrLabel);
    DashboardLabel("tf_val", panelX + valueCol, y, EnumToString(Period()), clrValue);
    y += lineHeight + 6;

    // ═══════════════════════════════════════════════════════════════════
    // --- CONNECTION (live — updates every tick from cached globals) ---
    // ═══════════════════════════════════════════════════════════════════
    DashboardBackground("bg_conn", panelX + 8, y - 3, panelWidth - 16, 106, C'15,25,15', 255);
    DashboardLabel("sep_conn", panelX + leftMargin, y, "--- CONNECTION ---", clrHeader, 9);
    y += lineHeight;

    // ── Python Bridge: LIVE / OFFLINE ────────────────────────────────
    string pyConnStr;
    color  pyConnClr;
    if(g_pyHeartbeatAge < 0)
    {
        pyConnStr = "● OFFLINE  (Python not started)";
        pyConnClr = clrRed;
    }
    else if(g_pyHeartbeatAge <= InpMaxHeartbeatAge)
    {
        pyConnStr = "● LIVE    " + IntegerToString(g_pyHeartbeatAge) + "s ago";
        pyConnClr = clrLime;
    }
    else
    {
        pyConnStr = "● OFFLINE  " + IntegerToString(g_pyHeartbeatAge) + "s ago";
        pyConnClr = clrRed;
    }
    DashboardLabel("conn_py_lbl", panelX + leftMargin, y, "Python Bridge:", clrLabel);
    DashboardLabel("conn_py_val", panelX + valueCol,   y, pyConnStr, pyConnClr);
    y += lineHeight;

    // ── Last Signal age (live second counter) ────────────────────────
    // g_lastSignalTime is parsed by StringToTime() from Python's local-time
    // timestamp, so compare against TimeLocal() — NOT TimeCurrent() (broker
    // time) which caused the ~7-hour offset shown on the dashboard.
    int    sigAgeSec = (g_lastSignalTime > 0)
                       ? (int)(TimeLocal() - g_lastSignalTime) : -1;
    string sigAgeStr = (sigAgeSec >= 0)
                       ? IntegerToString(sigAgeSec) + "s ago" : "---";
    color  sigAgeClr = (sigAgeSec >= 0 && sigAgeSec < 5)  ? clrLime
                     : (sigAgeSec >= 0 && sigAgeSec < 30)  ? clrWhite
                     :                                        clrSilver;
    DashboardLabel("conn_sig_lbl", panelX + leftMargin, y, "Last Signal:", clrLabel);
    DashboardLabel("conn_sig_val", panelX + valueCol,   y, sigAgeStr, sigAgeClr);
    y += lineHeight;

    // ── Brain: SL override ───────────────────────────────────────────
    string brainSlStr  = (g_brain_sl > 0)
                         ? "$" + DoubleToString(g_brain_sl, 2) + "  (brain)"
                         : "input  ($" + DoubleToString(InpFixedSL, 2) + ")";
    color  brainSlClr  = (g_brain_sl > 0) ? clrGold : clrSilver;
    DashboardLabel("conn_sl_lbl", panelX + leftMargin, y, "Brain SL:", clrLabel);
    DashboardLabel("conn_sl_val", panelX + valueCol,   y, brainSlStr, brainSlClr);
    y += lineHeight;

    // ── Brain: Lot multiplier ────────────────────────────────────────
    string brainLotStr;
    color  brainLotClr;
    if(g_brain_lot_mult > 1.05)
        { brainLotStr = "x" + DoubleToString(g_brain_lot_mult,2) + "  HOT ▲";  brainLotClr = clrLime;   }
    else if(g_brain_lot_mult < 0.95)
        { brainLotStr = "x" + DoubleToString(g_brain_lot_mult,2) + "  COLD ▼"; brainLotClr = clrRed;    }
    else
        { brainLotStr = "x1.00  neutral";                                       brainLotClr = clrSilver; }
    DashboardLabel("conn_lot_lbl", panelX + leftMargin, y, "Brain Lot:", clrLabel);
    DashboardLabel("conn_lot_val", panelX + valueCol,   y, brainLotStr, brainLotClr);
    y += lineHeight;

    // ── Brain: State (ACTIVE / PAUSED) ───────────────────────────────
    string brainStateStr = (g_brain_session_active == 0) ? "● PAUSED  (drawdown/poor session)"
                                                         : "● ACTIVE";
    color  brainStateClr = (g_brain_session_active == 0) ? clrRed : clrLime;
    DashboardLabel("conn_bst_lbl", panelX + leftMargin, y, "Brain State:", clrLabel);
    DashboardLabel("conn_bst_val", panelX + valueCol,   y, brainStateStr, brainStateClr);
    y += lineHeight + 6;
    // ═══════════════════════════════════════════════════════════════════

    // --- Separator ---
    DashboardLabel("sep1", panelX + leftMargin, y, "--- SIGNAL ---", clrHeader, 9);
    y += lineHeight;

    // Signal section
    color actionClr = clrValue;
    if(g_lastAction == "BUY") actionClr = clrLime;
    else if(g_lastAction == "SELL") actionClr = clrRed;

    DashboardLabel("sig_act_lbl", panelX + leftMargin, y, "Action:", clrLabel);
    DashboardLabel("sig_act_val", panelX + valueCol, y, g_lastAction, actionClr);
    y += lineHeight;
    DashboardLabel("sig_conf_lbl", panelX + leftMargin, y, "Confidence:", clrLabel);
    DashboardLabel("sig_conf_val", panelX + valueCol, y, DoubleToString(g_lastConfidence * 100, 1) + "%", clrValue);
    y += lineHeight;
    DashboardLabel("sig_mod_lbl", panelX + leftMargin, y, "Model:", clrLabel);
    DashboardLabel("sig_mod_val", panelX + valueCol, y, (g_lastModel == "" ? "---" : g_lastModel), clrValue);
    y += lineHeight;
    DashboardLabel("sig_reg_lbl", panelX + leftMargin, y, "Regime:", clrLabel);
    DashboardLabel("sig_reg_val", panelX + valueCol, y, (g_lastRegime == "" ? "---" : g_lastRegime), clrValue);
    y += lineHeight + 6;

    // --- Trade section ---
    DashboardLabel("sep2", panelX + leftMargin, y, "--- TRADE ---", clrHeader, 9);
    y += lineHeight;

    DashboardLabel("trd_lot_lbl", panelX + leftMargin, y, "Lot Size:", clrLabel);
    DashboardLabel("trd_lot_val", panelX + valueCol, y, DoubleToString(g_lastLotSize, 2), clrValue);
    y += lineHeight;
    DashboardLabel("trd_sl_lbl", panelX + leftMargin, y, "SL (pips):", clrLabel);
    DashboardLabel("trd_sl_val", panelX + valueCol, y, DoubleToString(g_lastSLPips, 1), clrValue);
    y += lineHeight;
    DashboardLabel("trd_tp_lbl", panelX + leftMargin, y, "TP Mode:", clrLabel);
    string tpDisplay = (g_lastTPPips >= 9990) ? "Dynamic Trail" : DoubleToString(g_lastTPPips, 1) + " pips";
    DashboardLabel("trd_tp_val", panelX + valueCol, y, tpDisplay, (g_lastTPPips >= 9990) ? clrGold : clrValue);
    y += lineHeight;
    DashboardLabel("trd_trail_lbl", panelX + leftMargin, y, "Trail:", clrLabel);
    DashboardLabel("trd_trail_val", panelX + valueCol, y, g_trailStatus, clrGold);
    y += lineHeight + 6;

    // --- Statistics section ---
    DashboardLabel("sep3", panelX + leftMargin, y, "--- STATISTICS ---", clrHeader, 9);
    y += lineHeight;

    DashboardLabel("st_sig_lbl", panelX + leftMargin, y, "Signals Read:", clrLabel);
    DashboardLabel("st_sig_val", panelX + valueCol, y, IntegerToString(g_signalsRead), clrValue);
    y += lineHeight;
    DashboardLabel("st_trd_lbl", panelX + leftMargin, y, "Trades Exec:", clrLabel);
    DashboardLabel("st_trd_val", panelX + valueCol, y, IntegerToString(g_tradesExecuted), clrValue);
    y += lineHeight;

    int openPos = CountOpenPositions();
    DashboardLabel("st_pos_lbl", panelX + leftMargin, y, "Open Positions:", clrLabel);
    DashboardLabel("st_pos_val", panelX + valueCol, y, IntegerToString(openPos) + " / " + IntegerToString(InpMaxOpenTrades), clrValue);
    y += lineHeight;

    double floatingPL = CalculateFloatingPL();
    color plColor = (floatingPL >= 0) ? clrLime : clrRed;
    string plSign = (floatingPL >= 0) ? "+" : "";
    DashboardLabel("st_pl_lbl", panelX + leftMargin, y, "Floating P/L:", clrLabel);
    DashboardLabel("st_pl_val", panelX + valueCol, y, plSign + "$" + DoubleToString(floatingPL, 2), plColor);
    y += lineHeight + 6;

    // --- Scalper config ---
    DashboardLabel("sep4", panelX + leftMargin, y, "--- CONFIG ---", clrHeader, 9);
    y += lineHeight;

    DashboardLabel("cfg1_lbl", panelX + leftMargin, y, "Cycle:", clrLabel);
    DashboardLabel("cfg1_val", panelX + valueCol, y, "Every tick | ATR Cap: $5", clrValue);
    y += lineHeight;
    DashboardLabel("cfg2_lbl", panelX + leftMargin, y, "Exit:", clrLabel);
    DashboardLabel("cfg2_val", panelX + valueCol, y, "Dynamic Trail", clrGold);
    y += lineHeight;
    DashboardLabel("cfg3_lbl", panelX + leftMargin, y, "Emergency:", clrLabel);
    DashboardLabel("cfg3_val", panelX + valueCol, y, "$50 loss stop", clrWarning);
    y += lineHeight + 6;

    // --- Status line ---
    DashboardLabel("sep5", panelX + leftMargin, y, "--- STATUS ---", clrHeader, 9);
    y += lineHeight;

    color statusClr = clrRunning;
    if(StringFind(g_status, "EMERGENCY") >= 0 || StringFind(g_status, "ERROR") >= 0)
        statusClr = clrWarning;
    else if(StringFind(g_status, "Ready") >= 0 || StringFind(g_status, "Running") >= 0 || StringFind(g_status, "RUNNING") >= 0)
        statusClr = clrRunning;

    DashboardLabel("status_val", panelX + leftMargin, y, g_status, statusClr);
    y += lineHeight;

    // --- News/Warning section from Python bridge status file ---
    if(g_statusType == "NEWS")
    {
        y += 4;
        DashboardLabel("sep6", panelX + leftMargin, y, "--- NEWS FILTER ---", clrYellow, 9);
        y += lineHeight;
        // Truncate long messages for display
        string newsMsg = g_newsWarning;
        if(StringLen(newsMsg) > 38)
            newsMsg = StringSubstr(newsMsg, 0, 35) + "...";
        DashboardLabel("news_val", panelX + leftMargin, y, newsMsg, clrOrange);
        y += lineHeight;
        DashboardLabel("news_warn", panelX + leftMargin, y, "Trading paused - waiting for event", clrYellow);
        y += lineHeight;
    }
    else if(g_statusType == "WARNING" || g_statusType == "ERROR")
    {
        y += 4;
        DashboardLabel("sep6", panelX + leftMargin, y, "--- WARNING ---", clrRed, 9);
        y += lineHeight;
        string warnMsg = g_newsWarning;
        if(StringLen(warnMsg) > 38)
            warnMsg = StringSubstr(warnMsg, 0, 35) + "...";
        DashboardLabel("news_val", panelX + leftMargin, y, warnMsg, clrRed);
        y += lineHeight;
        // Clear the "Trading paused" line when not in NEWS mode
        DashboardLabel("news_warn", panelX + leftMargin, y, "", clrNONE);
    }
    else
    {
        // OK status - show in green, clear news labels
        DashboardLabel("sep6", panelX + leftMargin, y, "", clrNONE);
        DashboardLabel("news_val", panelX + leftMargin, y, "", clrNONE);
        DashboardLabel("news_warn", panelX + leftMargin, y, "", clrNONE);
    }

    // Signal time
    string sigTimeStr = (g_lastSignalTime > 0) ? TimeToString(g_lastSignalTime, TIME_DATE | TIME_MINUTES) : "---";
    DashboardLabel("sigtime_lbl", panelX + leftMargin, y, "Last Signal:", clrLabel);
    DashboardLabel("sigtime_val", panelX + valueCol, y, sigTimeStr, clrValue);

    // Adjust panel height dynamically
    int finalHeight = (y - panelY) + lineHeight + 10;
    if(finalHeight != panelHeight)
    {
        string bgName = "PB_bg_main";
        ObjectSetInteger(0, bgName, OBJPROP_YSIZE, finalHeight);
    }

    ChartRedraw(0);
}
//+------------------------------------------------------------------+
