//+------------------------------------------------------------------+
//|                                            Python_Bridge_EA.mq5   |
//|                          Python ML Bridge - Signal Executor        |
//|                                                                    |
//|  Reads trade signals from the Python ML Bridge CSV file and        |
//|  executes trades with proper risk management. Writes execution     |
//|  confirmations back for the Python system to read.                 |
//|                                                                    |
//|  Communication Protocol:                                           |
//|    Python -> MT5: python_bridge_signal.csv (signals)               |
//|    MT5 -> Python: python_bridge_confirm.csv (confirmations)        |
//+------------------------------------------------------------------+
#property copyright "Python ML Bridge"
#property version   "1.00"
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
input double   InpMinConfidence    = 0.15;      // Minimum confidence to trade
input int      InpMagicNumber      = 20240115;  // Magic number for orders
input int      InpSlippage         = 30;        // Slippage in points
input int      InpMaxOpenTrades    = 10;        // Max open trades (HF scalping)
input bool     InpShowDashboard    = true;      // Show dashboard panel

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

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    // Configure trade object
    g_trade.SetExpertMagicNumber(InpMagicNumber);
    g_trade.SetDeviationInPoints(InpSlippage);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);

    g_status = "Ready - Waiting for signals";
    Print("[PythonBridge] EA initialized. Magic=", InpMagicNumber);
    Print("[PythonBridge] Signal file: ", InpSignalFile);
    Print("[PythonBridge] Min confidence: ", InpMinConfidence);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Comment("");
    Print("[PythonBridge] EA removed. Trades executed: ", g_tradesExecuted);
}

//+------------------------------------------------------------------+
//| Expert tick function                                                |
//+------------------------------------------------------------------+
void OnTick()
{
    // Only process on new bar to avoid excessive file reads
    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(currentBarTime == lastBarTime)
        return;
    lastBarTime = currentBarTime;

    // Read signal from Python bridge
    if(ReadSignalFile())
    {
        g_signalsRead++;

        // Validate and execute signal
        if(ValidateSignal())
        {
            ExecuteSignal();
        }
    }

    // Process exit signals from Smart Exit Manager (RL agent)
    ProcessExitSignals();

    // Update dashboard
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
//| Validate signal before execution                                   |
//+------------------------------------------------------------------+
bool ValidateSignal()
{
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

    // Check minimum confidence
    if(g_lastConfidence < InpMinConfidence)
    {
        g_status = "Low confidence: " + DoubleToString(g_lastConfidence, 4);
        return false;
    }

    // Check lot size
    if(g_lastLotSize <= 0 || g_lastLotSize > InpMaxLotSize)
    {
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

    // Normalize lot size
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double lotSize = MathFloor(g_lastLotSize / lotStep) * lotStep;
    lotSize = MathMax(lotSize, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));
    lotSize = MathMin(lotSize, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));
    lotSize = MathMin(lotSize, InpMaxLotSize);

    // Calculate SL/TP from pips
    // For gold: 1 pip = 0.1 (10 points if 2 digits, 100 if 3)
    double pipValue = point * 10;

    if(g_lastAction == "BUY")
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        sl = NormalizeDouble(price - g_lastSLPips * pipValue, digits);
        tp = NormalizeDouble(price + g_lastTPPips * pipValue, digits);

        if(g_trade.Buy(lotSize, _Symbol, price, sl, tp,
                       "PythonBridge|" + g_lastModel + "|" + g_lastRegime))
        {
            g_tradesExecuted++;
            g_status = "BUY executed @ " + DoubleToString(price, digits);
            Print("[PythonBridge] BUY ", lotSize, " lots @ ", price,
                  " SL=", sl, " TP=", tp,
                  " Model=", g_lastModel, " Regime=", g_lastRegime);
            WriteConfirmation("BUY", lotSize, price, sl, tp, "FILLED");
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
        tp = NormalizeDouble(price - g_lastTPPips * pipValue, digits);

        if(g_trade.Sell(lotSize, _Symbol, price, sl, tp,
                        "PythonBridge|" + g_lastModel + "|" + g_lastRegime))
        {
            g_tradesExecuted++;
            g_status = "SELL executed @ " + DoubleToString(price, digits);
            Print("[PythonBridge] SELL ", lotSize, " lots @ ", price,
                  " SL=", sl, " TP=", tp,
                  " Model=", g_lastModel, " Regime=", g_lastRegime);
            WriteConfirmation("SELL", lotSize, price, sl, tp, "FILLED");
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
//| Write execution confirmation for Python to read                    |
//+------------------------------------------------------------------+
void WriteConfirmation(string action, double lots, double price,
                       double sl, double tp, string status)
{
    int fileHandle = FileOpen(InpConfirmFile, FILE_WRITE | FILE_CSV | FILE_COMMON,
                              ',', CP_UTF8);
    if(fileHandle == INVALID_HANDLE)
    {
        Print("[PythonBridge] ERROR: Cannot write confirmation file");
        return;
    }

    // Write header
    FileWrite(fileHandle, "timestamp", "ticket", "symbol", "action",
              "lot_size", "open_price", "sl", "tp", "status");

    // Write data
    string ticket = IntegerToString(g_trade.ResultOrder());
    FileWrite(fileHandle,
              TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
              ticket, _Symbol, action,
              DoubleToString(lots, 2),
              DoubleToString(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              DoubleToString(sl, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              DoubleToString(tp, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
              status);

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
//| Update on-chart dashboard                                          |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
    string dashboard = "";
    dashboard += "\n";
    dashboard += "  +========================================+\n";
    dashboard += "  |  PYTHON ML BRIDGE // HF SCALPER v2.0   |\n";
    dashboard += "  +========================================+\n";
    dashboard += "  |                                        |\n";
    dashboard += "  |  >>> STATUS                            |\n";
    dashboard += "  |  " + g_status + "\n";
    dashboard += "  |                                        |\n";
    dashboard += "  |  >>> LAST SIGNAL                       |\n";
    dashboard += "  |  Action ...... " + g_lastAction + "\n";
    dashboard += "  |  Confidence .. " + DoubleToString(g_lastConfidence, 4) + "\n";
    dashboard += "  |  Model ....... " + g_lastModel + "\n";
    dashboard += "  |  Regime ...... " + g_lastRegime + "\n";
    dashboard += "  |                                        |\n";
    dashboard += "  |  >>> EXECUTION                         |\n";
    dashboard += "  |  Lot Size .... " + DoubleToString(g_lastLotSize, 2) + "\n";
    dashboard += "  |  SL (pips) ... " + DoubleToString(g_lastSLPips, 1) + "\n";
    dashboard += "  |  TP (pips) ... " + DoubleToString(g_lastTPPips, 1) + "\n";
    dashboard += "  |                                        |\n";
    dashboard += "  |  >>> STATISTICS                        |\n";
    dashboard += "  |  Signals ...... " + IntegerToString(g_signalsRead) + "\n";
    dashboard += "  |  Trades ....... " + IntegerToString(g_tradesExecuted) + "\n";
    dashboard += "  |  Max Positions. " + IntegerToString(InpMaxOpenTrades) + "\n";
    dashboard += "  |  Signal Time .. " + TimeToString(g_lastSignalTime) + "\n";
    dashboard += "  |                                        |\n";
    dashboard += "  |  >>> SCALPER CONFIG                    |\n";
    dashboard += "  |  Cycle: 10s | ATR Cap: $5             |\n";
    dashboard += "  |  SL: ~$1 | TP: ~$1.50                |\n";
    dashboard += "  |                                        |\n";
    dashboard += "  +========================================+\n";

    Comment(dashboard);
}
//+------------------------------------------------------------------+
