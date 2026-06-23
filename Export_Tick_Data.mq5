//+------------------------------------------------------------------+
//|                                              Export_Tick_Data.mq5 |
//|                    Export M1 Bar Data with Spread Info            |
//|               For use with Python ML Bridge Backtester           |
//|                                                                  |
//| USAGE: Drop on any chart in MT5, set InpDays, run.               |
//| OUTPUT: CSV file in MT5 Common Files folder                      |
//|         Filename: bars_export_SYMBOL.csv                         |
//|         Columns: timestamp,open,high,low,close,volume,spread     |
//+------------------------------------------------------------------+
#property copyright "Python ML Bridge - Data Export"
#property version   "1.00"
#property script_show_inputs
#property strict

//+------------------------------------------------------------------+
//| INPUTS                                                            |
//+------------------------------------------------------------------+
input int InpDays = 30;   // Number of days to export

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   // Get symbol name
   string symbol = _Symbol;

   // Calculate number of bars to copy
   int bars_to_copy = InpDays * 24 * 60;  // M1 bars: 1440 per day

   // Copy M1 rates
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(symbol, PERIOD_M1, 0, bars_to_copy, rates);

   if(copied <= 0)
   {
      Print("ERROR: Failed to copy rates. Error=", GetLastError());
      return;
   }

   // Build filename
   string filename = "bars_export_" + symbol + ".csv";

   // Open file in Common Files folder
   int file_handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_COMMON | FILE_ANSI, ',');

   if(file_handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open file ", filename, ". Error=", GetLastError());
      return;
   }

   // Write header
   FileWrite(file_handle, "timestamp", "open", "high", "low", "close", "volume", "spread");

   // Write data (oldest first, so reverse the series order)
   for(int i = copied - 1; i >= 0; i--)
   {
      string ts = TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES | TIME_SECONDS);
      FileWrite(file_handle,
         ts,
         DoubleToString(rates[i].open, _Digits),
         DoubleToString(rates[i].high, _Digits),
         DoubleToString(rates[i].low, _Digits),
         DoubleToString(rates[i].close, _Digits),
         IntegerToString(rates[i].tick_volume),
         IntegerToString(rates[i].spread)
      );
   }

   // Close file
   FileClose(file_handle);

   // Get the common files path for the success message
   string common_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + filename;

   Print("SUCCESS: Exported ", copied, " M1 bars for ", symbol,
         " (", InpDays, " days) to: ", common_path);
   Print("Columns: timestamp,open,high,low,close,volume,spread");
   Print("Spread is in POINTS (for ", symbol, " with ", _Digits, " digits)");
}
//+------------------------------------------------------------------+
