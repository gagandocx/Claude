//+------------------------------------------------------------------+
//|                                              ExportRealTicks.mq5 |
//|           Export Real Tick Data from MT5 Tick Cache               |
//|                                                                  |
//| PURPOSE: Export the EXACT same tick data that MT5's Strategy      |
//|          Tester uses when "Every tick based on real ticks" is     |
//|          selected. Uses CopyTicksRange() which reads from the    |
//|          same .tkc files the backtester reads from.              |
//|                                                                  |
//| USAGE:                                                            |
//|  1. First run a backtest with "Every tick based on real ticks"   |
//|     to force MT5 to download/sync the tick history               |
//|  2. Then run this script on a chart of the target symbol         |
//|  3. Output CSV is saved to MT5's Files folder                    |
//|  4. Use analyze_ticks.py to process the exported data            |
//|                                                                  |
//| OUTPUT FORMAT (CSV):                                              |
//|  time_msc,bid,ask,last,volume,flags,volume_real                  |
//|                                                                  |
//| NOTE: Exports in daily chunks to avoid memory overflow when      |
//|       handling 47M+ ticks.                                       |
//+------------------------------------------------------------------+
#property copyright "GaganEA - Real Tick Exporter"
#property version   "1.00"
#property script_show_inputs

//+------------------------------------------------------------------+
//| INPUTS                                                            |
//+------------------------------------------------------------------+
input string   InpSymbol       = "XAUUSD";        // Symbol to export
input datetime InpDateFrom     = D'2024.01.01';   // Start date (inclusive)
input datetime InpDateTo       = D'2024.12.31';   // End date (inclusive)
input string   InpFileName     = "";              // Output filename (blank = auto)
input bool     InpIncludeHeader= true;            // Include CSV header row
input bool     InpVerboseLog   = true;            // Print progress to Experts tab

//+------------------------------------------------------------------+
//| Tick flag decoder - returns human-readable flag string            |
//+------------------------------------------------------------------+
string DecodeFlagsToString(uint flags)
  {
   string result = "";
   if((flags & TICK_FLAG_BID)    != 0) result += "BID|";
   if((flags & TICK_FLAG_ASK)    != 0) result += "ASK|";
   if((flags & TICK_FLAG_LAST)   != 0) result += "LAST|";
   if((flags & TICK_FLAG_VOLUME) != 0) result += "VOL|";
   if((flags & TICK_FLAG_BUY)    != 0) result += "BUY|";
   if((flags & TICK_FLAG_SELL)   != 0) result += "SELL|";
   // Remove trailing pipe
   if(StringLen(result) > 0)
      result = StringSubstr(result, 0, StringLen(result) - 1);
   else
      result = "NONE";
   return result;
  }

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
  {
   //--- Validate symbol
   string symbol = InpSymbol;
   if(symbol == "" || symbol == "NULL")
      symbol = _Symbol;

   if(!SymbolSelect(symbol, true))
     {
      Print("ERROR: Symbol '", symbol, "' not found or cannot be selected.");
      return;
     }

   //--- Validate date range
   if(InpDateFrom >= InpDateTo)
     {
      Print("ERROR: Start date must be before end date.");
      return;
     }

   //--- Build output filename
   string filename;
   if(InpFileName != "")
      filename = InpFileName;
   else
     {
      // Auto-generate: XAUUSD_RealTicks_20240101_20241231.csv
      string from_str = TimeToString(InpDateFrom, TIME_DATE);
      string to_str   = TimeToString(InpDateTo, TIME_DATE);
      StringReplace(from_str, ".", "");
      StringReplace(to_str, ".", "");
      filename = symbol + "_RealTicks_" + from_str + "_" + to_str + ".csv";
     }

   //--- Open file for writing
   int file_handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(file_handle == INVALID_HANDLE)
     {
      Print("ERROR: Cannot open file '", filename, "' for writing. Error: ", GetLastError());
      return;
     }

   Print("=============================================================");
   Print("  ExportRealTicks - Starting Export");
   Print("=============================================================");
   Print("  Symbol     : ", symbol);
   Print("  Date range : ", TimeToString(InpDateFrom), " to ", TimeToString(InpDateTo));
   Print("  Output file: ", filename);
   Print("  Location   : ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL5\\Files\\");
   Print("=============================================================");

   //--- Write CSV header
   if(InpIncludeHeader)
     {
      FileWrite(file_handle, "time_msc", "bid", "ask", "last", "volume", "flags", "flags_str", "volume_real");
     }

   //--- Calculate daily chunks
   //    We export day by day to avoid requesting millions of ticks at once
   //    which can cause MT5 memory issues
   datetime current_day = InpDateFrom;
   long     total_ticks_exported = 0;
   int      total_days = 0;
   int      days_with_data = 0;
   uint     start_time_ms = GetTickCount();

   while(current_day < InpDateTo)
     {
      //--- Define this day's range (start of day to end of day)
      datetime day_start = current_day;
      datetime day_end   = current_day + 86400 - 1;  // 23:59:59 of same day

      // Don't exceed user's end date
      if(day_end > InpDateTo)
         day_end = InpDateTo;

      //--- Convert to milliseconds for CopyTicksRange
      long ms_from = (long)day_start * 1000;
      long ms_to   = (long)day_end   * 1000 + 999;  // Include last millisecond

      //--- Request ticks using CopyTicksRange
      //    This reads from the SAME tick cache (.tkc files) that the
      //    Strategy Tester uses for "Every tick based on real ticks"
      MqlTick ticks[];
      int copied = CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, ms_from, ms_to);

      total_days++;

      if(copied > 0)
        {
         days_with_data++;

         //--- Write each tick to CSV
         for(int i = 0; i < copied; i++)
           {
            string line = StringFormat("%I64d,%s,%s,%s,%I64d,%u,%s,%s",
                                       ticks[i].time_msc,
                                       DoubleToString(ticks[i].bid, 5),
                                       DoubleToString(ticks[i].ask, 5),
                                       DoubleToString(ticks[i].last, 5),
                                       ticks[i].volume,
                                       ticks[i].flags,
                                       DecodeFlagsToString(ticks[i].flags),
                                       DoubleToString(ticks[i].volume_real, 2));
            FileWriteString(file_handle, line + "\n");
           }

         total_ticks_exported += copied;

         //--- Progress logging
         if(InpVerboseLog && (total_days % 7 == 0 || copied > 500000))
           {
            Print("  Day ", total_days, " (", TimeToString(day_start, TIME_DATE), "): ",
                  copied, " ticks | Total so far: ", total_ticks_exported);
           }
        }
      else if(copied < 0)
        {
         int err = GetLastError();
         if(InpVerboseLog)
            Print("  WARNING: CopyTicksRange failed for ", TimeToString(day_start, TIME_DATE),
                  " Error: ", err);
        }

      //--- Flush periodically to avoid data loss
      if(total_days % 30 == 0)
         FileFlush(file_handle);

      //--- Move to next day
      current_day += 86400;
     }

   //--- Close file
   FileClose(file_handle);

   //--- Calculate elapsed time
   uint elapsed_ms = GetTickCount() - start_time_ms;
   double elapsed_sec = elapsed_ms / 1000.0;

   //--- Final summary
   Print("=============================================================");
   Print("  EXPORT COMPLETE");
   Print("=============================================================");
   Print("  Total ticks exported : ", total_ticks_exported);
   Print("  Days processed       : ", total_days);
   Print("  Days with tick data  : ", days_with_data);
   Print("  Time elapsed         : ", DoubleToString(elapsed_sec, 1), " seconds");
   if(elapsed_sec > 0)
      Print("  Export speed         : ", (long)(total_ticks_exported / elapsed_sec), " ticks/sec");
   Print("  Output file          : ", filename);
   Print("  File location        : ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL5\\Files\\", filename);
   Print("=============================================================");
   Print("");
   Print("  NEXT STEPS:");
   Print("  1. Copy the CSV file to your Python analysis folder");
   Print("  2. Update CSV_FILE in analyze_ticks.py to point to this file");
   Print("  3. Run: python analyze_ticks.py");
   Print("");
   Print("  NOTE: The CSV columns are:");
   Print("    time_msc    - Millisecond timestamp (same precision as backtester)");
   Print("    bid         - Bid price");
   Print("    ask         - Ask price");
   Print("    last        - Last deal price");
   Print("    volume      - Tick volume");
   Print("    flags       - Numeric tick flags");
   Print("    flags_str   - Human-readable flags (BID|ASK|LAST|BUY|SELL)");
   Print("    volume_real - Real volume with decimal precision");
   Print("=============================================================");

   //--- Alert user
   Alert("Export complete! ", total_ticks_exported, " ticks saved to ", filename);
  }
//+------------------------------------------------------------------+
