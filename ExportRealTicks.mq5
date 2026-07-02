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
//|  1. Run this script on a chart of the target symbol              |
//|  2. The script will attempt to sync tick history from server     |
//|  3. Output CSV is saved to MT5's Files folder                    |
//|  4. Use analyze_ticks.py to process the exported data            |
//|                                                                  |
//| OUTPUT FORMAT (CSV):                                              |
//|  time_msc,bid,ask,last,volume,flags,volume_real                  |
//|                                                                  |
//| NOTE: Exports in daily chunks to avoid memory overflow when      |
//|       handling 47M+ ticks.                                       |
//+------------------------------------------------------------------+
#property copyright "GaganEA - Real Tick Exporter v2.0"
#property version   "2.00"
#property script_show_inputs

//+------------------------------------------------------------------+
//| INPUTS                                                            |
//| Symbol defaults to "" which means use chart symbol (_Symbol)     |
//| Dates default to 0 which means auto-calculate previous month     |
//+------------------------------------------------------------------+
input string   InpSymbol       = "";              // Symbol (blank = chart symbol)
input datetime InpDateFrom     = 0;              // Start date (0 = prev month start)
input datetime InpDateTo       = 0;              // End date (0 = prev month end)
input string   InpFileName     = "";              // Output filename (blank = auto)
input bool     InpIncludeHeader= true;            // Include CSV header row
input bool     InpVerboseLog   = true;            // Print progress to Experts tab
input int      InpRetryDelay   = 5;              // Retry delay seconds if 0 ticks
input int      InpMaxRetries   = 3;              // Max retries per day on 0 ticks

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
//| Calculate previous month start and end dates                      |
//+------------------------------------------------------------------+
void GetPreviousMonthRange(datetime &date_from, datetime &date_to)
  {
   MqlDateTime now;
   TimeCurrent(now);

   int year  = now.year;
   int month = now.mon - 1;

   if(month < 1)
     {
      month = 12;
      year--;
     }

   // Start of previous month
   MqlDateTime dt_from;
   dt_from.year  = year;
   dt_from.mon   = month;
   dt_from.day   = 1;
   dt_from.hour  = 0;
   dt_from.min   = 0;
   dt_from.sec   = 0;
   date_from = StructToTime(dt_from);

   // End of previous month (last day)
   int days_in_month = 31;
   if(month == 4 || month == 6 || month == 9 || month == 11)
      days_in_month = 30;
   else if(month == 2)
     {
      if(year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))
         days_in_month = 29;
      else
         days_in_month = 28;
     }

   MqlDateTime dt_to;
   dt_to.year  = year;
   dt_to.mon   = month;
   dt_to.day   = days_in_month;
   dt_to.hour  = 23;
   dt_to.min   = 59;
   dt_to.sec   = 59;
   date_to = StructToTime(dt_to);
  }

//+------------------------------------------------------------------+
//| Wait for symbol tick data to synchronize from server              |
//| Returns true if synchronized, false if timeout                    |
//+------------------------------------------------------------------+
bool WaitForTickSync(string symbol, int timeout_seconds = 30)
  {
   Print("  Checking tick data synchronization for ", symbol, "...");

   // Check if already synchronized
   if(SymbolIsSynchronized(symbol))
     {
      Print("  Symbol ", symbol, " is already synchronized.");
      return true;
     }

   Print("  Symbol not synchronized. Waiting up to ", timeout_seconds, " seconds...");
   uint start = GetTickCount();
   uint timeout_ms = (uint)timeout_seconds * 1000;

   while(GetTickCount() - start < timeout_ms)
     {
      if(SymbolIsSynchronized(symbol))
        {
         Print("  Symbol ", symbol, " synchronized after ",
               (GetTickCount() - start) / 1000, " seconds.");
         return true;
        }
      Sleep(500);
     }

   Print("  WARNING: Symbol synchronization timed out after ", timeout_seconds, " seconds.");
   Print("  Will attempt export anyway - data may be partial.");
   return false;
  }

//+------------------------------------------------------------------+
//| Force tick history download by requesting a small range           |
//| This triggers MT5 to fetch data from the trade server            |
//+------------------------------------------------------------------+
bool ForceTickHistoryDownload(string symbol, datetime date_from)
  {
   Print("  Triggering tick history download for ", symbol, "...");

   // Request just 1 second of ticks to trigger server download
   MqlTick test_ticks[];
   long ms_from = (long)date_from * 1000;
   long ms_to   = ms_from + 1000;  // 1 second range

   int result = CopyTicksRange(symbol, test_ticks, COPY_TICKS_ALL, ms_from, ms_to);
   int err = GetLastError();

   Print("  Initial trigger request: CopyTicksRange returned ", result,
         " (Error code: ", err, ")");

   if(result > 0)
     {
      Print("  Good - tick data is available. Got ", result, " tick(s) from trigger request.");
      return true;
     }

   // If no data, wait a bit for server to respond
   Print("  No immediate data. Waiting ", InpRetryDelay, " seconds for server to deliver history...");
   Sleep(InpRetryDelay * 1000);

   // Try again
   result = CopyTicksRange(symbol, test_ticks, COPY_TICKS_ALL, ms_from, ms_to);
   err = GetLastError();
   Print("  Second trigger attempt: CopyTicksRange returned ", result,
         " (Error code: ", err, ")");

   if(result > 0)
     {
      Print("  Tick data now available after wait.");
      return true;
     }

   // Try requesting the most recent ticks to check if ANY tick data exists
   MqlTick recent_ticks[];
   int recent = CopyTicks(symbol, recent_ticks, COPY_TICKS_ALL, 0, 1);
   err = GetLastError();
   Print("  CopyTicks (most recent 1 tick): returned ", recent, " (Error: ", err, ")");

   if(recent > 0)
     {
      Print("  Recent ticks exist. Historical range may need time to download.");
      Print("  Tip: Run a backtest first with 'Every tick based on real ticks' to force download.");
      return true;  // Symbol has tick data, just maybe not in requested range
     }

   Print("  WARNING: No tick data available for this symbol at all.");
   return false;
  }

//+------------------------------------------------------------------+
//| Print symbol diagnostic information                               |
//+------------------------------------------------------------------+
void PrintSymbolDiagnostics(string symbol)
  {
   Print("--- Symbol Diagnostics ---");
   Print("  Symbol name        : ", symbol);
   Print("  Chart symbol       : ", _Symbol);
   Print("  Symbol description : ", SymbolInfoString(symbol, SYMBOL_DESCRIPTION));
   Print("  Symbol path        : ", SymbolInfoString(symbol, SYMBOL_PATH));

   // Data availability
   datetime first_date = (datetime)SeriesInfoInteger(symbol, PERIOD_M1, SERIES_FIRSTDATE);
   datetime server_first = (datetime)SeriesInfoInteger(symbol, PERIOD_M1, SERIES_SERVER_FIRSTDATE);
   Print("  Local first date   : ", (first_date > 0 ? TimeToString(first_date) : "N/A"));
   Print("  Server first date  : ", (server_first > 0 ? TimeToString(server_first) : "N/A"));

   // Check if symbol is synchronized
   Print("  Synchronized       : ", (SymbolIsSynchronized(symbol) ? "Yes" : "No"));

   // Current tick
   MqlTick last_tick;
   if(SymbolInfoTick(symbol, last_tick))
     {
      Print("  Last tick time     : ", TimeToString((datetime)(last_tick.time_msc / 1000), TIME_DATE|TIME_SECONDS));
      Print("  Last bid/ask       : ", DoubleToString(last_tick.bid, 5), " / ", DoubleToString(last_tick.ask, 5));
     }
   else
      Print("  Last tick          : UNAVAILABLE");

   Print("--------------------------");
  }

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
  {
   //--- Determine symbol (default to chart symbol)
   string symbol = InpSymbol;
   if(symbol == "" || symbol == "NULL")
      symbol = _Symbol;

   if(!SymbolSelect(symbol, true))
     {
      Print("ERROR: Symbol '", symbol, "' not found or cannot be selected.");
      Print("  Available chart symbol: ", _Symbol);
      Print("  Try leaving the Symbol input blank to use the chart symbol.");
      return;
     }

   //--- Determine date range (default to previous month)
   datetime date_from = InpDateFrom;
   datetime date_to   = InpDateTo;

   if(date_from == 0 || date_to == 0)
     {
      GetPreviousMonthRange(date_from, date_to);
      Print("  Auto date range: previous month = ",
            TimeToString(date_from, TIME_DATE), " to ",
            TimeToString(date_to, TIME_DATE));
     }

   //--- Validate date range
   if(date_from >= date_to)
     {
      Print("ERROR: Start date must be before end date.");
      Print("  From: ", TimeToString(date_from), " To: ", TimeToString(date_to));
      return;
     }

   //--- Check if date range is in the future
   datetime now = TimeCurrent();
   if(date_from > now)
     {
      Print("ERROR: Start date is in the FUTURE. No tick data can exist.");
      Print("  Start date: ", TimeToString(date_from));
      Print("  Current time: ", TimeToString(now));
      Print("  Did you forget to update the date range?");
      return;
     }

   if(date_to > now)
     {
      Print("  NOTE: End date is after current time. Adjusting to now.");
      date_to = now;
     }

   //--- Print diagnostics
   PrintSymbolDiagnostics(symbol);

   //--- Wait for symbol synchronization
   WaitForTickSync(symbol, 30);

   //--- Force tick history download attempt
   bool has_data = ForceTickHistoryDownload(symbol, date_from);
   if(!has_data)
     {
      Print("  WARNING: Could not confirm tick data availability.");
      Print("  The export will proceed but may return 0 ticks.");
      Print("  SOLUTION: Run a backtest with 'Every tick based on real ticks'");
      Print("  for this symbol/date range first, then re-run this script.");
     }

   //--- Build output filename
   string filename;
   if(InpFileName != "")
      filename = InpFileName;
   else
     {
      // Auto-generate: XAUUSD_RealTicks_20240101_20241231.csv
      string from_str = TimeToString(date_from, TIME_DATE);
      string to_str   = TimeToString(date_to, TIME_DATE);
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
   Print("  ExportRealTicks v2.0 - Starting Export");
   Print("=============================================================");
   Print("  Symbol     : ", symbol);
   Print("  Date range : ", TimeToString(date_from), " to ", TimeToString(date_to));
   Print("  Output file: ", filename);
   Print("  Location   : ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL5\\Files\\");
   Print("  Retry delay: ", InpRetryDelay, "s | Max retries: ", InpMaxRetries);
   Print("=============================================================");

   //--- Write CSV header
   if(InpIncludeHeader)
     {
      FileWrite(file_handle, "time_msc", "bid", "ask", "last", "volume", "flags", "flags_str", "volume_real");
     }

   //--- Calculate daily chunks
   //    We export day by day to avoid requesting millions of ticks at once
   //    which can cause MT5 memory issues
   datetime current_day = date_from;
   long     total_ticks_exported = 0;
   int      total_days = 0;
   int      days_with_data = 0;
   int      days_with_errors = 0;
   uint     start_time_ms = GetTickCount();

   while(current_day < date_to)
     {
      //--- Define this day's range (start of day to end of day)
      datetime day_start = current_day;
      datetime day_end   = current_day + 86400 - 1;  // 23:59:59 of same day

      // Don't exceed user's end date
      if(day_end > date_to)
         day_end = date_to;

      //--- Convert to milliseconds for CopyTicksRange
      long ms_from = (long)day_start * 1000;
      long ms_to   = (long)day_end   * 1000 + 999;  // Include last millisecond

      //--- Request ticks using CopyTicksRange with retry logic
      MqlTick ticks[];
      int copied = -1;
      int err = 0;
      int attempts = 0;

      for(int retry = 0; retry <= InpMaxRetries; retry++)
        {
         copied = CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, ms_from, ms_to);
         err = GetLastError();
         attempts = retry + 1;

         if(copied > 0)
            break;  // Success

         if(copied == 0 && retry < InpMaxRetries)
           {
            // 0 ticks could mean data is still downloading from server
            if(InpVerboseLog && retry == 0)
               Print("  Day ", TimeToString(day_start, TIME_DATE),
                     ": 0 ticks (Error: ", err, "). Retrying in ", InpRetryDelay, "s...");
            Sleep(InpRetryDelay * 1000);
           }
         else if(copied < 0)
           {
            // Negative means actual error, don't retry
            break;
           }
        }

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
      else if(copied == 0)
        {
         // Print error for EVERY day that returns 0 (after retries)
         if(InpVerboseLog)
            Print("  Day ", TimeToString(day_start, TIME_DATE),
                  ": 0 ticks after ", attempts, " attempt(s). Error code: ", err,
                  " (may be weekend/holiday or data not cached)");
        }
      else // copied < 0
        {
         days_with_errors++;
         if(InpVerboseLog)
            Print("  ERROR: Day ", TimeToString(day_start, TIME_DATE),
                  ": CopyTicksRange returned ", copied, ". Error code: ", err);
        }

      //--- Flush periodically to avoid data loss
      if(total_days % 30 == 0)
         FileFlush(file_handle);

      //--- Move to next day
      current_day += 86400;
     }

   //--- Close file
   FileClose(file_handle);

   //--- If zero ticks exported, delete the empty file
   if(total_ticks_exported == 0)
     {
      FileDelete(filename);
      Print("");
      Print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Print("  ZERO TICKS EXPORTED - FILE DELETED");
      Print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Print("");
      Print("  TROUBLESHOOTING:");
      Print("  1. The tick data for this date range is NOT in your local cache.");
      Print("  2. Run a BACKTEST first:");
      Print("     - Open Strategy Tester (Ctrl+R)");
      Print("     - Select symbol: ", symbol);
      Print("     - Set date range: ", TimeToString(date_from, TIME_DATE),
            " to ", TimeToString(date_to, TIME_DATE));
      Print("     - Set model: 'Every tick based on real ticks'");
      Print("     - Run any EA (even a blank one) - this forces tick download");
      Print("  3. After backtest completes, run this script again.");
      Print("  4. Make sure the date range is not in the future!");
      Print("     Current server time: ", TimeToString(TimeCurrent()));
      Print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Alert("FAILED: 0 ticks exported. See Experts tab for troubleshooting steps.");
      return;
     }

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
   Print("  Days with errors     : ", days_with_errors);
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
