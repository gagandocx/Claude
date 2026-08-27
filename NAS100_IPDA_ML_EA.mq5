//+------------------------------------------------------------------+
//|                                         NAS100_IPDA_ML_EA.mq5   |
//|          NAS100 IPDA + Machine Learning Expert Advisor           |
//|       Neural Network Inference with ICT Methodology              |
//|                                                                  |
//| STRATEGY:                                                        |
//|  1. Load pre-trained ML model weights from CSV files             |
//|  2. Compute IPDA features in real-time (M1 chart)               |
//|  3. Run neural network forward pass for signal prediction        |
//|  4. Combine ML confidence with IPDA confluence filters           |
//|  5. Execute with adaptive trailing stops and partial closes      |
//|                                                                  |
//| ML ARCHITECTURE: Input(50) -> 128 -> 64 -> 32 -> 3              |
//|  - Hidden layers: ReLU activation                                |
//|  - Output: Softmax (BUY / SELL / NEUTRAL probabilities)         |
//|                                                                  |
//| IPDA FEATURES:                                                   |
//|  - Market Structure (BOS/CHoCH) on M1, M15, H1                  |
//|  - Order Blocks (bullish/bearish with proximity)                 |
//|  - Fair Value Gaps (with size metric)                            |
//|  - Liquidity Sweeps (session high/low raids)                     |
//|  - Premium/Discount Zones                                        |
//|  - NAS100 Killzones (Pre-market, Regular, Power Hour)           |
//|  - Technical: EMA(8,21,50,200), RSI(14), ATR(14), Volume       |
//+------------------------------------------------------------------+
#property copyright "NAS100 IPDA ML EA"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

CTrade        trade;
CPositionInfo pos;

//+------------------------------------------------------------------+
//| ENUMERATIONS                                                      |
//+------------------------------------------------------------------+
enum ENUM_MARKET_BIAS { BIAS_BULLISH=1, BIAS_BEARISH=-1, BIAS_NEUTRAL=0 };
enum ENUM_SIGNAL_TYPE { SIG_BUY=1, SIG_SELL=-1, SIG_NEUTRAL=0 };
enum ENUM_OB_TYPE     { OB_BULLISH=1,  OB_BEARISH=-1 };

//+------------------------------------------------------------------+
//| STRUCTS                                                           |
//+------------------------------------------------------------------+
struct OrderBlock
{
   double         top;
   double         bottom;
   double         mid;
   ENUM_OB_TYPE   type;
   datetime       time;
   bool           valid;
   bool           traded;
};

struct FVG
{
   double         top;
   double         bottom;
   double         mid;
   bool           bullish;
   datetime       time;
   bool           filled;
};

struct SwingPoint
{
   double         price;
   datetime       time;
   bool           is_high;
};

struct NormParams
{
   double         min_val;
   double         max_val;
};

struct MLModel
{
   double         w1[];    // weights layer 1: input_size x 128 (flattened)
   double         b1[];    // biases layer 1: 128
   double         w2[];    // weights layer 2: 128 x 64 (flattened)
   double         b2[];    // biases layer 2: 64
   double         w3[];    // weights layer 3: 64 x 32 (flattened)
   double         b3[];    // biases layer 3: 32
   double         wout[];  // weights output: 32 x 3 (flattened)
   double         bout[];  // biases output: 3
   NormParams     norms[]; // normalization params per feature
   int            input_size;
   int            h1_size;
   int            h2_size;
   int            h3_size;
   int            output_size;
   bool           loaded;
};

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ML MODEL SETTINGS ==="
input string Model_Path           = "NAS100_ML_Model";   // Model directory in MQL5/Files/
input double ML_Confidence_Thresh = 0.55;                 // Minimum ML prediction confidence
input double ML_Strong_Thresh     = 0.70;                 // Strong signal threshold (skip IPDA)
input int    ML_Feature_Count     = 50;                   // Number of input features

input group "=== IPDA SETTINGS ==="
input int    Swing_Lookback       = 10;    // Bars for swing point detection
input int    Swing_Strength       = 3;     // Candles each side for swing confirmation
input int    OB_Lookback          = 50;    // Max bars back for order block search
input int    Max_OB_Count         = 5;     // Max OBs to track per direction
input double OB_Displacement_Pts  = 50.0;  // Min displacement after OB (points)
input int    FVG_Lookback         = 30;    // Bars back for FVG detection
input int    Max_FVG_Count        = 5;     // Max FVGs to track per direction
input double FVG_Min_Size_Pts     = 10.0;  // Minimum FVG size in points
input int    IPDA_Min_Confluence  = 2;     // Min IPDA confirmations needed

input group "=== KILLZONE SETTINGS (UTC) ==="
input int    PreMarket_Start      = 13;    // Pre-market start (08:00 ET = 13:00 UTC)
input int    PreMarket_End        = 14;    // Pre-market end   (09:30 ET ~ 14:30 UTC)
input int    RegSession_Start     = 14;    // Regular session  (09:30 ET = 14:30 UTC)
input int    RegSession_End       = 16;    // NY momentum end  (11:30 ET = 16:30 UTC)
input int    PowerHour_Start      = 20;    // Power hour start (15:00 ET = 20:00 UTC)
input int    PowerHour_End        = 21;    // Power hour end   (16:00 ET = 21:00 UTC)
input bool   Use_Killzones        = true;  // Only trade in killzones

input group "=== TRADE SETTINGS ==="
input ENUM_TIMEFRAMES Entry_TF    = PERIOD_M1;   // Entry timeframe
input ENUM_TIMEFRAMES HTF         = PERIOD_M15;  // Higher TF for structure
input ENUM_TIMEFRAMES HTF2        = PERIOD_H1;   // Second HTF for trend
input int    Max_Trades           = 3;            // Max simultaneous trades
input int    Max_Spread_Pts       = 50;           // Max spread in points
input bool   Trade_Buy            = true;         // Allow BUY trades
input bool   Trade_Sell           = true;         // Allow SELL trades

input group "=== RISK MANAGEMENT ==="
input double Risk_Percent         = 1.0;   // Risk % per trade
input double Manual_Lot           = 0.0;   // Manual lot (0=auto)
input double Max_Lot              = 5.0;   // Max lot size
input double SL_ATR_Mult          = 2.0;   // SL distance = ATR x Multiplier
input double TP_ATR_Mult          = 3.0;   // TP distance = ATR x Multiplier

input group "=== TRAILING STOP ==="
input bool   Use_Trailing         = true;         // Enable trailing stop
input double BE_ATR_Mult          = 1.0;          // Break-even at ATR x Mult profit
input double Trail_ATR_Mult       = 1.5;          // Trail distance = ATR x Mult
input double Trail_Step_Pts       = 5.0;          // Minimum trail step in points
input double Partial_Close_Pct    = 40.0;         // % to close at first target
input double Partial_ATR_Mult     = 1.5;          // Close partial at ATR x Mult profit
input bool   Use_Adaptive_Trail   = true;         // Tighten trail as profit grows

input group "=== EQUITY PROTECTION ==="
input bool   Use_Equity_Protect   = true;  // Enable equity protection
input double Max_DD_Percent       = 5.0;   // Max drawdown % (close all)
input double Daily_Loss_Limit     = 3.0;   // Daily loss limit % (stop trading)

input group "=== HFT CONTROLS ==="
input int    Min_Bars_Between     = 5;     // Minimum bars between entries
input int    Max_Trades_Per_Hour  = 5;     // Max trades per hour
input int    Loss_Cooldown_Bars   = 10;    // Cooldown bars after consecutive losses
input int    Max_Consec_Losses    = 3;     // Max consecutive losses before cooldown
input bool   Momentum_Reentry    = true;   // Allow rapid re-entry after quick profit
input int    Max_Trade_Duration   = 60;    // Max bars before time-based exit

input group "=== DISPLAY ==="
input bool   Show_Panel           = true;   // Show dashboard panel
input bool   Draw_Zones           = true;   // Draw OB/FVG zones on chart
input int    Magic                = 88001;  // Magic number
input string EA_Comment           = "NAS100_ML"; // Trade comment
input int    Slippage             = 15;     // Max slippage points

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
// ML Model
MLModel model;

// Indicator handles
int h_ema8, h_ema21, h_ema50, h_ema200;
int h_ema8_htf, h_ema21_htf, h_ema50_htf, h_ema200_htf;
int h_rsi, h_atr;
int h_ema8_h1, h_ema21_h1;

// Indicator buffers
double buf_ema8[], buf_ema21[], buf_ema50[], buf_ema200[];
double buf_ema8_htf[], buf_ema21_htf[], buf_ema50_htf[], buf_ema200_htf[];
double buf_ema8_h1[], buf_ema21_h1[];
double buf_rsi[], buf_atr[];

// Price data
double point_size;

// Market structure
SwingPoint swing_highs[];
SwingPoint swing_lows[];
ENUM_MARKET_BIAS market_bias     = BIAS_NEUTRAL;
ENUM_MARKET_BIAS market_bias_htf = BIAS_NEUTRAL;
double last_bos_level = 0;
bool   choch_detected = false;

// Order blocks and FVGs
OrderBlock bull_obs[];
OrderBlock bear_obs[];
FVG bull_fvgs[];
FVG bear_fvgs[];

// Trading state
datetime last_bar_time;
datetime last_htf_bar_time;
int      last_entry_bar     = 0;
int      current_bar_index  = 0;
int      open_buys          = 0;
int      open_sells         = 0;
double   float_pnl          = 0;
int      consec_losses      = 0;
int      cooldown_until_bar = 0;
int      trades_this_hour   = 0;
int      last_trade_hour    = -1;

// Break-even and partial close tracking
ulong    be_tickets[];
ulong    partial_tickets[];

// Session tracking
double   session_high = 0;
double   session_low  = DBL_MAX;
datetime session_date = 0;

// P&L
double   pnl_today, pnl_week, pnl_month;
datetime pnl_cache_time;
double   day_start_balance = 0;
bool     daily_limit_hit   = false;

// Dashboard
string   lbl = "NML_";
string   panel_signal    = "INIT";
color    panel_sig_col   = clrGray;
string   panel_ml_pred   = "---";
string   panel_bias_str  = "---";
string   panel_kz_str    = "---";
string   panel_ipda_str  = "---";
double   panel_ml_conf   = 0.0;

// Trade stats
int      total_trades    = 0;
int      winning_trades  = 0;
double   total_profit    = 0;
double   total_loss      = 0;

//+------------------------------------------------------------------+
//| OnInit - Load model, create indicators, setup dashboard          |
//+------------------------------------------------------------------+
int OnInit()
{
   point_size = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // Initialize arrays
   ArrayResize(be_tickets, 0);
   ArrayResize(partial_tickets, 0);
   ArrayResize(bull_obs, 0);
   ArrayResize(bear_obs, 0);
   ArrayResize(bull_fvgs, 0);
   ArrayResize(bear_fvgs, 0);
   ArrayResize(swing_highs, 0);
   ArrayResize(swing_lows, 0);

   // Set series direction for buffers
   ArraySetAsSeries(buf_ema8, true);
   ArraySetAsSeries(buf_ema21, true);
   ArraySetAsSeries(buf_ema50, true);
   ArraySetAsSeries(buf_ema200, true);
   ArraySetAsSeries(buf_ema8_htf, true);
   ArraySetAsSeries(buf_ema21_htf, true);
   ArraySetAsSeries(buf_ema50_htf, true);
   ArraySetAsSeries(buf_ema200_htf, true);
   ArraySetAsSeries(buf_ema8_h1, true);
   ArraySetAsSeries(buf_ema21_h1, true);
   ArraySetAsSeries(buf_rsi, true);
   ArraySetAsSeries(buf_atr, true);

   // Create indicator handles - Entry TF (M1)
   h_ema8   = iMA(_Symbol, Entry_TF, 8,   0, MODE_EMA, PRICE_CLOSE);
   h_ema21  = iMA(_Symbol, Entry_TF, 21,  0, MODE_EMA, PRICE_CLOSE);
   h_ema50  = iMA(_Symbol, Entry_TF, 50,  0, MODE_EMA, PRICE_CLOSE);
   h_ema200 = iMA(_Symbol, Entry_TF, 200, 0, MODE_EMA, PRICE_CLOSE);
   h_rsi    = iRSI(_Symbol, Entry_TF, 14, PRICE_CLOSE);
   h_atr    = iATR(_Symbol, Entry_TF, 14);

   // HTF indicators (M15)
   h_ema8_htf  = iMA(_Symbol, HTF, 8,   0, MODE_EMA, PRICE_CLOSE);
   h_ema21_htf = iMA(_Symbol, HTF, 21,  0, MODE_EMA, PRICE_CLOSE);
   h_ema50_htf = iMA(_Symbol, HTF, 50,  0, MODE_EMA, PRICE_CLOSE);
   h_ema200_htf= iMA(_Symbol, HTF, 200, 0, MODE_EMA, PRICE_CLOSE);

   // H1 indicators
   h_ema8_h1  = iMA(_Symbol, HTF2, 8,  0, MODE_EMA, PRICE_CLOSE);
   h_ema21_h1 = iMA(_Symbol, HTF2, 21, 0, MODE_EMA, PRICE_CLOSE);

   // Validate handles
   if(h_ema8==INVALID_HANDLE || h_ema21==INVALID_HANDLE ||
      h_ema50==INVALID_HANDLE || h_ema200==INVALID_HANDLE ||
      h_rsi==INVALID_HANDLE || h_atr==INVALID_HANDLE ||
      h_ema8_htf==INVALID_HANDLE || h_ema21_htf==INVALID_HANDLE ||
      h_ema50_htf==INVALID_HANDLE || h_ema200_htf==INVALID_HANDLE ||
      h_ema8_h1==INVALID_HANDLE || h_ema21_h1==INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   // Load ML model
   if(!LoadMLModel())
   {
      Print("WARNING: ML model not loaded. EA will use IPDA signals only.");
      model.loaded = false;
   }

   // Setup trade object
   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(Slippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   // Record starting balance for daily limit
   day_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);

   // Create dashboard
   if(Show_Panel) CreateDashboard();

   Print("NAS100 IPDA ML EA initialized | Magic=", Magic,
         " | Model loaded=", model.loaded);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   IndicatorRelease(h_ema8);
   IndicatorRelease(h_ema21);
   IndicatorRelease(h_ema50);
   IndicatorRelease(h_ema200);
   IndicatorRelease(h_rsi);
   IndicatorRelease(h_atr);
   IndicatorRelease(h_ema8_htf);
   IndicatorRelease(h_ema21_htf);
   IndicatorRelease(h_ema50_htf);
   IndicatorRelease(h_ema200_htf);
   IndicatorRelease(h_ema8_h1);
   IndicatorRelease(h_ema21_h1);

   // Clean chart objects
   ObjectsDeleteAll(0, lbl);
   ObjectsDeleteAll(0, "BullOB_");
   ObjectsDeleteAll(0, "BearOB_");
   ObjectsDeleteAll(0, "BullFVG_");
   ObjectsDeleteAll(0, "BearFVG_");
}


//+------------------------------------------------------------------+
//| ML MODEL LOADING                                                  |
//| Reads CSV weight files from MQL5/Files/NAS100_ML_Model/          |
//+------------------------------------------------------------------+
bool LoadMLModel()
{
   model.input_size  = ML_Feature_Count;
   model.h1_size     = 128;
   model.h2_size     = 64;
   model.h3_size     = 32;
   model.output_size = 3;

   // Allocate arrays
   ArrayResize(model.w1,    model.input_size * model.h1_size);
   ArrayResize(model.b1,    model.h1_size);
   ArrayResize(model.w2,    model.h1_size * model.h2_size);
   ArrayResize(model.b2,    model.h2_size);
   ArrayResize(model.w3,    model.h2_size * model.h3_size);
   ArrayResize(model.b3,    model.h3_size);
   ArrayResize(model.wout,  model.h3_size * model.output_size);
   ArrayResize(model.bout,  model.output_size);
   ArrayResize(model.norms, model.input_size);

   // Load weights layer 1
   if(!LoadWeightFile(Model_Path + "/weights_layer1.csv", model.w1, model.input_size * model.h1_size))
      return false;
   if(!LoadBiasFile(Model_Path + "/biases_layer1.csv", model.b1, model.h1_size))
      return false;

   // Load weights layer 2
   if(!LoadWeightFile(Model_Path + "/weights_layer2.csv", model.w2, model.h1_size * model.h2_size))
      return false;
   if(!LoadBiasFile(Model_Path + "/biases_layer2.csv", model.b2, model.h2_size))
      return false;

   // Load weights layer 3
   if(!LoadWeightFile(Model_Path + "/weights_layer3.csv", model.w3, model.h2_size * model.h3_size))
      return false;
   if(!LoadBiasFile(Model_Path + "/biases_layer3.csv", model.b3, model.h3_size))
      return false;

   // Load output weights
   if(!LoadWeightFile(Model_Path + "/weights_output.csv", model.wout, model.h3_size * model.output_size))
      return false;
   if(!LoadBiasFile(Model_Path + "/biases_output.csv", model.bout, model.output_size))
      return false;

   // Load normalization parameters
   if(!LoadNormParams(Model_Path + "/normalization_params.csv"))
      return false;

   model.loaded = true;
   Print("ML Model loaded successfully: ", model.input_size, " inputs -> ",
         model.h1_size, " -> ", model.h2_size, " -> ", model.h3_size,
         " -> ", model.output_size, " outputs");
   return true;
}

bool LoadWeightFile(string filename, double &arr[], int expected_size)
{
   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_ANSI, ",");
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open weight file: ", filename, " Error=", GetLastError());
      return false;
   }

   int idx = 0;
   while(!FileIsEnding(handle) && idx < expected_size)
   {
      arr[idx] = StringToDouble(FileReadString(handle));
      idx++;
   }
   FileClose(handle);

   if(idx < expected_size)
   {
      Print("WARNING: Weight file ", filename, " has ", idx,
            " values, expected ", expected_size);
      // Zero-fill remaining
      for(int i = idx; i < expected_size; i++)
         arr[i] = 0.0;
   }
   return true;
}

bool LoadBiasFile(string filename, double &arr[], int expected_size)
{
   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_ANSI, ",");
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open bias file: ", filename, " Error=", GetLastError());
      return false;
   }

   int idx = 0;
   while(!FileIsEnding(handle) && idx < expected_size)
   {
      arr[idx] = StringToDouble(FileReadString(handle));
      idx++;
   }
   FileClose(handle);

   if(idx < expected_size)
   {
      Print("WARNING: Bias file ", filename, " has ", idx,
            " values, expected ", expected_size);
      for(int i = idx; i < expected_size; i++)
         arr[i] = 0.0;
   }
   return true;
}

bool LoadNormParams(string filename)
{
   int handle = FileOpen(filename, FILE_READ|FILE_CSV|FILE_ANSI, ",");
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open normalization file: ", filename);
      return false;
   }

   // Skip header line (feature_name,min,max)
   if(!FileIsEnding(handle))
   {
      FileReadString(handle); // skip "feature_name"
      FileReadString(handle); // skip "min"
      FileReadString(handle); // skip "max"
   }

   int idx = 0;
   while(!FileIsEnding(handle) && idx < model.input_size)
   {
      string feat_name = FileReadString(handle);  // feature name
      if(StringLen(feat_name) == 0) break;
      model.norms[idx].min_val = StringToDouble(FileReadString(handle));
      model.norms[idx].max_val = StringToDouble(FileReadString(handle));
      idx++;
   }
   FileClose(handle);

   if(idx < model.input_size)
   {
      Print("WARNING: Norm params has ", idx, " entries, expected ", model.input_size);
      for(int i = idx; i < model.input_size; i++)
      {
         model.norms[i].min_val = 0.0;
         model.norms[i].max_val = 1.0;
      }
   }
   Print("Normalization params loaded: ", idx, " features");
   return true;
}

//+------------------------------------------------------------------+
//| NEURAL NETWORK FORWARD PASS                                       |
//| Architecture: input -> 128(ReLU) -> 64(ReLU) -> 32(ReLU) -> 3   |
//| Output: softmax probabilities [SELL, NEUTRAL, BUY]               |
//+------------------------------------------------------------------+
void MLPredict(double &features[], double &output[])
{
   ArrayResize(output, model.output_size);

   // Normalize input features using min-max scaling
   double norm_input[];
   ArrayResize(norm_input, model.input_size);
   for(int i = 0; i < model.input_size; i++)
   {
      double range = model.norms[i].max_val - model.norms[i].min_val;
      if(range > 0.0)
         norm_input[i] = (features[i] - model.norms[i].min_val) / range;
      else
         norm_input[i] = 0.0;
      // Clip to [0, 1]
      if(norm_input[i] < 0.0) norm_input[i] = 0.0;
      if(norm_input[i] > 1.0) norm_input[i] = 1.0;
   }

   // Layer 1: input -> 128 (ReLU)
   double h1[];
   ArrayResize(h1, model.h1_size);
   for(int j = 0; j < model.h1_size; j++)
   {
      double sum = model.b1[j];
      for(int i = 0; i < model.input_size; i++)
         sum += norm_input[i] * model.w1[i * model.h1_size + j];
      h1[j] = MathMax(0.0, sum); // ReLU
   }

   // Layer 2: 128 -> 64 (ReLU)
   double h2[];
   ArrayResize(h2, model.h2_size);
   for(int j = 0; j < model.h2_size; j++)
   {
      double sum = model.b2[j];
      for(int i = 0; i < model.h1_size; i++)
         sum += h1[i] * model.w2[i * model.h2_size + j];
      h2[j] = MathMax(0.0, sum); // ReLU
   }

   // Layer 3: 64 -> 32 (ReLU)
   double h3[];
   ArrayResize(h3, model.h3_size);
   for(int j = 0; j < model.h3_size; j++)
   {
      double sum = model.b3[j];
      for(int i = 0; i < model.h2_size; i++)
         sum += h2[i] * model.w3[i * model.h3_size + j];
      h3[j] = MathMax(0.0, sum); // ReLU
   }

   // Output layer: 32 -> 3 (linear before softmax)
   double logits[];
   ArrayResize(logits, model.output_size);
   for(int j = 0; j < model.output_size; j++)
   {
      double sum = model.bout[j];
      for(int i = 0; i < model.h3_size; i++)
         sum += h3[i] * model.wout[i * model.output_size + j];
      logits[j] = sum;
   }

   // Softmax activation
   double max_logit = logits[0];
   for(int i = 1; i < model.output_size; i++)
      if(logits[i] > max_logit) max_logit = logits[i];

   double exp_sum = 0.0;
   for(int i = 0; i < model.output_size; i++)
   {
      output[i] = MathExp(logits[i] - max_logit); // subtract max for numerical stability
      exp_sum += output[i];
   }
   for(int i = 0; i < model.output_size; i++)
      output[i] /= exp_sum;
}

//+------------------------------------------------------------------+
//| FEATURE COMPUTATION                                               |
//| Builds the 50-feature vector matching Python trainer output      |
//+------------------------------------------------------------------+
void BuildFeatureVector(double &features[])
{
   ArrayResize(features, ML_Feature_Count);
   ArrayInitialize(features, 0.0);

   // Refresh indicator buffers
   if(CopyBuffer(h_ema8,   0, 0, 10, buf_ema8)   < 10) return;
   if(CopyBuffer(h_ema21,  0, 0, 10, buf_ema21)  < 10) return;
   if(CopyBuffer(h_ema50,  0, 0, 10, buf_ema50)  < 10) return;
   if(CopyBuffer(h_ema200, 0, 0, 10, buf_ema200) < 10) return;
   if(CopyBuffer(h_rsi,    0, 0, 10, buf_rsi)    < 10) return;
   if(CopyBuffer(h_atr,    0, 0, 10, buf_atr)    < 10) return;
   if(CopyBuffer(h_ema8_htf,  0, 0, 5, buf_ema8_htf)   < 5) return;
   if(CopyBuffer(h_ema21_htf, 0, 0, 5, buf_ema21_htf)  < 5) return;
   if(CopyBuffer(h_ema50_htf, 0, 0, 5, buf_ema50_htf)  < 5) return;
   if(CopyBuffer(h_ema200_htf,0, 0, 5, buf_ema200_htf) < 5) return;
   if(CopyBuffer(h_ema8_h1,   0, 0, 5, buf_ema8_h1)    < 5) return;
   if(CopyBuffer(h_ema21_h1,  0, 0, 5, buf_ema21_h1)   < 5) return;

   double close = iClose(_Symbol, Entry_TF, 1);
   double high  = iHigh(_Symbol, Entry_TF, 1);
   double low   = iLow(_Symbol, Entry_TF, 1);
   double open_p= iOpen(_Symbol, Entry_TF, 1);
   double atr   = buf_atr[1];
   if(atr <= 0) atr = point_size * 100; // fallback

   int idx = 0;

   // --- 1. Technical Indicators (features 0-3) ---
   features[idx++] = buf_ema8[1];    // ema_8
   features[idx++] = buf_ema21[1];   // ema_21
   features[idx++] = buf_ema50[1];   // ema_50
   features[idx++] = buf_ema200[1];  // ema_200

   // --- ATR (feature 4) ---
   features[idx++] = atr;            // atr_14

   // --- Distance from EMAs normalized by ATR (features 5-8) ---
   // Clipped to [-5, 5] to match Python trainer's np.clip(dist, -5.0, 5.0)
   features[idx++] = MathMax(-5.0, MathMin(5.0, (close - buf_ema8[1]) / atr));    // dist_from_ema_8
   features[idx++] = MathMax(-5.0, MathMin(5.0, (close - buf_ema21[1]) / atr));   // dist_from_ema_21
   features[idx++] = MathMax(-5.0, MathMin(5.0, (close - buf_ema50[1]) / atr));   // dist_from_ema_50
   features[idx++] = MathMax(-5.0, MathMin(5.0, (close - buf_ema200[1]) / atr));  // dist_from_ema_200

   // --- EMA slopes (features 9-12) ---
   // Clipped to [-2, 2] to match Python trainer's np.clip(slope, -2.0, 2.0)
   features[idx++] = MathMax(-2.0, MathMin(2.0, (buf_ema8[1] - buf_ema8[6]) / (5.0 * atr)));   // ema_8_slope
   features[idx++] = MathMax(-2.0, MathMin(2.0, (buf_ema21[1] - buf_ema21[6]) / (5.0 * atr))); // ema_21_slope
   features[idx++] = MathMax(-2.0, MathMin(2.0, (buf_ema50[1] - buf_ema50[6]) / (5.0 * atr))); // ema_50_slope
   features[idx++] = MathMax(-2.0, MathMin(2.0, (buf_ema200[1] - buf_ema200[6]) / (5.0 * atr)));// ema_200_slope

   // --- RSI (feature 13) ---
   features[idx++] = buf_rsi[1] / 100.0;  // rsi_14 normalized 0-1

   // --- VWAP distance proxy (feature 14) ---
   // Compute VWAP proxy: sum(typical_price*volume) / sum(volume) over 50-bar rolling window
   // This matches the Python trainer's compute_vwap_proxy(window=50) approach
   int vwap_period = 50;
   double cum_tp_vol = 0.0;
   double cum_vol_vwap = 0.0;
   for(int i = 1; i <= vwap_period; i++)
   {
      double c_i = iClose(_Symbol, Entry_TF, i);
      double h_i = iHigh(_Symbol, Entry_TF, i);
      double l_i = iLow(_Symbol, Entry_TF, i);
      double tp_i = (h_i + l_i + c_i) / 3.0;
      long v_i = iVolume(_Symbol, Entry_TF, i);
      cum_tp_vol += tp_i * (double)v_i;
      cum_vol_vwap += (double)v_i;
   }
   double vwap_proxy = (cum_vol_vwap > 0) ? cum_tp_vol / cum_vol_vwap : close;
   features[idx++] = MathMax(-5.0, MathMin(5.0, (close - vwap_proxy) / atr));  // dist_from_vwap

   // --- Volume ratio (feature 15) ---
   long vol_cur = iVolume(_Symbol, Entry_TF, 1);
   double vol_avg = 0;
   for(int i = 1; i <= 20; i++) vol_avg += (double)iVolume(_Symbol, Entry_TF, i);
   vol_avg /= 20.0;
   features[idx++] = (vol_avg > 0) ? (double)vol_cur / vol_avg : 1.0; // volume_ratio

   // --- Candle metrics (features 16-17) ---
   double body = MathAbs(close - open_p);
   double full_range = high - low;
   features[idx++] = (full_range > 0) ? body / full_range : 0.5; // body_ratio
   features[idx++] = (close > open_p) ? 1.0 : 0.0;              // is_bullish_candle

   // --- ATR normalized (feature 18) ---
   double atr_avg50 = 0;
   for(int i = 1; i <= 50; i++)
   {
      double h_i = iHigh(_Symbol, Entry_TF, i);
      double l_i = iLow(_Symbol, Entry_TF, i);
      atr_avg50 += (h_i - l_i);
   }
   atr_avg50 /= 50.0;
   features[idx++] = (atr_avg50 > 0) ? atr / atr_avg50 : 1.0; // atr_normalized

   // --- 2. Market Structure on M1 (features 19-25) ---
   double sh_price = 0, sl_price = 0;
   bool has_bos_bull = false, has_bos_bear = false;
   bool has_choch_bull = false, has_choch_bear = false;
   double trend_m1 = 0;
   ComputeMarketStructure(Entry_TF, sh_price, sl_price,
                          has_bos_bull, has_bos_bear,
                          has_choch_bull, has_choch_bear, trend_m1);

   features[idx++] = (sh_price > 0) ? 1.0 : 0.0;   // swing_high (binary detection flag)
   features[idx++] = (sl_price > 0) ? 1.0 : 0.0;   // swing_low (binary detection flag)
   features[idx++] = has_bos_bull ? 1.0 : 0.0;   // bos_bullish
   features[idx++] = has_bos_bear ? 1.0 : 0.0;   // bos_bearish
   features[idx++] = has_choch_bull ? 1.0 : 0.0;  // choch_bullish
   features[idx++] = has_choch_bear ? 1.0 : 0.0;  // choch_bearish
   features[idx++] = trend_m1;                     // trend_m1

   // --- 3. Higher TF Trends (features 26-28) ---
   // M5 trend approximation
   features[idx++] = (buf_ema8[1] > buf_ema21[1]) ? 1.0 :
                     (buf_ema8[1] < buf_ema21[1]) ? -1.0 : 0.0; // trend_m5

   // M15 trend
   features[idx++] = (buf_ema8_htf[1] > buf_ema21_htf[1]) ? 1.0 :
                     (buf_ema8_htf[1] < buf_ema21_htf[1]) ? -1.0 : 0.0; // trend_m15

   // H1 trend
   features[idx++] = (buf_ema8_h1[1] > buf_ema21_h1[1]) ? 1.0 :
                     (buf_ema8_h1[1] < buf_ema21_h1[1]) ? -1.0 : 0.0; // trend_h1

   // --- 4. Order Blocks (features 29-32) ---
   double ob_bull_active = 0, ob_bear_active = 0;
   double ob_dist_bull = 0, ob_dist_bear = 0;
   ComputeOBFeatures(close, atr, ob_bull_active, ob_bear_active,
                     ob_dist_bull, ob_dist_bear);
   features[idx++] = ob_bull_active;   // ob_bullish_active
   features[idx++] = ob_bear_active;   // ob_bearish_active
   features[idx++] = ob_dist_bull;     // ob_distance_bull
   features[idx++] = ob_dist_bear;     // ob_distance_bear

   // --- 5. Fair Value Gaps (features 33-36) ---
   double fvg_bull_active = 0, fvg_bear_active = 0;
   double fvg_size_bull = 0, fvg_size_bear = 0;
   ComputeFVGFeatures(close, atr, fvg_bull_active, fvg_bear_active,
                      fvg_size_bull, fvg_size_bear);
   features[idx++] = fvg_bull_active;  // fvg_bullish_active
   features[idx++] = fvg_bear_active;  // fvg_bearish_active
   features[idx++] = fvg_size_bull;    // fvg_bullish_size
   features[idx++] = fvg_size_bear;    // fvg_bearish_size

   // --- 6. Liquidity Sweeps (features 37-40) ---
   double sweep_bull = 0, sweep_bear = 0;
   double bars_since_bull_sweep = 0, bars_since_bear_sweep = 0;
   ComputeLiquiditySweeps(sweep_bull, sweep_bear,
                          bars_since_bull_sweep, bars_since_bear_sweep);
   features[idx++] = sweep_bull;              // sweep_bullish
   features[idx++] = sweep_bear;              // sweep_bearish
   features[idx++] = bars_since_bull_sweep;   // bars_since_bull_sweep
   features[idx++] = bars_since_bear_sweep;   // bars_since_bear_sweep

   // --- 7. Session/Killzone features (features 41-45) ---
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   features[idx++] = IsPreMarket() ? 1.0 : 0.0;        // is_premarket
   features[idx++] = IsRegularSession() ? 1.0 : 0.0;   // is_regular_session
   features[idx++] = IsPowerHour() ? 1.0 : 0.0;        // is_power_hour
   features[idx++] = (double)(dt.hour * 60 + dt.min) / 1440.0;  // time_of_day (minute resolution, normalized 0-1)
   // day_of_week: MQL5 day_of_week is 0=Sun,1=Mon,...,5=Fri
   // Python uses weekday()/4.0 where Mon=0, Fri=1.0
   // Match Python: (day_of_week - 1) / 4.0 gives Mon=0.0, Tue=0.25, ..., Fri=1.0
   features[idx++] = (double)(dt.day_of_week - 1) / 4.0;  // day_of_week (normalized, Mon=0 Fri=1)

   // --- 8. Premium/Discount (features 46-49) ---
   double pd_position = 0, eq_distance = 0;
   double is_premium = 0, is_discount = 0;
   ComputePremiumDiscount(close, pd_position, is_premium, is_discount, eq_distance);
   features[idx++] = pd_position;   // pd_position
   features[idx++] = is_premium;    // is_premium
   features[idx++] = is_discount;   // is_discount
   features[idx++] = eq_distance;   // equilibrium_distance
}

//+------------------------------------------------------------------+
//| MARKET STRUCTURE COMPUTATION                                       |
//| Detect swing highs/lows, BOS, CHoCH on specified timeframe       |
//+------------------------------------------------------------------+
void ComputeMarketStructure(ENUM_TIMEFRAMES tf,
                            double &sh_price, double &sl_price,
                            bool &bos_bull, bool &bos_bear,
                            bool &choch_bull, bool &choch_bear,
                            double &trend)
{
   bos_bull = false; bos_bear = false;
   choch_bull = false; choch_bear = false;
   sh_price = 0; sl_price = 0;
   trend = 0;

   int bars = iBars(_Symbol, tf);
   if(bars < Swing_Lookback + Swing_Strength + 5) return;

   // Find swing highs and lows
   double swing_h1 = 0, swing_h2 = 0;
   double swing_l1 = 0, swing_l2 = 0;
   int found_h = 0, found_l = 0;

   for(int i = Swing_Strength; i < Swing_Lookback + Swing_Strength && (found_h < 2 || found_l < 2); i++)
   {
      double h = iHigh(_Symbol, tf, i);
      double l = iLow(_Symbol, tf, i);
      bool is_sh = true, is_sl = true;

      for(int j = 1; j <= Swing_Strength; j++)
      {
         if(iHigh(_Symbol, tf, i-j) >= h || iHigh(_Symbol, tf, i+j) >= h) is_sh = false;
         if(iLow(_Symbol, tf, i-j) <= l || iLow(_Symbol, tf, i+j) <= l) is_sl = false;
      }

      if(is_sh && found_h < 2)
      {
         if(found_h == 0) swing_h1 = h;
         else             swing_h2 = h;
         found_h++;
      }
      if(is_sl && found_l < 2)
      {
         if(found_l == 0) swing_l1 = l;
         else             swing_l2 = l;
         found_l++;
      }
   }

   if(found_h < 2 || found_l < 2) return;

   sh_price = swing_h1;
   sl_price = swing_l1;

   double cur_close = iClose(_Symbol, tf, 1);

   // Determine trend and structure breaks
   // Higher highs + higher lows = bullish
   if(swing_h1 > swing_h2 && swing_l1 > swing_l2) trend = 1.0;
   else if(swing_h1 < swing_h2 && swing_l1 < swing_l2) trend = -1.0;

   // BOS: break in same direction as trend
   if(trend > 0 && cur_close > swing_h1) bos_bull = true;
   if(trend < 0 && cur_close < swing_l1) bos_bear = true;

   // CHoCH: break in opposite direction
   if(trend <= 0 && cur_close > swing_h1) choch_bull = true;
   if(trend >= 0 && cur_close < swing_l1) choch_bear = true;
}


//+------------------------------------------------------------------+
//| MARKET STRUCTURE UPDATE (for IPDA trading logic)                  |
//| Updates global bias used by the signal engine                     |
//+------------------------------------------------------------------+
void UpdateMarketStructure()
{
   int bars = iBars(_Symbol, HTF);
   if(bars < Swing_Lookback + Swing_Strength + 5) return;

   ArrayResize(swing_highs, 0);
   ArrayResize(swing_lows, 0);

   for(int i = Swing_Strength; i < Swing_Lookback + Swing_Strength; i++)
   {
      double h = iHigh(_Symbol, HTF, i);
      double l = iLow(_Symbol, HTF, i);
      bool is_sh = true, is_sl = true;

      for(int j = 1; j <= Swing_Strength; j++)
      {
         if(iHigh(_Symbol, HTF, i-j) >= h || iHigh(_Symbol, HTF, i+j) >= h) is_sh = false;
         if(iLow(_Symbol, HTF, i-j) <= l || iLow(_Symbol, HTF, i+j) <= l) is_sl = false;
      }

      if(is_sh)
      {
         int sz = ArraySize(swing_highs);
         ArrayResize(swing_highs, sz+1);
         swing_highs[sz].price = h;
         swing_highs[sz].time = iTime(_Symbol, HTF, i);
         swing_highs[sz].is_high = true;
      }
      if(is_sl)
      {
         int sz = ArraySize(swing_lows);
         ArrayResize(swing_lows, sz+1);
         swing_lows[sz].price = l;
         swing_lows[sz].time = iTime(_Symbol, HTF, i);
         swing_lows[sz].is_high = false;
      }
   }

   if(ArraySize(swing_highs) < 2 || ArraySize(swing_lows) < 2) return;

   double cur_close = iClose(_Symbol, HTF, 1);
   double last_sh = swing_highs[0].price;
   double last_sl = swing_lows[0].price;

   ENUM_MARKET_BIAS prev_bias = market_bias;

   // CHoCH detection
   if(cur_close > last_sh && market_bias != BIAS_BULLISH)
   {
      market_bias = BIAS_BULLISH;
      choch_detected = true;
      panel_bias_str = "CHoCH BULLISH";
   }
   else if(cur_close < last_sl && market_bias != BIAS_BEARISH)
   {
      market_bias = BIAS_BEARISH;
      choch_detected = true;
      panel_bias_str = "CHoCH BEARISH";
   }
   // BOS detection
   else if(cur_close > last_sh && market_bias == BIAS_BULLISH)
   {
      last_bos_level = last_sh;
      panel_bias_str = "BULLISH BOS";
   }
   else if(cur_close < last_sl && market_bias == BIAS_BEARISH)
   {
      last_bos_level = last_sl;
      panel_bias_str = "BEARISH BOS";
   }

   // Initial bias
   if(market_bias == BIAS_NEUTRAL)
   {
      if(swing_highs[0].price > swing_highs[1].price &&
         swing_lows[0].price > swing_lows[1].price)
      { market_bias = BIAS_BULLISH; panel_bias_str = "BULLISH"; }
      else if(swing_highs[0].price < swing_highs[1].price &&
              swing_lows[0].price < swing_lows[1].price)
      { market_bias = BIAS_BEARISH; panel_bias_str = "BEARISH"; }
      else
      { panel_bias_str = "NEUTRAL"; }
   }
}

//+------------------------------------------------------------------+
//| ORDER BLOCK DETECTION                                             |
//| Bullish OB: last bearish candle before strong up move            |
//| Bearish OB: last bullish candle before strong down move          |
//+------------------------------------------------------------------+
void DetectOrderBlocks()
{
   ArrayResize(bull_obs, 0);
   ArrayResize(bear_obs, 0);

   int bars = iBars(_Symbol, HTF);
   if(bars < OB_Lookback + 5) return;

   double min_move = OB_Displacement_Pts * point_size;

   for(int i = 2; i < OB_Lookback; i++)
   {
      double o1 = iOpen(_Symbol, HTF, i);
      double c1 = iClose(_Symbol, HTF, i);
      double h1 = iHigh(_Symbol, HTF, i);
      double l1 = iLow(_Symbol, HTF, i);

      // Look at next 3 candles for displacement
      double disp_up = 0, disp_down = 0;
      for(int j = 1; j <= 3; j++)
      {
         if(i - j < 1) break;
         double move_up   = iClose(_Symbol, HTF, i-j) - iHigh(_Symbol, HTF, i);
         double move_down = iLow(_Symbol, HTF, i) - iClose(_Symbol, HTF, i-j);
         if(move_up > disp_up)     disp_up = move_up;
         if(move_down > disp_down) disp_down = move_down;
      }

      // Bullish OB: bearish candle + strong up displacement
      if(c1 < o1 && disp_up >= min_move && ArraySize(bull_obs) < Max_OB_Count)
      {
         double ob_top = o1;
         double ob_bot = c1;
         bool mitigated = false;

         for(int k = i-1; k >= 1; k--)
         {
            if(iLow(_Symbol, HTF, k) <= ob_top && iLow(_Symbol, HTF, k) >= ob_bot)
            { mitigated = true; break; }
         }
         if(!mitigated)
         {
            int sz = ArraySize(bull_obs);
            ArrayResize(bull_obs, sz+1);
            bull_obs[sz].top    = ob_top;
            bull_obs[sz].bottom = ob_bot;
            bull_obs[sz].mid    = (ob_top + ob_bot) / 2.0;
            bull_obs[sz].type   = OB_BULLISH;
            bull_obs[sz].time   = iTime(_Symbol, HTF, i);
            bull_obs[sz].valid  = true;
            bull_obs[sz].traded = false;

            if(Draw_Zones)
               DrawOBBox("BullOB_"+IntegerToString(i),
                  iTime(_Symbol,HTF,i), ob_bot, iTime(_Symbol,HTF,i-1), ob_top,
                  C'0,100,0', true);
         }
      }

      // Bearish OB: bullish candle + strong down displacement
      if(c1 > o1 && disp_down >= min_move && ArraySize(bear_obs) < Max_OB_Count)
      {
         double ob_top = c1;
         double ob_bot = o1;
         bool mitigated = false;

         for(int k = i-1; k >= 1; k--)
         {
            if(iHigh(_Symbol, HTF, k) >= ob_bot && iHigh(_Symbol, HTF, k) <= ob_top)
            { mitigated = true; break; }
         }
         if(!mitigated)
         {
            int sz = ArraySize(bear_obs);
            ArrayResize(bear_obs, sz+1);
            bear_obs[sz].top    = ob_top;
            bear_obs[sz].bottom = ob_bot;
            bear_obs[sz].mid    = (ob_top + ob_bot) / 2.0;
            bear_obs[sz].type   = OB_BEARISH;
            bear_obs[sz].time   = iTime(_Symbol, HTF, i);
            bear_obs[sz].valid  = true;
            bear_obs[sz].traded = false;

            if(Draw_Zones)
               DrawOBBox("BearOB_"+IntegerToString(i),
                  iTime(_Symbol,HTF,i), ob_bot, iTime(_Symbol,HTF,i-1), ob_top,
                  C'100,0,0', false);
         }
      }
   }
}

void ComputeOBFeatures(double close, double atr,
                       double &bull_active, double &bear_active,
                       double &dist_bull, double &dist_bear)
{
   bull_active = 0; bear_active = 0;
   dist_bull = 5.0; dist_bear = 5.0; // default far distance

   // Find nearest bullish OB below price
   for(int i = 0; i < ArraySize(bull_obs); i++)
   {
      if(!bull_obs[i].valid) continue;
      if(close >= bull_obs[i].bottom && close <= bull_obs[i].top)
      {
         bull_active = 1.0;
         dist_bull = 0.0;
         break;
      }
      else if(close > bull_obs[i].top)
      {
         double d = (close - bull_obs[i].top) / atr;
         if(d < dist_bull) dist_bull = d;
      }
   }
   if(ArraySize(bull_obs) > 0 && bull_active < 1.0)
      bull_active = (dist_bull < 2.0) ? 0.5 : 0.0;

   // Find nearest bearish OB above price
   for(int i = 0; i < ArraySize(bear_obs); i++)
   {
      if(!bear_obs[i].valid) continue;
      if(close >= bear_obs[i].bottom && close <= bear_obs[i].top)
      {
         bear_active = 1.0;
         dist_bear = 0.0;
         break;
      }
      else if(close < bear_obs[i].bottom)
      {
         double d = (bear_obs[i].bottom - close) / atr;
         if(d < dist_bear) dist_bear = d;
      }
   }
   if(ArraySize(bear_obs) > 0 && bear_active < 1.0)
      bear_active = (dist_bear < 2.0) ? 0.5 : 0.0;
}

//+------------------------------------------------------------------+
//| FAIR VALUE GAP DETECTION                                          |
//| 3-candle imbalance pattern                                        |
//+------------------------------------------------------------------+
void DetectFVGs()
{
   ArrayResize(bull_fvgs, 0);
   ArrayResize(bear_fvgs, 0);

   int bars = iBars(_Symbol, HTF);
   if(bars < FVG_Lookback + 5) return;

   double min_size = FVG_Min_Size_Pts * point_size;

   for(int i = 2; i < FVG_Lookback; i++)
   {
      double h3 = iHigh(_Symbol, HTF, i+1); // candle 3 (oldest)
      double l3 = iLow(_Symbol, HTF, i+1);
      double h1 = iHigh(_Symbol, HTF, i-1); // candle 1 (newest)
      double l1 = iLow(_Symbol, HTF, i-1);

      // Bullish FVG: gap between candle 3 high and candle 1 low
      if(l1 > h3 + min_size && ArraySize(bull_fvgs) < Max_FVG_Count)
      {
         bool filled = false;
         for(int k = i-2; k >= 1; k--)
            if(iLow(_Symbol, HTF, k) <= l1 && iHigh(_Symbol, HTF, k) >= h3)
            { filled = true; break; }

         if(!filled)
         {
            int sz = ArraySize(bull_fvgs);
            ArrayResize(bull_fvgs, sz+1);
            bull_fvgs[sz].top     = l1;
            bull_fvgs[sz].bottom  = h3;
            bull_fvgs[sz].mid     = (l1 + h3) / 2.0;
            bull_fvgs[sz].bullish = true;
            bull_fvgs[sz].time    = iTime(_Symbol, HTF, i);
            bull_fvgs[sz].filled  = false;

            if(Draw_Zones)
               DrawFVGBox("BullFVG_"+IntegerToString(i),
                  iTime(_Symbol,HTF,i+1), h3, iTime(_Symbol,HTF,i-1), l1,
                  C'0,50,100');
         }
      }

      // Bearish FVG: gap between candle 1 high and candle 3 low
      if(h1 + min_size < l3 && ArraySize(bear_fvgs) < Max_FVG_Count)
      {
         bool filled = false;
         for(int k = i-2; k >= 1; k--)
            if(iHigh(_Symbol, HTF, k) >= h1 && iLow(_Symbol, HTF, k) <= l3)
            { filled = true; break; }

         if(!filled)
         {
            int sz = ArraySize(bear_fvgs);
            ArrayResize(bear_fvgs, sz+1);
            bear_fvgs[sz].top     = l3;
            bear_fvgs[sz].bottom  = h1;
            bear_fvgs[sz].mid     = (l3 + h1) / 2.0;
            bear_fvgs[sz].bullish = false;
            bear_fvgs[sz].time    = iTime(_Symbol, HTF, i);
            bear_fvgs[sz].filled  = false;

            if(Draw_Zones)
               DrawFVGBox("BearFVG_"+IntegerToString(i),
                  iTime(_Symbol,HTF,i+1), h1, iTime(_Symbol,HTF,i-1), l3,
                  C'100,50,0');
         }
      }
   }
}

void ComputeFVGFeatures(double close, double atr,
                        double &bull_active, double &bear_active,
                        double &size_bull, double &size_bear)
{
   bull_active = 0; bear_active = 0;
   size_bull = 0; size_bear = 0;

   // Check if price is inside or near a bullish FVG
   for(int i = 0; i < ArraySize(bull_fvgs); i++)
   {
      if(bull_fvgs[i].filled) continue;
      double fvg_size = (bull_fvgs[i].top - bull_fvgs[i].bottom) / atr;
      if(close >= bull_fvgs[i].bottom && close <= bull_fvgs[i].top)
      {
         bull_active = 1.0;
         size_bull = fvg_size;
         break;
      }
      else if(close > bull_fvgs[i].top && (close - bull_fvgs[i].top) / atr < 1.0)
      {
         bull_active = 0.5;
         if(fvg_size > size_bull) size_bull = fvg_size;
      }
   }

   // Check bearish FVGs
   for(int i = 0; i < ArraySize(bear_fvgs); i++)
   {
      if(bear_fvgs[i].filled) continue;
      double fvg_size = (bear_fvgs[i].top - bear_fvgs[i].bottom) / atr;
      if(close >= bear_fvgs[i].bottom && close <= bear_fvgs[i].top)
      {
         bear_active = 1.0;
         size_bear = fvg_size;
         break;
      }
      else if(close < bear_fvgs[i].bottom && (bear_fvgs[i].bottom - close) / atr < 1.0)
      {
         bear_active = 0.5;
         if(fvg_size > size_bear) size_bear = fvg_size;
      }
   }
}

//+------------------------------------------------------------------+
//| LIQUIDITY SWEEP DETECTION                                         |
//| Price takes session high/low then reverses within 3-5 bars       |
//+------------------------------------------------------------------+
void ComputeLiquiditySweeps(double &sweep_bull, double &sweep_bear,
                            double &bars_since_bull, double &bars_since_bear)
{
   sweep_bull = 0; sweep_bear = 0;
   bars_since_bull = 50.0; bars_since_bear = 50.0;

   // Update session high/low
   UpdateSessionRange();

   if(session_high == 0 || session_low >= DBL_MAX) return;

   // Check for bullish sweep (price takes low then reverses up)
   for(int i = 1; i <= 10; i++)
   {
      double bar_low  = iLow(_Symbol, Entry_TF, i);
      double bar_close = iClose(_Symbol, Entry_TF, i);

      // Sweep below session low, then close back above
      if(bar_low < session_low && bar_close > session_low)
      {
         // Confirm reversal: next bars move up
         bool reversal = false;
         for(int j = 1; j < i && j <= 5; j++)
         {
            if(iClose(_Symbol, Entry_TF, j) > iClose(_Symbol, Entry_TF, i))
            { reversal = true; break; }
         }
         if(reversal || i <= 3)
         {
            sweep_bull = 1.0;
            bars_since_bull = (double)i;
            break;
         }
      }
   }

   // Check for bearish sweep (price takes high then reverses down)
   for(int i = 1; i <= 10; i++)
   {
      double bar_high  = iHigh(_Symbol, Entry_TF, i);
      double bar_close = iClose(_Symbol, Entry_TF, i);

      // Sweep above session high, then close back below
      if(bar_high > session_high && bar_close < session_high)
      {
         bool reversal = false;
         for(int j = 1; j < i && j <= 5; j++)
         {
            if(iClose(_Symbol, Entry_TF, j) < iClose(_Symbol, Entry_TF, i))
            { reversal = true; break; }
         }
         if(reversal || i <= 3)
         {
            sweep_bear = 1.0;
            bars_since_bear = (double)i;
            break;
         }
      }
   }

   // Normalize bars_since to [0, 1] range for ML
   bars_since_bull = MathMin(bars_since_bull / 50.0, 1.0);
   bars_since_bear = MathMin(bars_since_bear / 50.0, 1.0);
}

void UpdateSessionRange()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d 00:00",
                    dt.year, dt.mon, dt.day));

   if(session_date == today) return; // already computed
   session_date = today;
   session_high = 0;
   session_low  = DBL_MAX;

   // Get previous session high/low (yesterday or day before)
   datetime prev_day = today - 86400;
   int bars = iBars(_Symbol, PERIOD_M15);
   for(int i = 0; i < bars; i++)
   {
      datetime btime = iTime(_Symbol, PERIOD_M15, i);
      if(btime < prev_day) break;
      if(btime >= today) continue;
      double h = iHigh(_Symbol, PERIOD_M15, i);
      double l = iLow(_Symbol, PERIOD_M15, i);
      if(h > session_high) session_high = h;
      if(l < session_low)  session_low = l;
   }
   if(session_low >= DBL_MAX) session_low = 0;
}


//+------------------------------------------------------------------+
//| PREMIUM/DISCOUNT ZONES                                            |
//| Dealing range from recent swing high to swing low                |
//| Premium > 50%, Discount < 50%                                    |
//+------------------------------------------------------------------+
void ComputePremiumDiscount(double close,
                            double &pd_position, double &is_premium,
                            double &is_discount, double &eq_distance)
{
   pd_position = 0.5;
   is_premium = 0; is_discount = 0;
   eq_distance = 0;

   // Use swing points from HTF structure
   double range_high = 0, range_low = DBL_MAX;

   // Get highest high and lowest low over lookback
   for(int i = 1; i <= Swing_Lookback * 3; i++)
   {
      double h = iHigh(_Symbol, HTF, i);
      double l = iLow(_Symbol, HTF, i);
      if(h > range_high) range_high = h;
      if(l < range_low) range_low = l;
   }

   if(range_low >= DBL_MAX || range_high <= range_low)
   {
      range_high = iHigh(_Symbol, HTF, 1);
      range_low  = iLow(_Symbol, HTF, 1);
   }

   double range = range_high - range_low;
   if(range > 0)
   {
      pd_position = (close - range_low) / range;
      pd_position = MathMax(0.0, MathMin(1.0, pd_position));

      // Match Python: abs(pd_position - 0.5) * 2.0 (bounded 0-1, unsigned)
      eq_distance = MathAbs(pd_position - 0.5) * 2.0;

      is_premium  = (pd_position > 0.5) ? 1.0 : 0.0;
      is_discount = (pd_position < 0.5) ? 1.0 : 0.0;
   }
}


//+------------------------------------------------------------------+
//| NAS100 KILLZONE FUNCTIONS                                         |
//| Pre-market:    08:00-09:30 ET = 13:00-14:30 UTC                  |
//| Regular open:  09:30-11:30 ET = 14:30-16:30 UTC                  |
//| Power hour:    15:00-16:00 ET = 20:00-21:00 UTC                  |
//+------------------------------------------------------------------+
bool IsPreMarket()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   return (h >= PreMarket_Start && h < PreMarket_End);
}

bool IsRegularSession()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   return (h >= RegSession_Start && h < RegSession_End);
}

bool IsPowerHour()
{
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   return (h >= PowerHour_Start && h < PowerHour_End);
}

bool IsAnyKillzone()
{
   return IsPreMarket() || IsRegularSession() || IsPowerHour();
}

string GetKillzoneName()
{
   if(IsPowerHour())      return "POWER HOUR";
   if(IsRegularSession()) return "REGULAR";
   if(IsPreMarket())      return "PRE-MARKET";
   return "OFF";
}

//+------------------------------------------------------------------+
//| SIGNAL GENERATION                                                 |
//| Combines ML prediction with IPDA confluence                      |
//| Requires: ML confidence > threshold AND 2+ IPDA confirmations    |
//+------------------------------------------------------------------+
ENUM_SIGNAL_TYPE GenerateSignal(double &ml_confidence)
{
   ml_confidence = 0;

   // --- ML Prediction ---
   double ml_output[];
   ENUM_SIGNAL_TYPE ml_signal = SIG_NEUTRAL;

   if(model.loaded)
   {
      double features[];
      BuildFeatureVector(features);
      MLPredict(features, ml_output);

      // Python labels: SELL=0, NEUTRAL=1, BUY=2 (via labels+1)
      // output[0] = SELL prob, output[1] = NEUTRAL prob, output[2] = BUY prob
      double sell_prob    = ml_output[0];
      double neutral_prob = ml_output[1];
      double buy_prob     = ml_output[2];

      if(buy_prob > sell_prob && buy_prob > neutral_prob)
      {
         ml_signal = SIG_BUY;
         ml_confidence = buy_prob;
      }
      else if(sell_prob > buy_prob && sell_prob > neutral_prob)
      {
         ml_signal = SIG_SELL;
         ml_confidence = sell_prob;
      }
      else
      {
         ml_confidence = neutral_prob;
      }

      panel_ml_pred = StringFormat("%s (%.1f%%)",
         ml_signal==SIG_BUY ? "BUY" : ml_signal==SIG_SELL ? "SELL" : "NEUTRAL",
         ml_confidence * 100.0);
      panel_ml_conf = ml_confidence;
   }
   else
   {
      // No model loaded - use IPDA only
      panel_ml_pred = "NO MODEL";
   }

   // --- IPDA Confluence ---
   int ipda_bull_score = 0;
   int ipda_bear_score = 0;

   // 1. Market Structure bias
   if(market_bias == BIAS_BULLISH) ipda_bull_score++;
   if(market_bias == BIAS_BEARISH) ipda_bear_score++;

   // 2. Price in/near Order Block
   double close = iClose(_Symbol, Entry_TF, 1);
   for(int i = 0; i < ArraySize(bull_obs); i++)
   {
      if(!bull_obs[i].valid) continue;
      if(close >= bull_obs[i].bottom && close <= bull_obs[i].top)
      { ipda_bull_score++; break; }
   }
   for(int i = 0; i < ArraySize(bear_obs); i++)
   {
      if(!bear_obs[i].valid) continue;
      if(close >= bear_obs[i].bottom && close <= bear_obs[i].top)
      { ipda_bear_score++; break; }
   }

   // 3. FVG confirmation
   for(int i = 0; i < ArraySize(bull_fvgs); i++)
   {
      if(!bull_fvgs[i].filled && close >= bull_fvgs[i].bottom && close <= bull_fvgs[i].top)
      { ipda_bull_score++; break; }
   }
   for(int i = 0; i < ArraySize(bear_fvgs); i++)
   {
      if(!bear_fvgs[i].filled && close >= bear_fvgs[i].bottom && close <= bear_fvgs[i].top)
      { ipda_bear_score++; break; }
   }

   // 4. Killzone active
   if(IsAnyKillzone())
   { ipda_bull_score++; ipda_bear_score++; }

   // 5. Premium/Discount
   double pd_pos = 0.5;
   if(ArraySize(swing_highs) > 0 && ArraySize(swing_lows) > 0)
   {
      double range_hi = swing_highs[0].price;
      double range_lo = swing_lows[0].price;
      double range = range_hi - range_lo;
      if(range > 0) pd_pos = (close - range_lo) / range;
   }
   if(pd_pos < 0.4) ipda_bull_score++; // discount = buy zone
   if(pd_pos > 0.6) ipda_bear_score++; // premium = sell zone

   // 6. Liquidity sweep
   double sw_bull=0, sw_bear=0, bs_bull=0, bs_bear=0;
   ComputeLiquiditySweeps(sw_bull, sw_bear, bs_bull, bs_bear);
   if(sw_bull > 0) ipda_bull_score++;
   if(sw_bear > 0) ipda_bear_score++;

   panel_ipda_str = StringFormat("Bull:%d Bear:%d", ipda_bull_score, ipda_bear_score);

   // --- Combine ML + IPDA ---
   ENUM_SIGNAL_TYPE final_signal = SIG_NEUTRAL;

   if(model.loaded)
   {
      // Strong ML signal (high confidence): skip IPDA min requirement
      if(ml_confidence >= ML_Strong_Thresh)
      {
         if(ml_signal == SIG_BUY && ipda_bull_score >= 1) final_signal = SIG_BUY;
         if(ml_signal == SIG_SELL && ipda_bear_score >= 1) final_signal = SIG_SELL;
      }
      // Normal ML signal: require full IPDA confluence
      else if(ml_confidence >= ML_Confidence_Thresh)
      {
         if(ml_signal == SIG_BUY && ipda_bull_score >= IPDA_Min_Confluence)
            final_signal = SIG_BUY;
         if(ml_signal == SIG_SELL && ipda_bear_score >= IPDA_Min_Confluence)
            final_signal = SIG_SELL;
      }
   }
   else
   {
      // No model: use IPDA only with higher threshold
      if(ipda_bull_score >= IPDA_Min_Confluence + 1 && market_bias == BIAS_BULLISH)
         final_signal = SIG_BUY;
      if(ipda_bear_score >= IPDA_Min_Confluence + 1 && market_bias == BIAS_BEARISH)
         final_signal = SIG_SELL;
      ml_confidence = (double)MathMax(ipda_bull_score, ipda_bear_score) / 6.0;
   }

   // Direction filter
   if(final_signal == SIG_BUY && !Trade_Buy) final_signal = SIG_NEUTRAL;
   if(final_signal == SIG_SELL && !Trade_Sell) final_signal = SIG_NEUTRAL;

   return final_signal;
}

//+------------------------------------------------------------------+
//| TRADE EXECUTION                                                   |
//| Entry with ATR-based or OB-based SL, next-liquidity TP           |
//+------------------------------------------------------------------+
void OpenTrade(ENUM_SIGNAL_TYPE signal, double confidence)
{
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   double atr    = buf_atr[1];
   if(atr <= 0) return;

   // Get broker minimum stop distance
   long   stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double min_stop    = (stops_level + 5) * point_size;

   double price, sl, tp, sl_dist;

   if(signal == SIG_BUY)
   {
      price = ask;
      sl = price - SL_ATR_Mult * atr;

      // Try to set SL below nearest bullish OB
      for(int i = 0; i < ArraySize(bull_obs); i++)
      {
         if(!bull_obs[i].valid || bull_obs[i].traded) continue;
         if(price >= bull_obs[i].bottom && price <= bull_obs[i].top * 1.005)
         {
            double ob_sl = bull_obs[i].bottom - 10.0 * point_size;
            if(price - ob_sl < SL_ATR_Mult * atr * 1.5) // dont let OB SL be too wide
               sl = ob_sl;
            bull_obs[i].traded = true;
            break;
         }
      }

      sl = NormalizeDouble(sl, _Digits);
      if(price - sl < min_stop) sl = NormalizeDouble(price - min_stop, _Digits);
      sl_dist = price - sl;
      if(sl_dist <= 0) return;

      // TP: next liquidity level or ATR-based
      tp = GetBuyTP(price, sl_dist, atr);
      if(tp - price < min_stop) tp = NormalizeDouble(price + TP_ATR_Mult * atr, _Digits);
   }
   else if(signal == SIG_SELL)
   {
      price = bid;
      sl = price + SL_ATR_Mult * atr;

      // Try to set SL above nearest bearish OB
      for(int i = 0; i < ArraySize(bear_obs); i++)
      {
         if(!bear_obs[i].valid || bear_obs[i].traded) continue;
         if(price >= bear_obs[i].bottom * 0.995 && price <= bear_obs[i].top)
         {
            double ob_sl = bear_obs[i].top + 10.0 * point_size + spread;
            if(ob_sl - price < SL_ATR_Mult * atr * 1.5)
               sl = ob_sl;
            bear_obs[i].traded = true;
            break;
         }
      }

      sl = NormalizeDouble(sl, _Digits);
      if(sl - price < min_stop) sl = NormalizeDouble(price + min_stop, _Digits);
      sl_dist = sl - price;
      if(sl_dist <= 0) return;

      tp = GetSellTP(price, sl_dist, atr);
      if(price - tp < min_stop) tp = NormalizeDouble(price - TP_ATR_Mult * atr, _Digits);
   }
   else return;

   double lot = CalcLot(sl_dist);
   ENUM_ORDER_TYPE type = (signal == SIG_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   Print("NAS100_ML | Signal: ", EnumToString(type),
         " | Conf=", DoubleToString(confidence*100,1), "%%",
         " | Bias=", panel_bias_str,
         " | IPDA=", panel_ipda_str,
         " | KZ=", GetKillzoneName());

   if(trade.PositionOpen(_Symbol, type, lot, price, sl, tp, EA_Comment))
   {
      last_entry_bar = current_bar_index;
      trades_this_hour++;
      Print("NAS100_ML | Trade OPENED: ", EnumToString(type),
            " lot=", lot, " price=", price,
            " sl=", sl, " tp=", tp);
   }
   else
   {
      Print("NAS100_ML | Trade FAILED: ", trade.ResultRetcodeDescription());
   }
}

double GetBuyTP(double entry, double sl_dist, double atr)
{
   // Try swing high above as TP (liquidity target)
   for(int i = 0; i < ArraySize(swing_highs); i++)
   {
      if(swing_highs[i].price > entry + sl_dist)
         return NormalizeDouble(swing_highs[i].price - 2 * point_size, _Digits);
   }
   // Try bearish FVG above as target
   for(int i = 0; i < ArraySize(bear_fvgs); i++)
   {
      if(!bear_fvgs[i].filled && bear_fvgs[i].bottom > entry + sl_dist)
         return NormalizeDouble(bear_fvgs[i].bottom, _Digits);
   }
   // Default: ATR-based TP
   return NormalizeDouble(entry + TP_ATR_Mult * atr, _Digits);
}

double GetSellTP(double entry, double sl_dist, double atr)
{
   // Try swing low below as TP
   for(int i = 0; i < ArraySize(swing_lows); i++)
   {
      if(swing_lows[i].price < entry - sl_dist)
         return NormalizeDouble(swing_lows[i].price + 2 * point_size, _Digits);
   }
   // Try bullish FVG below
   for(int i = 0; i < ArraySize(bull_fvgs); i++)
   {
      if(!bull_fvgs[i].filled && bull_fvgs[i].top < entry - sl_dist)
         return NormalizeDouble(bull_fvgs[i].top, _Digits);
   }
   return NormalizeDouble(entry - TP_ATR_Mult * atr, _Digits);
}

//+------------------------------------------------------------------+
//| RISK MANAGEMENT                                                   |
//+------------------------------------------------------------------+
double CalcLot(double sl_dist)
{
   if(Manual_Lot > 0.0) return NormLot(Manual_Lot);
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_money= balance * Risk_Percent / 100.0;
   double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(sl_dist <= 0 || tick_val <= 0 || tick_size <= 0) return NormLot(0.01);
   double lot = risk_money / (sl_dist / tick_size * tick_val);
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
//| EQUITY PROTECTION                                                 |
//+------------------------------------------------------------------+
bool CheckEquityProtection()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd_pct  = (balance > 0) ? (balance - equity) / balance * 100.0 : 0;

   if(dd_pct >= Max_DD_Percent)
   {
      Print("NAS100_ML | EQUITY PROTECTION: DD=", DoubleToString(dd_pct,2), "%%");
      CloseAllPositions();
      panel_signal = "EP TRIGGERED";
      panel_sig_col = clrRed;
      return true;
   }
   return false;
}

bool CheckDailyLossLimit()
{
   if(daily_limit_hit) return true;

   double cur_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double cur_equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double daily_loss  = (day_start_balance - cur_equity) / day_start_balance * 100.0;

   if(daily_loss >= Daily_Loss_Limit)
   {
      daily_limit_hit = true;
      Print("NAS100_ML | DAILY LOSS LIMIT: ", DoubleToString(daily_loss,2), "%%");
      panel_signal = "DAILY LIMIT";
      panel_sig_col = clrOrange;
      return true;
   }
   return false;
}

void CloseAllPositions()
{
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() == _Symbol && pos.Magic() == Magic)
         trade.PositionClose(pos.Ticket());
   }
}


//+------------------------------------------------------------------+
//| BREAK-EVEN MANAGEMENT                                             |
//| Move SL to entry + 1pt after price moves 1x ATR in favor        |
//+------------------------------------------------------------------+
void ManageBreakEven()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double atr = buf_atr[1];
   if(atr <= 0) return;

   double be_dist = BE_ATR_Mult * atr;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != Magic) continue;
      if(TicketInArr(be_tickets, pos.Ticket())) continue;

      bool   is_buy  = (pos.PositionType() == POSITION_TYPE_BUY);
      double open_px = pos.PriceOpen();
      double cur_sl  = pos.StopLoss();
      double cur_px  = is_buy ? bid : ask;
      double profit  = is_buy ? (cur_px - open_px) : (open_px - cur_px);

      if(profit >= be_dist)
      {
         double be_sl = is_buy
            ? NormalizeDouble(open_px + point_size * 5, _Digits)
            : NormalizeDouble(open_px - point_size * 5, _Digits);

         bool should_modify = is_buy ? (be_sl > cur_sl + point_size)
                                     : (cur_sl < point_size || be_sl < cur_sl - point_size);
         if(should_modify)
         {
            if(trade.PositionModify(pos.Ticket(), be_sl, pos.TakeProfit()))
            {
               AddTicket(be_tickets, pos.Ticket());
               Print("NAS100_ML | Break-even set: ticket=", pos.Ticket(),
                     " new_sl=", be_sl);
            }
         }
      }
   }
}


//+------------------------------------------------------------------+
//| ADAPTIVE TRAILING STOP                                            |
//| Trail distance tightens as profit grows (adaptive)               |
//| NOTE: Trailing only activates after break-even is reached (1x    |
//| ATR profit). This is intentional design - positions must reach   |
//| break-even before trailing engages. Positions below the BE       |
//| threshold rely on the initial SL for protection. This gap        |
//| prevents premature trailing from closing trades during normal    |
//| retracement before the move develops.                            |
//+------------------------------------------------------------------+
void ManageTrailing()
{
   if(!Use_Trailing) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double atr = buf_atr[1];
   if(atr <= 0) return;

   double trail_base = Trail_ATR_Mult * atr;
   double trail_step = Trail_Step_Pts * point_size;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != Magic) continue;
      if(!TicketInArr(be_tickets, pos.Ticket())) continue; // Only after BE

      bool   is_buy  = (pos.PositionType() == POSITION_TYPE_BUY);
      double open_px = pos.PriceOpen();
      double cur_sl  = pos.StopLoss();
      double cur_px  = is_buy ? bid : ask;
      double profit  = is_buy ? (cur_px - open_px) : (open_px - cur_px);

      // Adaptive trail: tighten distance as profit grows
      double trail_dist = trail_base;
      if(Use_Adaptive_Trail && profit > 0 && atr > 0)
      {
         double profit_atr_ratio = profit / atr;
         // At 1x ATR profit: full trail distance
         // At 3x ATR profit: 60% of trail distance
         // At 5x+ ATR profit: 40% of trail distance
         double tighten_factor = 1.0;
         if(profit_atr_ratio > 5.0)      tighten_factor = 0.4;
         else if(profit_atr_ratio > 3.0) tighten_factor = 0.6;
         else if(profit_atr_ratio > 2.0) tighten_factor = 0.8;
         trail_dist = trail_base * tighten_factor;
      }

      if(is_buy)
      {
         double new_sl = NormalizeDouble(bid - trail_dist, _Digits);
         double floor  = NormalizeDouble(open_px + point_size * 5, _Digits);
         new_sl = MathMax(new_sl, floor);
         if(new_sl > cur_sl + trail_step)
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
      }
      else
      {
         double new_sl = NormalizeDouble(ask + trail_dist, _Digits);
         double floor  = NormalizeDouble(open_px - point_size * 5, _Digits);
         new_sl = MathMin(new_sl, floor);
         if(cur_sl < point_size || new_sl < cur_sl - trail_step)
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
      }
   }
}


//+------------------------------------------------------------------+
//| PARTIAL CLOSE                                                     |
//| Close 40% at 1.5x ATR profit                                    |
//+------------------------------------------------------------------+
void ManagePartialClose()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double atr = buf_atr[1];
   if(atr <= 0) return;

   double partial_dist = Partial_ATR_Mult * atr;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != Magic) continue;
      if(TicketInArr(partial_tickets, pos.Ticket())) continue;

      bool   is_buy  = (pos.PositionType() == POSITION_TYPE_BUY);
      double open_px = pos.PriceOpen();
      double cur_px  = is_buy ? bid : ask;
      double profit  = is_buy ? (cur_px - open_px) : (open_px - cur_px);
      double vol     = pos.Volume();
      double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

      if(profit >= partial_dist)
      {
         double close_vol = NormLot(vol * Partial_Close_Pct / 100.0);
         if(close_vol >= min_vol)
         {
            trade.PositionClosePartial(pos.Ticket(), close_vol);
            Print("NAS100_ML | Partial close: ticket=", pos.Ticket(),
                  " closed=", close_vol, " at profit=", DoubleToString(profit/_Digits,1));
         }
         AddTicket(partial_tickets, pos.Ticket());
      }
   }
}


//+------------------------------------------------------------------+
//| TIME-BASED EXIT                                                   |
//| Close if trade open more than Max_Trade_Duration bars             |
//+------------------------------------------------------------------+
void ManageTimeExit()
{
   datetime cur_time = iTime(_Symbol, Entry_TF, 0);

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != Magic) continue;

      datetime open_time = (datetime)pos.Time();
      int bars_open = iBars(_Symbol, Entry_TF, open_time, cur_time);

      if(bars_open >= Max_Trade_Duration)
      {
         double profit = pos.Profit() + pos.Swap();
         // Only time-exit if not significantly in profit
         if(profit < buf_atr[1] * 2.0 * pos.Volume())
         {
            trade.PositionClose(pos.Ticket());
            Print("NAS100_ML | Time exit: ticket=", pos.Ticket(),
                  " bars=", bars_open, " pnl=", DoubleToString(profit,2));
         }
      }
   }
}

//+------------------------------------------------------------------+
//| HFT CONTROLS                                                      |
//| Prevent overtrading while allowing momentum entries              |
//+------------------------------------------------------------------+
bool CheckHFTControls()
{
   // Minimum bars between trades
   if(current_bar_index - last_entry_bar < Min_Bars_Between)
      return false;

   // Max trades per hour
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   if(dt.hour != last_trade_hour)
   {
      last_trade_hour = dt.hour;
      trades_this_hour = 0;
   }
   if(trades_this_hour >= Max_Trades_Per_Hour)
      return false;

   // Cooldown after consecutive losses
   if(current_bar_index < cooldown_until_bar)
      return false;

   // Max open trades
   if(open_buys + open_sells >= Max_Trades)
      return false;

   return true;
}

void UpdateConsecLosses()
{
   // Check last closed trade for loss
   datetime now = TimeCurrent();
   if(!HistorySelect(now - 300, now)) return; // last 5 minutes

   for(int i = HistoryDealsTotal()-1; i >= 0; i--)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(ticket, DEAL_MAGIC) != Magic) continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      if(profit < 0)
      {
         consec_losses++;
         if(consec_losses >= Max_Consec_Losses)
         {
            cooldown_until_bar = current_bar_index + Loss_Cooldown_Bars;
            Print("NAS100_ML | Cooldown activated: ", consec_losses,
                  " consecutive losses. Cooling until bar ", cooldown_until_bar);
         }
      }
      else if(profit > 0)
      {
         consec_losses = 0;
      }
      break; // only check most recent
   }
}

bool CheckMomentumReentry()
{
   if(!Momentum_Reentry) return false;

   // Allow rapid re-entry if last trade was a quick profit
   datetime now = TimeCurrent();
   if(!HistorySelect(now - 120, now)) return false; // last 2 minutes

   for(int i = HistoryDealsTotal()-1; i >= 0; i--)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(ticket, DEAL_MAGIC) != Magic) continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      if(profit > 0) return true; // recent profit - allow quick re-entry
      break;
   }
   return false;
}


//+------------------------------------------------------------------+
//| SPREAD FILTER                                                     |
//+------------------------------------------------------------------+
bool CheckSpread()
{
   double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread_pts = spread / point_size;
   return (spread_pts <= Max_Spread_Pts);
}

//+------------------------------------------------------------------+
//| POSITION COUNTING                                                 |
//+------------------------------------------------------------------+
void CountPositions()
{
   open_buys = 0; open_sells = 0; float_pnl = 0;
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      if(!pos.SelectByIndex(i)) continue;
      if(pos.Symbol() != _Symbol || pos.Magic() != Magic) continue;
      float_pnl += pos.Profit() + pos.Swap();
      if(pos.PositionType() == POSITION_TYPE_BUY) open_buys++;
      else                                         open_sells++;
   }
}


//+------------------------------------------------------------------+
//| TICKET ARRAY HELPERS                                              |
//+------------------------------------------------------------------+
bool TicketInArr(ulong &arr[], ulong t)
{
   for(int i = 0; i < ArraySize(arr); i++)
      if(arr[i] == t) return true;
   return false;
}

void AddTicket(ulong &arr[], ulong t)
{
   if(TicketInArr(arr, t)) return;
   int s = ArraySize(arr);
   ArrayResize(arr, s+1);
   arr[s] = t;
}

void CleanTicketArrays()
{
   ulong n1[], n2[];
   ArrayResize(n1, 0);
   ArrayResize(n2, 0);

   for(int i = 0; i < ArraySize(be_tickets); i++)
      if(pos.SelectByTicket(be_tickets[i]))
      { int s = ArraySize(n1); ArrayResize(n1, s+1); n1[s] = be_tickets[i]; }

   for(int i = 0; i < ArraySize(partial_tickets); i++)
      if(pos.SelectByTicket(partial_tickets[i]))
      { int s = ArraySize(n2); ArrayResize(n2, s+1); n2[s] = partial_tickets[i]; }

   ArrayResize(be_tickets, ArraySize(n1));
   if(ArraySize(n1) > 0) ArrayCopy(be_tickets, n1);

   ArrayResize(partial_tickets, ArraySize(n2));
   if(ArraySize(n2) > 0) ArrayCopy(partial_tickets, n2);
}


//+------------------------------------------------------------------+
//| P&L TRACKING                                                      |
//+------------------------------------------------------------------+
void RefreshPnL()
{
   datetime now = TimeCurrent();
   if(now - pnl_cache_time < 60) return;
   pnl_cache_time = now;

   MqlDateTime dt; TimeToStruct(now, dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime today = StructToTime(dt);
   datetime week  = today - (datetime)((dt.day_of_week > 0 ? dt.day_of_week-1 : 6) * 86400);
   dt.day = 1;
   datetime month = StructToTime(dt);

   pnl_today = GetPnL(today, now);
   pnl_week  = GetPnL(week, now);
   pnl_month = GetPnL(month, now);

   // Update trade stats
   UpdateTradeStats(today, now);
}

double GetPnL(datetime from, datetime to)
{
   double pnl = 0;
   if(!HistorySelect(from, to)) return 0;
   for(int i = 0; i < HistoryDealsTotal(); i++)
   {
      ulong t = HistoryDealGetTicket(i);
      if(t == 0) continue;
      if(HistoryDealGetString(t, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(t, DEAL_MAGIC) != Magic) continue;
      if(HistoryDealGetInteger(t, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         pnl += HistoryDealGetDouble(t, DEAL_PROFIT)
               +HistoryDealGetDouble(t, DEAL_SWAP)
               +HistoryDealGetDouble(t, DEAL_COMMISSION);
   }
   return pnl;
}

void UpdateTradeStats(datetime from, datetime to)
{
   total_trades = 0; winning_trades = 0;
   total_profit = 0; total_loss = 0;

   if(!HistorySelect(from, to)) return;
   for(int i = 0; i < HistoryDealsTotal(); i++)
   {
      ulong t = HistoryDealGetTicket(i);
      if(t == 0) continue;
      if(HistoryDealGetString(t, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(t, DEAL_MAGIC) != Magic) continue;
      if(HistoryDealGetInteger(t, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      double profit = HistoryDealGetDouble(t, DEAL_PROFIT);
      total_trades++;
      if(profit > 0)
      {
         winning_trades++;
         total_profit += profit;
      }
      else
      {
         total_loss += MathAbs(profit);
      }
   }
}

//+------------------------------------------------------------------+
//| OnTick - Main trading logic                                       |
//+------------------------------------------------------------------+
void OnTick()
{
   // Refresh buffers
   if(CopyBuffer(h_atr, 0, 0, 10, buf_atr) < 10) return;
   if(CopyBuffer(h_rsi, 0, 0, 10, buf_rsi) < 10) return;

   CountPositions();

   // Equity protection every tick
   if(Use_Equity_Protect && CheckEquityProtection()) { if(Show_Panel) UpdateDashboard(); return; }
   if(CheckDailyLossLimit()) { if(Show_Panel) UpdateDashboard(); return; }

   // Manage open positions every tick
   ManageBreakEven();
   ManageTrailing();
   ManagePartialClose();
   ManageTimeExit();

   // Reset daily limit on new day
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   static int last_day = -1;
   if(dt.day != last_day)
   {
      last_day = dt.day;
      daily_limit_hit = false;
      day_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   }

   // HTF bar update - structure, OBs, FVGs
   datetime htf_bar = iTime(_Symbol, HTF, 0);
   if(htf_bar != last_htf_bar_time)
   {
      last_htf_bar_time = htf_bar;
      UpdateMarketStructure();
      DetectOrderBlocks();
      DetectFVGs();
      CleanTicketArrays();
      UpdateConsecLosses();
   }

   // Entry logic on Entry TF (M1) bar
   datetime entry_bar = iTime(_Symbol, Entry_TF, 0);
   if(entry_bar != last_bar_time)
   {
      last_bar_time = entry_bar;
      current_bar_index++;

      // Update killzone display
      panel_kz_str = GetKillzoneName();

      // Killzone filter
      if(Use_Killzones && !IsAnyKillzone())
      {
         panel_signal = "OUT OF KZ";
         panel_sig_col = clrGray;
         if(Show_Panel) UpdateDashboard();
         return;
      }

      // Spread filter
      if(!CheckSpread())
      {
         panel_signal = "SPREAD WIDE";
         panel_sig_col = clrOrange;
         if(Show_Panel) UpdateDashboard();
         return;
      }

      // HFT controls (allow momentum re-entry override)
      bool hft_ok = CheckHFTControls();
      bool momentum_ok = CheckMomentumReentry();
      if(!hft_ok && !momentum_ok)
      {
         panel_signal = "HFT COOLDOWN";
         panel_sig_col = clrOrange;
         if(Show_Panel) UpdateDashboard();
         return;
      }

      // Generate signal
      double ml_conf = 0;
      ENUM_SIGNAL_TYPE signal = GenerateSignal(ml_conf);

      if(signal == SIG_BUY)
      {
         panel_signal = StringFormat("BUY (%.0f%%)", ml_conf*100);
         panel_sig_col = clrLime;
         OpenTrade(signal, ml_conf);
      }
      else if(signal == SIG_SELL)
      {
         panel_signal = StringFormat("SELL (%.0f%%)", ml_conf*100);
         panel_sig_col = clrTomato;
         OpenTrade(signal, ml_conf);
      }
      else
      {
         panel_signal = IsAnyKillzone() ? "IN KZ - WAITING" : "WAITING";
         panel_sig_col = IsAnyKillzone() ? clrYellow : clrGray;
      }
   }

   if(Show_Panel) UpdateDashboard();
}

//+------------------------------------------------------------------+
//| DASHBOARD                                                         |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   ObjectsDeleteAll(0, lbl);
   int x = 15, y = 25, row = 15;
   color bg = C'15,20,35', border = C'30,80,120';

   ObjRect(lbl+"bg", x-8, y-8, 320, 520, bg, border, 1);

   // Title
   ObjLbl(lbl+"dot",   "\x25CF",                  x,    y,    C'0,200,150', 12, true);
   ObjLbl(lbl+"title", " NAS100 IPDA + ML EA",      x+15, y+1,  clrWhite,      9,  true);
   ObjLbl(lbl+"sub",   " Neural Network + ICT",     x+15, y+13, C'100,120,150',7, false);
   ObjLine(lbl+"d0", x, y+27, 300);

   int r = y+36;
   ObjLbl(lbl+"l_sym",  "Symbol",     x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_sym",  _Symbol,      x+140, r,   clrWhite, 8);
   ObjLbl(lbl+"l_tf",   "Entry TF",   x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_tf",   "M1",         x+140, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_htf",  "HTF",        x, r+row*2, clrSilver, 8);
   ObjLbl(lbl+"v_htf",  "M15 / H1",   x+140, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"l_mdl",  "Model",      x, r+row*3, clrSilver, 8);
   ObjLbl(lbl+"v_mdl",  "---",        x+140, r+row*3, clrWhite, 8);
   ObjLine(lbl+"d1", x, r+row*4+2, 300);

   r = r+row*4+10;
   ObjLbl(lbl+"l_sig",  "Signal",       x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_sig",  "---",          x+140, r,   clrWhite, 8);
   ObjLbl(lbl+"l_ml",   "ML Predict",   x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_ml",   "---",          x+140, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_bias", "IPDA Bias",    x, r+row*2, clrSilver, 8);
   ObjLbl(lbl+"v_bias", "---",          x+140, r+row*2, clrWhite, 8);
   ObjLbl(lbl+"l_ipda", "IPDA Score",   x, r+row*3, clrSilver, 8);
   ObjLbl(lbl+"v_ipda", "---",          x+140, r+row*3, clrWhite, 8);
   ObjLbl(lbl+"l_kz",   "Killzone",     x, r+row*4, clrSilver, 8);
   ObjLbl(lbl+"v_kz",   "---",          x+140, r+row*4, clrWhite, 8);
   ObjLine(lbl+"d2", x, r+row*5+2, 300);

   r = r+row*5+10;
   ObjLbl(lbl+"l_tr",   "Open Trades",  x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_tr",   "0",            x+140, r,   clrWhite, 8);
   ObjLbl(lbl+"l_fp",   "Float P/L",    x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_fp",   "---",          x+140, r+row, clrWhite, 8);
   ObjLine(lbl+"d3", x, r+row*2+2, 300);

   r = r+row*2+10;
   ObjLbl(lbl+"l_td",   "Today P/L",    x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_td",   "---",          x+140, r,   clrWhite, 8);
   ObjLbl(lbl+"l_tw",   "Week P/L",     x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_tw",   "---",          x+140, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_tm",   "Month P/L",    x, r+row*2, clrSilver, 8);
   ObjLbl(lbl+"v_tm",   "---",          x+140, r+row*2, clrWhite, 8);
   ObjLine(lbl+"d4", x, r+row*3+2, 300);

   r = r+row*3+10;
   ObjLbl(lbl+"l_wr",   "Win Rate",     x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_wr",   "---",          x+140, r,   clrWhite, 8);
   ObjLbl(lbl+"l_tt",   "Trades Today", x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_tt",   "---",          x+140, r+row, clrWhite, 8);
   ObjLbl(lbl+"l_aw",   "Avg W/L",      x, r+row*2, clrSilver, 8);
   ObjLbl(lbl+"v_aw",   "---",          x+140, r+row*2, clrWhite, 8);
   ObjLine(lbl+"d5", x, r+row*3+2, 300);

   r = r+row*3+10;
   ObjLbl(lbl+"l_ri",   "Risk/Trade",   x, r,       clrSilver, 8);
   ObjLbl(lbl+"v_ri",   Manual_Lot > 0 ? StringFormat("%.2f lot", Manual_Lot)
                                         : StringFormat("Auto %.1f%%", Risk_Percent),
                                           x+140, r, clrWhite, 8);
   ObjLbl(lbl+"l_st",   "Status",       x, r+row,   clrSilver, 8);
   ObjLbl(lbl+"v_st",   "RUNNING",      x+140, r+row, clrLime, 8);

   ChartRedraw(0);
}

void UpdateDashboard()
{
   if(!Show_Panel) return;

   // Model status
   ObjSet(lbl+"v_mdl", model.loaded ? "LOADED (50 features)" : "NOT LOADED",
          model.loaded ? clrLime : clrOrange);

   // Signal info
   ObjSet(lbl+"v_sig",  panel_signal, panel_sig_col);
   ObjSet(lbl+"v_ml",   panel_ml_pred,
          panel_ml_conf >= ML_Strong_Thresh ? clrLime :
          panel_ml_conf >= ML_Confidence_Thresh ? clrYellow : clrGray);
   ObjSet(lbl+"v_bias", panel_bias_str,
          market_bias == BIAS_BULLISH ? clrLime :
          market_bias == BIAS_BEARISH ? clrTomato : clrGray);
   ObjSet(lbl+"v_ipda", panel_ipda_str, clrWhite);
   ObjSet(lbl+"v_kz",   panel_kz_str,
          IsAnyKillzone() ? clrLime : clrGray);

   // Trade info
   ObjSet(lbl+"v_tr", StringFormat("%d (B:%d S:%d)",
          open_buys+open_sells, open_buys, open_sells), clrWhite);
   ObjSet(lbl+"v_fp", StringFormat("%.2f", float_pnl),
          float_pnl >= 0 ? clrLime : clrTomato);

   // P&L
   RefreshPnL();
   ObjSet(lbl+"v_td", StringFormat("%.2f", pnl_today), pnl_today >= 0 ? clrLime : clrTomato);
   ObjSet(lbl+"v_tw", StringFormat("%.2f", pnl_week),  pnl_week >= 0 ? clrLime : clrTomato);
   ObjSet(lbl+"v_tm", StringFormat("%.2f", pnl_month), pnl_month >= 0 ? clrLime : clrTomato);

   // Stats
   double win_rate = (total_trades > 0) ? (double)winning_trades / total_trades * 100.0 : 0;
   double avg_win  = (winning_trades > 0) ? total_profit / winning_trades : 0;
   double avg_loss = (total_trades - winning_trades > 0) ?
                     total_loss / (total_trades - winning_trades) : 0;

   ObjSet(lbl+"v_wr", StringFormat("%.1f%% (%d/%d)", win_rate, winning_trades, total_trades),
          win_rate >= 50 ? clrLime : clrOrange);
   ObjSet(lbl+"v_tt", IntegerToString(total_trades), clrWhite);
   ObjSet(lbl+"v_aw", StringFormat("+%.1f / -%.1f", avg_win, avg_loss),
          avg_win > avg_loss ? clrLime : clrOrange);

   // Status
   string status = "RUNNING";
   color  st_col = clrLime;
   if(panel_signal == "EP TRIGGERED")    { status = "EP STOPPED";    st_col = clrRed; }
   else if(panel_signal == "DAILY LIMIT"){ status = "DAILY STOPPED"; st_col = clrOrange; }
   else if(!model.loaded)                { status = "IPDA ONLY";     st_col = clrYellow; }
   ObjSet(lbl+"v_st", status, st_col);

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| CHART DRAWING HELPERS                                             |
//+------------------------------------------------------------------+
void DrawOBBox(string name, datetime t1, double p1, datetime t2, double p2, color clr, bool bull)
{
   if(ObjectFind(0, name) >= 0) ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, bull ? "Bullish OB" : "Bearish OB");
}

void DrawFVGBox(string name, datetime t1, double p1, datetime t2, double p2, color clr)
{
   if(ObjectFind(0, name) >= 0) ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, "Fair Value Gap");
}


//+------------------------------------------------------------------+
//| PANEL OBJECT HELPERS                                              |
//+------------------------------------------------------------------+
void ObjLbl(string name, string txt, int x, int y, color c, int fs=8, bool bold=false)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fs);
   ObjectSetString(0, name, OBJPROP_FONT, bold ? "Arial Bold" : "Arial");
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}

void ObjSet(string name, string txt, color c)
{
   if(ObjectFind(0, name) < 0) return;
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
}

void ObjLine(string name, int x, int y, int w)
{
   string d = "";
   int n = (int)(w / 5.5);
   for(int i = 0; i < n; i++) d += "-";
   ObjLbl(name, d, x, y, C'30,80,120', 6);
}

void ObjRect(string name, int x, int y, int w, int h, color bg, color border, int bw)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_COLOR, border);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, bw);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}
//+------------------------------------------------------------------+
