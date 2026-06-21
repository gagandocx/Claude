//+------------------------------------------------------------------+
//|                                              AI_Adaptive_EA.mq5  |
//|                        AI-Based Adaptive Expert Advisor           |
//|                        Self-Learning Market Adaptation System     |
//+------------------------------------------------------------------+
//| METHODOLOGY:                                                      |
//|                                                                    |
//| This Expert Advisor implements a multi-layered artificial         |
//| intelligence system that continuously learns from live market     |
//| data and adapts its trading behavior in real-time.                |
//|                                                                    |
//| CORE COMPONENTS:                                                   |
//|                                                                    |
//| 1. ONLINE LEARNING ENGINE                                          |
//|    A lightweight feedforward neural network (perceptron-style)    |
//|    that processes market features and outputs trade signals.      |
//|    Features include: price momentum, volatility ratio, trend      |
//|    strength, mean-reversion signal, and volume profile.           |
//|    Weights are updated via simplified stochastic gradient         |
//|    descent after each trade outcome (reward/penalty signal).      |
//|                                                                    |
//| 2. MARKET REGIME DETECTION                                         |
//|    Classifies market state into Trending, Ranging, or Volatile   |
//|    using: ADX (trend strength), ATR percentile (volatility),     |
//|    and Hurst exponent approximation (persistence vs mean-        |
//|    reversion). Outputs a probability distribution over states.   |
//|                                                                    |
//| 3. ADAPTIVE STRATEGY SELECTION                                     |
//|    Three sub-strategies compete for capital allocation:           |
//|    (a) Trend Following with adaptive moving average periods      |
//|    (b) Mean Reversion with dynamic Bollinger Band deviations     |
//|    (c) Breakout with self-adjusting confirmation thresholds      |
//|    Each strategy's weight is updated based on rolling P&L.       |
//|                                                                    |
//| 4. DYNAMIC POSITION SIZING                                         |
//|    Kelly Criterion approximation adjusted by drawdown level,     |
//|    current volatility regime, and serial correlation of trades.  |
//|                                                                    |
//| 5. PROFIT BOOKING SYSTEM                                           |
//|    Multi-level take-profit with ATR-adaptive trailing stops,     |
//|    partial closure at key levels, and time-based exit for        |
//|    trades that fail to perform within expected timeframes.       |
//|                                                                    |
//| 6. SELF-LEARNING PERFORMANCE TRACKER                               |
//|    Maintains rolling statistics (Sharpe ratio, win rate, profit  |
//|    factor) per regime. Uses these metrics to continuously        |
//|    refine strategy weights and neural network parameters.        |
//|                                                                    |
//| CONFIGURATION GUIDE:                                               |
//|    - LearningRate: Controls how fast the network adapts          |
//|      (lower = more stable, higher = faster adaptation)           |
//|    - RegimeLookback: Bars used for regime classification         |
//|    - MaxRiskPercent: Maximum risk per trade as % of equity       |
//|    - AdaptationSpeed: How quickly strategy weights shift         |
//|    - TrailingATRMultiplier: ATR multiplier for trailing stops    |
//|    - PartialClosePercent: Portion closed at first TP level      |
//+------------------------------------------------------------------+
#property copyright "AI Adaptive Trading System"
#property link      ""
#property version   "1.00"
#property description "Self-learning EA with neural network, regime detection, and adaptive strategies"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                   |
//+------------------------------------------------------------------+
// --- Neural Network Parameters ---
input double   InpLearningRate        = 0.01;    // Learning Rate for weight updates
input int      InpNetworkInputs       = 5;       // Number of input features
input int      InpHiddenNodes         = 8;       // Hidden layer nodes
input double   InpMomentumFactor      = 0.9;     // Momentum for gradient descent
input double   InpWeightDecay         = 0.001;   // L2 regularization strength

// --- Regime Detection Parameters ---
input int      InpRegimeLookback      = 50;      // Bars for regime detection
input int      InpADXPeriod           = 14;      // ADX period
input int      InpATRPeriod           = 14;      // ATR period
input double   InpTrendThreshold      = 25.0;    // ADX threshold for trending
input double   InpVolatileThreshold   = 75.0;    // ATR percentile for volatile
input int      InpHurstPeriod         = 100;     // Period for Hurst exponent calc

// --- Strategy Parameters ---
input int      InpFastMAPeriod        = 10;      // Fast MA base period
input int      InpSlowMAPeriod        = 30;      // Slow MA base period
input int      InpBBPeriod            = 20;      // Bollinger Bands period
input double   InpBBDeviation         = 2.0;     // BB base deviation
input int      InpBreakoutPeriod      = 20;      // Breakout lookback period
input double   InpBreakoutConfirm     = 1.5;     // Breakout confirmation multiplier
input double   InpAdaptationSpeed     = 0.1;     // Strategy weight adaptation speed

// --- Risk Management Parameters ---
input double   InpMaxRiskPercent      = 2.0;     // Max risk per trade (%)
input double   InpMaxDrawdownPercent  = 15.0;    // Max drawdown before reducing size (%)
input double   InpKellyFraction       = 0.25;    // Kelly fraction (conservative)
input int      InpMaxOpenTrades       = 3;       // Maximum concurrent trades
input double   InpMinLotSize          = 0.01;    // Minimum lot size
input double   InpMaxLotSize          = 10.0;    // Maximum lot size

// --- Profit Booking Parameters ---
input double   InpTP1_ATRMultiple     = 1.5;     // First TP level (ATR multiple)
input double   InpTP2_ATRMultiple     = 3.0;     // Second TP level (ATR multiple)
input double   InpTP3_ATRMultiple     = 5.0;     // Third TP level (ATR multiple)
input double   InpTrailingATRMult     = 1.0;     // Trailing stop ATR multiplier
input double   InpPartialClose1       = 0.4;     // Partial close at TP1 (fraction)
input double   InpPartialClose2       = 0.3;     // Partial close at TP2 (fraction)
input int      InpStaleTradeHours     = 48;      // Hours before time-based exit
input double   InpSLATRMultiple       = 2.0;     // Stop loss ATR multiplier

// --- Performance Tracking ---
input int      InpPerformanceWindow   = 50;      // Rolling window for stats
input int      InpMinTradesForLearning= 10;      // Min trades before adaptation

// --- General ---
input int      InpMagicNumber         = 202401;  // Magic number
input bool     InpShowDashboard       = true;    // Show chart dashboard
input ENUM_TIMEFRAMES InpTimeframe    = PERIOD_H1; // Operating timeframe

//+------------------------------------------------------------------+
//| CONSTANTS                                                          |
//+------------------------------------------------------------------+
#define MAX_NETWORK_INPUTS  5
#define MAX_HIDDEN_NODES    8
#define MAX_TRADE_HISTORY   500
#define NUM_STRATEGIES      3
#define NUM_REGIMES         3

// Regime enumerations
enum ENUM_MARKET_REGIME
{
   REGIME_TRENDING = 0,
   REGIME_RANGING  = 1,
   REGIME_VOLATILE = 2
};

// Strategy enumerations
enum ENUM_STRATEGY
{
   STRATEGY_TREND_FOLLOW = 0,
   STRATEGY_MEAN_REVERT  = 1,
   STRATEGY_BREAKOUT     = 2
};

// Trade signal
enum ENUM_AI_SIGNAL
{
   SIGNAL_NONE  = 0,
   SIGNAL_BUY   = 1,
   SIGNAL_SELL  = -1
};

//+------------------------------------------------------------------+
//| NEURAL NETWORK STRUCTURE                                           |
//+------------------------------------------------------------------+
struct NeuralNetwork
{
   double inputWeights[MAX_HIDDEN_NODES][MAX_NETWORK_INPUTS];  // Input to hidden
   double hiddenBias[MAX_HIDDEN_NODES];                         // Hidden biases
   double outputWeights[MAX_HIDDEN_NODES];                      // Hidden to output
   double outputBias;                                           // Output bias
   double prevDeltaIW[MAX_HIDDEN_NODES][MAX_NETWORK_INPUTS];   // Momentum for input weights
   double prevDeltaHB[MAX_HIDDEN_NODES];                        // Momentum for hidden biases
   double prevDeltaOW[MAX_HIDDEN_NODES];                        // Momentum for output weights
   double prevDeltaOB;                                          // Momentum for output bias
   double lastHiddenOutput[MAX_HIDDEN_NODES];                   // Cached forward pass
   double lastOutput;                                           // Last network output
};

//+------------------------------------------------------------------+
//| REGIME DETECTION STRUCTURE                                         |
//+------------------------------------------------------------------+
struct RegimeState
{
   double probTrending;     // Probability of trending regime
   double probRanging;      // Probability of ranging regime
   double probVolatile;     // Probability of volatile regime
   ENUM_MARKET_REGIME current; // Current dominant regime
   double adxValue;         // Current ADX value
   double atrPercentile;    // Current ATR percentile
   double hurstExponent;    // Hurst exponent approximation
};

//+------------------------------------------------------------------+
//| STRATEGY STATE STRUCTURE                                           |
//+------------------------------------------------------------------+
struct StrategyState
{
   double weights[NUM_STRATEGIES];         // Current strategy weights
   double performance[NUM_STRATEGIES];     // Rolling performance score
   int    tradeCount[NUM_STRATEGIES];      // Trades per strategy
   double pnl[NUM_STRATEGIES];             // Cumulative PnL per strategy
   double adaptiveFastMA;                  // Adaptive fast MA period
   double adaptiveSlowMA;                  // Adaptive slow MA period
   double adaptiveBBDev;                   // Adaptive BB deviation
   double adaptiveBreakoutConf;            // Adaptive breakout confirmation
   ENUM_STRATEGY activeStrategy;           // Currently selected strategy
   double confidence;                      // Confidence in current signal
};

//+------------------------------------------------------------------+
//| TRADE RECORD STRUCTURE                                             |
//+------------------------------------------------------------------+
struct TradeRecord
{
   datetime openTime;
   datetime closeTime;
   double   profit;
   double   profitPoints;
   ENUM_MARKET_REGIME regime;
   ENUM_STRATEGY strategy;
   int      direction;       // 1 = buy, -1 = sell
   double   entryPrice;
   double   exitPrice;
   double   lotSize;
   bool     isWin;
};

//+------------------------------------------------------------------+
//| PERFORMANCE METRICS STRUCTURE                                       |
//+------------------------------------------------------------------+
struct PerformanceMetrics
{
   double winRate;
   double profitFactor;
   double sharpeRatio;
   double avgWin;
   double avgLoss;
   double maxDrawdown;
   double currentDrawdown;
   double peakEquity;
   int    consecutiveWins;
   int    consecutiveLosses;
   double recentCorrelation;   // Serial correlation of trade outcomes
};

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                    |
//+------------------------------------------------------------------+
NeuralNetwork      g_network;
RegimeState        g_regime;
StrategyState      g_strategy;
PerformanceMetrics g_metrics;
TradeRecord        g_tradeHistory[];
int                g_tradeHistoryCount = 0;

// Trade management
CTrade             g_trade;
CPositionInfo      g_position;
CAccountInfo       g_account;
CSymbolInfo        g_symbol;

// Indicator handles
int  g_handleADX;
int  g_handleATR;
int  g_handleBBUpper;
int  g_handleBBLower;
int  g_handleBBMiddle;
int  g_handleBB;
int  g_handleFastMA;
int  g_handleSlowMA;

// State tracking
datetime g_lastBarTime       = 0;
double   g_initialBalance    = 0;
int      g_totalTrades       = 0;
bool     g_isNewBar          = false;
double   g_atrBuffer[];
double   g_adxBuffer[];
double   g_bbUpperBuffer[];
double   g_bbLowerBuffer[];
double   g_bbMiddleBuffer[];
double   g_fastMABuffer[];
double   g_slowMABuffer[];

// Dashboard objects
string   g_dashPrefix = "AI_DASH_";


//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize symbol info
   g_symbol.Name(_Symbol);
   g_symbol.Refresh();
   
   // Set magic number
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   // Store initial balance
   g_initialBalance = g_account.Balance();
   g_metrics.peakEquity = g_initialBalance;
   
   // Initialize neural network with small random weights
   InitializeNetwork();
   
   // Initialize strategy weights (equal initially)
   for(int i = 0; i < NUM_STRATEGIES; i++)
   {
      g_strategy.weights[i] = 1.0 / NUM_STRATEGIES;
      g_strategy.performance[i] = 0.0;
      g_strategy.tradeCount[i] = 0;
      g_strategy.pnl[i] = 0.0;
   }
   g_strategy.adaptiveFastMA = InpFastMAPeriod;
   g_strategy.adaptiveSlowMA = InpSlowMAPeriod;
   g_strategy.adaptiveBBDev  = InpBBDeviation;
   g_strategy.adaptiveBreakoutConf = InpBreakoutConfirm;
   
   // Create indicator handles
   g_handleADX = iADX(_Symbol, InpTimeframe, InpADXPeriod);
   g_handleATR = iATR(_Symbol, InpTimeframe, InpATRPeriod);
   g_handleBB  = iBands(_Symbol, InpTimeframe, InpBBPeriod, 0, InpBBDeviation, PRICE_CLOSE);
   g_handleFastMA = iMA(_Symbol, InpTimeframe, InpFastMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   g_handleSlowMA = iMA(_Symbol, InpTimeframe, InpSlowMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   
   if(g_handleADX == INVALID_HANDLE || g_handleATR == INVALID_HANDLE ||
      g_handleBB == INVALID_HANDLE || g_handleFastMA == INVALID_HANDLE ||
      g_handleSlowMA == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return(INIT_FAILED);
   }
   
   // Allocate trade history
   ArrayResize(g_tradeHistory, MAX_TRADE_HISTORY);
   ArrayResize(g_atrBuffer, InpRegimeLookback + 10);
   ArrayResize(g_adxBuffer, InpRegimeLookback + 10);
   ArrayResize(g_bbUpperBuffer, InpRegimeLookback + 10);
   ArrayResize(g_bbLowerBuffer, InpRegimeLookback + 10);
   ArrayResize(g_bbMiddleBuffer, InpRegimeLookback + 10);
   ArrayResize(g_fastMABuffer, InpRegimeLookback + 10);
   ArrayResize(g_slowMABuffer, InpRegimeLookback + 10);
   
   // Set buffer series direction
   ArraySetAsSeries(g_atrBuffer, true);
   ArraySetAsSeries(g_adxBuffer, true);
   ArraySetAsSeries(g_bbUpperBuffer, true);
   ArraySetAsSeries(g_bbLowerBuffer, true);
   ArraySetAsSeries(g_bbMiddleBuffer, true);
   ArraySetAsSeries(g_fastMABuffer, true);
   ArraySetAsSeries(g_slowMABuffer, true);
   
   // Initialize regime
   g_regime.probTrending = 0.33;
   g_regime.probRanging  = 0.34;
   g_regime.probVolatile = 0.33;
   g_regime.current = REGIME_RANGING;
   
   // Initialize performance metrics
   g_metrics.winRate = 0.5;
   g_metrics.profitFactor = 1.0;
   g_metrics.sharpeRatio = 0.0;
   g_metrics.maxDrawdown = 0.0;
   g_metrics.currentDrawdown = 0.0;
   g_metrics.consecutiveWins = 0;
   g_metrics.consecutiveLosses = 0;
   g_metrics.recentCorrelation = 0.0;
   
   Print("AI Adaptive EA initialized successfully");
   Print("Learning Rate: ", InpLearningRate, " | Hidden Nodes: ", InpHiddenNodes);
   Print("Max Risk: ", InpMaxRiskPercent, "% | Magic: ", InpMagicNumber);
   
   if(InpShowDashboard)
      CreateDashboard();
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   if(g_handleADX != INVALID_HANDLE)    IndicatorRelease(g_handleADX);
   if(g_handleATR != INVALID_HANDLE)    IndicatorRelease(g_handleATR);
   if(g_handleBB != INVALID_HANDLE)     IndicatorRelease(g_handleBB);
   if(g_handleFastMA != INVALID_HANDLE) IndicatorRelease(g_handleFastMA);
   if(g_handleSlowMA != INVALID_HANDLE) IndicatorRelease(g_handleSlowMA);
   
   // Remove dashboard objects
   if(InpShowDashboard)
      RemoveDashboard();
   
   // Log final performance summary
   Print("=== AI Adaptive EA Final Summary ===");
   Print("Total Trades: ", g_totalTrades);
   Print("Win Rate: ", DoubleToString(g_metrics.winRate * 100, 1), "%");
   Print("Profit Factor: ", DoubleToString(g_metrics.profitFactor, 2));
   Print("Sharpe Ratio: ", DoubleToString(g_metrics.sharpeRatio, 2));
   Print("Max Drawdown: ", DoubleToString(g_metrics.maxDrawdown, 2), "%");
   Print("Strategy Weights - Trend: ", DoubleToString(g_strategy.weights[0], 3),
         " | MeanRev: ", DoubleToString(g_strategy.weights[1], 3),
         " | Breakout: ", DoubleToString(g_strategy.weights[2], 3));
   Print("===================================");
}


//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   g_isNewBar = IsNewBar();
   
   if(!g_isNewBar)
   {
      // On every tick: manage existing positions (trailing stops, time exits)
      ManageOpenPositions();
      return;
   }
   
   // Refresh symbol data
   g_symbol.Refresh();
   g_symbol.RefreshRates();
   
   // Copy indicator buffers
   if(!RefreshIndicatorBuffers())
      return;
   
   // Step 1: Detect market regime
   DetectMarketRegime();
   
   // Step 2: Compute neural network features
   double features[MAX_NETWORK_INPUTS];
   ComputeFeatures(features);
   
   // Step 3: Forward pass through neural network
   double networkOutput = ForwardPass(features);
   
   // Step 4: Get strategy signals
   ENUM_AI_SIGNAL trendSignal    = GetTrendFollowingSignal();
   ENUM_AI_SIGNAL meanRevSignal  = GetMeanReversionSignal();
   ENUM_AI_SIGNAL breakoutSignal = GetBreakoutSignal();
   
   // Step 5: Combine signals using adaptive weights and network output
   ENUM_AI_SIGNAL finalSignal = CombineSignals(trendSignal, meanRevSignal, breakoutSignal, networkOutput);
   
   // Step 6: Check if we can trade
   if(finalSignal != SIGNAL_NONE && CanOpenNewTrade())
   {
      // Step 7: Calculate position size
      double lotSize = CalculatePositionSize();
      
      // Step 8: Execute trade with multi-level TP
      ExecuteTrade(finalSignal, lotSize);
   }
   
   // Step 9: Update dashboard
   if(InpShowDashboard)
      UpdateDashboard();
}

//+------------------------------------------------------------------+
//| Trade event handler                                                |
//+------------------------------------------------------------------+
void OnTrade()
{
   // Check for newly closed positions
   ProcessClosedTrades();
}

//+------------------------------------------------------------------+
//| NEURAL NETWORK FUNCTIONS                                           |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Initialize network with small random weights                       |
//+------------------------------------------------------------------+
void InitializeNetwork()
{
   MathSrand((int)TimeLocal());
   
   for(int h = 0; h < InpHiddenNodes && h < MAX_HIDDEN_NODES; h++)
   {
      for(int i = 0; i < InpNetworkInputs && i < MAX_NETWORK_INPUTS; i++)
      {
         // Xavier initialization: random in [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
         double limit = MathSqrt(6.0 / (InpNetworkInputs + InpHiddenNodes));
         g_network.inputWeights[h][i] = RandomDouble(-limit, limit);
         g_network.prevDeltaIW[h][i] = 0.0;
      }
      g_network.hiddenBias[h] = 0.0;
      g_network.prevDeltaHB[h] = 0.0;
      
      double limit2 = MathSqrt(6.0 / (InpHiddenNodes + 1));
      g_network.outputWeights[h] = RandomDouble(-limit2, limit2);
      g_network.prevDeltaOW[h] = 0.0;
      g_network.lastHiddenOutput[h] = 0.0;
   }
   g_network.outputBias = 0.0;
   g_network.prevDeltaOB = 0.0;
   g_network.lastOutput = 0.0;
}

//+------------------------------------------------------------------+
//| Compute input features for the neural network                      |
//+------------------------------------------------------------------+
void ComputeFeatures(double &features[])
{
   // Feature 1: Price Momentum (normalized rate of change)
   double close0 = iClose(_Symbol, InpTimeframe, 0);
   double close10 = iClose(_Symbol, InpTimeframe, 10);
   double momentum = 0.0;
   if(close10 > 0)
      momentum = (close0 - close10) / close10;
   features[0] = TanhActivation(momentum * 100.0); // Normalize to [-1, 1]
   
   // Feature 2: Volatility Ratio (current ATR vs average ATR)
   double currentATR = g_atrBuffer[0];
   double avgATR = 0.0;
   for(int i = 0; i < InpRegimeLookback && i < ArraySize(g_atrBuffer); i++)
      avgATR += g_atrBuffer[i];
   avgATR /= InpRegimeLookback;
   double volRatio = 0.0;
   if(avgATR > 0)
      volRatio = currentATR / avgATR - 1.0;
   features[1] = TanhActivation(volRatio * 2.0);
   
   // Feature 3: Trend Strength (ADX normalized)
   features[2] = TanhActivation((g_adxBuffer[0] - 25.0) / 25.0);
   
   // Feature 4: Mean Reversion Signal (distance from BB middle)
   double bbMiddle = g_bbMiddleBuffer[0];
   double bbUpper  = g_bbUpperBuffer[0];
   double bbRange  = bbUpper - bbMiddle;
   double meanRevSignal = 0.0;
   if(bbRange > 0)
      meanRevSignal = (close0 - bbMiddle) / bbRange;
   features[3] = TanhActivation(meanRevSignal);
   
   // Feature 5: Volume Profile (tick volume relative to average)
   long currentVolume = iVolume(_Symbol, InpTimeframe, 0);
   long avgVolume = 0;
   for(int i = 1; i <= 20; i++)
      avgVolume += iVolume(_Symbol, InpTimeframe, i);
   avgVolume /= 20;
   double volProfile = 0.0;
   if(avgVolume > 0)
      volProfile = (double)currentVolume / (double)avgVolume - 1.0;
   features[4] = TanhActivation(volProfile);
}

//+------------------------------------------------------------------+
//| Forward pass through the neural network                            |
//+------------------------------------------------------------------+
double ForwardPass(double &features[])
{
   // Hidden layer computation
   for(int h = 0; h < InpHiddenNodes && h < MAX_HIDDEN_NODES; h++)
   {
      double sum = g_network.hiddenBias[h];
      for(int i = 0; i < InpNetworkInputs && i < MAX_NETWORK_INPUTS; i++)
      {
         sum += g_network.inputWeights[h][i] * features[i];
      }
      g_network.lastHiddenOutput[h] = TanhActivation(sum);
   }
   
   // Output layer computation
   double outputSum = g_network.outputBias;
   for(int h = 0; h < InpHiddenNodes && h < MAX_HIDDEN_NODES; h++)
   {
      outputSum += g_network.outputWeights[h] * g_network.lastHiddenOutput[h];
   }
   
   g_network.lastOutput = TanhActivation(outputSum);
   return g_network.lastOutput;
}

//+------------------------------------------------------------------+
//| Update network weights via simplified gradient descent             |
//+------------------------------------------------------------------+
void UpdateNetworkWeights(double reward, double &features[])
{
   // reward: +1 for profitable trade, -1 for loss, scaled by magnitude
   // This implements a policy gradient-like update
   
   double lr = InpLearningRate;
   double momentum = InpMomentumFactor;
   double decay = InpWeightDecay;
   
   // Compute output error gradient
   double outputError = reward - g_network.lastOutput;
   double outputGrad = outputError * TanhDerivative(g_network.lastOutput);
   
   // Update output weights with momentum and weight decay
   for(int h = 0; h < InpHiddenNodes && h < MAX_HIDDEN_NODES; h++)
   {
      double delta = lr * outputGrad * g_network.lastHiddenOutput[h] 
                     - decay * g_network.outputWeights[h]
                     + momentum * g_network.prevDeltaOW[h];
      g_network.outputWeights[h] += delta;
      g_network.prevDeltaOW[h] = delta;
   }
   
   // Update output bias
   double deltaBias = lr * outputGrad + momentum * g_network.prevDeltaOB;
   g_network.outputBias += deltaBias;
   g_network.prevDeltaOB = deltaBias;
   
   // Backpropagate to hidden layer
   for(int h = 0; h < InpHiddenNodes && h < MAX_HIDDEN_NODES; h++)
   {
      double hiddenError = outputGrad * g_network.outputWeights[h];
      double hiddenGrad = hiddenError * TanhDerivative(g_network.lastHiddenOutput[h]);
      
      // Update input weights
      for(int i = 0; i < InpNetworkInputs && i < MAX_NETWORK_INPUTS; i++)
      {
         double deltaW = lr * hiddenGrad * features[i]
                         - decay * g_network.inputWeights[h][i]
                         + momentum * g_network.prevDeltaIW[h][i];
         g_network.inputWeights[h][i] += deltaW;
         g_network.prevDeltaIW[h][i] = deltaW;
      }
      
      // Update hidden bias
      double deltaHB = lr * hiddenGrad + momentum * g_network.prevDeltaHB[h];
      g_network.hiddenBias[h] += deltaHB;
      g_network.prevDeltaHB[h] = deltaHB;
   }
}


//+------------------------------------------------------------------+
//| MARKET REGIME DETECTION                                            |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Detect current market regime                                       |
//+------------------------------------------------------------------+
void DetectMarketRegime()
{
   // Get ADX value for trend strength
   g_regime.adxValue = g_adxBuffer[0];
   
   // Calculate ATR percentile for volatility classification
   g_regime.atrPercentile = CalculateATRPercentile();
   
   // Calculate Hurst exponent for persistence detection
   g_regime.hurstExponent = CalculateHurstExponent();
   
   // Compute regime probabilities using fuzzy logic
   // Trending: High ADX + Hurst > 0.5 (persistent)
   double trendScore = 0.0;
   if(g_regime.adxValue > InpTrendThreshold)
      trendScore = MathMin(1.0, (g_regime.adxValue - InpTrendThreshold) / 25.0);
   if(g_regime.hurstExponent > 0.55)
      trendScore *= 1.0 + (g_regime.hurstExponent - 0.55) * 2.0;
   
   // Ranging: Low ADX + Hurst < 0.5 (mean-reverting)
   double rangeScore = 0.0;
   if(g_regime.adxValue < InpTrendThreshold)
      rangeScore = MathMin(1.0, (InpTrendThreshold - g_regime.adxValue) / 15.0);
   if(g_regime.hurstExponent < 0.45)
      rangeScore *= 1.0 + (0.45 - g_regime.hurstExponent) * 2.0;
   
   // Volatile: High ATR percentile regardless of direction
   double volatileScore = 0.0;
   if(g_regime.atrPercentile > InpVolatileThreshold)
      volatileScore = MathMin(1.0, (g_regime.atrPercentile - InpVolatileThreshold) / 25.0);
   
   // Normalize to probability distribution
   double totalScore = trendScore + rangeScore + volatileScore;
   if(totalScore <= 0.0)
   {
      // Default: assume ranging
      g_regime.probTrending = 0.2;
      g_regime.probRanging  = 0.6;
      g_regime.probVolatile = 0.2;
   }
   else
   {
      g_regime.probTrending = trendScore / totalScore;
      g_regime.probRanging  = rangeScore / totalScore;
      g_regime.probVolatile = volatileScore / totalScore;
   }
   
   // Determine dominant regime
   if(g_regime.probTrending >= g_regime.probRanging && g_regime.probTrending >= g_regime.probVolatile)
      g_regime.current = REGIME_TRENDING;
   else if(g_regime.probRanging >= g_regime.probTrending && g_regime.probRanging >= g_regime.probVolatile)
      g_regime.current = REGIME_RANGING;
   else
      g_regime.current = REGIME_VOLATILE;
}

//+------------------------------------------------------------------+
//| Calculate ATR percentile over lookback period                       |
//+------------------------------------------------------------------+
double CalculateATRPercentile()
{
   double currentATR = g_atrBuffer[0];
   int countBelow = 0;
   int total = MathMin(InpRegimeLookback, ArraySize(g_atrBuffer));
   
   for(int i = 1; i < total; i++)
   {
      if(g_atrBuffer[i] < currentATR)
         countBelow++;
   }
   
   return (total > 1) ? (double)countBelow / (double)(total - 1) * 100.0 : 50.0;
}

//+------------------------------------------------------------------+
//| Calculate Hurst Exponent approximation via Rescaled Range          |
//+------------------------------------------------------------------+
double CalculateHurstExponent()
{
   int periods = MathMin(InpHurstPeriod, iBars(_Symbol, InpTimeframe) - 1);
   if(periods < 20) return 0.5; // Not enough data
   
   // Calculate log returns
   double returns[];
   ArrayResize(returns, periods);
   for(int i = 0; i < periods; i++)
   {
      double closeI = iClose(_Symbol, InpTimeframe, i);
      double closeI1 = iClose(_Symbol, InpTimeframe, i + 1);
      if(closeI1 > 0)
         returns[i] = MathLog(closeI / closeI1);
      else
         returns[i] = 0.0;
   }
   
   // Use multiple sub-period lengths for R/S analysis
   double sumLogRS = 0.0;
   double sumLogN = 0.0;
   double sumLogRS_LogN = 0.0;
   double sumLogN2 = 0.0;
   int numPoints = 0;
   
   int subLengths[] = {10, 15, 20, 30, 40, 50};
   int numLengths = ArraySize(subLengths);
   
   for(int s = 0; s < numLengths; s++)
   {
      int subLen = subLengths[s];
      if(subLen > periods) continue;
      
      int numSubs = periods / subLen;
      if(numSubs < 1) continue;
      
      double avgRS = 0.0;
      int validSubs = 0;
      
      for(int sub = 0; sub < numSubs; sub++)
      {
         int startIdx = sub * subLen;
         
         // Calculate mean of sub-period
         double mean = 0.0;
         for(int i = 0; i < subLen; i++)
            mean += returns[startIdx + i];
         mean /= subLen;
         
         // Calculate cumulative deviations and standard deviation
         double cumDev = 0.0;
         double maxCumDev = -DBL_MAX;
         double minCumDev = DBL_MAX;
         double sumSq = 0.0;
         
         for(int i = 0; i < subLen; i++)
         {
            double dev = returns[startIdx + i] - mean;
            cumDev += dev;
            sumSq += dev * dev;
            if(cumDev > maxCumDev) maxCumDev = cumDev;
            if(cumDev < minCumDev) minCumDev = cumDev;
         }
         
         double range = maxCumDev - minCumDev;
         double stdDev = MathSqrt(sumSq / subLen);
         
         if(stdDev > 0.0)
         {
            avgRS += range / stdDev;
            validSubs++;
         }
      }
      
      if(validSubs > 0)
      {
         avgRS /= validSubs;
         double logRS = MathLog(avgRS);
         double logN  = MathLog((double)subLen);
         
         sumLogRS += logRS;
         sumLogN  += logN;
         sumLogRS_LogN += logRS * logN;
         sumLogN2 += logN * logN;
         numPoints++;
      }
   }
   
   // Linear regression to estimate Hurst exponent
   if(numPoints < 2) return 0.5;
   
   double hurst = (numPoints * sumLogRS_LogN - sumLogRS * sumLogN) / 
                  (numPoints * sumLogN2 - sumLogN * sumLogN);
   
   // Clamp to valid range [0, 1]
   hurst = MathMax(0.0, MathMin(1.0, hurst));
   
   return hurst;
}


//+------------------------------------------------------------------+
//| ADAPTIVE STRATEGY FUNCTIONS                                        |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Strategy 1: Trend Following with Adaptive MA Periods               |
//+------------------------------------------------------------------+
ENUM_AI_SIGNAL GetTrendFollowingSignal()
{
   double fastMA = g_fastMABuffer[0];
   double slowMA = g_slowMABuffer[0];
   double fastMA1 = g_fastMABuffer[1];
   double slowMA1 = g_slowMABuffer[1];
   double close0 = iClose(_Symbol, InpTimeframe, 0);
   
   // Adaptive period adjustment based on regime
   // In strong trends, use shorter fast MA for quicker signals
   // In weak trends, use longer periods to avoid whipsaws
   double adxNorm = g_regime.adxValue / 50.0; // Normalize ADX
   g_strategy.adaptiveFastMA = InpFastMAPeriod * (2.0 - adxNorm);
   g_strategy.adaptiveSlowMA = InpSlowMAPeriod * (2.0 - adxNorm * 0.5);
   
   // Clamp periods
   g_strategy.adaptiveFastMA = MathMax(5, MathMin(50, g_strategy.adaptiveFastMA));
   g_strategy.adaptiveSlowMA = MathMax(20, MathMin(100, g_strategy.adaptiveSlowMA));
   
   // MA crossover signal
   bool bullishCross = (fastMA > slowMA) && (fastMA1 <= slowMA1);
   bool bearishCross = (fastMA < slowMA) && (fastMA1 >= slowMA1);
   
   // Confirm with price above/below both MAs
   bool bullishConfirm = (close0 > fastMA) && (close0 > slowMA);
   bool bearishConfirm = (close0 < fastMA) && (close0 < slowMA);
   
   // Additional: ADX must show trend strength
   bool trendPresent = g_regime.adxValue > (InpTrendThreshold * 0.8);
   
   if(bullishCross && bullishConfirm && trendPresent)
      return SIGNAL_BUY;
   else if(bearishCross && bearishConfirm && trendPresent)
      return SIGNAL_SELL;
   
   // Continuation signal: price pulling back to fast MA in a trend
   if(trendPresent && fastMA > slowMA)
   {
      double prevLow = iLow(_Symbol, InpTimeframe, 1);
      if(prevLow <= fastMA * 1.001 && close0 > fastMA)
         return SIGNAL_BUY;
   }
   else if(trendPresent && fastMA < slowMA)
   {
      double prevHigh = iHigh(_Symbol, InpTimeframe, 1);
      if(prevHigh >= fastMA * 0.999 && close0 < fastMA)
         return SIGNAL_SELL;
   }
   
   return SIGNAL_NONE;
}

//+------------------------------------------------------------------+
//| Strategy 2: Mean Reversion with Dynamic Bollinger Bands            |
//+------------------------------------------------------------------+
ENUM_AI_SIGNAL GetMeanReversionSignal()
{
   double close0 = iClose(_Symbol, InpTimeframe, 0);
   double close1 = iClose(_Symbol, InpTimeframe, 1);
   double bbUpper = g_bbUpperBuffer[0];
   double bbLower = g_bbLowerBuffer[0];
   double bbMiddle = g_bbMiddleBuffer[0];
   
   // Dynamic deviation adjustment: wider in volatile, tighter in calm
   double volFactor = g_regime.atrPercentile / 50.0; // > 1 in high vol
   g_strategy.adaptiveBBDev = InpBBDeviation * MathMax(0.5, MathMin(2.0, volFactor));
   
   // Calculate dynamic bands using adaptive deviation
   double bbRange = bbUpper - bbMiddle;
   double adaptiveRange = bbRange * (g_strategy.adaptiveBBDev / InpBBDeviation);
   double adaptiveUpper = bbMiddle + adaptiveRange;
   double adaptiveLower = bbMiddle - adaptiveRange;
   
   // Mean reversion conditions
   // Buy when price touches/crosses below lower band and shows reversal
   bool oversold = close1 <= adaptiveLower && close0 > adaptiveLower;
   // Sell when price touches/crosses above upper band and shows reversal
   bool overbought = close1 >= adaptiveUpper && close0 < adaptiveUpper;
   
   // Additional confirmation: RSI-like momentum divergence
   double momentum3 = close0 - iClose(_Symbol, InpTimeframe, 3);
   double momentum6 = iClose(_Symbol, InpTimeframe, 3) - iClose(_Symbol, InpTimeframe, 6);
   
   // Look for momentum divergence at bands
   if(oversold)
   {
      // Price at lower band but momentum improving
      if(momentum3 > momentum6 || close0 > close1)
         return SIGNAL_BUY;
   }
   else if(overbought)
   {
      // Price at upper band but momentum fading
      if(momentum3 < momentum6 || close0 < close1)
         return SIGNAL_SELL;
   }
   
   // Also check for extreme extensions (more than 1.5x normal deviation)
   double extremeUpper = bbMiddle + adaptiveRange * 1.5;
   double extremeLower = bbMiddle - adaptiveRange * 1.5;
   
   if(close0 < extremeLower && close0 > close1)
      return SIGNAL_BUY;
   else if(close0 > extremeUpper && close0 < close1)
      return SIGNAL_SELL;
   
   return SIGNAL_NONE;
}

//+------------------------------------------------------------------+
//| Strategy 3: Breakout with Adaptive Confirmation                    |
//+------------------------------------------------------------------+
ENUM_AI_SIGNAL GetBreakoutSignal()
{
   double close0 = iClose(_Symbol, InpTimeframe, 0);
   double close1 = iClose(_Symbol, InpTimeframe, 1);
   
   // Find support and resistance levels
   double highestHigh = -DBL_MAX;
   double lowestLow = DBL_MAX;
   
   int lookback = InpBreakoutPeriod;
   for(int i = 1; i <= lookback; i++)
   {
      double high = iHigh(_Symbol, InpTimeframe, i);
      double low  = iLow(_Symbol, InpTimeframe, i);
      if(high > highestHigh) highestHigh = high;
      if(low < lowestLow)    lowestLow = low;
   }
   
   // Adaptive confirmation threshold based on volatility
   double atr = g_atrBuffer[0];
   double confirmThreshold = atr * g_strategy.adaptiveBreakoutConf;
   
   // Adjust confirmation based on regime
   if(g_regime.current == REGIME_VOLATILE)
      confirmThreshold *= 1.5; // Require more confirmation in volatile markets
   else if(g_regime.current == REGIME_TRENDING)
      confirmThreshold *= 0.8; // Less confirmation needed in trending markets
   
   g_strategy.adaptiveBreakoutConf = InpBreakoutConfirm;
   if(g_regime.current == REGIME_VOLATILE)
      g_strategy.adaptiveBreakoutConf = InpBreakoutConfirm * 1.5;
   else if(g_regime.current == REGIME_TRENDING)
      g_strategy.adaptiveBreakoutConf = InpBreakoutConfirm * 0.8;
   
   // Breakout above resistance
   bool bullBreakout = (close0 > highestHigh) && 
                       (close0 - highestHigh > confirmThreshold * 0.3) &&
                       (close1 <= highestHigh);
   
   // Breakout below support
   bool bearBreakout = (close0 < lowestLow) && 
                       (lowestLow - close0 > confirmThreshold * 0.3) &&
                       (close1 >= lowestLow);
   
   // Volume confirmation: breakout should have above-average volume
   long currentVol = iVolume(_Symbol, InpTimeframe, 0);
   long avgVol = 0;
   for(int i = 1; i <= 20; i++)
      avgVol += iVolume(_Symbol, InpTimeframe, i);
   avgVol /= 20;
   
   bool volumeConfirm = (currentVol > (long)(avgVol * 1.2));
   
   if(bullBreakout && volumeConfirm)
      return SIGNAL_BUY;
   else if(bearBreakout && volumeConfirm)
      return SIGNAL_SELL;
   
   // Breakout re-test: price broke out and is retesting the level
   double prevHighest = -DBL_MAX;
   for(int i = 2; i <= lookback + 5; i++)
   {
      double high = iHigh(_Symbol, InpTimeframe, i);
      if(high > prevHighest) prevHighest = high;
   }
   
   // Retest of broken resistance as support
   if(close1 <= prevHighest * 1.002 && close0 > prevHighest && 
      iClose(_Symbol, InpTimeframe, lookback) > prevHighest)
      return SIGNAL_BUY;
   
   return SIGNAL_NONE;
}

//+------------------------------------------------------------------+
//| Combine strategy signals using adaptive weights and NN output      |
//+------------------------------------------------------------------+
ENUM_AI_SIGNAL CombineSignals(ENUM_AI_SIGNAL trendSig, ENUM_AI_SIGNAL meanRevSig, 
                               ENUM_AI_SIGNAL breakoutSig, double networkOutput)
{
   // Calculate weighted signal score
   double signalScore = 0.0;
   
   signalScore += g_strategy.weights[STRATEGY_TREND_FOLLOW] * (double)trendSig;
   signalScore += g_strategy.weights[STRATEGY_MEAN_REVERT]  * (double)meanRevSig;
   signalScore += g_strategy.weights[STRATEGY_BREAKOUT]     * (double)breakoutSig;
   
   // Incorporate neural network output as a filter/amplifier
   // Network output is in [-1, 1]: positive = bullish bias, negative = bearish
   double networkInfluence = 0.3; // 30% influence from network
   signalScore = signalScore * (1.0 - networkInfluence) + networkOutput * networkInfluence;
   
   // Apply regime-based modulation
   // In volatile regime, require stronger signal
   double threshold = 0.3;
   if(g_regime.current == REGIME_VOLATILE)
      threshold = 0.5;
   else if(g_regime.current == REGIME_TRENDING && trendSig != SIGNAL_NONE)
      threshold = 0.2; // Lower threshold for trend signals in trending regime
   
   // Determine which strategy contributed most
   double maxContrib = 0.0;
   if(MathAbs(g_strategy.weights[0] * (double)trendSig) > maxContrib)
   { maxContrib = MathAbs(g_strategy.weights[0] * (double)trendSig); g_strategy.activeStrategy = STRATEGY_TREND_FOLLOW; }
   if(MathAbs(g_strategy.weights[1] * (double)meanRevSig) > maxContrib)
   { maxContrib = MathAbs(g_strategy.weights[1] * (double)meanRevSig); g_strategy.activeStrategy = STRATEGY_MEAN_REVERT; }
   if(MathAbs(g_strategy.weights[2] * (double)breakoutSig) > maxContrib)
   { maxContrib = MathAbs(g_strategy.weights[2] * (double)breakoutSig); g_strategy.activeStrategy = STRATEGY_BREAKOUT; }
   
   g_strategy.confidence = MathAbs(signalScore);
   
   if(signalScore > threshold)
      return SIGNAL_BUY;
   else if(signalScore < -threshold)
      return SIGNAL_SELL;
   
   return SIGNAL_NONE;
}


//+------------------------------------------------------------------+
//| DYNAMIC POSITION SIZING                                            |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Calculate position size using Kelly Criterion approximation        |
//+------------------------------------------------------------------+
double CalculatePositionSize()
{
   double equity = g_account.Equity();
   double balance = g_account.Balance();
   
   // Calculate current drawdown
   if(equity > g_metrics.peakEquity)
      g_metrics.peakEquity = equity;
   g_metrics.currentDrawdown = (g_metrics.peakEquity - equity) / g_metrics.peakEquity * 100.0;
   
   // Base Kelly fraction calculation
   // Kelly f* = (bp - q) / b
   // where b = avg win/avg loss, p = win probability, q = 1-p
   double kellyFraction = InpKellyFraction; // Default conservative Kelly
   
   if(g_totalTrades >= InpMinTradesForLearning)
   {
      double winRate = g_metrics.winRate;
      double avgWin = MathMax(g_metrics.avgWin, 0.0001);
      double avgLoss = MathMax(MathAbs(g_metrics.avgLoss), 0.0001);
      double b = avgWin / avgLoss; // Win/loss ratio
      
      double fullKelly = (b * winRate - (1.0 - winRate)) / b;
      fullKelly = MathMax(0.0, fullKelly);
      
      // Use fractional Kelly (more conservative)
      kellyFraction = fullKelly * InpKellyFraction;
   }
   
   // Adjust for drawdown: reduce size as drawdown increases
   double drawdownAdjustment = 1.0;
   if(g_metrics.currentDrawdown > 5.0)
      drawdownAdjustment = MathMax(0.25, 1.0 - (g_metrics.currentDrawdown / InpMaxDrawdownPercent));
   
   // Adjust for volatility regime
   double regimeAdjustment = 1.0;
   if(g_regime.current == REGIME_VOLATILE)
      regimeAdjustment = 0.5; // Halve size in volatile regimes
   else if(g_regime.current == REGIME_TRENDING)
      regimeAdjustment = 1.2; // Slightly larger in trends
   
   // Adjust for serial correlation (reduce after consecutive losses)
   double correlationAdjust = 1.0;
   if(g_metrics.consecutiveLosses >= 3)
      correlationAdjust = MathMax(0.3, 1.0 - g_metrics.consecutiveLosses * 0.15);
   else if(g_metrics.consecutiveWins >= 3 && g_metrics.recentCorrelation > 0.3)
      correlationAdjust = MathMin(1.3, 1.0 + g_metrics.consecutiveWins * 0.05);
   
   // Calculate risk amount
   double riskPercent = InpMaxRiskPercent * kellyFraction * drawdownAdjustment * 
                        regimeAdjustment * correlationAdjust;
   riskPercent = MathMax(0.1, MathMin(InpMaxRiskPercent, riskPercent));
   
   double riskAmount = equity * riskPercent / 100.0;
   
   // Calculate lot size based on ATR stop loss
   double atr = g_atrBuffer[0];
   double slDistance = atr * InpSLATRMultiple;
   
   // Convert to lots
   double tickValue = g_symbol.TickValue();
   double tickSize  = g_symbol.TickSize();
   double point     = g_symbol.Point();
   
   if(tickValue <= 0 || tickSize <= 0 || slDistance <= 0)
      return InpMinLotSize;
   
   double slPoints = slDistance / point;
   double lotSize = riskAmount / (slPoints * tickValue / (tickSize / point));
   
   // Normalize and clamp lot size
   double lotStep = g_symbol.LotsStep();
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(InpMinLotSize, MathMin(InpMaxLotSize, lotSize));
   lotSize = MathMax(g_symbol.LotsMin(), MathMin(g_symbol.LotsMax(), lotSize));
   
   return NormalizeDouble(lotSize, 2);
}

//+------------------------------------------------------------------+
//| TRADE EXECUTION                                                    |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Execute a trade with multi-level take profit                       |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_AI_SIGNAL signal, double lotSize)
{
   double atr = g_atrBuffer[0];
   double ask = g_symbol.Ask();
   double bid = g_symbol.Bid();
   double point = g_symbol.Point();
   int digits = g_symbol.Digits();
   
   // Calculate SL/TP levels based on ATR
   double slDistance = atr * InpSLATRMultiple;
   double tp1Distance = atr * InpTP1_ATRMultiple;
   double tp2Distance = atr * InpTP2_ATRMultiple;
   double tp3Distance = atr * InpTP3_ATRMultiple;
   
   // Ensure minimum stop distance
   long minStopLevel = g_symbol.StopsLevel();
   double minStopDist = minStopLevel * point;
   slDistance = MathMax(slDistance, minStopDist * 1.5);
   tp1Distance = MathMax(tp1Distance, minStopDist * 1.5);
   
   double entryPrice, sl, tp;
   
   if(signal == SIGNAL_BUY)
   {
      entryPrice = ask;
      sl = NormalizeDouble(entryPrice - slDistance, digits);
      tp = NormalizeDouble(entryPrice + tp1Distance, digits); // Set initial TP at level 1
      
      string comment = StringFormat("AI_BUY|%d|%.1f|%.3f", 
                       (int)g_regime.current, g_strategy.confidence, lotSize);
      
      if(!g_trade.Buy(lotSize, _Symbol, entryPrice, sl, tp, comment))
      {
         Print("ERROR: Buy order failed - ", g_trade.ResultRetcodeDescription());
         return;
      }
      
      Print("BUY Signal Executed: Lot=", lotSize, " SL=", sl, " TP=", tp,
            " Regime=", EnumToString(g_regime.current), 
            " Strategy=", EnumToString(g_strategy.activeStrategy),
            " Confidence=", DoubleToString(g_strategy.confidence, 3));
   }
   else if(signal == SIGNAL_SELL)
   {
      entryPrice = bid;
      sl = NormalizeDouble(entryPrice + slDistance, digits);
      tp = NormalizeDouble(entryPrice - tp1Distance, digits);
      
      string comment = StringFormat("AI_SELL|%d|%.1f|%.3f", 
                       (int)g_regime.current, g_strategy.confidence, lotSize);
      
      if(!g_trade.Sell(lotSize, _Symbol, entryPrice, sl, tp, comment))
      {
         Print("ERROR: Sell order failed - ", g_trade.ResultRetcodeDescription());
         return;
      }
      
      Print("SELL Signal Executed: Lot=", lotSize, " SL=", sl, " TP=", tp,
            " Regime=", EnumToString(g_regime.current),
            " Strategy=", EnumToString(g_strategy.activeStrategy),
            " Confidence=", DoubleToString(g_strategy.confidence, 3));
   }
   
   g_totalTrades++;
}


//+------------------------------------------------------------------+
//| PROFIT BOOKING AND POSITION MANAGEMENT                             |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Manage open positions: trailing stops, partial close, time exit    |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   int totalPositions = PositionsTotal();
   
   for(int i = totalPositions - 1; i >= 0; i--)
   {
      if(!g_position.SelectByIndex(i))
         continue;
      
      // Only manage our positions
      if(g_position.Magic() != InpMagicNumber)
         continue;
      if(g_position.Symbol() != _Symbol)
         continue;
      
      double currentPrice = g_position.PriceCurrent();
      double openPrice = g_position.PriceOpen();
      double sl = g_position.StopLoss();
      double tp = g_position.TakeProfit();
      double volume = g_position.Volume();
      ulong ticket = g_position.Ticket();
      datetime openTime = g_position.Time();
      ENUM_POSITION_TYPE posType = g_position.PositionType();
      
      double atr = g_atrBuffer[0];
      double point = g_symbol.Point();
      int digits = g_symbol.Digits();
      
      // Calculate profit in ATR units
      double profitDistance = 0.0;
      if(posType == POSITION_TYPE_BUY)
         profitDistance = currentPrice - openPrice;
      else
         profitDistance = openPrice - currentPrice;
      
      double profitATR = (atr > 0) ? profitDistance / atr : 0.0;
      
      // === TIME-BASED EXIT ===
      // Close stale trades that haven't reached TP
      int hoursOpen = (int)((TimeCurrent() - openTime) / 3600);
      if(hoursOpen >= InpStaleTradeHours && profitATR < 0.5)
      {
         Print("TIME EXIT: Position ", ticket, " open for ", hoursOpen, 
               " hours with insufficient progress");
         g_trade.PositionClose(ticket);
         continue;
      }
      
      // === PARTIAL CLOSE AT TP LEVELS ===
      // Check TP2 level (close partial)
      double tp2Distance = atr * InpTP2_ATRMultiple;
      if(profitDistance >= tp2Distance && volume > InpMinLotSize * 2)
      {
         double closeVolume = NormalizeDouble(volume * InpPartialClose2, 2);
         closeVolume = MathMax(closeVolume, g_symbol.LotsMin());
         double lotStep = g_symbol.LotsStep();
         closeVolume = MathFloor(closeVolume / lotStep) * lotStep;
         
         if(closeVolume >= g_symbol.LotsMin() && closeVolume < volume)
         {
            if(g_trade.PositionClosePartial(ticket, closeVolume))
            {
               Print("PARTIAL CLOSE TP2: Closed ", closeVolume, " lots at profit ATR=", 
                     DoubleToString(profitATR, 2));
            }
         }
      }
      // Check TP1 level (close partial)
      else if(profitDistance >= atr * InpTP1_ATRMultiple && volume > InpMinLotSize * 2)
      {
         double closeVolume = NormalizeDouble(volume * InpPartialClose1, 2);
         closeVolume = MathMax(closeVolume, g_symbol.LotsMin());
         double lotStep = g_symbol.LotsStep();
         closeVolume = MathFloor(closeVolume / lotStep) * lotStep;
         
         if(closeVolume >= g_symbol.LotsMin() && closeVolume < volume)
         {
            if(g_trade.PositionClosePartial(ticket, closeVolume))
            {
               Print("PARTIAL CLOSE TP1: Closed ", closeVolume, " lots at profit ATR=", 
                     DoubleToString(profitATR, 2));
            }
         }
      }
      
      // === ADAPTIVE TRAILING STOP ===
      // Trail after price moves beyond 1 ATR in profit
      if(profitATR > 1.0)
      {
         double trailDistance = atr * InpTrailingATRMult;
         
         // Tighten trail as profit increases
         if(profitATR > 3.0)
            trailDistance *= 0.7; // Tighter trail at high profit
         else if(profitATR > 2.0)
            trailDistance *= 0.85;
         
         double newSL = 0.0;
         if(posType == POSITION_TYPE_BUY)
         {
            newSL = NormalizeDouble(currentPrice - trailDistance, digits);
            // Only move SL up, never down
            if(newSL > sl && newSL < currentPrice)
            {
               if(g_trade.PositionModify(ticket, newSL, tp))
               {
                  // Trail successful
               }
            }
         }
         else // SELL
         {
            newSL = NormalizeDouble(currentPrice + trailDistance, digits);
            // Only move SL down, never up
            if(newSL < sl && newSL > currentPrice)
            {
               if(g_trade.PositionModify(ticket, newSL, tp))
               {
                  // Trail successful
               }
            }
         }
      }
      
      // === TP3 FULL EXIT ===
      // If price reaches TP3, close everything
      double tp3Distance = atr * InpTP3_ATRMultiple;
      if(profitDistance >= tp3Distance)
      {
         Print("TP3 FULL EXIT: Position ", ticket, " reached target at ", 
               DoubleToString(profitATR, 2), " ATR");
         g_trade.PositionClose(ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| PERFORMANCE TRACKING AND SELF-LEARNING                             |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Process closed trades and update learning systems                   |
//+------------------------------------------------------------------+
void ProcessClosedTrades()
{
   // Check deal history for our closed trades
   datetime fromTime = TimeCurrent() - 86400; // Last 24 hours
   datetime toTime = TimeCurrent();
   
   if(!HistorySelect(fromTime, toTime))
      return;
   
   int totalDeals = HistoryDealsTotal();
   
   for(int i = totalDeals - 1; i >= 0; i--)
   {
      ulong dealTicket = HistoryDealGetTicket(i);
      if(dealTicket == 0) continue;
      
      // Check if this is our deal
      long magic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
      if(magic != InpMagicNumber) continue;
      
      // Check if it's an exit deal
      long dealEntry = HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
      if(dealEntry != DEAL_ENTRY_OUT && dealEntry != DEAL_ENTRY_INOUT) continue;
      
      string symbol = HistoryDealGetString(dealTicket, DEAL_SYMBOL);
      if(symbol != _Symbol) continue;
      
      // Get deal details
      double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
      double volume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
      datetime dealTime = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
      
      // Check if we already recorded this trade
      bool alreadyRecorded = false;
      for(int t = 0; t < g_tradeHistoryCount; t++)
      {
         if(g_tradeHistory[t].closeTime == dealTime && 
            MathAbs(g_tradeHistory[t].profit - profit) < 0.01)
         {
            alreadyRecorded = true;
            break;
         }
      }
      
      if(alreadyRecorded) continue;
      
      // Record the trade
      if(g_tradeHistoryCount < MAX_TRADE_HISTORY)
      {
         TradeRecord record;
         record.closeTime = dealTime;
         record.profit = profit;
         record.lotSize = volume;
         record.regime = g_regime.current;
         record.strategy = g_strategy.activeStrategy;
         record.isWin = (profit > 0);
         record.openTime = dealTime; // Approximate
         record.entryPrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
         record.exitPrice = record.entryPrice;
         record.profitPoints = profit / (volume > 0 ? volume : 1.0);
         record.direction = (HistoryDealGetInteger(dealTicket, DEAL_TYPE) == DEAL_TYPE_BUY) ? -1 : 1;
         
         g_tradeHistory[g_tradeHistoryCount] = record;
         g_tradeHistoryCount++;
         
         // Update all learning systems
         OnTradeOutcome(record);
      }
   }
}

//+------------------------------------------------------------------+
//| Process a trade outcome - update all adaptive systems              |
//+------------------------------------------------------------------+
void OnTradeOutcome(TradeRecord &record)
{
   // Update performance metrics
   UpdatePerformanceMetrics(record);
   
   // Update strategy weights based on outcome
   UpdateStrategyWeights(record);
   
   // Update neural network weights
   double features[MAX_NETWORK_INPUTS];
   ComputeFeatures(features);
   
   // Reward signal: normalized profit
   double reward = 0.0;
   if(record.isWin)
      reward = MathMin(1.0, record.profit / (g_account.Balance() * 0.01));
   else
      reward = MathMax(-1.0, record.profit / (g_account.Balance() * 0.01));
   
   UpdateNetworkWeights(reward, features);
   
   // Log learning update
   Print("LEARNING UPDATE: Trade P&L=", DoubleToString(record.profit, 2),
         " Regime=", EnumToString(record.regime),
         " Strategy=", EnumToString(record.strategy),
         " Reward=", DoubleToString(reward, 3));
}


//+------------------------------------------------------------------+
//| Update rolling performance metrics                                 |
//+------------------------------------------------------------------+
void UpdatePerformanceMetrics(TradeRecord &record)
{
   // Calculate metrics over rolling window
   int windowSize = MathMin(g_tradeHistoryCount, InpPerformanceWindow);
   int startIdx = g_tradeHistoryCount - windowSize;
   
   int wins = 0;
   double totalWin = 0.0;
   double totalLoss = 0.0;
   double sumReturns = 0.0;
   double sumReturnsSq = 0.0;
   
   for(int i = startIdx; i < g_tradeHistoryCount; i++)
   {
      if(g_tradeHistory[i].isWin)
      {
         wins++;
         totalWin += g_tradeHistory[i].profit;
      }
      else
      {
         totalLoss += MathAbs(g_tradeHistory[i].profit);
      }
      
      double returnPct = g_tradeHistory[i].profit / g_account.Balance() * 100.0;
      sumReturns += returnPct;
      sumReturnsSq += returnPct * returnPct;
   }
   
   // Win rate
   g_metrics.winRate = (windowSize > 0) ? (double)wins / windowSize : 0.5;
   
   // Average win/loss
   g_metrics.avgWin = (wins > 0) ? totalWin / wins : 0.0;
   g_metrics.avgLoss = (windowSize - wins > 0) ? -totalLoss / (windowSize - wins) : 0.0;
   
   // Profit factor
   g_metrics.profitFactor = (totalLoss > 0) ? totalWin / totalLoss : 
                            (totalWin > 0) ? 99.9 : 1.0;
   
   // Sharpe ratio (annualized approximation)
   if(windowSize > 1)
   {
      double meanReturn = sumReturns / windowSize;
      double variance = (sumReturnsSq / windowSize) - (meanReturn * meanReturn);
      double stdDev = MathSqrt(MathMax(0, variance));
      g_metrics.sharpeRatio = (stdDev > 0) ? (meanReturn / stdDev) * MathSqrt(252.0) : 0.0;
   }
   
   // Consecutive wins/losses
   if(record.isWin)
   {
      g_metrics.consecutiveWins++;
      g_metrics.consecutiveLosses = 0;
   }
   else
   {
      g_metrics.consecutiveLosses++;
      g_metrics.consecutiveWins = 0;
   }
   
   // Max drawdown update
   double equity = g_account.Equity();
   if(equity > g_metrics.peakEquity)
      g_metrics.peakEquity = equity;
   double dd = (g_metrics.peakEquity - equity) / g_metrics.peakEquity * 100.0;
   if(dd > g_metrics.maxDrawdown)
      g_metrics.maxDrawdown = dd;
   g_metrics.currentDrawdown = dd;
   
   // Serial correlation of trade outcomes
   g_metrics.recentCorrelation = CalculateSerialCorrelation();
}

//+------------------------------------------------------------------+
//| Calculate serial correlation of recent trade outcomes              |
//+------------------------------------------------------------------+
double CalculateSerialCorrelation()
{
   int n = MathMin(g_tradeHistoryCount, 20);
   if(n < 5) return 0.0;
   
   int startIdx = g_tradeHistoryCount - n;
   
   double sumXY = 0.0, sumX = 0.0, sumY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
   
   for(int i = startIdx; i < g_tradeHistoryCount - 1; i++)
   {
      double x = g_tradeHistory[i].profit;
      double y = g_tradeHistory[i + 1].profit;
      sumXY += x * y;
      sumX += x;
      sumY += y;
      sumX2 += x * x;
      sumY2 += y * y;
   }
   
   int pairs = n - 1;
   double numerator = pairs * sumXY - sumX * sumY;
   double denominator = MathSqrt((pairs * sumX2 - sumX * sumX) * (pairs * sumY2 - sumY * sumY));
   
   if(denominator == 0.0) return 0.0;
   
   return numerator / denominator;
}

//+------------------------------------------------------------------+
//| Update strategy weights based on trade outcome                     |
//+------------------------------------------------------------------+
void UpdateStrategyWeights(TradeRecord &record)
{
   int stratIdx = (int)record.strategy;
   if(stratIdx < 0 || stratIdx >= NUM_STRATEGIES) return;
   
   // Update per-strategy metrics
   g_strategy.tradeCount[stratIdx]++;
   g_strategy.pnl[stratIdx] += record.profit;
   
   // Calculate per-strategy performance score (rolling)
   double alpha = InpAdaptationSpeed;
   double outcome = record.isWin ? 1.0 : -1.0;
   
   // Exponential moving average of outcomes
   g_strategy.performance[stratIdx] = 
      g_strategy.performance[stratIdx] * (1.0 - alpha) + outcome * alpha;
   
   // Update weights using softmax-like normalization
   double maxPerf = -DBL_MAX;
   for(int i = 0; i < NUM_STRATEGIES; i++)
   {
      if(g_strategy.performance[i] > maxPerf)
         maxPerf = g_strategy.performance[i];
   }
   
   double sumExp = 0.0;
   double expWeights[NUM_STRATEGIES];
   for(int i = 0; i < NUM_STRATEGIES; i++)
   {
      // Temperature-scaled softmax
      double temperature = 2.0; // Higher = more uniform, lower = more greedy
      expWeights[i] = MathExp((g_strategy.performance[i] - maxPerf) / temperature);
      sumExp += expWeights[i];
   }
   
   // Normalize to get probabilities
   if(sumExp > 0)
   {
      for(int i = 0; i < NUM_STRATEGIES; i++)
      {
         g_strategy.weights[i] = expWeights[i] / sumExp;
         // Ensure minimum weight (never completely abandon a strategy)
         g_strategy.weights[i] = MathMax(0.1, g_strategy.weights[i]);
      }
      
      // Re-normalize after applying minimum
      double totalWeight = 0.0;
      for(int i = 0; i < NUM_STRATEGIES; i++)
         totalWeight += g_strategy.weights[i];
      for(int i = 0; i < NUM_STRATEGIES; i++)
         g_strategy.weights[i] /= totalWeight;
   }
   
   // Adapt strategy parameters based on regime performance
   AdaptStrategyParameters();
}

//+------------------------------------------------------------------+
//| Adapt strategy parameters based on accumulated learning            |
//+------------------------------------------------------------------+
void AdaptStrategyParameters()
{
   if(g_totalTrades < InpMinTradesForLearning) return;
   
   // Adapt trend following: adjust MA periods based on trend strategy performance
   if(g_strategy.performance[STRATEGY_TREND_FOLLOW] > 0.3)
   {
      // Strategy working well: make it more aggressive (shorter periods)
      g_strategy.adaptiveFastMA *= 0.95;
      g_strategy.adaptiveSlowMA *= 0.95;
   }
   else if(g_strategy.performance[STRATEGY_TREND_FOLLOW] < -0.3)
   {
      // Strategy struggling: make it more conservative (longer periods)
      g_strategy.adaptiveFastMA *= 1.05;
      g_strategy.adaptiveSlowMA *= 1.05;
   }
   g_strategy.adaptiveFastMA = MathMax(5, MathMin(50, g_strategy.adaptiveFastMA));
   g_strategy.adaptiveSlowMA = MathMax(20, MathMin(100, g_strategy.adaptiveSlowMA));
   
   // Adapt mean reversion: adjust BB deviation
   if(g_strategy.performance[STRATEGY_MEAN_REVERT] > 0.3)
   {
      // Working well: can use tighter bands (more signals)
      g_strategy.adaptiveBBDev *= 0.97;
   }
   else if(g_strategy.performance[STRATEGY_MEAN_REVERT] < -0.3)
   {
      // Struggling: wider bands (fewer, higher quality signals)
      g_strategy.adaptiveBBDev *= 1.03;
   }
   g_strategy.adaptiveBBDev = MathMax(1.0, MathMin(4.0, g_strategy.adaptiveBBDev));
   
   // Adapt breakout: adjust confirmation threshold
   if(g_strategy.performance[STRATEGY_BREAKOUT] > 0.3)
   {
      // Lower threshold for more breakout trades
      g_strategy.adaptiveBreakoutConf *= 0.95;
   }
   else if(g_strategy.performance[STRATEGY_BREAKOUT] < -0.3)
   {
      // Higher threshold for fewer, stronger breakouts only
      g_strategy.adaptiveBreakoutConf *= 1.05;
   }
   g_strategy.adaptiveBreakoutConf = MathMax(0.5, MathMin(3.0, g_strategy.adaptiveBreakoutConf));
}


//+------------------------------------------------------------------+
//| UTILITY FUNCTIONS                                                  |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Check for new bar                                                  |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime currentBarTime = iTime(_Symbol, InpTimeframe, 0);
   if(currentBarTime != g_lastBarTime)
   {
      g_lastBarTime = currentBarTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check if we can open a new trade                                   |
//+------------------------------------------------------------------+
bool CanOpenNewTrade()
{
   // Count our open positions
   int ourPositions = 0;
   int totalPositions = PositionsTotal();
   
   for(int i = 0; i < totalPositions; i++)
   {
      if(g_position.SelectByIndex(i))
      {
         if(g_position.Magic() == InpMagicNumber && g_position.Symbol() == _Symbol)
            ourPositions++;
      }
   }
   
   // Check max trades limit
   if(ourPositions >= InpMaxOpenTrades)
      return false;
   
   // Check maximum drawdown limit - stop trading if too much drawdown
   if(g_metrics.currentDrawdown >= InpMaxDrawdownPercent)
   {
      Print("RISK LIMIT: Current drawdown ", DoubleToString(g_metrics.currentDrawdown, 1),
            "% exceeds max ", DoubleToString(InpMaxDrawdownPercent, 1), "%");
      return false;
   }
   
   // Check minimum confidence
   if(g_strategy.confidence < 0.15)
      return false;
   
   // Spread check: don't trade during extremely wide spreads
   double spread = g_symbol.Spread() * g_symbol.Point();
   double atr = g_atrBuffer[0];
   if(atr > 0 && spread / atr > 0.3) // Spread > 30% of ATR
   {
      Print("SPREAD FILTER: Spread too wide relative to ATR");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Refresh indicator buffers                                          |
//+------------------------------------------------------------------+
bool RefreshIndicatorBuffers()
{
   int copied = 0;
   int barsNeeded = InpRegimeLookback + 5;
   
   copied = CopyBuffer(g_handleATR, 0, 0, barsNeeded, g_atrBuffer);
   if(copied < barsNeeded) { Print("ERROR: Failed to copy ATR buffer"); return false; }
   
   copied = CopyBuffer(g_handleADX, 0, 0, barsNeeded, g_adxBuffer);
   if(copied < barsNeeded) { Print("ERROR: Failed to copy ADX buffer"); return false; }
   
   copied = CopyBuffer(g_handleBB, 1, 0, barsNeeded, g_bbUpperBuffer);
   if(copied < barsNeeded) { Print("ERROR: Failed to copy BB Upper buffer"); return false; }
   
   copied = CopyBuffer(g_handleBB, 2, 0, barsNeeded, g_bbLowerBuffer);
   if(copied < barsNeeded) { Print("ERROR: Failed to copy BB Lower buffer"); return false; }
   
   copied = CopyBuffer(g_handleBB, 0, 0, barsNeeded, g_bbMiddleBuffer);
   if(copied < barsNeeded) { Print("ERROR: Failed to copy BB Middle buffer"); return false; }
   
   copied = CopyBuffer(g_handleFastMA, 0, 0, barsNeeded, g_fastMABuffer);
   if(copied < barsNeeded) { Print("ERROR: Failed to copy Fast MA buffer"); return false; }
   
   copied = CopyBuffer(g_handleSlowMA, 0, 0, barsNeeded, g_slowMABuffer);
   if(copied < barsNeeded) { Print("ERROR: Failed to copy Slow MA buffer"); return false; }
   
   return true;
}

//+------------------------------------------------------------------+
//| Tanh activation function                                           |
//+------------------------------------------------------------------+
double TanhActivation(double x)
{
   if(x > 20.0) return 1.0;
   if(x < -20.0) return -1.0;
   double e2x = MathExp(2.0 * x);
   return (e2x - 1.0) / (e2x + 1.0);
}

//+------------------------------------------------------------------+
//| Tanh derivative for backpropagation                                |
//+------------------------------------------------------------------+
double TanhDerivative(double tanhOutput)
{
   return 1.0 - tanhOutput * tanhOutput;
}

//+------------------------------------------------------------------+
//| Generate random double in range [min, max]                         |
//+------------------------------------------------------------------+
double RandomDouble(double minVal, double maxVal)
{
   return minVal + (maxVal - minVal) * ((double)MathRand() / 32767.0);
}


//+------------------------------------------------------------------+
//| CHART DASHBOARD VISUALIZATION                                      |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Create dashboard objects on chart                                   |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   int x = 10, y = 30;
   int lineHeight = 18;
   
   CreateLabel(g_dashPrefix + "Title", "AI Adaptive EA Dashboard", x, y, clrGold, 12);
   y += lineHeight + 5;
   CreateLabel(g_dashPrefix + "Regime", "Regime: Initializing...", x, y, clrWhite, 10);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "RegimeProb", "P(T/R/V): --/--/--", x, y, clrSilver, 9);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "Hurst", "Hurst: --", x, y, clrSilver, 9);
   y += lineHeight + 5;
   CreateLabel(g_dashPrefix + "Strategy", "Active: --", x, y, clrAqua, 10);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "Confidence", "Confidence: --", x, y, clrSilver, 9);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "Weights", "W(T/M/B): --/--/--", x, y, clrSilver, 9);
   y += lineHeight + 5;
   CreateLabel(g_dashPrefix + "Network", "NN Output: --", x, y, clrLime, 10);
   y += lineHeight + 5;
   CreateLabel(g_dashPrefix + "Trades", "Trades: 0", x, y, clrWhite, 10);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "WinRate", "Win Rate: --", x, y, clrSilver, 9);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "PF", "Profit Factor: --", x, y, clrSilver, 9);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "Sharpe", "Sharpe: --", x, y, clrSilver, 9);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "DD", "Drawdown: --", x, y, clrSilver, 9);
   y += lineHeight + 5;
   CreateLabel(g_dashPrefix + "PositionSize", "Lot Size: --", x, y, clrYellow, 10);
   y += lineHeight;
   CreateLabel(g_dashPrefix + "KellyInfo", "Kelly Adj: --", x, y, clrSilver, 9);
   
   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Update dashboard with current AI state                             |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   // Regime display
   string regimeStr = "RANGING";
   color regimeColor = clrYellow;
   if(g_regime.current == REGIME_TRENDING) { regimeStr = "TRENDING"; regimeColor = clrLime; }
   else if(g_regime.current == REGIME_VOLATILE) { regimeStr = "VOLATILE"; regimeColor = clrRed; }
   
   ObjectSetString(0, g_dashPrefix + "Regime", OBJPROP_TEXT, 
                   "Regime: " + regimeStr + " (ADX:" + DoubleToString(g_regime.adxValue, 1) + ")");
   ObjectSetInteger(0, g_dashPrefix + "Regime", OBJPROP_COLOR, regimeColor);
   
   ObjectSetString(0, g_dashPrefix + "RegimeProb", OBJPROP_TEXT,
                   "P(T/R/V): " + DoubleToString(g_regime.probTrending * 100, 0) + "/" +
                   DoubleToString(g_regime.probRanging * 100, 0) + "/" +
                   DoubleToString(g_regime.probVolatile * 100, 0) + "%");
   
   ObjectSetString(0, g_dashPrefix + "Hurst", OBJPROP_TEXT,
                   "Hurst: " + DoubleToString(g_regime.hurstExponent, 3));
   
   // Strategy display
   string stratStr = "TREND_FOLLOW";
   if(g_strategy.activeStrategy == STRATEGY_MEAN_REVERT) stratStr = "MEAN_REVERT";
   else if(g_strategy.activeStrategy == STRATEGY_BREAKOUT) stratStr = "BREAKOUT";
   
   ObjectSetString(0, g_dashPrefix + "Strategy", OBJPROP_TEXT, "Active: " + stratStr);
   ObjectSetString(0, g_dashPrefix + "Confidence", OBJPROP_TEXT,
                   "Confidence: " + DoubleToString(g_strategy.confidence * 100, 1) + "%");
   ObjectSetString(0, g_dashPrefix + "Weights", OBJPROP_TEXT,
                   "W(T/M/B): " + DoubleToString(g_strategy.weights[0] * 100, 0) + "/" +
                   DoubleToString(g_strategy.weights[1] * 100, 0) + "/" +
                   DoubleToString(g_strategy.weights[2] * 100, 0) + "%");
   
   // Network output
   color nnColor = (g_network.lastOutput > 0) ? clrLime : clrRed;
   ObjectSetString(0, g_dashPrefix + "Network", OBJPROP_TEXT,
                   "NN Output: " + DoubleToString(g_network.lastOutput, 4));
   ObjectSetInteger(0, g_dashPrefix + "Network", OBJPROP_COLOR, nnColor);
   
   // Performance metrics
   ObjectSetString(0, g_dashPrefix + "Trades", OBJPROP_TEXT,
                   "Trades: " + IntegerToString(g_totalTrades));
   ObjectSetString(0, g_dashPrefix + "WinRate", OBJPROP_TEXT,
                   "Win Rate: " + DoubleToString(g_metrics.winRate * 100, 1) + "%");
   ObjectSetString(0, g_dashPrefix + "PF", OBJPROP_TEXT,
                   "Profit Factor: " + DoubleToString(g_metrics.profitFactor, 2));
   ObjectSetString(0, g_dashPrefix + "Sharpe", OBJPROP_TEXT,
                   "Sharpe: " + DoubleToString(g_metrics.sharpeRatio, 2));
   
   color ddColor = (g_metrics.currentDrawdown > 10) ? clrRed : 
                   (g_metrics.currentDrawdown > 5) ? clrOrange : clrLime;
   ObjectSetString(0, g_dashPrefix + "DD", OBJPROP_TEXT,
                   "Drawdown: " + DoubleToString(g_metrics.currentDrawdown, 1) + "% (Max: " +
                   DoubleToString(g_metrics.maxDrawdown, 1) + "%)");
   ObjectSetInteger(0, g_dashPrefix + "DD", OBJPROP_COLOR, ddColor);
   
   // Position sizing info
   double nextLot = CalculatePositionSize();
   ObjectSetString(0, g_dashPrefix + "PositionSize", OBJPROP_TEXT,
                   "Next Lot: " + DoubleToString(nextLot, 2));
   ObjectSetString(0, g_dashPrefix + "KellyInfo", OBJPROP_TEXT,
                   "ConsW:" + IntegerToString(g_metrics.consecutiveWins) + 
                   " ConsL:" + IntegerToString(g_metrics.consecutiveLosses) +
                   " Corr:" + DoubleToString(g_metrics.recentCorrelation, 2));
   
   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Remove dashboard objects from chart                                 |
//+------------------------------------------------------------------+
void RemoveDashboard()
{
   ObjectsDeleteAll(0, g_dashPrefix);
   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Helper: Create a text label on chart                               |
//+------------------------------------------------------------------+
void CreateLabel(string name, string text, int x, int y, color clr, int fontSize)
{
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

//+------------------------------------------------------------------+
//| END OF AI ADAPTIVE EXPERT ADVISOR                                  |
//+------------------------------------------------------------------+
