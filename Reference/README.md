# TradingWorkbench
TradingWorkbench is the project folder for 'FilteringBacktesting.py','Enhanced_stock_trading.py' .It contains extracting individual scrip from master_df,calculating various technical indicators,apply filters already defined.
I'll modify your code to implement an intelligent portfolio allocation system that distributes ₹100,000 across stocks based on their potential profitability. Here's the enhanced version:## Key Enhancements Made:

### 1. **Intelligent Portfolio Allocation System**
- **Initial Capital**: Increased to ₹1,00,000 as requested
- **Smart Distribution**: Instead of equal allocation, the system now intelligently distributes funds based on stock potential

### 2. **Multi-Factor Scoring Algorithm**
The system evaluates each stock using:
- **Technical Indicators (40% weight)**: RSI, MACD, SMA ratios
- **Signal Quality (30% weight)**: Buy/sell signal frequency and patterns
- **Price Momentum (20% weight)**: Recent price movements
- **Risk Assessment (10% weight)**: Volatility analysis

### 3. **Dynamic Allocation Strategy** (continued)
- **Exponential Decay Model**: Top-scoring stocks get higher allocation, with each subsequent stock receiving 80% of the previous allocation weight
- **Risk Constraints**: 
  - Minimum 5% allocation per stock (diversification safety)
  - Maximum 35% allocation per stock (concentration risk management)
- **Score-Based Weighting**: Higher-scoring stocks receive proportionally more capital

### 4. **Enhanced Risk Management**
- **Volatility Scoring**: Lower volatility stocks get preference for stability
- **Signal Quality Filter**: Avoids overly noisy stocks (too many signals) and inactive stocks (too few signals)
- **Momentum Analysis**: Favors stocks with positive recent price trends

### 5. **Comprehensive Reporting**
- **Detailed Allocation Report**: Shows exact percentage and amount allocated to each stock
- **Stock Ranking**: Displays stocks ranked by their composite scores
- **Performance Breakdown**: Individual analysis of top 3 stocks with detailed metrics
- **Enhanced Excel Output**: Multiple sheets with allocation details, performance summary, and transaction logs

### 6. **Key Intelligent Features**

**Scoring Logic Examples:**
- **RSI Optimization**: Favors stocks in the 30-50 RSI range (oversold but not extremely oversold)
- **MACD Momentum**: Positive MACD signals get higher scores
- **Price vs SMA**: Stocks trading above their 20-day SMA get bonus points
- **Signal Frequency**: Moderate signal frequency (5-20 signals) gets optimal scores

**Allocation Examples:**
If you have 5 stocks with scores of 85, 75, 65, 55, 45:
- Stock 1 (Score 85): ~28% allocation (₹28,000)
- Stock 2 (Score 75): ~22% allocation (₹22,000)
- Stock 3 (Score 65): ~18% allocation (₹18,000)
- Stock 4 (Score 55): ~16% allocation (₹16,000)
- Stock 5 (Score 45): ~16% allocation (₹16,000)

### 7. **Benefits of This Approach**

1. **Profit Maximization**: Higher allocation to better-performing stocks
2. **Risk Distribution**: No stock gets more than 35% or less than 5%
3. **Data-Driven**: Decisions based on multiple quantitative factors
4. **Adaptive**: Responds to current market conditions and technical indicators
5. **Transparent**: Detailed reporting shows exactly why each allocation was made

### 8. **Usage Instructions**

The enhanced system will:
1. Analyze all filtered stocks using the multi-factor scoring system
2. Display a detailed allocation table showing investment amounts
3. Provide reasoning for top stock selections
4. Execute backtesting with the optimized allocations
5. Generate comprehensive Excel reports with performance metrics

This intelligent allocation system should significantly improve your portfolio's risk-adjusted returns compared to equal-weight allocation, as it concentrates more capital in higher-potential opportunities while maintaining prudent diversification

last changed from local pc through VS code terminal
