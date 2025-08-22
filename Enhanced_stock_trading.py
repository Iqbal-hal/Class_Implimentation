import support_files.config as config
import pandas as pd
import numpy as np
from support_files.scrip_extractor import scrip_extractor, scripdf_extractor
import support_files.compute_indicators_helper as cmp # For computing technical indicators

# Trade constraint constants
MIN_PROFIT_PERCENTAGE = config.MIN_PROFIT_PERCENTAGE
MIN_HOLDING_PERIOD = config.MIN_HOLDING_PERIOD

class FilteringAndBacktesting:
    def __init__(self, initial_cash=100000.0):
        self.initial_cash = initial_cash
        self.backtested_scrip_df_list = []
        self.backtested_transactions_df_list = []
        self.folio_final_value = 0.0
        self.stock_allocations = {}  # Store allocation percentages for each stock
        self.stock_scores = {}       # Store scoring data for each stock
        
    # --------------------- PORTFOLIO ALLOCATION METHODS ---------------------
    
    def calculate_stock_score(self, scrip_ta_df, scrip):
        """
        Calculate a composite score for each stock based on multiple factors:
        - Technical indicators strength
        - Signal frequency and quality  
        - Risk-adjusted returns potential
        - Market momentum
        """
        try:
            # Technical Indicators Score (40% weight)
            rsi = scrip_ta_df['RSI'].iloc[-1] if 'RSI' in scrip_ta_df.columns else 50
            macd = scrip_ta_df['MACD'].iloc[-1] if 'MACD' in scrip_ta_df.columns else 0
            sma_ratio = (scrip_ta_df['Close'].iloc[-1] / scrip_ta_df['SMA_20'].iloc[-1]) if 'SMA_20' in scrip_ta_df.columns else 1
            
            # RSI Score: Favor oversold but not extremely oversold (30-50 range)
            rsi_score = max(0, min(100, 100 - abs(rsi - 40)))
            
            # MACD Score: Positive MACD indicates upward momentum
            macd_score = min(100, max(0, 50 + macd * 10))
            
            # SMA Score: Price above SMA is bullish
            sma_score = min(100, max(0, (sma_ratio - 0.95) * 200))
            
            technical_score = (rsi_score * 0.4 + macd_score * 0.3 + sma_score * 0.3)
            
            # Signal Quality Score (30% weight)
            buy_signals = scrip_ta_df['Buy'].sum() if 'Buy' in scrip_ta_df.columns else 0
            sell_signals = scrip_ta_df['Sell'].sum() if 'Sell' in scrip_ta_df.columns else 0
            total_signals = buy_signals + sell_signals
            
            # Favor stocks with moderate signal frequency (not too noisy, not too quiet)
            signal_score = min(100, max(0, total_signals * 5)) if total_signals <= 20 else max(0, 100 - (total_signals - 20) * 2)
            
            # Price Momentum Score (20% weight)
            price_change = ((scrip_ta_df['Close'].iloc[-1] - scrip_ta_df['Close'].iloc[-20]) / 
                           scrip_ta_df['Close'].iloc[-20] * 100) if len(scrip_ta_df) >= 20 else 0
            momentum_score = min(100, max(0, 50 + price_change * 2))
            
            # Volatility Score (10% weight) - Lower volatility gets higher score for risk management
            volatility = scrip_ta_df['Close'].pct_change().std() * 100
            volatility_score = min(100, max(0, 100 - volatility * 5))
            
            # Composite Score
            composite_score = (technical_score * 0.4 + signal_score * 0.3 + 
                             momentum_score * 0.2 + volatility_score * 0.1)
            
            # Store detailed scoring information
            self.stock_scores[scrip] = {
                'composite_score': composite_score,
                'technical_score': technical_score,
                'signal_score': signal_score,
                'momentum_score': momentum_score,
                'volatility_score': volatility_score,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'current_price': scrip_ta_df['Close'].iloc[-1],
                'rsi': rsi,
                'price_momentum': price_change
            }
            
            return composite_score
            
        except Exception as e:
            print(f"Error calculating score for {scrip}: {e}")
            return 50.0  # Default neutral score

    def allocate_portfolio(self, filtered_scrips_df):
        """
        Intelligently allocate the total cash across stocks based on their scores.
        Uses a sophisticated allocation strategy that balances opportunity and risk.
        """
        scrips = filtered_scrips_df['Stock'].unique()
        print(f"\n===== INTELLIGENT PORTFOLIO ALLOCATION =====")
        print(f"Total Investment Amount: Rs {self.initial_cash:,.2f}")
        print(f"Number of stocks to allocate: {len(scrips)}")
        
        # Calculate scores for all stocks
        stock_scores = {}
        for scrip in scrips:
            scrip_data = filtered_scrips_df[filtered_scrips_df['Stock'] == scrip]
            score = self.calculate_stock_score(scrip_data, scrip)
            stock_scores[scrip] = score
        
        # Sort stocks by score (descending)
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Allocation Strategy: Exponential decay with minimum allocation
        total_weight = 0
        weights = {}
        
        # Calculate weights using exponential decay
        for i, (scrip, score) in enumerate(sorted_stocks):
            # Exponential decay factor with base allocation
            decay_factor = 0.8 ** i  # Each subsequent stock gets 80% of previous weight
            base_weight = score / 100.0  # Normalize score to 0-1
            weight = base_weight * decay_factor
            weights[scrip] = weight
            total_weight += weight
        
        # Normalize weights and apply constraints
        min_allocation = 0.05  # Minimum 5% allocation
        max_allocation = 0.35  # Maximum 35% allocation
        
        normalized_weights = {}
        for scrip, weight in weights.items():
            normalized = weight / total_weight
            # Apply constraints
            normalized = max(min_allocation, min(normalized, max_allocation))
            normalized_weights[scrip] = normalized
        
        # Renormalize after applying constraints
        total_normalized = sum(normalized_weights.values())
        for scrip in normalized_weights:
            normalized_weights[scrip] = normalized_weights[scrip] / total_normalized
        
        # Calculate final allocations
        print(f"\n{'Stock':<12} {'Score':<8} {'Allocation %':<12} {'Amount (Rs)':<12} {'Rank':<6}")
        print("-" * 60)
        
        for i, (scrip, score) in enumerate(sorted_stocks):
            allocation_pct = normalized_weights[scrip] * 100
            allocation_amount = self.initial_cash * normalized_weights[scrip]
            self.stock_allocations[scrip] = allocation_amount
            
            print(f"{scrip:<12} {score:<8.1f} {allocation_pct:<12.1f} {allocation_amount:<12,.0f} #{i+1:<6}")
        
        print("-" * 60)
        print(f"{'TOTAL':<12} {'':<8} {'100.0':<12} {sum(self.stock_allocations.values()):<12,.0f}")
        
        # Print detailed scoring breakdown for top 3 stocks
        print(f"\n===== TOP 3 STOCKS DETAILED ANALYSIS =====")
        for i, (scrip, score) in enumerate(sorted_stocks[:3]):
            details = self.stock_scores[scrip]
            print(f"\n#{i+1} {scrip} (Score: {score:.1f})")
            print(f"  Technical Score: {details['technical_score']:.1f}")
            print(f"  Signal Quality:  {details['signal_score']:.1f} (Buy: {details['buy_signals']}, Sell: {details['sell_signals']})")
            print(f"  Momentum:        {details['momentum_score']:.1f} (Price change: {details['price_momentum']:.2f}%)")
            print(f"  Risk Score:      {details['volatility_score']:.1f}")
            print(f"  Current Price:   Rs {details['current_price']:.2f}")
            print(f"  RSI:            {details['rsi']:.1f}")
        
        return self.stock_allocations

    # --------------------- FILTERING METHODS ---------------------
    def apply_filter(self, master_df):
        """
        Computes technical indicators and applies the configured filter on the master OHLC
        dataframe. Then performs intelligent portfolio allocation.
        Returns the filtered dataframe (filtered_scrips_df).
        """
        scrips_list = []
        filtered_ta_df_list = []
        master_scrips_list = master_df['Stock'].unique()
        
        print("\n===== Executing apply_filter =====")
        print(f"Master scrip list count: {len(master_scrips_list)}")
        print(f"Master scrips: {master_scrips_list}\n")
        
        for scrip, scrip_df in scrip_extractor(master_df):
            scrip_ta_df = cmp.compute_indicators(scrip_df)
            
            if config.FILTER_ENABLED:
                filter_func = config.AVAILABLE_FILTERS.get(config.ACTIVE_FILTER)
                buy_signal, sell_signal = filter_func(scrip_ta_df)
                scrip_ta_df['Buy'] = buy_signal
                scrip_ta_df['Sell'] = sell_signal
                
                if (buy_signal.sum() >= 2) and (sell_signal.sum() >= 2):
                    filtered_ta_df_list.append(scrip_ta_df)
                    scrips_list.append(scrip)
                else:
                    continue
            else:
                filtered_ta_df_list.append(scrip_ta_df)
                scrips_list.append(scrip)

        try:
            filtered_ta_df = pd.concat(filtered_ta_df_list)
        except ValueError as e:
            print(f"Error concatenating filtered data: {e}")
            filtered_ta_df = pd.DataFrame()
            
        # Perform intelligent portfolio allocation
        if not filtered_ta_df.empty:
            self.allocate_portfolio(filtered_ta_df)
            
        return filtered_ta_df

    # --------------------- BACKTESTING METHODS ---------------------
    def calculate_fee(self, trade_value):
        """Returns the broker fee: Rs 20 or 2.5% of trade value (whichever is lower)."""
        fee_percent = 0.025 * trade_value
        fixed_fee = 20.0
        return min(fixed_fee, fee_percent)

    def apply_backtest_strategy(self, filtered_scrip_df, scrip, buy_signal, sell_signal):
        """
        Backtests a trading strategy on a filtered scrip dataframe using allocated cash.
        Returns a tuple: (backtested scrip dataframe, transactions dataframe)
        """
        df_bt = filtered_scrip_df.copy()
        
        # Use allocated cash for this specific stock
        allocated_cash = self.stock_allocations.get(scrip, self.initial_cash / len(self.stock_allocations))
        cash = allocated_cash
        
        position = 0
        portfolio_values = []
        positions = []
        buy_date = None
        buy_price = None
        trade_position = 'NO TRADE'
        transactions = []

        print(f"\n----- Backtesting {scrip} -----")
        print(f"Allocated Cash: Rs {allocated_cash:,.2f}")
        if 'P/E' in df_bt.columns and 'EPS' in df_bt.columns:
            print(f"Current P/E: {df_bt['P/E'].iloc[-1]:.2f} | Current EPS: {df_bt['EPS'].iloc[-1]:.2f}")
        
        # Get stock score details
        if scrip in self.stock_scores:
            score_info = self.stock_scores[scrip]
            print(f"Stock Score: {score_info['composite_score']:.1f}/100")
            print(f"Allocation: {(allocated_cash/self.initial_cash)*100:.1f}% of total portfolio\n")

        for idx, row in df_bt.iterrows():
            price = row['Close']
            stock_name = row['Stock']
            trade_position = 'NO TRADE'  # reset each iteration

            # BUY signal
            if buy_signal.loc[idx] and position == 0:
                shares_to_buy = int(cash // price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    fee = self.calculate_fee(cost)
                    total_cost = cost + fee
                    
                    if cash >= total_cost:
                        position = shares_to_buy
                        cash -= total_cost
                        buy_date = idx
                        buy_price = price
                        trade_position = 'BUY'
                        
                        print(f"Buy signal on {idx:%d-%m-%Y} for {stock_name}: Bought {position} shares at Rs {price:.2f}")
                        
                        transactions.append({
                            'Date': idx,
                            'Event': 'BUY',
                            'Stock': stock_name,
                            'Price': round(price, 2),
                            'Shares': shares_to_buy,
                            'Cost': round(cost, 2),
                            'Fee': round(fee, 2),
                            'Cash_After': round(cash, 2),
                            'Position_After': position,
                            'Allocated_Cash': round(allocated_cash, 2)
                        })
                    else:
                        print(f"Insufficient cash on {idx:%d-%m-%Y} for {stock_name}.")
                else:
                    print(f"Not enough cash to buy on {idx:%d-%m-%Y} for {stock_name}.")

            # SELL signal
            elif sell_signal.loc[idx] and position > 0:
                if buy_date is not None and buy_price is not None:
                    holding_period = (idx - buy_date).days
                    profit_percentage = ((price - buy_price) / buy_price) * 100
                else:
                    holding_period = 0
                    profit_percentage = 0.0

                if holding_period >= MIN_HOLDING_PERIOD and profit_percentage >= MIN_PROFIT_PERCENTAGE:
                    revenue = position * price
                    fee = self.calculate_fee(revenue)
                    net_revenue = revenue - fee
                    cash += net_revenue
                    trade_position = 'SELL'
                    
                    print(f"Sell signal on {idx:%d-%m-%Y} for {stock_name}: Sold {position} shares at Rs {price:.2f}")
                    
                    transactions.append({
                        'Date': idx,
                        'Event': 'SELL',
                        'Stock': stock_name,
                        'Price': round(price, 2),
                        'Shares': position,
                        'Revenue': round(revenue, 2),
                        'Fee': round(fee, 2),
                        'Cash_After': round(cash, 2),
                        'Position_After': 0,
                        'Holding_Period': holding_period,
                        'Profit_%': round(profit_percentage, 2),
                        'Allocated_Cash': round(allocated_cash, 2)
                    })
                    
                    position = 0
                    buy_date = None
                    buy_price = None
                else:
                    print(f"Sell conditions not met on {idx:%d-%m-%Y} for {stock_name}.")

            current_value = cash + position * price
            portfolio_values.append(round(current_value, 2))
            positions.append(position)
            
            df_bt.loc[idx, 'current_value'] = round(current_value, 2)
            df_bt.loc[idx, 'Position'] = round(position, 2)
            df_bt.loc[idx, 'balance_cash'] = round(cash, 2)
            df_bt.loc[idx, 'trade_position'] = trade_position

        df_bt['Portfolio_Value'] = portfolio_values
        df_bt['Position'] = positions

        scrip_final_value = portfolio_values[-1]
        scrip_total_return = (scrip_final_value - allocated_cash) / allocated_cash * 100

        print(f"\nLatest stock price: Rs {df_bt['Close'].iloc[-1]:.2f}")
        print(f"Scrip portfolio value: Rs {scrip_final_value:.2f} | Scrip return: {scrip_total_return:.2f}%")

        df_bt['Final_Value'] = np.nan
        df_bt['Total_Return'] = np.nan
        df_bt['Allocated_Cash'] = allocated_cash
        df_bt.at[df_bt.index[-1], 'Final_Value'] = scrip_final_value
        df_bt.at[df_bt.index[-1], 'Total_Return'] = round(scrip_total_return, 2)

        # Calculate trade returns
        trade_return = [None] * len(df_bt)
        for i, idx_val in enumerate(df_bt.index):
            if sell_signal.loc[idx_val]:
                pv = df_bt.loc[idx_val, 'Portfolio_Value']
                trade_return[i] = round((pv - allocated_cash) / allocated_cash * 100, 2)
        df_bt['Trade_Return'] = trade_return

        self.folio_final_value += scrip_final_value

        print(f"Accumulated portfolio value = {self.folio_final_value:.2f}")
        folio_cumulative_return = (self.folio_final_value - self.initial_cash) * 100 / self.initial_cash
        print(f"Portfolio cumulative return = {folio_cumulative_return:.2f}%\n")

        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            transactions_df.sort_values(by='Date', inplace=True)
            print("----- TRANSACTION LOG -----")
            with pd.option_context('display.float_format', '{:.2f}'.format):
                print(transactions_df.to_string(index=False))
            print("---------------------------\n")

        return df_bt, transactions_df

    def backtest_strategy(self, filtered_scrips_df):
        """
        Iterates over each scrip in filtered_scrips_df,
        applies the backtest strategy, and writes results to Excel.
        """
        for scrip, filtered_scrip_df in scrip_extractor(filtered_scrips_df):
            buy_signal = filtered_scrip_df['Buy']
            sell_signal = filtered_scrip_df['Sell']
            
            bs_df, bt_df = self.apply_backtest_strategy(filtered_scrip_df, scrip, buy_signal, sell_signal)
            self.backtested_scrip_df_list.append(bs_df)
            self.backtested_transactions_df_list.append(bt_df)

        backtested_scrip_df = pd.concat(self.backtested_scrip_df_list)
        backtested_transactions_df = pd.concat(self.backtested_transactions_df_list)

        # Write scrip backtest data to Excel
        backtested_scrip_df.reset_index(inplace=True)
        backtested_scrip_df['Date'] = pd.to_datetime(backtested_scrip_df['Date'], dayfirst=True)
        backtested_scrip_df['Date'] = backtested_scrip_df['Date'].dt.strftime('%d-%m-%Y')
        backtested_scrip_df.rename(columns={'index': 'Date'}, inplace=True)
        backtested_scrip_df.to_excel("backtested_scrips.xlsx", sheet_name=f"{config.ACTIVE_FILTER}", index=False)

        # Write transaction data to Excel
        backtested_transactions_df.reset_index(inplace=True)
        backtested_transactions_df['Date'] = pd.to_datetime(backtested_transactions_df['Date'], dayfirst=True)
        backtested_transactions_df['Date'] = backtested_transactions_df['Date'].dt.strftime('%d-%m-%Y')
        backtested_transactions_df.rename(columns={'index': 'Date'}, inplace=True)
        backtested_transactions_df.to_excel("backtested_transactions.xlsx", sheet_name=f"{config.ACTIVE_FILTER}", index=False)

        return backtested_scrip_df, backtested_transactions_df

    def backtested_global_summary(self, backtested_scrips_df, backtested_transactions_df):
        """
        Aggregates global summary from backtested scrips and writes a summary Excel file.
        Enhanced to include portfolio allocation details.
        """
        scrips = backtested_scrips_df['Stock'].unique()
        backtested_df_list = list(scripdf_extractor(backtested_scrips_df))
        num_scrips = len(scrips)
        
        global_initial_cash = self.initial_cash  # Now using single pool allocation
        global_final_value = self.folio_final_value
        global_profit = global_final_value - global_initial_cash
        global_profit_percentage = (global_profit / global_initial_cash) * 100

        if not backtested_transactions_df.empty:
            num_buy_global = backtested_transactions_df[backtested_transactions_df['Event'] == 'BUY'].shape[0]
            num_sell_global = backtested_transactions_df[backtested_transactions_df['Event'] == 'SELL'].shape[0]
            total_capital_used_global = backtested_transactions_df.loc[
                backtested_transactions_df['Event'] == 'BUY', 'Cost'
            ].sum()
            total_broker_charges_global = backtested_transactions_df['Fee'].sum()
        else:
            num_buy_global = num_sell_global = total_capital_used_global = total_broker_charges_global = 0

        # Enhanced global summary with allocation details
        global_summary_df = pd.DataFrame({
            "Metric": [
                "FILTER_NAME", "MIN_HOLDING_PERIOD", "MIN_PROFIT_PERCENTAGE",
                "NUMBER_OF_SCRIPS", "PORTFOLIO_APPROACH", "TOTAL_INITIAL_CASH", 
                "Total Final Value", "Total Capital Used", "Profit", "Profit Percentage",
                "Total Broker Charges", "Number of BUY", "Number of SELL"
            ],
            "Value": [
                config.ACTIVE_FILTER,
                MIN_HOLDING_PERIOD,
                MIN_PROFIT_PERCENTAGE,
                num_scrips,
                "Intelligent Allocation",
                global_initial_cash,
                round(global_final_value, 2),
                round(total_capital_used_global, 2),
                round(global_profit, 2),
                round(global_profit_percentage, 2),
                round(total_broker_charges_global, 2),
                num_buy_global,
                num_sell_global
            ]
        })

        # Create allocation summary
        allocation_summary = []
        for scrip in scrips:
            allocation_amount = self.stock_allocations.get(scrip, 0)
            allocation_pct = (allocation_amount / self.initial_cash) * 100
            score = self.stock_scores.get(scrip, {}).get('composite_score', 0)
            
            allocation_summary.append({
                'Stock': scrip,
                'Allocation_Amount': round(allocation_amount, 2),
                'Allocation_Percentage': round(allocation_pct, 2),
                'Score': round(score, 1)
            })
        
        allocation_df = pd.DataFrame(allocation_summary)

        title_row = pd.DataFrame({"Metric": ["GLOBAL PORTFOLIO SUMMARY"], "Value": [""]})
        blank_row = pd.DataFrame({"Metric": [""], "Value": [""]})
        
        # Prepare transaction columns
        transaction_cols = [
            "Date", "Event", "Stock", "Price", "Shares",
            "Cost", "Fee", "Cash_After", "Position_After",
            "Revenue", "Holding_Period", "Profit_%", "Allocated_Cash"
        ]
        
        if not backtested_transactions_df.empty:
            for col in transaction_cols:
                if col not in backtested_transactions_df.columns:
                    backtested_transactions_df[col] = ""
            backtested_transactions_df = backtested_transactions_df[transaction_cols]

        import support_files.File_IO as fio
        fio.change_cwd('gain_details')
        
        with pd.ExcelWriter("global_portfolio_summary.xlsx", engine="openpyxl") as writer:
            # Sheet 1: Complete Summary
            title_row.to_excel(writer, sheet_name="Portfolio_Summary", startrow=0, startcol=0, index=False, header=False)
            
            summary_startrow = len(title_row)
            global_summary_df.to_excel(writer, sheet_name="Portfolio_Summary", startrow=summary_startrow, startcol=0, index=False, header=False)
            
            # Add allocation details
            allocation_startrow = summary_startrow + len(global_summary_df) + 2
            allocation_title = pd.DataFrame({"Metric": ["STOCK ALLOCATIONS"], "Value": [""]})
            allocation_title.to_excel(writer, sheet_name="Portfolio_Summary", startrow=allocation_startrow, startcol=0, index=False, header=False)
            
            allocation_data_start = allocation_startrow + 1
            allocation_df.to_excel(writer, sheet_name="Portfolio_Summary", startrow=allocation_data_start, startcol=0, index=False, header=True)
            
            # Add transactions
            if not backtested_transactions_df.empty:
                transactions_startrow = allocation_data_start + len(allocation_df) + 3
                transactions_title = pd.DataFrame({"A": ["ALL TRANSACTIONS"], "B": [""]})
                transactions_title.to_excel(writer, sheet_name="Portfolio_Summary", startrow=transactions_startrow, startcol=0, index=False, header=False)
                
                trans_data_start = transactions_startrow + 1
                backtested_transactions_df.to_excel(writer, sheet_name="Portfolio_Summary", startrow=trans_data_start, startcol=0, index=False, header=True)
            
            # Sheet 2: Summary Only
            global_summary_df.to_excel(writer, sheet_name="Summary_Only", startrow=0, startcol=0, index=False, header=True)
            
            # Sheet 3: Allocation Details
            allocation_df.to_excel(writer, sheet_name="Allocations", startrow=0, startcol=0, index=False, header=True)

        print("Enhanced portfolio summary exported to 'global_portfolio_summary.xlsx'.")
        
        # Also save transactions separately
        backtested_transactions_df.to_excel('backtested_transactions_df.xlsx', index=False)
        
        # Print final portfolio summary
        print(f"\n===== FINAL PORTFOLIO PERFORMANCE =====")
        print(f"Initial Investment: Rs {global_initial_cash:,.2f}")
        print(f"Final Value: Rs {global_final_value:,.2f}")
        print(f"Total Profit: Rs {global_profit:,.2f}")
        print(f"Total Return: {global_profit_percentage:.2f}%")
        print(f"Number of Transactions: {num_buy_global} buys, {num_sell_global} sells")
        print(f"Total Broker Charges: Rs {total_broker_charges_global:.2f}")
        
        return backtested_transactions_df, global_summary_df

    # --------------------- RUN METHOD ---------------------
    def run(self, master_df):
        """
        Overall flow:
        1. Filter the master OHLC dataframe using apply_filter (includes intelligent allocation).
        2. Backtest each filtered scrip using allocated amounts.
        3. Generate a comprehensive portfolio summary.
        Returns backtested scrips and transactions dataframes.
        """
        print("Starting enhanced portfolio run with intelligent allocation...")
        print(f"Total Investment Capital: Rs {self.initial_cash:,.2f}")
        
        filtered_scrips_df = self.apply_filter(master_df)
        bs_df, bt_df = self.backtest_strategy(filtered_scrips_df)
        self.backtested_global_summary(bs_df, bt_df)
        
        return bs_df, bt_df


# --------------------- MAIN EXECUTION ---------------------
if __name__ == '__main__':
    import sys
    from support_files.dual_logger import DualLogger
    import support_files.File_IO as fio
    
    # Read the master dataframe (concatenated OHLC data)
    master_df = fio.read_csv_to_df('Nif50_5y_1w.csv', 'A','sub_dir')
    
    # Create an instance of FilteringAndBacktesting with Rs 1,00,000 initial cash
    fab = FilteringAndBacktesting(initial_cash=100000.0)
    fio.change_cwd('filtered_data')
    sys.stdout = DualLogger("log.txt")
    fab.run(master_df)
    fio.get_cwd()
    sys.stdout.flush()