import os
import support_files.config as config
import pandas as pd
import numpy as np
from support_files.scrip_extractor import scrip_extractor
import support_files.compute_indicators_helper as cmp  # For computing technical indicators

# Trade constraint constants
MIN_PROFIT_PERCENTAGE = config.MIN_PROFIT_PERCENTAGE
MIN_HOLDING_PERIOD = config.MIN_HOLDING_PERIOD


class FilteringAndBacktesting:
    def __init__(self, initial_cash=100000.0):
        self.initial_cash = initial_cash
        self.backtested_scrip_df_list = []
        self.backtested_transactions_df_list = []
        self.portfolio_value = 0.0
        self.stock_allocations = {}  # Store allocation amounts for each stock
        self.stock_scores = {}  # Store scoring data for each stock
        # Control verbose detailed prints: show full calculation details once (at start)
        self._detailed_print_shown = False

    # --------------------- PORTFOLIO ALLOCATION METHODS ---------------------
    def calculate_stock_score(self, scrip_ta_df, scrip):
        """
        Calculate a composite score for each stock based on multiple factors.
        Stores detailed score components in self.stock_scores[scrip].
        """
        try:
            rsi = scrip_ta_df['RSI'].iloc[-1] if 'RSI' in scrip_ta_df.columns else 50.0
            macd = scrip_ta_df['MACD'].iloc[-1] if 'MACD' in scrip_ta_df.columns else 0.0
            sma_ratio = (
                scrip_ta_df['Close'].iloc[-1] / scrip_ta_df['SMA_20'].iloc[-1]
                if 'SMA_20' in scrip_ta_df.columns and scrip_ta_df['SMA_20'].iloc[-1] != 0
                else 1.0
            )

            rsi_score = max(0.0, min(100.0, 100.0 - abs(rsi - 40.0)))
            macd_score = min(100.0, max(0.0, 50.0 + (macd * 10.0)))
            sma_score = min(100.0, max(0.0, (sma_ratio - 0.95) * 200.0))
            technical_score = (rsi_score * 0.4 + macd_score * 0.3 + sma_score * 0.3)

            buy_signals = int(scrip_ta_df['Buy'].sum()) if 'Buy' in scrip_ta_df.columns else 0
            sell_signals = int(scrip_ta_df['Sell'].sum()) if 'Sell' in scrip_ta_df.columns else 0
            total_signals = buy_signals + sell_signals
            if total_signals <= 20:
                signal_score = min(100.0, max(0.0, total_signals * 5.0))
            else:
                signal_score = max(0.0, 100.0 - (total_signals - 20) * 2.0)

            if len(scrip_ta_df) >= 20 and scrip_ta_df['Close'].iloc[-20] != 0:
                price_change = ((scrip_ta_df['Close'].iloc[-1] - scrip_ta_df['Close'].iloc[-20])
                                / scrip_ta_df['Close'].iloc[-20] * 100.0)
            else:
                price_change = 0.0
            momentum_score = min(100.0, max(0.0, 50.0 + price_change * 2.0))

            volatility = scrip_ta_df['Close'].pct_change().std() * 100.0 if 'Close' in scrip_ta_df.columns else 0.0
            volatility_score = min(100.0, max(0.0, 100.0 - volatility * 5.0))

            composite_score = (technical_score * 0.4 + signal_score * 0.3 +
                               momentum_score * 0.2 + volatility_score * 0.1)

            self.stock_scores[scrip] = {
                'composite_score': composite_score,
                'technical_score': technical_score,
                'signal_score': signal_score,
                'momentum_score': momentum_score,
                'volatility_score': volatility_score,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'current_price': float(scrip_ta_df['Close'].iloc[-1]) if 'Close' in scrip_ta_df.columns else 0.0,
                'rsi': rsi,
                'price_momentum': price_change
            }

            return composite_score
        except Exception as e:
            print(f"Error calculating score for {scrip}: {e}")
            return 50.0

    def allocate_portfolio(self, filtered_scrips_df):
        """
        Intelligently allocate the total cash across stocks based on their scores.
        Prints detailed calculation steps for allocation using standard trading terminology.
        """
        scrips = filtered_scrips_df['Stock'].unique()
        print(f"\n" + "="*80)
        print(f"PORTFOLIO ALLOCATION STRATEGY".center(80))
        print(f"="*80)
        print(f"Total Capital Available: ₹{self.initial_cash:,.2f}")
        print(f"Number of Securities Selected: {len(scrips)}")
        print(f"Allocation Method: Risk-Weighted Dynamic Allocation")
        
        # Calculate scores for all stocks
        stock_scores = {}
        for scrip in scrips:
            scrip_data = filtered_scrips_df[filtered_scrips_df['Stock'] == scrip]
            score = self.calculate_stock_score(scrip_data, scrip)
            stock_scores[scrip] = score
        
        # Sort stocks by score (descending)
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Allocation Strategy: Exponential decay with minimum allocation
        total_weight = 0.0
        weights = {}
        
        # Calculate weights using exponential decay
        for i, (scrip, score) in enumerate(sorted_stocks):
            decay_factor = 0.8 ** i  # Each subsequent stock gets 80% of previous weight
            base_weight = score / 100.0  # Normalize score to 0-1
            weight = base_weight * decay_factor
            weights[scrip] = weight
            total_weight += weight

        # Print weight calculation details
        print(f"\nRISK-WEIGHTED ALLOCATION CALCULATION:")
        print(f"{'Security':<15} {'Score':>8} {'Rank':>6} {'Weight':>10} {'Raw Alloc':>12}")
        print("-" * 60)
        for i, (scrip, score) in enumerate(sorted_stocks):
            decay_factor = 0.8 ** i
            base_weight = score / 100.0
            raw_w = weights[scrip]
            print(f"{scrip:<15} {score:8.1f} #{i+1:>5} {raw_w:10.6f} {raw_w/total_weight*100:11.2f}%")
        print(f"Total Weight Sum: {total_weight:.6f}")

        # Normalize weights to sum to 1
        min_allocation = 0.05  # 5% minimum
        max_allocation = 0.35  # 35% maximum

        normalized_weights = {}
        print(f"\nAPPLYING POSITION SIZE LIMITS:")
        print(f"Minimum Position Size: {min_allocation*100:.1f}%")
        print(f"Maximum Position Size: {max_allocation*100:.1f}%")
        
        for scrip, weight in weights.items():
            normalized = weight / total_weight if total_weight else 0.0
            normalized_weights[scrip] = normalized

        # Apply min/max constraints
        constrained_weights = {}
        print(f"\n{'Security':<15} {'Before':<10} {'After':<10} {'Status':<15}")
        print("-" * 55)
        for scrip, normalized in normalized_weights.items():
            constrained = max(min_allocation, min(normalized, max_allocation))
            constrained_weights[scrip] = constrained
            if constrained != normalized:
                if constrained == min_allocation:
                    status = "Min Applied"
                else:
                    status = "Max Applied"
            else:
                status = "No Change"
            print(f"{scrip:<15} {normalized*100:9.2f}% {constrained*100:9.2f}% {status:<15}")

        total_constrained = sum(constrained_weights.values()) or 1.0

        # Renormalize constrained weights to sum to 1
        for scrip in constrained_weights:
            normalized_weights[scrip] = constrained_weights[scrip] / total_constrained

        # Calculate final allocations and print detailed steps
        print(f"\n" + "="*80)
        print(f"FINAL PORTFOLIO ALLOCATION".center(80))
        print(f"="*80)
        print(f"{'Security':<15} {'Score':<8} {'Weight':<10} {'Capital Alloc':<15} {'Rank':<6}")
        print("-" * 65)
        
        for i, (scrip, score) in enumerate(sorted_stocks):
            allocation_pct = normalized_weights[scrip] * 100.0
            allocation_amount = self.initial_cash * normalized_weights[scrip]
            self.stock_allocations[scrip] = allocation_amount
            
            print(f"{scrip:<15} {score:<8.1f} {allocation_pct:<9.2f}% ₹{allocation_amount:<14,.0f} #{i+1:<6}")
        
        print("-" * 65)
        total_alloc = sum(self.stock_allocations.values())
        print(f"{'TOTAL':<15} {'':<8} {'100.0%':<10} ₹{total_alloc:<14,.0f}")
        
        # Print detailed scoring breakdown for top 3 stocks (only once)
        if not self._detailed_print_shown:
            print(f"\n" + "="*60)
            print(f"TOP 3 SECURITIES - DETAILED ANALYSIS".center(60))
            print(f"="*60)
            for i, (scrip, score) in enumerate(sorted_stocks[:3]):
                details = self.stock_scores[scrip]
                print(f"\n#{i+1} {scrip} - Overall Score: {score:.1f}/100")
                print(f"  Technical Strength: {details['technical_score']:.1f}/100")
                print(f"  Signal Quality:     {details['signal_score']:.1f}/100")
                print(f"  Price Momentum:     {details['momentum_score']:.1f}/100 ({details['price_momentum']:.2f}% change)")
                print(f"  Risk Assessment:    {details['volatility_score']:.1f}/100")
                print(f"  Current Market Price: ₹{details['current_price']:.2f}")
                print(f"  RSI Level:          {details['rsi']:.1f}")
                print(f"  Trading Signals:    {details['buy_signals']} Buy, {details['sell_signals']} Sell")
            # mark detailed calculations as shown so later per-stock runs remain concise
            self._detailed_print_shown = True
        else:
            print("\nDetailed analysis already shown; subsequent logs will be concise.")
        
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

        print("\n" + "="*60)
        print("STOCK SCREENING & FILTERING".center(60))
        print("="*60)
        print(f"Universe Size: {len(master_scrips_list)} securities")
        print(f"Screening Filter: {config.ACTIVE_FILTER}")

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
            filtered_ta_df = pd.concat(filtered_ta_df_list) if filtered_ta_df_list else pd.DataFrame()
        except ValueError as e:
            print(f"Error in screening process: {e}")
            filtered_ta_df = pd.DataFrame()

        print(f"Securities Passed Screening: {len(scrips_list)}")
        print(f"Filter Success Rate: {len(scrips_list)/len(master_scrips_list)*100:.1f}%")

        if not filtered_ta_df.empty:
            self.allocate_portfolio(filtered_ta_df)

        return filtered_ta_df

    # --------------------- BACKTESTING METHODS ---------------------
    def calculate_fee(self, trade_value):
        """Returns the broker fee: ₹20 or 2.5% of trade value (whichever is lower)."""
        fee_percent = 0.025 * trade_value
        fixed_fee = 20.0
        return min(fixed_fee, fee_percent)

    def apply_backtest_strategy(self, filtered_scrip_df, scrip, buy_signal, sell_signal):
        """
        Backtests a trading strategy on a filtered scrip dataframe using allocated cash.
        Returns a tuple: (backtested scrip dataframe, transactions dataframe)
        Uses standard trading terminology in logs.
        """
        df_bt = filtered_scrip_df.copy()
        allocated_capital = self.stock_allocations.get(scrip,
                                                   self.initial_cash / len(self.stock_allocations) if self.stock_allocations else self.initial_cash)
        available_cash = float(allocated_capital)

        print(f"\n" + "="*70)
        print(f"BACKTESTING: {scrip}".center(70))
        print("="*70)
        
        # Print allocation info concisely if detailed already shown, otherwise verbose
        if not self._detailed_print_shown:
            print("Capital Allocation Details:")
            if scrip in self.stock_allocations:
                print(f"  Allocated Capital: ₹{self.stock_allocations[scrip]:,.2f}")
            else:
                fallback = self.initial_cash / len(self.stock_allocations) if self.stock_allocations else self.initial_cash
                print(f"  Default Allocation: ₹{fallback:,.2f} (equal weight fallback)")
            print(f"  Starting Cash Available: ₹{allocated_capital:,.2f}")
        else:
            allocation_pct = (allocated_capital / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            print(f"Allocated Capital: ₹{allocated_capital:,.2f} ({allocation_pct:.1f}% of portfolio)")

        position_qty = 0
        portfolio_values = []
        positions = []
        entry_date = None
        entry_price = None
        trade_status = 'NO POSITION'
        transactions = []

        if 'P/E' in df_bt.columns and 'EPS' in df_bt.columns:
            print(f"Fundamental Data - P/E: {df_bt['P/E'].iloc[-1]:.2f} | EPS: ₹{df_bt['EPS'].iloc[-1]:.2f}")

        if scrip in self.stock_scores:
            score_info = self.stock_scores[scrip]
            print(f"Investment Score: {score_info['composite_score']:.1f}/100")
            print(f"  • Technical Analysis: {score_info['technical_score']:.1f}/100")
            print(f"  • Signal Quality: {score_info['signal_score']:.1f}/100") 
            print(f"  • Price Momentum: {score_info['momentum_score']:.1f}/100")
            print(f"  • Risk Assessment: {score_info['volatility_score']:.1f}/100")
            allocation_pct = (allocated_capital / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            print(f"Portfolio Weight: {allocation_pct:.1f}%")

        print(f"\nTRADING ACTIVITY LOG:")
        print("-" * 70)

        for idx, row in df_bt.iterrows():
            market_price = float(row['Close'])
            stock_name = row['Stock']
            trade_status = 'NO ACTION'

            # BUY LOGIC
            if buy_signal.loc[idx] and position_qty == 0:
                shares_affordable = int(available_cash // market_price)
                if shares_affordable > 0:
                    gross_cost = shares_affordable * market_price
                    brokerage = self.calculate_fee(gross_cost)
                    total_investment = gross_cost + brokerage

                    if available_cash >= total_investment:
                        position_qty = shares_affordable
                        available_cash -= total_investment
                        entry_date = idx
                        entry_price = market_price
                        trade_status = 'LONG ENTRY'

                        # Record BUY transaction
                        transactions.append({
                            'Date': idx,
                            'Event': 'BUY',
                            'Stock': stock_name,
                            'Price': round(market_price, 2),
                            'Shares': position_qty,
                            'Cost': round(gross_cost, 2),
                            'Fee': round(brokerage, 2),
                            'Cash_After': round(available_cash, 2),
                            'Position_After': position_qty,
                            'Holding_Period': 0,
                            'Profit_%': 0.0,
                            'Allocated_Cash': round(allocated_capital, 2)
                        })

                        if not self._detailed_print_shown:
                            print(f"\n📈 LONG ENTRY - {idx:%d-%b-%Y}")
                            print(f"   Security: {stock_name}")
                            print(f"   Entry Price: ₹{market_price:.2f}")
                            print(f"   Quantity: {position_qty:,} shares")
                            print(f"   Gross Investment: ₹{gross_cost:,.2f}")
                            print(f"   Brokerage: ₹{brokerage:.2f}")
                            print(f"   Total Investment: ₹{total_investment:,.2f}")
                            print(f"   Cash Remaining: ₹{available_cash:,.2f}")
                        else:
                            print(f"📈 {idx:%d-%b-%Y} | LONG ENTRY | {position_qty:,} shares @ ₹{market_price:.2f} | Cash: ₹{available_cash:,.0f}")

                else:
                    print(f"⚠️  {idx:%d-%b-%Y} | Insufficient funds for {stock_name} @ ₹{market_price:.2f}")

            # SELL LOGIC
            elif sell_signal.loc[idx] and position_qty > 0:
                if entry_date is not None and entry_price is not None:
                    holding_period = (idx - entry_date).days
                    unrealized_pnl_pct = ((market_price - entry_price) / entry_price) * 100.0
                else:
                    holding_period = 0
                    unrealized_pnl_pct = 0.0

                # Check exit conditions
                if holding_period >= MIN_HOLDING_PERIOD and unrealized_pnl_pct >= MIN_PROFIT_PERCENTAGE:
                    gross_proceeds = position_qty * market_price
                    brokerage = self.calculate_fee(gross_proceeds)
                    net_proceeds = gross_proceeds - brokerage
                    available_cash += net_proceeds
                    
                    # Calculate realized P&L
                    total_cost = position_qty * entry_price + self.calculate_fee(position_qty * entry_price)
                    realized_pnl = net_proceeds - total_cost
                    realized_pnl_pct = (realized_pnl / total_cost) * 100.0
                    
                    trade_status = 'LONG EXIT'

                    if not self._detailed_print_shown:
                        print(f"\n📉 LONG EXIT - {idx:%d-%b-%Y}")
                        print(f"   Security: {stock_name}")
                        print(f"   Exit Price: ₹{market_price:.2f}")
                        print(f"   Quantity: {position_qty:,} shares")
                        print(f"   Holding Period: {holding_period} days")
                        print(f"   Gross Proceeds: ₹{gross_proceeds:,.2f}")
                        print(f"   Brokerage: ₹{brokerage:.2f}")
                        print(f"   Net Proceeds: ₹{net_proceeds:,.2f}")
                        print(f"   Realized P&L: ₹{realized_pnl:,.2f} ({realized_pnl_pct:+.2f}%)")
                        print(f"   Total Cash: ₹{available_cash:,.2f}")
                    else:
                        print(f"📉 {idx:%d-%b-%Y} | LONG EXIT | {position_qty:,} shares @ ₹{market_price:.2f} | P&L: ₹{realized_pnl:+.0f} ({realized_pnl_pct:+.1f}%) | Cash: ₹{available_cash:,.0f}")

                    transactions.append({
                        'Date': idx,
                        'Event': 'SELL',
                        'Stock': stock_name,
                        'Price': round(market_price, 2),
                        'Shares': position_qty,
                        'Revenue': round(gross_proceeds, 2),
                        'Fee': round(brokerage, 2),
                        'Cash_After': round(available_cash, 2),
                        'Position_After': 0,
                        'Holding_Period': holding_period,
                        'Profit_%': round(unrealized_pnl_pct, 2),
                        'Allocated_Cash': round(allocated_capital, 2)
                    })

                    position_qty = 0
                    entry_date = None
                    entry_price = None
                else:
                    print(f"⏳ {idx:%d-%b-%Y} | Position held | Days: {holding_period} | Unrealized P&L: {unrealized_pnl_pct:+.1f}% | Conditions not met")

            # Calculate current portfolio value for this stock
            current_position_value = position_qty * market_price
            total_portfolio_value = available_cash + current_position_value
            portfolio_values.append(round(total_portfolio_value, 2))
            positions.append(position_qty)

            df_bt.loc[idx, 'current_value'] = round(total_portfolio_value, 2)
            df_bt.loc[idx, 'Position'] = round(position_qty, 2)
            df_bt.loc[idx, 'balance_cash'] = round(available_cash, 2)
            df_bt.loc[idx, 'trade_position'] = trade_status

        df_bt['Portfolio_Value'] = portfolio_values
        df_bt['Position'] = positions

        # Final position summary
        last_idx = df_bt.index[-1]
        final_cash = float(df_bt.loc[last_idx, 'balance_cash'])
        final_position = int(df_bt.loc[last_idx, 'Position'])
        current_market_price = float(df_bt.loc[last_idx, 'Close'])
        
        final_position_value = final_position * current_market_price
        final_portfolio_value = final_cash + final_position_value
        
        total_return_amount = final_portfolio_value - allocated_capital
        total_return_pct = (total_return_amount / allocated_capital) * 100.0 if allocated_capital else 0.0

        print(f"\n" + "-"*70)
        print(f"POSITION SUMMARY as of {last_idx:%d-%b-%Y}")
        print(f"-"*70)
        print(f"Current Market Price: ₹{current_market_price:.2f}")
        print(f"Position: {final_position:,} shares")
        print(f"Position Value: ₹{final_position_value:,.2f}")
        print(f"Available Cash: ₹{final_cash:,.2f}")
        print(f"Total Portfolio Value: ₹{final_portfolio_value:,.2f}")
        print(f"Total Return: ₹{total_return_amount:+,.2f} ({total_return_pct:+.2f}%)")

        df_bt['Final_Value'] = np.nan
        df_bt['Total_Return'] = np.nan
        df_bt['Allocated_Cash'] = allocated_capital
        df_bt.at[df_bt.index[-1], 'Final_Value'] = final_portfolio_value
        df_bt.at[df_bt.index[-1], 'Total_Return'] = round(total_return_pct, 2)

        trade_return = [None] * len(df_bt)
        for i, idx_val in enumerate(df_bt.index):
            if sell_signal.loc[idx_val]:
                pv = df_bt.loc[idx_val, 'Portfolio_Value']
                trade_return[i] = round((pv - allocated_capital) / allocated_capital * 100.0, 2)
        df_bt['Trade_Return'] = trade_return

        # Update global portfolio value
        prev_portfolio = self.portfolio_value
        print(f"\nGLOBAL PORTFOLIO UPDATE:")
        print(f"Previous Portfolio Value: ₹{prev_portfolio:,.2f}")
        self.portfolio_value += final_portfolio_value
        print(f"Added from {scrip}: ₹{final_portfolio_value:,.2f}")
        print(f"New Portfolio Value: ₹{self.portfolio_value:,.2f}")
        global_return = (self.portfolio_value - self.initial_cash) * 100.0 / self.initial_cash if self.initial_cash else 0.0
        print(f"Overall Portfolio Return: {global_return:+.2f}%")

        # Transaction log
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            transactions_df.sort_values(by='Date', inplace=True)
            print(f"\n" + "="*60)
            print("TRANSACTION HISTORY".center(60))
            print("="*60)
            with pd.option_context('display.float_format', '{:.2f}'.format):
                print(transactions_df.to_string(index=False))

        return df_bt, transactions_df

    def backtest_strategy(self, filtered_scrips_df):
        """
        Iterates over each scrip in filtered_scrips_df,
        applies the backtest strategy, and writes results to Excel.
        """
        print(f"\n" + "="*80)
        print("PORTFOLIO BACKTESTING STARTED".center(80))
        print("="*80)

        for scrip, filtered_scrip_df in scrip_extractor(filtered_scrips_df):
            buy_signal = filtered_scrip_df['Buy']
            sell_signal = filtered_scrip_df['Sell']

            bs_df, bt_df = self.apply_backtest_strategy(filtered_scrip_df, scrip, buy_signal, sell_signal)
            self.backtested_scrip_df_list.append(bs_df)
            self.backtested_transactions_df_list.append(bt_df)

        backtested_scrip_df = pd.concat(self.backtested_scrip_df_list) if self.backtested_scrip_df_list else pd.DataFrame()
        backtested_transactions_df = pd.concat(self.backtested_transactions_df_list) if self.backtested_transactions_df_list else pd.DataFrame()

        if not backtested_scrip_df.empty:
            backtested_scrip_df.reset_index(inplace=True)
            backtested_scrip_df['Date'] = pd.to_datetime(backtested_scrip_df['Date'], dayfirst=True)
            backtested_scrip_df['Date'] = backtested_scrip_df['Date'].dt.strftime('%d-%m-%Y')
            backtested_scrip_df.rename(columns={'index': 'Date'}, inplace=True)
            backtested_scrip_df.to_excel("backtested_scrips.xlsx", sheet_name=f"{config.ACTIVE_FILTER}", index=False)

        if not backtested_transactions_df.empty:
            backtested_transactions_df.reset_index(inplace=True)
            
            # Handle duplicate Date columns properly
            if 'Date' in backtested_transactions_df.columns:
                date_columns = [col for col in backtested_transactions_df.columns if col == 'Date']
                if len(date_columns) > 1:
                    # Combine all Date columns (take first non-null value)
                    date_data = backtested_transactions_df[date_columns].bfill(axis=1).iloc[:, 0]
                    # Remove all Date columns
                    backtested_transactions_df = backtested_transactions_df.loc[:, ~(backtested_transactions_df.columns == 'Date')]
                    # Add back single Date column
                    backtested_transactions_df['Date'] = date_data
            
            # Now safely format dates
            try:
                backtested_transactions_df['Date'] = pd.to_datetime(backtested_transactions_df['Date'], dayfirst=True)
                backtested_transactions_df['Date'] = backtested_transactions_df['Date'].dt.strftime('%d-%m-%Y')
            except Exception as e:
                print(f"Warning: Date formatting issue: {e}")
            
            # Handle index column naming
            if 'index' in backtested_transactions_df.columns:
                backtested_transactions_df.drop('index', axis=1, inplace=True)
            
            backtested_transactions_df.to_excel("backtested_transactions.xlsx", sheet_name=f"{config.ACTIVE_FILTER}", index=False)

        return backtested_scrip_df, backtested_transactions_df

    def backtested_global_summary(self, backtested_scrips_df, backtested_transactions_df, master_df=None):
        """
        Aggregates global summary from backtested scrips and writes a summary Excel file.
        Uses professional trading terminology and metrics.
        """
        backtested_scrips_df = backtested_scrips_df.copy() if not backtested_scrips_df.empty else pd.DataFrame()
        backtested_transactions_df = backtested_transactions_df.copy() if not backtested_transactions_df.empty else pd.DataFrame()

        scrips = backtested_scrips_df['Stock'].unique() if not backtested_scrips_df.empty else []
        num_positions = len(scrips)
        initial_capital = self.initial_cash

        print(f"\n" + "="*80)
        print("PORTFOLIO PERFORMANCE ANALYSIS".center(80))
        print("="*80)

        # Build map of final cash and positions per stock
        final_cash_map = {}
        final_position_map = {}
        for scrip, df_s in scrip_extractor(backtested_scrips_df):
            if df_s.empty:
                final_cash_map[scrip] = 0.0
                final_position_map[scrip] = 0
                continue
            try:
                last_row = df_s.loc[df_s.index.max()]
            except Exception:
                last_row = df_s.iloc[-1] if not df_s.empty else None
            if last_row is None:
                final_cash_map[scrip] = 0.0
                final_position_map[scrip] = 0
            else:
                if hasattr(last_row, "get"):
                    final_cash_map[scrip] = float(last_row.get('balance_cash', 0.0) or 0.0)
                    final_position_map[scrip] = int(last_row.get('Position', 0) or 0)
                else:
                    final_cash_map[scrip] = float(last_row['balance_cash']) if 'balance_cash' in last_row.index else 0.0
                    final_position_map[scrip] = int(last_row['Position']) if 'Position' in last_row.index else 0

        # Get latest market prices
        current_market_prices = {}
        if master_df is not None:
            try:
                latest_data = master_df.sort_values(by=['Stock', 'Date']).groupby('Stock').last()
                for scrip in scrips:
                    current_market_prices[scrip] = float(latest_data.loc[scrip]['Close']) if scrip in latest_data.index else None
            except Exception:
                current_market_prices = {}
        
        # Fallback to backtested data if master_df not available
        for scrip in scrips:
            if scrip not in current_market_prices or current_market_prices.get(scrip) is None:
                try:
                    df_s = backtested_scrips_df[backtested_scrips_df['Stock'] == scrip]
                    current_market_prices[scrip] = float(df_s['Close'].iloc[-1]) if not df_s.empty else 0.0
                except Exception:
                    current_market_prices[scrip] = 0.0

        # Calculate portfolio components
        total_cash = sum(final_cash_map.values())
        unrealized_market_value = 0.0
        
        print("CURRENT PORTFOLIO HOLDINGS:")
        print("-" * 80)
        print(f"{'Security':<15} {'Shares':<10} {'Market Price':<12} {'Market Value':<15} {'Cash':<12}")
        print("-" * 80)
        
        for scrip in scrips:
            shares = final_position_map.get(scrip, 0)
            market_price = current_market_prices.get(scrip, 0.0) or 0.0
            market_value = shares * market_price
            cash = final_cash_map.get(scrip, 0.0)
            unrealized_market_value += market_value
            
            print(f"{scrip:<15} {shares:<10,} ₹{market_price:<11.2f} ₹{market_value:<14,.0f} ₹{cash:<11,.0f}")

        print("-" * 80)
        print(f"{'TOTALS':<15} {'':<10} {'':<12} ₹{unrealized_market_value:<14,.0f} ₹{total_cash:<11,.0f}")

        total_portfolio_value = total_cash + unrealized_market_value
        print(f"\nPORTFOLIO VALUATION SUMMARY:")
        print(f"Cash Balance: ₹{total_cash:,.2f}")
        print(f"Unrealized Market Value: ₹{unrealized_market_value:,.2f}")
        print(f"Total Portfolio Value: ₹{total_portfolio_value:,.2f}")

        # Calculate Realized P&L using transaction history (FIFO method)
        realized_pnl = 0.0
        print(f"\n" + "="*60)
        print("REALIZED P&L CALCULATION (FIFO Method)".center(60))
        print("="*60)
        
        if not backtested_transactions_df.empty:
            tx_df = backtested_transactions_df.copy()
            
            # Handle duplicate Date columns properly
            if 'Date' in tx_df.columns:
                date_columns = [col for col in tx_df.columns if col == 'Date']
                if len(date_columns) > 1:
                    # Combine all Date columns (take first non-null value)
                    date_data = tx_df[date_columns].bfill(axis=1).iloc[:, 0]
                    # Remove all Date columns
                    tx_df = tx_df.loc[:, ~(tx_df.columns == 'Date')]
                    # Add back single Date column
                    tx_df['Date'] = date_data
                
            # Convert to datetime safely
            try:
                tx_df['Date'] = pd.to_datetime(tx_df['Date'], dayfirst=True, errors='coerce')
            except Exception as e:
                print(f"Warning: Date conversion error in P&L calculation: {e}")
                tx_df['Date'] = pd.to_datetime('today')  # Fallback date
            
            tx_df = tx_df.sort_values(by=['Stock', 'Date'])
            
            print(f"{'Security':<15} {'Trade Type':<12} {'Quantity':<10} {'Price':<10} {'P&L':<15}")
            print("-" * 70)
            
            for stock, group in tx_df.groupby('Stock'):
                buy_positions = []  # FIFO queue for buy positions
                stock_realized_pnl = 0.0
                
                for _, transaction in group.iterrows():
                    event_type = str(transaction.get('Event', '')).upper()
                    quantity = int(transaction.get('Shares', 0) or 0)
                    price = float(transaction.get('Price', 0.0) or 0.0)
                    
                    if event_type == 'BUY' and quantity > 0:
                        total_cost = float(transaction.get('Revenue', 0.0) or 0.0)  # Actually cost for BUY
                        brokerage = float(transaction.get('Fee', 0.0) or 0.0)
                        buy_positions.append({
                            'quantity': quantity,
                            'avg_price': price,
                            'total_cost': total_cost + brokerage
                        })
                        print(f"{stock:<15} {'LONG ENTRY':<12} {quantity:<10,} ₹{price:<9.2f} {'-':<15}")
                    
                    elif event_type == 'SELL' and quantity > 0:
                        gross_proceeds = float(transaction.get('Revenue', 0.0) or 0.0)
                        brokerage = float(transaction.get('Fee', 0.0) or 0.0)
                        net_proceeds = gross_proceeds - brokerage
                        
                        remaining_to_sell = quantity
                        trade_pnl = 0.0
                        
                        while remaining_to_sell > 0 and buy_positions:
                            buy_lot = buy_positions[0]
                            buy_qty = buy_lot['quantity']
                            
                            if buy_qty <= remaining_to_sell:
                                # Sell entire buy lot
                                sell_proportion = buy_qty / quantity
                                allocated_proceeds = net_proceeds * sell_proportion
                                lot_pnl = allocated_proceeds - buy_lot['total_cost']
                                trade_pnl += lot_pnl
                                
                                remaining_to_sell -= buy_qty
                                buy_positions.pop(0)  # Remove fully sold lot
                            else:
                                # Partially sell buy lot
                                sell_proportion = remaining_to_sell / quantity
                                allocated_proceeds = net_proceeds * sell_proportion
                                cost_proportion = (remaining_to_sell / buy_qty) * buy_lot['total_cost']
                                lot_pnl = allocated_proceeds - cost_proportion
                                trade_pnl += lot_pnl
                                
                                # Update remaining buy lot
                                buy_lot['quantity'] -= remaining_to_sell
                                buy_lot['total_cost'] -= cost_proportion
                                remaining_to_sell = 0
                        
                        stock_realized_pnl += trade_pnl
                        realized_pnl += trade_pnl
                        
                        print(f"{stock:<15} {'LONG EXIT':<12} {quantity:<10,} ₹{price:<9.2f} ₹{trade_pnl:<14,.0f}")
                
                if stock_realized_pnl != 0:
                    print(f"{stock + ' TOTAL':<15} {'':<12} {'':<10} {'':<10} ₹{stock_realized_pnl:<14,.0f}")
                    print("-" * 70)
        else:
            print("No completed transactions found.")
        
        print(f"TOTAL REALIZED P&L: ₹{realized_pnl:,.2f}")

        # Calculate CAGR and other metrics
        print(f"\n" + "="*60)
        print("PERFORMANCE METRICS".center(60))
        print("="*60)
        
        # Date range calculation
        start_date = None
        end_date = None
        if not backtested_transactions_df.empty and 'Date' in backtested_transactions_df.columns:
            try:
                # Handle potential duplicate Date columns
                tx_temp = backtested_transactions_df.copy()
                if 'Date' in tx_temp.columns:
                    date_columns = [col for col in tx_temp.columns if col == 'Date']
                    if len(date_columns) > 1:
                        date_data = tx_temp[date_columns].bfill(axis=1).iloc[:, 0]
                        start_date = pd.to_datetime(date_data, dayfirst=True, errors='coerce').min()
                    else:
                        start_date = pd.to_datetime(tx_temp['Date'], dayfirst=True, errors='coerce').min()
            except Exception as e:
                print(f"Warning: Start date calculation error: {e}")
                start_date = None
                
        if start_date is None and master_df is not None and 'Date' in master_df.columns:
            try:
                start_date = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce').min()
            except Exception:
                start_date = None
        
        if master_df is not None and 'Date' in master_df.columns:
            try:
                end_date = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce').max()
            except Exception:
                end_date = None

        # Calculate CAGR
        cagr = None
        if start_date and end_date and not pd.isna(start_date) and not pd.isna(end_date):
            days = (end_date - start_date).days
            years = max(days / 365.25, 1.0 / 365.25)  # Minimum 1 day = 1/365.25 years
            
            try:
                cagr = (total_portfolio_value / initial_capital) ** (1.0 / years) - 1.0
                print(f"Investment Period: {start_date.date()} to {end_date.date()} ({days} days, {years:.2f} years)")
            except Exception as e:
                cagr = None
                print(f"CAGR calculation error: {e}")

        # Portfolio performance summary
        total_return_amount = total_portfolio_value - initial_capital
        total_return_pct = (total_return_amount / initial_capital) * 100.0 if initial_capital else 0.0
        unrealized_pnl = total_return_amount - realized_pnl

        # Transaction statistics
        if not backtested_transactions_df.empty:
            buy_transactions = backtested_transactions_df[backtested_transactions_df.get('Event') == 'BUY'].shape[0]
            sell_transactions = backtested_transactions_df[backtested_transactions_df.get('Event') == 'SELL'].shape[0]

            # Calculate total brokerage
            total_brokerage = 0.0
            if 'Fee' in backtested_transactions_df.columns:
                total_brokerage = pd.to_numeric(backtested_transactions_df['Fee'], errors='coerce').fillna(0.0).sum()

            # Calculate capital deployment
            capital_deployed = 0.0
            buy_txns = backtested_transactions_df[backtested_transactions_df['Event'] == 'BUY']
            if not buy_txns.empty and 'Revenue' in buy_txns.columns:
                capital_deployed = pd.to_numeric(buy_txns['Revenue'], errors='coerce').fillna(0.0).sum()
            elif not buy_txns.empty and 'Price' in buy_txns.columns and 'Shares' in buy_txns.columns:
                prices = pd.to_numeric(buy_txns['Price'], errors='coerce').fillna(0.0)
                shares = pd.to_numeric(buy_txns['Shares'], errors='coerce').fillna(0.0)
                capital_deployed = (prices * shares).sum()
        else:
            buy_transactions = sell_transactions = total_brokerage = capital_deployed = 0

        print(f"\nPERFORMANCE SUMMARY:")
        print(f"{'Metric':<30} {'Value':<20}")
        print("-" * 50)
        print(f"{'Initial Capital':<30} ₹{initial_capital:>15,.2f}")
        print(f"{'Final Portfolio Value':<30} ₹{total_portfolio_value:>15,.2f}")
        print(f"{'Total Return (Amount)':<30} ₹{total_return_amount:>15,.2f}")
        print(f"{'Total Return (%)':<30} {total_return_pct:>15.2f}%")
        print(f"{'Realized P&L':<30} ₹{realized_pnl:>15,.2f}")
        print(f"{'Unrealized P&L':<30} ₹{unrealized_pnl:>15,.2f}")
        if cagr is not None:
            print(f"{'CAGR':<30} {cagr*100:>15.2f}%")
        print(f"{'Capital Deployed':<30} ₹{capital_deployed:>15,.2f}")
        print(f"{'Total Brokerage':<30} ₹{total_brokerage:>15,.2f}")
        print(f"{'Number of Positions':<30} {num_positions:>15}")
        print(f"{'Buy Transactions':<30} {buy_transactions:>15}")
        print(f"{'Sell Transactions':<30} {sell_transactions:>15}")

        # Create summary dataframe for Excel export
        global_summary_df = pd.DataFrame({
            "Metric": [
                "STRATEGY_NAME", "MIN_HOLDING_PERIOD_DAYS", "MIN_PROFIT_PERCENTAGE",
                "NUMBER_OF_POSITIONS", "INVESTMENT_APPROACH", "INITIAL_CAPITAL",
                "Final Portfolio Value", "Capital Deployed", "Total Return (Amount)", "Total Return (%)",
                "Total Brokerage Paid", "Buy Transactions", "Sell Transactions",
                "Realized P&L", "Unrealized P&L", "CAGR (%)"
            ],
            "Value": [
                config.ACTIVE_FILTER,
                MIN_HOLDING_PERIOD,
                MIN_PROFIT_PERCENTAGE,
                num_positions,
                "Risk-Weighted Allocation",
                initial_capital,
                round(total_portfolio_value, 2),
                round(capital_deployed, 2),
                round(total_return_amount, 2),
                round(total_return_pct, 2),
                round(total_brokerage, 2),
                buy_transactions,
                sell_transactions,
                round(realized_pnl, 2),
                round(unrealized_pnl, 2),
                round((cagr or 0.0) * 100.0, 2)
            ]
        })

        # Portfolio allocation summary
        allocation_summary = []
        for scrip in scrips:
            allocation_amount = self.stock_allocations.get(scrip, 0.0)
            allocation_pct = (allocation_amount / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            score = self.stock_scores.get(scrip, {}).get('composite_score', 0.0)
            current_price = current_market_prices.get(scrip, 0.0)
            position = final_position_map.get(scrip, 0)
            market_value = position * current_price
            cash = final_cash_map.get(scrip, 0.0)
            position_value = market_value + cash
            position_return = ((position_value - allocation_amount) / allocation_amount * 100.0) if allocation_amount else 0.0
            
            allocation_summary.append({
                'Security': scrip,
                'Initial_Allocation': round(allocation_amount, 2),
                'Allocation_%': round(allocation_pct, 2),
                'Investment_Score': round(score, 1),
                'Current_Price': round(current_price, 2),
                'Position_Qty': position,
                'Market_Value': round(market_value, 2),
                'Cash_Balance': round(cash, 2),
                'Total_Value': round(position_value, 2),
                'Return_%': round(position_return, 2)
            })
        
        allocation_df = pd.DataFrame(allocation_summary)

        # Export to Excel with professional formatting
        import support_files.File_IO as fio
        fio.change_cwd('gain_details')

        print(f"\n" + "="*60)
        print("EXPORTING PORTFOLIO ANALYSIS".center(60))
        print("="*60)

        # Create comprehensive Excel report
        with pd.ExcelWriter("portfolio_performance_report.xlsx", engine="openpyxl") as writer:
            # Performance Summary Sheet
            title_df = pd.DataFrame({"PORTFOLIO PERFORMANCE REPORT": [""]})
            title_df.to_excel(writer, sheet_name="Performance_Summary", startrow=0, index=False, header=False)
            
            summary_start = 2
            global_summary_df.to_excel(writer, sheet_name="Performance_Summary", startrow=summary_start, index=False)
            
            # Holdings Analysis Sheet
            allocation_df.to_excel(writer, sheet_name="Holdings_Analysis", index=False)
            
            # Transaction History Sheet
            if not backtested_transactions_df.empty:
                tx_export = backtested_transactions_df.copy()
                
                # Handle duplicate Date columns before datetime conversion
                if 'Date' in tx_export.columns:
                    date_columns = [col for col in tx_export.columns if col == 'Date']
                    if len(date_columns) > 1:
                        # Get all Date columns and combine them (take first non-null value)
                        date_data = tx_export[date_columns].bfill(axis=1).iloc[:, 0]
                        # Drop all Date columns
                        tx_export = tx_export.loc[:, ~(tx_export.columns == 'Date')]
                        # Add single Date column back
                        tx_export['Date'] = date_data
                
                # Now safely convert to datetime
                try:
                    tx_export['Date'] = pd.to_datetime(tx_export['Date'], errors='coerce').dt.strftime('%d-%b-%Y')
                except Exception as e:
                    print(f"Warning: Date formatting error in transactions export: {e}")
                    # Fallback: keep original date format
                    pass
                
                tx_export.to_excel(writer, sheet_name="Transaction_History", index=False)
            
            # Summary only for quick reference
            global_summary_df.to_excel(writer, sheet_name="Quick_Summary", index=False)

        print("✅ Portfolio Performance Report exported to 'portfolio_performance_report.xlsx'")
        
        if not backtested_transactions_df.empty:
            backtested_transactions_df.to_excel('detailed_transactions.xlsx', index=False)
            print("✅ Detailed transactions exported to 'detailed_transactions.xlsx'")

        print(f"\n" + "="*80)
        print("FINAL PORTFOLIO RESULTS".center(80))
        print("="*80)
        print(f"🎯 Strategy: {config.ACTIVE_FILTER}")
        print(f"💰 Initial Investment: ₹{initial_capital:,.2f}")
        print(f"💎 Current Portfolio Value: ₹{total_portfolio_value:,.2f}")
        print(f"📈 Total Return: ₹{total_return_amount:+,.2f} ({total_return_pct:+.2f}%)")
        print(f"💵 Realized Gains: ₹{realized_pnl:,.2f}")
        print(f"💹 Unrealized Gains: ₹{unrealized_pnl:,.2f}")
        if cagr is not None:
            print(f"📊 CAGR: {cagr*100:.2f}%")
        print(f"🏢 Active Positions: {sum(1 for pos in final_position_map.values() if pos > 0)}/{num_positions}")
        print(f"💸 Total Brokerage: ₹{total_brokerage:,.2f}")
        print("="*80)

        fio.get_cwd()
        return backtested_transactions_df, global_summary_df

    # --------------------- RUN METHOD ---------------------
    def run(self, master_df):
        """
        Complete portfolio management workflow:
        1. Screen and filter securities from universe
        2. Allocate capital using risk-weighted approach  
        3. Execute backtesting with realistic trading constraints
        4. Generate comprehensive performance analysis
        """
        print("🚀 STARTING PORTFOLIO MANAGEMENT SYSTEM")
        print(f"💰 Total Investment Capital: ₹{self.initial_cash:,.2f}")
        print(f"📊 Strategy: {config.ACTIVE_FILTER}")
        print(f"⏱️  Min Holding Period: {MIN_HOLDING_PERIOD} days")
        print(f"🎯 Min Profit Target: {MIN_PROFIT_PERCENTAGE}%")

        # Execute the complete workflow
        filtered_scrips_df = self.apply_filter(master_df)
        backtested_scrips_df, backtested_transactions_df = self.backtest_strategy(filtered_scrips_df)
        self.backtested_global_summary(backtested_scrips_df, backtested_transactions_df, master_df)

        return backtested_scrips_df, backtested_transactions_df


# --------------------- MAIN EXECUTION ---------------------
if __name__ == '__main__':
    import sys
    from support_files.dual_logger import DualLogger
    import support_files.File_IO as fio

    print("=" * 80)
    print("ALGORITHMIC TRADING PORTFOLIO MANAGEMENT SYSTEM".center(80))
    print("=" * 80)
    
    # Load market data
    master_df = fio.read_csv_to_df('Nif50_5y_1w.csv', 'A', 'sub_dir')
    
    # Initialize portfolio management system
    portfolio_manager = FilteringAndBacktesting(initial_cash=100000.0)
    
    # Setup logging
    fio.change_cwd('filtered_data')
    sys.stdout = DualLogger("portfolio_trading_log.txt")
    
    print(f"📁 Data Source: Nif50_5y_1w.csv")
    print(f"💼 Portfolio Manager Initialized")
    print(f"📝 Logging to: portfolio_trading_log.txt")
    
    # Execute portfolio strategy
    portfolio_manager.run(master_df)
    
    # Finalize
    fio.get_cwd()
    sys.stdout.flush()
    
    print("\n🎉 PORTFOLIO ANALYSIS COMPLETED SUCCESSFULLY!")
    print("📊 Check 'gain_details' folder for detailed reports")
    print("📝 Review 'portfolio_trading_log.txt' for complete trading history")