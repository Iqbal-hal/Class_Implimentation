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
        self.folio_final_value = 0.0
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
        Prints detailed calculation steps for allocation.
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
        print("\nWeight calculation steps (raw):")
        print(f"{'Stock':<20} {'Score':>6} {'Decay':>8} {'BaseW':>8} {'RawW':>10}")
        for i, (scrip, score) in enumerate(sorted_stocks):
            decay_factor = 0.8 ** i
            base_weight = score / 100.0
            raw_w = weights[scrip]
            print(f"{scrip:<20} {score:6.1f} {decay_factor:8.4f} {base_weight:8.4f} {raw_w:10.6f}")
        print(f"Total raw weight sum = {total_weight:.6f}")

        # Normalize weights to sum to 1
        min_allocation = 0.05
        max_allocation = 0.35

        normalized_weights = {}
        print("\nNormalized weights before constraints:")
        for scrip, weight in weights.items():
            normalized = weight / total_weight if total_weight else 0.0
            normalized_weights[scrip] = normalized
            print(f"{scrip:<20} normalized = {normalized:.6f}")

        # Apply min/max constraints
        constrained_weights = {}
        print("\nApplying min/max constraints (min 5%, max 35%):")
        for scrip, normalized in normalized_weights.items():
            constrained = max(min_allocation, min(normalized, max_allocation))
            constrained_weights[scrip] = constrained
            if constrained != normalized:
                print(f"{scrip:<20} was {normalized:.6f} -> constrained to {constrained:.6f}")
            else:
                print(f"{scrip:<20} remains {constrained:.6f}")

        total_constrained = sum(constrained_weights.values()) or 1.0
        print(f"Total after constraints = {total_constrained:.6f} (will renormalize to sum=1)")

        # Renormalize constrained weights to sum to 1
        for scrip in constrained_weights:
            normalized_weights[scrip] = constrained_weights[scrip] / total_constrained

        # Calculate final allocations and print detailed steps
        print(f"\n{'Stock':<20} {'Score':<8} {'Allocation %':<12} {'Amount (Rs)':<15} {'Rank':<6}")
        print("-" * 75)
        
        for i, (scrip, score) in enumerate(sorted_stocks):
            allocation_pct = normalized_weights[scrip] * 100.0
            allocation_amount = self.initial_cash * normalized_weights[scrip]
            self.stock_allocations[scrip] = allocation_amount
            
            print(f"{scrip:<20} {score:<8.1f} {allocation_pct:<12.2f} {allocation_amount:<15,.2f} #{i+1:<6}")
        
        print("-" * 75)
        total_alloc = sum(self.stock_allocations.values())
        print(f"{'TOTAL':<20} {'':<8} {'100.0':<12} {total_alloc:<15,.2f}")
        
        # Print detailed scoring breakdown for top 3 stocks (only once)
        if not self._detailed_print_shown:
            print(f"\n===== TOP 3 STOCKS DETAILED ANALYSIS =====")
            for i, (scrip, score) in enumerate(sorted_stocks[:3]):
                details = self.stock_scores[scrip]
                print(f"\n#{i+1} {scrip} (Score: {score:.1f})")
                print(f"  Technical Score: {details['technical_score']:.1f}")
                print(f"  Signal Quality:  {details['signal_score']:.1f} (Buy: {details['buy_signals']}, Sell: {details['sell_signals']})")
                print(f"  Momentum:        {details['momentum_score']:.1f} (Price change: {details['price_momentum']:.2f}%)")
                print(f"  Risk Score:      {details['volatility_score']:.1f}")
                print(f"  Current Price:   Rs {details['current_price']:.2f}")
                print(f"  RSI:             {details['rsi']:.1f}")
            # mark detailed calculations as shown so later per-stock runs remain concise
            self._detailed_print_shown = True
        else:
            print("\nDetailed breakdown already shown once; per-stock logs will be concise.")
        
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
            print(f"Error concatenating filtered data: {e}")
            filtered_ta_df = pd.DataFrame()

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
        allocated_cash = self.stock_allocations.get(scrip,
                                                   self.initial_cash / len(self.stock_allocations) if self.stock_allocations else self.initial_cash)
        cash = float(allocated_cash)

        print(f"\n----- Backtesting {scrip} -----")
        # Print allocation info concisely if detailed already shown, otherwise verbose
        if not self._detailed_print_shown:
            print("Allocated Cash calculation steps:")
            if scrip in self.stock_allocations:
                print(f"  Allocation found: allocation_amount = {self.stock_allocations[scrip]:,.2f}")
            else:
                fallback = self.initial_cash / len(self.stock_allocations) if self.stock_allocations else self.initial_cash
                print(f"  No allocation found, fallback used: initial_cash / n_allocated = {fallback:,.2f}")
            print(f"  Allocated Cash used = Rs {allocated_cash:,.2f}")
        else:
            print(f"Allocated Cash: Rs {allocated_cash:,.2f}  (concise)")

        position = 0
        portfolio_values = []
        positions = []
        buy_date = None
        buy_price = None
        trade_position = 'NO TRADE'
        transactions = []

        if 'P/E' in df_bt.columns and 'EPS' in df_bt.columns:
            print(f"Current P/E: {df_bt['P/E'].iloc[-1]:.2f} | Current EPS: {df_bt['EPS'].iloc[-1]:.2f}")

        if scrip in self.stock_scores:
            score_info = self.stock_scores[scrip]
            print(f"Stock Score breakdown:")
            print(f"  composite_score = {score_info['composite_score']:.2f}")
            print(f"  technical_score = {score_info['technical_score']:.2f}")
            print(f"  signal_score = {score_info['signal_score']:.2f}")
            print(f"  momentum_score = {score_info['momentum_score']:.2f}")
            print(f"  volatility_score = {score_info['volatility_score']:.2f}")
            allocation_pct = (allocated_cash / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            print(f"Allocation percent of total portfolio: {allocation_pct:.2f}%\n")

        for idx, row in df_bt.iterrows():
            price = float(row['Close'])
            stock_name = row['Stock']
            trade_position = 'NO TRADE'

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

                        if not self._detailed_print_shown:
                            print(f"Buy signal on {idx:%d-%m-%Y} for {stock_name}: Bought {position} shares at Rs {price:.2f}")
                            print(f"  Calculation steps:")
                            print(f"    available_cash_before = {cash + total_cost:.2f}")
                            print(f"    shares_to_buy = floor(available_cash_before / price) = {shares_to_buy}")
                            print(f"    cost = shares_to_buy * price = {cost:.2f}")
                            print(f"    fee = {fee:.2f}")
                            print(f"    cash_after_buy = {cash:.2f}")
                        else:
                            print(f"Buy {stock_name} on {idx:%d-%m-%Y}: {position} shares @ Rs {price:.2f}")

                else:
                    print(f"Not enough cash to buy on {idx:%d-%m-%Y} for {stock_name} (shares_to_buy=0).")

            elif sell_signal.loc[idx] and position > 0:
                if buy_date is not None and buy_price is not None:
                    holding_period = (idx - buy_date).days
                    profit_percentage = ((price - buy_price) / buy_price) * 100.0
                else:
                    holding_period = 0
                    profit_percentage = 0.0

                if holding_period >= MIN_HOLDING_PERIOD and profit_percentage >= MIN_PROFIT_PERCENTAGE:
                    revenue = position * price
                    fee = self.calculate_fee(revenue)
                    net_revenue = revenue - fee
                    cash += net_revenue
                    trade_position = 'SELL'

                    if not self._detailed_print_shown:
                        print(f"Sell signal on {idx:%d-%m-%Y} for {stock_name}: Sold {position} shares at Rs {price:.2f}")
                        print(f"  Calculation steps:")
                        print(f"    revenue = position * price = {position} * {price:.2f} = {revenue:.2f}")
                        print(f"    fee = {fee:.2f}")
                        print(f"    net_revenue_added = {net_revenue:.2f}")
                        print(f"    cash_after_sell = {cash:.2f}")
                    else:
                        print(f"Sell {stock_name} on {idx:%d-%m-%Y}: {position} shares @ Rs {price:.2f}  net +{net_revenue:.2f}")

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
                    print(f"Sell conditions not met on {idx:%d-%m-%Y} for {stock_name}. Holding period {holding_period}, profit% {profit_percentage:.2f}")

            current_value = cash + position * price
            portfolio_values.append(round(current_value, 2))
            positions.append(position)

            df_bt.loc[idx, 'current_value'] = round(current_value, 2)
            df_bt.loc[idx, 'Position'] = round(position, 2)
            df_bt.loc[idx, 'balance_cash'] = round(cash, 2)
            df_bt.loc[idx, 'trade_position'] = trade_position

        df_bt['Portfolio_Value'] = portfolio_values
        df_bt['Position'] = positions

        last_idx = df_bt.index[-1]
        last_cash = float(df_bt.loc[last_idx, 'balance_cash'])
        last_position = int(df_bt.loc[last_idx, 'Position'])
        last_price = float(df_bt.loc[last_idx, 'Close'])
        scrip_final_value = round(last_cash + last_position * last_price, 2)
        scrip_total_return = (scrip_final_value - allocated_cash) / allocated_cash * 100.0 if allocated_cash else 0.0

        print(f"\nLatest stock price used (from this scrip data): Rs {last_price:.2f}")
        print(f"Scrip portfolio value calculation steps:")
        print(f"  last_cash = {last_cash:.2f}")
        print(f"  last_position = {last_position}")
        print(f"  last_price = {last_price:.2f}")
        print(f"  scrip_final_value = last_cash + last_position * last_price = {last_cash:.2f} + {last_position} * {last_price:.2f} = {scrip_final_value:.2f}")
        print(f"  Scrip return = (scrip_final_value - allocated_cash) / allocated_cash * 100 = ({scrip_final_value:.2f} - {allocated_cash:.2f}) / {allocated_cash:.2f} * 100 = {scrip_total_return:.2f}%")

        df_bt['Final_Value'] = np.nan
        df_bt['Total_Return'] = np.nan
        df_bt['Allocated_Cash'] = allocated_cash
        df_bt.at[df_bt.index[-1], 'Final_Value'] = scrip_final_value
        df_bt.at[df_bt.index[-1], 'Total_Return'] = round(scrip_total_return, 2)

        trade_return = [None] * len(df_bt)
        for i, idx_val in enumerate(df_bt.index):
            if sell_signal.loc[idx_val]:
                pv = df_bt.loc[idx_val, 'Portfolio_Value']
                trade_return[i] = round((pv - allocated_cash) / allocated_cash * 100.0, 2)
        df_bt['Trade_Return'] = trade_return

        prev_folio = self.folio_final_value
        print(f"\nAccumulated portfolio value before adding this scrip = {prev_folio:.2f}")
        self.folio_final_value += scrip_final_value
        print(f"Added scrip_final_value ({scrip_final_value:.2f}) to accumulated portfolio.")
        print(f"New Accumulated portfolio value = {self.folio_final_value:.2f}")
        folio_cumulative_return = (self.folio_final_value - self.initial_cash) * 100.0 / self.initial_cash if self.initial_cash else 0.0
        print(f"Portfolio cumulative return calculation:")
        print(f"  = (Accumulated - Initial) / Initial * 100 = ({self.folio_final_value:.2f} - {self.initial_cash:.2f}) / {self.initial_cash:.2f} * 100 = {folio_cumulative_return:.2f}%\n")

        print(f"Accumulated portfolio value = {self.folio_final_value:.2f}")
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
            # Ensure single Date column and format
            if 'Date' in backtested_transactions_df.columns:
                # if duplicates exist, keep first non-null
                date_cols = [c for c in backtested_transactions_df.columns if c == 'Date']
                if len(date_cols) > 1:
                    date_df = backtested_transactions_df.iloc[:, [i for i, c in enumerate(backtested_transactions_df.columns) if c == 'Date']]
                    combined = date_df.bfill(axis=1).iloc[:, 0]
                    backtested_transactions_df = backtested_transactions_df.drop(columns=[c for c in backtested_transactions_df.columns if c == 'Date'])
                    backtested_transactions_df['Date'] = combined.values
            backtested_transactions_df['Date'] = pd.to_datetime(backtested_transactions_df['Date'], dayfirst=True)
            backtested_transactions_df['Date'] = backtested_transactions_df['Date'].dt.strftime('%d-%m-%Y')
            backtested_transactions_df.rename(columns={'index': 'Date'}, inplace=True)
            backtested_transactions_df.to_excel("backtested_transactions.xlsx", sheet_name=f"{config.ACTIVE_FILTER}", index=False)

        return backtested_scrip_df, backtested_transactions_df

    def backtested_global_summary(self, backtested_scrips_df, backtested_transactions_df, master_df=None):
        """
        Aggregates global summary from backtested scrips and writes a summary Excel file.
        Prints detailed calculation steps and computes:
          - Realized profit (from closed trades)
          - Final portfolio value (cash + unrealized using latest prices from master_df if provided)
          - CAGR
        """
        backtested_scrips_df = backtested_scrips_df.copy() if not backtested_scrips_df.empty else pd.DataFrame()
        backtested_transactions_df = backtested_transactions_df.copy() if not backtested_transactions_df.empty else pd.DataFrame()

        scrips = backtested_scrips_df['Stock'].unique() if not backtested_scrips_df.empty else []
        num_scrips = len(scrips)
        global_initial_cash = self.initial_cash

        # Build map of last cash and last position per scrip
        last_cash_map = {}
        last_pos_map = {}
        for scrip, df_s in scrip_extractor(backtested_scrips_df):
            if df_s.empty:
                last_cash_map[scrip] = 0.0
                last_pos_map[scrip] = 0
                continue
            try:
                last_row = df_s.loc[df_s.index.max()]
            except Exception:
                last_row = df_s.iloc[-1] if not df_s.empty else None
            if last_row is None:
                last_cash_map[scrip] = 0.0
                last_pos_map[scrip] = 0
            else:
                if hasattr(last_row, "get"):
                    last_cash_map[scrip] = float(last_row.get('balance_cash', 0.0) or 0.0)
                    last_pos_map[scrip] = int(last_row.get('Position', 0) or 0)
                else:
                    last_cash_map[scrip] = float(last_row['balance_cash']) if 'balance_cash' in last_row.index else 0.0
                    last_pos_map[scrip] = int(last_row['Position']) if 'Position' in last_row.index else 0

        # latest prices from master_df if provided
        latest_price_map = {}
        if master_df is not None:
            try:
                master_last = master_df.sort_values(by=['Stock', 'Date']).groupby('Stock').last()
                for scrip in scrips:
                    latest_price_map[scrip] = float(master_last.loc[scrip]['Close']) if scrip in master_last.index else None
            except Exception:
                latest_price_map = {}
        for scrip in scrips:
            if scrip not in latest_price_map or latest_price_map.get(scrip) is None:
                try:
                    df_s = backtested_scrips_df[backtested_scrips_df['Stock'] == scrip]
                    latest_price_map[scrip] = float(df_s['Close'].iloc[-1]) if not df_s.empty else 0.0
                except Exception:
                    latest_price_map[scrip] = 0.0

        cash_total = sum(last_cash_map.values())
        unrealized_value = 0.0
        print("\n===== FINAL PORTFOLIO VALUE (UNREALIZED) CALCULATION =====")
        for scrip, shares in last_pos_map.items():
            last_price = latest_price_map.get(scrip, 0.0) or 0.0
            mv = shares * last_price
            unrealized_value += mv
            print(f"{scrip:<20} shares={shares} * last_price={last_price:.2f} -> {mv:.2f}")

        print(f"Sum of cash across scrips (cash_total) = {cash_total:.2f}")
        print(f"Unrealized (market) value of open positions = {unrealized_value:.2f}")

        total_portfolio_value = cash_total + unrealized_value
        print(f"Total current portfolio value = cash_total + unrealized_value = {cash_total:.2f} + {unrealized_value:.2f} = {total_portfolio_value:.2f}")

        # Compute realized profit using transaction history (FIFO)
        realized_profit = 0.0
        print("\n===== REALIZED PROFIT CALCULATION (FIFO) =====")
        if not backtested_transactions_df.empty:
            tx_df = backtested_transactions_df.copy()
            # merge duplicate Date cols if any
            if 'Date' in tx_df.columns:
                date_col_indices = [i for i, c in enumerate(tx_df.columns) if c == 'Date']
                if len(date_col_indices) > 1:
                    date_df = tx_df.iloc[:, date_col_indices]
                    combined_dates = date_df.bfill(axis=1).iloc[:, 0]
                    tx_df = tx_df.drop(columns=[c for c in tx_df.columns if c == 'Date'])
                    tx_df['Date'] = combined_dates.values
                else:
                    if not isinstance(tx_df['Date'], pd.Series):
                        tx_df['Date'] = tx_df['Date'].astype(str)
                tx_df['Date'] = pd.to_datetime(tx_df['Date'], dayfirst=True, errors='coerce')
            tx_df = tx_df.sort_values(by=['Stock', 'Date'])
            for stock, group in tx_df.groupby('Stock'):
                buy_lots = []
                for _, r in group.iterrows():
                    evt = str(r.get('Event', '')).upper()
                    shares = int(r.get('Shares', 0) or 0)
                    if evt == 'BUY' and shares > 0:
                        buy_lots.append({
                            'shares': shares,
                            'cost_total': float(r.get('Cost', 0.0) or 0.0),
                            'fee_total': float(r.get('Fee', 0.0) or 0.0)
                        })
                        print(f"BUY  {stock}: shares={shares} cost_total={r.get('Cost',0.0):.2f} fee={r.get('Fee',0.0):.2f}")
                    elif evt == 'SELL' and shares > 0:
                        sell_revenue_total = float(r.get('Revenue', 0.0) or 0.0)
                        sell_fee_total = float(r.get('Fee', 0.0) or 0.0)
                        shares_to_sell = shares
                        print(f"SELL {stock}: shares={shares} revenue_total={sell_revenue_total:.2f} fee={sell_fee_total:.2f}")
                        while shares_to_sell > 0 and buy_lots:
                            lot = buy_lots[0]
                            lot_shares = lot['shares']
                            matched = min(lot_shares, shares_to_sell)
                            if lot_shares:
                                buy_cost_per_share = lot['cost_total'] / lot_shares
                                buy_fee_per_share = lot['fee_total'] / lot_shares
                            else:
                                buy_cost_per_share = buy_fee_per_share = 0.0
                            sell_revenue_part = sell_revenue_total * (matched / shares) if shares else 0.0
                            sell_fee_part = sell_fee_total * (matched / shares) if shares else 0.0
                            buy_cost_part = buy_cost_per_share * matched
                            buy_fee_part = buy_fee_per_share * matched
                            lot_profit = sell_revenue_part - sell_fee_part - buy_cost_part - buy_fee_part
                            realized_profit += lot_profit
                            print(f"  matched {matched} shares -> sell_rev_part={sell_revenue_part:.2f}, sell_fee_part={sell_fee_part:.2f}, buy_cost_part={buy_cost_part:.2f}, buy_fee_part={buy_fee_part:.2f}, profit={lot_profit:.2f}")
                            lot['shares'] -= matched
                            lot['cost_total'] -= buy_cost_part
                            lot['fee_total'] -= buy_fee_part
                            shares_to_sell -= matched
                            if lot['shares'] == 0:
                                buy_lots.pop(0)
                        if shares_to_sell > 0:
                            remaining_revenue = sell_revenue_total * (shares_to_sell / shares) if shares else 0.0
                            remaining_fee = sell_fee_total * (shares_to_sell / shares) if shares else 0.0
                            extra_profit = remaining_revenue - remaining_fee
                            realized_profit += extra_profit
                            print(f"  Warning: unmatched sell of {shares_to_sell} shares -> extra_profit={extra_profit:.2f}")
        else:
            print("No transactions available to compute realized profit.")
        print(f"Total realized profit = {realized_profit:.2f}")

        # Compute CAGR
        print("\n===== CAGR CALCULATION =====")
        first_trade_date = None
        if not backtested_transactions_df.empty and 'Date' in backtested_transactions_df.columns:
            try:
                first_trade_date = pd.to_datetime(backtested_transactions_df['Date'], dayfirst=True, errors='coerce').min()
            except Exception:
                first_trade_date = None
        if first_trade_date is None and master_df is not None and 'Date' in master_df.columns:
            try:
                first_trade_date = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce').min()
            except Exception:
                first_trade_date = None
        last_date = None
        if master_df is not None and 'Date' in master_df.columns:
            try:
                last_date = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce').max()
            except Exception:
                last_date = None
        if first_trade_date is None or last_date is None or pd.isna(first_trade_date) or pd.isna(last_date):
            cagr = None
            print("Insufficient date information to compute CAGR.")
        else:
            days = (last_date - first_trade_date).days
            years = max(days / 365.25, 1.0 / 365.25)
            starting_value = global_initial_cash
            ending_value = total_portfolio_value
            try:
                cagr = (ending_value / starting_value) ** (1.0 / years) - 1.0
                print(f"Start date = {first_trade_date.date()}  End date = {last_date.date()}  Years = {years:.4f}")
                print(f"Starting value = {starting_value:.2f}  Ending value = {ending_value:.2f}")
                print(f"CAGR = (Ending/Starting)^(1/years) - 1 = {cagr*100:.2f}%")
            except Exception as e:
                cagr = None
                print(f"Error computing CAGR: {e}")

        global_final_value = total_portfolio_value
        global_profit = global_final_value - global_initial_cash
        global_profit_percentage = (global_profit / global_initial_cash) * 100.0 if global_initial_cash else 0.0

        if not backtested_transactions_df.empty:
            # safe counts
            num_buy_global = backtested_transactions_df[backtested_transactions_df.get('Event') == 'BUY'].shape[0]
            num_sell_global = backtested_transactions_df[backtested_transactions_df.get('Event') == 'SELL'].shape[0]

            # total capital used for BUYs: prefer 'Cost' column, fallback to Price * Shares
            total_capital_used_global = 0.0
            if 'Cost' in backtested_transactions_df.columns:
                total_capital_used_global = pd.to_numeric(
                    backtested_transactions_df.loc[backtested_transactions_df['Event'] == 'BUY', 'Cost'],
                    errors='coerce'
                ).fillna(0.0).sum()
            else:
                # fallback compute using Price * Shares if available
                if ('Price' in backtested_transactions_df.columns) and ('Shares' in backtested_transactions_df.columns):
                    price_series = pd.to_numeric(
                        backtested_transactions_df.loc[backtested_transactions_df['Event'] == 'BUY', 'Price'],
                        errors='coerce'
                    ).fillna(0.0)
                    shares_series = pd.to_numeric(
                        backtested_transactions_df.loc[backtested_transactions_df['Event'] == 'BUY', 'Shares'],
                        errors='coerce'
                    ).fillna(0.0)
                    total_capital_used_global = (price_series * shares_series).sum()
                else:
                    total_capital_used_global = 0.0

            # total broker charges: prefer 'Fee' column, try common alternatives
            total_broker_charges_global = 0.0
            if 'Fee' in backtested_transactions_df.columns:
                total_broker_charges_global = pd.to_numeric(backtested_transactions_df['Fee'], errors='coerce').fillna(0.0).sum()
            else:
                for alt in ('Broker_Fee', 'Broker_Charges', 'Charges'):
                    if alt in backtested_transactions_df.columns:
                        total_broker_charges_global = pd.to_numeric(backtested_transactions_df[alt], errors='coerce').fillna(0.0).sum()
                        break
        else:
             num_buy_global = num_sell_global = total_capital_used_global = total_broker_charges_global = 0

        global_summary_df = pd.DataFrame({
            "Metric": [
                "FILTER_NAME", "MIN_HOLDING_PERIOD", "MIN_PROFIT_PERCENTAGE",
                "NUMBER_OF_SCRIPS", "PORTFOLIO_APPROACH", "TOTAL_INITIAL_CASH",
                "Total Final Value", "Total Capital Used", "Profit", "Profit Percentage",
                "Total Broker Charges", "Number of BUY", "Number of SELL",
                "Realized Profit", "Unrealized (market) Value", "CAGR (%)"
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
                num_sell_global,
                round(realized_profit, 2),
                round(unrealized_value, 2),
                round((cagr or 0.0) * 100.0, 2)
            ]
        })

        allocation_summary = []
        for scrip in scrips:
            allocation_amount = self.stock_allocations.get(scrip, 0.0)
            allocation_pct = (allocation_amount / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            score = self.stock_scores.get(scrip, {}).get('composite_score', 0.0)
            allocation_summary.append({
                'Stock': scrip,
                'Allocation_Amount': round(allocation_amount, 2),
                'Allocation_Percentage': round(allocation_pct, 2),
                'Score': round(score, 1)
            })
        allocation_df = pd.DataFrame(allocation_summary)

        import support_files.File_IO as fio
        fio.change_cwd('gain_details')

        title_row = pd.DataFrame({"Metric": ["GLOBAL PORTFOLIO SUMMARY"], "Value": [""]})
        with pd.ExcelWriter("global_portfolio_summary.xlsx", engine="openpyxl") as writer:
            title_row.to_excel(writer, sheet_name="Portfolio_Summary", startrow=0, startcol=0, index=False, header=False)
            summary_startrow = len(title_row)
            global_summary_df.to_excel(writer, sheet_name="Portfolio_Summary", startrow=summary_startrow, startcol=0, index=False, header=False)
            allocation_startrow = summary_startrow + len(global_summary_df) + 2
            allocation_title = pd.DataFrame({"Metric": ["STOCK ALLOCATIONS"], "Value": [""]})
            allocation_title.to_excel(writer, sheet_name="Portfolio_Summary", startrow=allocation_startrow, startcol=0, index=False, header=False)
            allocation_data_start = allocation_startrow + 1
            allocation_df.to_excel(writer, sheet_name="Portfolio_Summary", startrow=allocation_data_start, startcol=0, index=False, header=True)
            if not backtested_transactions_df.empty:
                transactions_startrow = allocation_data_start + len(allocation_df) + 3
                backtested_transactions_df.to_excel(writer, sheet_name="Portfolio_Summary", startrow=transactions_startrow, startcol=0, index=False, header=True)
            global_summary_df.to_excel(writer, sheet_name="Summary_Only", startrow=0, startcol=0, index=False, header=True)
            allocation_df.to_excel(writer, sheet_name="Allocations", startrow=0, startcol=0, index=False, header=True)

        print("\nEnhanced portfolio summary exported to 'gain_details/global_portfolio_summary.xlsx' (cwd changed to gain_details).")
        if not backtested_transactions_df.empty:
            backtested_transactions_df.to_excel('backtested_transactions_df.xlsx', index=False)

        print(f"\n===== FINAL PORTFOLIO PERFORMANCE =====")
        print(f"Initial Investment: Rs {global_initial_cash:,.2f}")
        print(f"Cash (realized) available (sum of last cash across scrips) = Rs {cash_total:,.2f}")
        print(f"Unrealized (market) value of open positions (using latest prices) = Rs {unrealized_value:,.2f}")
        print(f"Final Total Portfolio Value (cash + unrealized) = Rs {global_final_value:,.2f}")
        print(f"Realized Profit (from closed trades) = Rs {realized_profit:,.2f}")
        print(f"Total Profit (final - initial) = Rs {global_final_value - global_initial_cash:,.2f}")
        if cagr is not None:
            print(f"CAGR = {cagr*100.0:.2f}%")
        else:
            print("CAGR: N/A (insufficient date information)")
        print(f"Number of Transactions: {num_buy_global} buys, {num_sell_global} sells")
        print(f"Total Broker Charges: Rs {total_broker_charges_global:.2f}")

        fio.get_cwd()
        return backtested_transactions_df, global_summary_df

    # --------------------- RUN METHOD ---------------------
    def run(self, master_df):
        """
        Overall flow:
        1. Filter the master OHLC dataframe using apply_filter (includes intelligent allocation).
        2. Backtest each filtered scrip using allocated amounts.
        3. Generate a comprehensive portfolio summary (uses master_df latest prices).
        Returns backtested scrips and transactions dataframes.
        """
        print("Starting enhanced portfolio run with intelligent allocation...")
        print(f"Total Investment Capital: Rs {self.initial_cash:,.2f}")

        filtered_scrips_df = self.apply_filter(master_df)
        bs_df, bt_df = self.backtest_strategy(filtered_scrips_df)
        self.backtested_global_summary(bs_df, bt_df, master_df)

        return bs_df, bt_df


# --------------------- MAIN EXECUTION ---------------------
if __name__ == '__main__':
    import sys
    from support_files.dual_logger import DualLogger
    import support_files.File_IO as fio

    master_df = fio.read_csv_to_df('Nif50_5y_1w.csv', 'A', 'sub_dir')

    fab = FilteringAndBacktesting(initial_cash=100000.0)
    fio.change_cwd('filtered_data')
    sys.stdout = DualLogger("log.txt")
    fab.run(master_df)
    fio.get_cwd()
    sys.stdout.flush()
