import support_files.config as config
import pandas as pd
import numpy as np
from support_files.scrip_extractor import scrip_extractor, scripdf_extractor
import support_files.compute_indicators_helper as cmp  # For computing technical indicators

# Trade constraint constants
MIN_PROFIT_PERCENTAGE = config.MIN_PROFIT_PERCENTAGE
MIN_HOLDING_PERIOD = config.MIN_HOLDING_PERIOD

class FilteringAndBacktesting:
    def __init__(self, initial_cash=10000.0):
        self.initial_cash = initial_cash
        self.backtested_scrip_df_list = []
        self.backtested_transactions_df_list = []
        self.folio_final_value = 0.0

    # --------------------- FILTERING METHODS ---------------------
    def apply_filter(self, master_df):
        """
        Computes technical indicators and applies the configured filter on the master OHLC dataframe.
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
        return filtered_ta_df

    # --------------------- BACKTESTING METHODS ---------------------
    def calculate_fee(self, trade_value):
        """Returns the broker fee: â‚¹20 or 2.5% of trade value (whichever is lower)."""
        fee_percent = 0.025 * trade_value
        fixed_fee = 20.0
        return min(fixed_fee, fee_percent)

    def apply_backtest_strategy(self, filtered_scrip_df, scrip, buy_signal, sell_signal):
        """
        Backtests a trading strategy on a filtered scrip dataframe.
        Returns a tuple: (backtested scrip dataframe, transactions dataframe)
        """
        df_bt = filtered_scrip_df.copy()
        cash = self.initial_cash
        position = 0
        portfolio_values = []
        positions = []
        buy_date = None
        buy_price = None
        trade_position = 'NO TRADE'
        transactions = []

        print(f"\n----- Backtesting {scrip} -----")
        print(f"Current P/E: {df_bt['P/E'].iloc[-1]:.2f} | Current EPS: {df_bt['EPS'].iloc[-1]:.2f}\n")
        cash_cumulative = self.initial_cash + config.counter * self.initial_cash
        print(f"cash_cumulative = {cash_cumulative:.2f}")
        config.counter += 1
        print(f"Counter updated to: {config.counter}")

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
                            'Position_After': position
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
                        'Profit_%': round(profit_percentage, 2)
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
        scrip_total_return = (scrip_final_value - self.initial_cash) / self.initial_cash * 100

        print(f"\nLatest stock price: Rs {df_bt['Close'].iloc[-1]:.2f}")
        print(f"Scrip portfolio value: Rs {scrip_final_value:.2f} | Scrip return: {scrip_total_return:.2f}%")

        df_bt['Final_Value'] = np.nan
        df_bt['Total_Return'] = np.nan
        df_bt.at[df_bt.index[-1], 'Final_Value'] = scrip_final_value
        df_bt.at[df_bt.index[-1], 'Total_Return'] = round(scrip_total_return, 2)

        trade_return = [None] * len(df_bt)
        for i, idx_val in enumerate(df_bt.index):
            if sell_signal.loc[idx_val]:
                pv = df_bt.loc[idx_val, 'Portfolio_Value']
                trade_return[i] = round((pv - self.initial_cash) / self.initial_cash * 100, 2)
        df_bt['Trade_Return'] = trade_return

        self.folio_final_value += scrip_final_value
        folio_cumulative_return = (self.folio_final_value - cash_cumulative) * 100 / cash_cumulative
        print(f"Accumulated folio_final_value = {self.folio_final_value:.2f}")
        print(f"folio_cumulative_return = {folio_cumulative_return:.2f}%\n")

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
        """
        initial_cash_per_stock = self.initial_cash
        scrips = backtested_scrips_df['Stock'].unique()
        backtested_df_list = list(scripdf_extractor(backtested_scrips_df))
        
        num_scrips = len(scrips)
        global_initial_cash = num_scrips * initial_cash_per_stock
        global_final_value = sum(df['Final_Value'].iloc[-1] for df in backtested_df_list)
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
        
        global_summary_df = pd.DataFrame({
            "Metric": ["FILTER_NAME", "MIN_HOLDING_PERIOD", "MIN_PROFIT_PERCENTAGE", "NUMBER_OF_SCRIPS",
                       "TOTAL_INITIAL_CASH", "Total Capital Used", "Profit", "Profit Percentage",
                       "Total Broker Charges", "Number of BUY", "Number of SELL"],
            "Value": [
                config.ACTIVE_FILTER,                
                MIN_HOLDING_PERIOD,
                MIN_PROFIT_PERCENTAGE,
                num_scrips,
                global_initial_cash,
                round(total_capital_used_global, 2),
                round(global_profit, 2),
                round(global_profit_percentage, 2),
                round(total_broker_charges_global, 2),
                num_buy_global,
                num_sell_global  
            ]
        })
       
        title_row = pd.DataFrame({"Metric": ["GLOBAL SUMMARY"], "Value": [""]})
        blank_row = pd.DataFrame({"Metric": [""], "Value": [""]})
        transaction_cols = [
            "Date", "Event", "Stock", "Price", "Shares", 
            "Cost", "Fee", "Cash_After", "Position_After", 
            "Revenue", "Holding_Period", "Profit_%"
        ]
        if not backtested_transactions_df.empty:
            for col in transaction_cols:
                if col not in backtested_transactions_df.columns:
                    backtested_transactions_df[col] = ""
            backtested_transactions_df = backtested_transactions_df[transaction_cols]
        
        import support_files.File_IO as fio
        # write outputs (excel/logs) into unified output folder
        fio.change_cwd('output_data')
        with pd.ExcelWriter("global_transactions_summary.xlsx", engine="openpyxl") as writer:
            title_row.to_excel(writer, sheet_name="Sheet1", startrow=0, startcol=0, index=False, header=False)
            summary_startrow = len(title_row)
            global_summary_df.to_excel(writer, sheet_name="Sheet1", startrow=summary_startrow, startcol=0, index=False, header=False)
            blank_startrow = summary_startrow + len(global_summary_df)
            blank_row.to_excel(writer, sheet_name="Sheet1", startrow=blank_startrow, startcol=0, index=False, header=False)
            if not backtested_transactions_df.empty:
                transactions_startrow = blank_startrow + len(blank_row)
                backtested_transactions_df.to_excel(writer, sheet_name="Sheet1", startrow=transactions_startrow, startcol=0, index=False, header=True)
            global_summary_df.to_excel(writer, sheet_name="Sheet2", startrow=0, startcol=0, index=False, header=True)
        print("Global transactions summary exported to 'global_transactions_summary.xlsx'.")
        backtested_transactions_df.to_excel('backtested_transactions_df.xlsx', index=False)
        return backtested_transactions_df, global_summary_df

    # --------------------- RUN METHOD ---------------------
    def run(self, master_df):
        """
        Overall flow:
         1. Filter the master OHLC dataframe using apply_filter.
         2. Backtest each filtered scrip.
         3. Generate a global summary.
         Returns backtested scrips and transactions dataframes.
        """
        print("Starting overall run: filtering master dataframe...")
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
    # read inputs from input_data folder
    master_df = fio.read_csv_to_df('Nif50_5y_1w.csv', 'A', 'input_data')
    
    # Create an instance of FilteringAndBacktesting and run the full process
    fab = FilteringAndBacktesting(initial_cash=10000.0)
    # ensure all generated outputs go into output_data
    fio.change_cwd('output_data')
    sys.stdout = DualLogger("portfolio_trading_log.txt")
    
    fab.run(master_df)
    
    fio.get_cwd()
    sys.stdout.flush()
