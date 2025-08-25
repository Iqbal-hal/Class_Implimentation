# TradeScanner.py
import pandas as pd
import support_files.config as con
from FilteringBacktesting import FilteringAndBacktesting  
import support_files.File_IO as fio  # For reading CSV and any file I/O

class TradeScanner:
    def __init__(self, master_df, initial_cash=10000.0):
        self.master_df = master_df
        self.initial_cash = initial_cash

    def run_param_scan(self):
        """
        Systematically vary FILTER_NAME, MIN_HOLDING_PERIOD, and MIN_PROFIT_PERCENTAGE,
        run the complete filtering and backtesting process, and consolidate the global summary results.
        """
        # filter_names = [
        #     "filter_basic",
        #     "filter_aggressive",
        #     "filter_momentum",
        #     "filter_breakout",
        #     "filter_mean_reversion",
        #     "filter_trend_following",
        #     "filter_volume_surge",
        #     "filter_vwap",
        #     "filter_golden_death_cross",
        #     "filter_divergence",
        #     "filter_adx",
        #     "filter_supertrend"
        # ]

        filter_names = [
            "filter_momentum",
            "filter_adx"            
        ]
        # For example, holding periods in days
        holding_periods = range(300, 361, 60)  # 120, 180, 240, 300, 360 days
        profit_percentages = range(40, 61, 20)   # 20, 30, 40%

        results = []

        for f_name in filter_names:
            # Dynamically update the configuration
            con.FILTER_NAME = f_name

            for hp in holding_periods:
                con.MIN_HOLDING_PERIOD = hp

                for pp in profit_percentages:
                    con.MIN_PROFIT_PERCENTAGE = pp

                    print(f"\nCurrently testing: FILTER_NAME={f_name}, "
                          f"MIN_HOLDING_PERIOD={hp}, MIN_PROFIT_PERCENTAGE={pp}")
                    
                    # Create a new instance for each run to avoid residual state
                    fab = FilteringAndBacktesting(initial_cash=self.initial_cash)
                    
                    # Run filtering and backtesting process
                    bs_df, bt_df = fab.run(self.master_df)
                    
                    # Option 1: Capture global summary by re-calling the method.
                    # (Since fab.run() already calls backtested_global_summary, you can also modify run() to return it.)
                    transactions_df, global_summary_df = fab.backtested_global_summary(bs_df, bt_df)

                    # Global summary is assumed to be a DataFrame with columns ["Metric", "Value"]
                    summary_indexed = global_summary_df.set_index("Metric")
                    total_cap_used = summary_indexed.loc["Total Capital Used", "Value"]
                    profit = summary_indexed.loc["Profit", "Value"]
                    profit_pct = summary_indexed.loc["Profit Percentage", "Value"]
                    broker_charges = summary_indexed.loc["Total Broker Charges", "Value"]
                    num_buy = summary_indexed.loc["Number of BUY", "Value"]
                    num_sell = summary_indexed.loc["Number of SELL", "Value"]

                    results.append([
                        f_name,
                        hp,
                        pp,
                        total_cap_used,
                        profit,
                        profit_pct,
                        broker_charges,
                        num_buy,
                        num_sell
                    ])

        # Build a DataFrame of all runs
        columns = [
            "FILTER_NAME", "MIN_HOLDING_PERIOD", "MIN_PROFIT_PERCENTAGE",
            "Total Capital Used", "Profit", "Profit Percentage",
            "Total Broker Charges", "Number of BUY", "Number of SELL"
        ]
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_excel("scan_results.xlsx", index=False)
        print("\nAll parameter combinations have been scanned!")
        print("Consolidated results saved to 'scan_results.xlsx'.")
        return results_df

if __name__ == "__main__":
    # Read the master dataframe (concatenated OHLC data)
    master_df = fio.read_csv_to_df('50NIF_2020FEB10_2025FEB10.csv', 'A','sub_dir')
    
    # Instantiate TradeScanner with the master dataframe and desired initial cash amount
    scanner = TradeScanner(master_df, initial_cash=10000.0)
    scanner.run_param_scan()
