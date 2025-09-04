# param_scan.py
import pandas as pd
import support_files.config as config
import backtester as bt  
from filterstock_processor_r3 import filter_stocks

def run_param_scan(master_df):
    """
    Systematically vary FILTER_NAME, MIN_HOLDING_PERIOD, and MIN_PROFIT_PERCENTAGE,
    run filter_stocks, and gather the global summary results into one final DataFrame.
    """
    filter_names = [
    "filter_basic",
    "filter_aggressive",
    "filter_momentum",
    "filter_breakout",
    "filter_mean_reversion",
    "filter_trend_following",
    "filter_volume_surge",
    "filter_vwap",
    "filter_golden_death_cross",
    "filter_divergence",
    "filter_adx",
    "filter_supertrend"
]
    holding_periods = range(120, 361, 60)        # 10, 20, 30, ... 360
    profit_percentages = range(20, 50, 10)     # 10, 20, 30, ... 100

    results = []

    for f_name in filter_names:
        # Dynamically set FILTER_NAME in config
        config.FILTER_NAME = f_name

        for hp in holding_periods:
            # Set the global MIN_HOLDING_PERIOD in your backtester
            bt.MIN_HOLDING_PERIOD = hp

            for pp in profit_percentages:
                # Set the global MIN_PROFIT_PERCENTAGE in your backtester
                bt.MIN_PROFIT_PERCENTAGE = pp

                # Run filter_stocks with the current combination
                # (Be sure filter_stocks returns global_summary_df now)
                print(f"Currently testing: FILTER_NAME={f_name}, "
                f"MIN_HOLDING_PERIOD={hp}, MIN_PROFIT_PERCENTAGE={pp}")

                global_summary_df = filter_stocks(master_df)[4]

                # global_summary_df has columns ["Metric", "Value"] with rows like:
                # FILTER_NAME, MIN_HOLDING_PERIOD, MIN_PROFIT_PERCENTAGE, ...
                # We convert it to an index for easy access:
                summary_indexed = global_summary_df.set_index("Metric")

                # Extract the desired values
                total_cap_used     = summary_indexed.loc["Total Capital Used", "Value"]
                profit             = summary_indexed.loc["Profit", "Value"]
                profit_pct         = summary_indexed.loc["Profit Percentage", "Value"]
                broker_charges     = summary_indexed.loc["Total Broker Charges", "Value"]
                num_buy            = summary_indexed.loc["Number of BUY", "Value"]
                num_sell           = summary_indexed.loc["Number of SELL", "Value"]

                # Append them to our results list
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

    # Export to Excel
    results_df.to_excel("scan_results.xlsx", index=False)
    print("All parameter combinations have been scanned!")
    print("Consolidated results saved to 'scan_results.xlsx'.")

if __name__ == "__main__":
    print("")
