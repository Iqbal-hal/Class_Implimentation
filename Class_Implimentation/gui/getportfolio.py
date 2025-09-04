
import config as config
import json
import pandas as pd

# -----------------------------------------------------------------------------
# Function to load portfolio details from a JSON file.
def load_portfolio_from_json(file_path="portfolio.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Convert "Date Invested" from string to pd.Timestamp for each trade.
    for ticker, trades in data.items():
        for trade in trades:
            trade["Date Invested"] = pd.to_datetime(trade["Date Invested"], format="%Y-%m-%d")
    return data

# -----------------------------------------------------------------------------
