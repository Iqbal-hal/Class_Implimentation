# config_gui.py
import streamlit as st
import importlib
import re
from pathlib import Path

# Path to your config.py file
CONFIG_PATH = Path(__file__).parent / 'support_files' / 'config.py'

def load_config_module():
    # Dynamically load the config module
    import support_files.config as cfg
    importlib.reload(cfg)
    return cfg

@st.cache_data
def read_config_file():
    # Read and split lines of the config file
    return CONFIG_PATH.read_text().splitlines()

@st.cache_data
def write_config_file(lines):
    # Write updated lines back to the config file
    CONFIG_PATH.write_text("\n".join(lines))

def update_config_values(lines, updates):
    updated = []
    pattern = re.compile(r"^(?P<key>\w+)\s*=.*$")
    for line in lines:
        m = pattern.match(line)
        if m and m.group('key') in updates:
            key = m.group('key')
            val = updates[key]
            # support None, strings, numbers
            if val is None:
                updated.append(f"{key} = None")
            elif isinstance(val, str) and not val.startswith(("\"", "'")):
                updated.append(f"{key} = '{val}'")
            else:
                updated.append(f"{key} = {val}")
        else:
            updated.append(line)
    return updated

def main():
    st.title("Configuration GUI")
    cfg = load_config_module()
    lines = read_config_file()

    # ================= Logging & Filtering =================
    st.header("Core Settings")
    logging_enabled = st.checkbox(
        "Enable Logging (LOGGING_ENABLED)",
        value=cfg.LOGGING_ENABLED,
        help="Toggle detailed console logging during backtests."
    )
    filter_enabled = st.checkbox(
        "Enable Filtering (FILTER_ENABLED)",
        value=cfg.FILTER_ENABLED,
        help="Include only stocks passing filter criteria when enabled."
    )
    detailed_logging = st.checkbox(
        "Enable Detailed Logging (ENABLE_DETAILED_LOGGING)",
        value=getattr(cfg, 'ENABLE_DETAILED_LOGGING', False),
        help="Toggle transaction-level logging on or off."
    )
    filter_options = [None] + list(cfg.AVAILABLE_FILTERS.keys())
    default_idx = 0 if cfg.ACTIVE_FILTER not in cfg.AVAILABLE_FILTERS else filter_options.index(cfg.ACTIVE_FILTER)
    active_filter = st.selectbox(
        "Active Filter (ACTIVE_FILTER)",
        options=filter_options,
        index=default_idx,
        format_func=lambda x: "All Filters" if x is None else x,
        help="Select a specific filter to apply, or 'All Filters' to scan all."
    )

    # ================= Risk Management =================
    st.header("Risk Management")
    use_support = st.checkbox(
        "Use Support Trailing Exits (USE_SUPPORT_TRAILING_EXITS)",
        value=getattr(cfg, 'USE_SUPPORT_TRAILING_EXITS', False),
        help="Exit trades when price falls below support levels if enabled."
    )
    support_type = st.selectbox(
        "Support Type (SUPPORT_TYPE)",
        options=["min", "single", "pivot", "fractal"],
        index=["min", "single", "pivot", "fractal"].index(getattr(cfg, 'SUPPORT_TYPE', 'min')),
        help="Choose the method for computing support (min, single, pivot, fractal, etc.)."
    )
    min_holding = st.number_input(
        "Minimum Holding Period (MIN_HOLDING_PERIOD) [days]",
        value=getattr(cfg, 'MIN_HOLDING_PERIOD', 40),
        min_value=1,
        help="Minimum number of days to hold a position before selling."
    )
    min_profit = st.number_input(
        "Minimum Profit Percentage (MIN_PROFIT_PERCENTAGE) [%]",
        value=getattr(cfg, 'MIN_PROFIT_PERCENTAGE', 20.0),
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        help="Minimum percent gain required before selling a position."
    )
    min_filter_sell = st.number_input(  # New parameter
        "Minimum Filter Sell Profit (MIN_FILTER_SELL_PROFIT) [%]",
        value=getattr(cfg, 'MIN_FILTER_SELL_PROFIT', 5.0),
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        help="Minimum profit percentage required for filter-based sells."
    )
    trailing_stop = st.number_input(
        "Trailing Stop Percentage (TRAILING_STOP_PERCENT) [%]",
        value=getattr(cfg, 'TRAILING_STOP_PERCENT', 3.0),
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        help="Percentage drop from peak to trigger a trailing stop exit."
    )

    # ================= Scan Parameter Ranges =================
    st.header("Scan Parameter Ranges")
    st.write("Specify start, end, and step for each numerical parameter in a single row layout.")
    
    # Holding Period Range
    col1, col2, col3 = st.columns(3)
    with col1:
        holding_start = st.number_input(
            "Holding Start (days)",
            value=getattr(cfg, 'HOLDING_START', 120),
            min_value=1,
            help="Start (inclusive) of holding period range."
        )
    with col2:
        holding_end = st.number_input(
            "Holding End (days)",
            value=getattr(cfg, 'HOLDING_END', 360),
            min_value=holding_start,
            help="End (inclusive) of holding period range."
        )
    with col3:
        holding_step = st.number_input(
            "Holding Step (days)",
            value=getattr(cfg, 'HOLDING_STEP', 60),
            min_value=1,
            help="Step increment for holding period range."
        )
    
    # Profit Percentage Range
    col4, col5, col6 = st.columns(3)
    with col4:
        profit_start = st.number_input(
            "Profit Start (%)",
            value=getattr(cfg, 'PROFIT_START', 20.0),
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            help="Start (inclusive) of profit target percentage range."
        )
    with col5:
        profit_end = st.number_input(
            "Profit End (%)",
            value=getattr(cfg, 'PROFIT_END', 40.0),
            min_value=profit_start,
            max_value=100.0,
            step=0.1,
            help="End (inclusive) of profit target percentage range."
        )
    with col6:
        profit_step = st.number_input(
            "Profit Step (%)",
            value=getattr(cfg, 'PROFIT_STEP', 10.0),
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            help="Step increment for profit target percentage range."
        )
    
    # Trailing Stop Range
    col7, col8, col9 = st.columns(3)
    with col7:
        trail_start = st.number_input(
            "Trail Start (%)",
            value=getattr(cfg, 'TRAIL_START', 3.0),
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            help="Start (inclusive) of trailing stop percentage range."
        )
    with col8:
        trail_end = st.number_input(
            "Trail End (%)",
            value=getattr(cfg, 'TRAIL_END', 7.5),
            min_value=trail_start,
            max_value=100.0,
            step=0.1,
            help="End (inclusive) of trailing stop percentage range."
        )
    with col9:
        trail_step = st.number_input(
            "Trail Step (%)",
            value=getattr(cfg, 'TRAIL_STEP', 2.5),
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            help="Step increment for trailing stop percentage range."
        )

    if st.button("Save Configuration", help="Write these settings back to config.py"):
        updates = {
            'LOGGING_ENABLED': logging_enabled,
            'FILTER_ENABLED': filter_enabled,
            'ENABLE_DETAILED_LOGGING': detailed_logging,
            'ACTIVE_FILTER': active_filter,
            'USE_SUPPORT_TRAILING_EXITS': use_support,
            'SUPPORT_TYPE': support_type,
            'MIN_HOLDING_PERIOD': int(min_holding),
            'MIN_PROFIT_PERCENTAGE': float(min_profit),
            'MIN_FILTER_SELL_PROFIT': float(min_filter_sell),  # New entry
            'TRAILING_STOP_PERCENT': float(trailing_stop),
            'HOLDING_START': int(holding_start),
            'HOLDING_END': int(holding_end),
            'HOLDING_STEP': int(holding_step),
            'PROFIT_START': float(profit_start),
            'PROFIT_END': float(profit_end),
            'PROFIT_STEP': float(profit_step),
            'TRAIL_START': float(trail_start),
            'TRAIL_END': float(trail_end),
            'TRAIL_STEP': float(trail_step)
        }
        new_lines = update_config_values(lines, updates)
        write_config_file(new_lines)
        st.success("Configuration saved! Restart the app to apply the new settings.")

if __name__ == '__main__':
    main()
