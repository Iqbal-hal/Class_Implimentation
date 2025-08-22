#====================================================================
# annotation_utils.py
# ====================================================================
import config as config

def annotation(df, df_scrip_gain):
    # Extract values from the gain dataframe.
    buy_date   = df_scrip_gain['Buy Date']
    buy_price  = df_scrip_gain['Buy Price']
    buy_rsi    = df.loc[buy_date, 'RSI'] if buy_date in df.index else None
    buy_pe     = df.loc[buy_date, 'P/E'] if (buy_date in df.index and 'P/E' in df.columns) else None

    sell_date  = df_scrip_gain['Sell Date']
    sell_price = df_scrip_gain['Sell Price']
    sell_rsi   = df.loc[sell_date, 'RSI'] if sell_date in df.index else None
    sell_pe    = df.loc[sell_date, 'P/E'] if (sell_date in df.index and 'P/E' in df.columns) else None

    min_date   = df_scrip_gain['Min Date']
    min_price  = df_scrip_gain['Min Price']
    min_rsi    = df.loc[min_date, 'RSI'] if min_date in df.index else None
    min_pe     = df.loc[min_date, 'P/E'] if (min_date in df.index and 'P/E' in df.columns) else None

    max_date   = df_scrip_gain['Max Date']
    max_price  = df_scrip_gain['Max Price']
    max_rsi    = df.loc[max_date, 'RSI'] if max_date in df.index else None
    max_pe     = df.loc[max_date, 'P/E'] if (max_date in df.index and 'P/E' in df.columns) else None

    trade_gain      = df_scrip_gain['Trade Gain']
    max_gain        = df_scrip_gain['Max Gain']
    max_gain_signal = df_scrip_gain['Signal Max Gain']

    min_date_signal  = df_scrip_gain['Signal Min Date']
    min_price_signal = df_scrip_gain['Signal Min Price']
    min_rsi_signal   = df.loc[min_date_signal, 'RSI'] if min_date_signal in df.index else None
    min_pe_signal    = df.loc[min_date_signal, 'P/E'] if (min_date_signal in df.index and 'P/E' in df.columns) else None

    max_date_signal  = df_scrip_gain['Signal Max Date']
    max_price_signal = df_scrip_gain['Signal Max Price']
    max_rsi_signal   = df.loc[max_date_signal, 'RSI'] if max_date_signal in df.index else None
    max_pe_signal    = df.loc[max_date_signal, 'P/E'] if (max_date_signal in df.index and 'P/E' in df.columns) else None

    dates = [buy_date, sell_date, min_date, max_date, min_date_signal, max_date_signal]
    prices = [buy_price, sell_price, min_price, max_price, min_price_signal, max_price_signal]
    rsi_values = [buy_rsi, sell_rsi, min_rsi, max_rsi, min_rsi_signal, max_rsi_signal]
    pe_values  = [buy_pe, sell_pe, min_pe, max_pe, min_pe_signal, max_pe_signal]
    gains = [trade_gain, trade_gain, max_gain, max_gain, max_gain_signal, max_gain_signal]
    notes = ['Buy', 'Sell', 'Min', 'Max', 'MinSig', 'MaxSig']
    styles = ['--', '--', '-.', '-.', ':', ':']
    textcolors = ['', '', '', '', '', '']

    if buy_date < sell_date:
        b = 'green'
        s = 'red'
        notes[0] = ' (1,Sig_ind,B,V)'
        notes[1] = ' (1,Sig_ind,S,V)'
        textcolors[0] = 'green'
        textcolors[1] = 'red'
    else:
        b = 'green'
        s = 'red'
        notes[0] = ' (1,Sig_ind,B,InV)'
        notes[1] = ' (1,Sig_ind,S,InV)'
        textcolors[0] = 'magenta'
        textcolors[1] = 'magenta'

    if min_date < max_date:
        min_ = 'green'
        max_ = 'red'
        notes[2] = ' (2,Pd_min,B,V)'
        notes[3] = ' (2,Pd_max,S,V)'
        textcolors[2] = 'green'
        textcolors[3] = 'red'
    else:
        min_ = 'green'
        max_ = 'red'
        notes[2] = ' (2,Pd_min,B,InV)'
        notes[3] = ' (2,Pd_max,S,InV)'
        textcolors[2] = 'magenta'
        textcolors[3] = 'magenta'

    if min_date_signal < max_date_signal:
        min_sig = 'green'
        max_sig = 'red'
        notes[4] = ' (3,Sig_Min,B,V)'
        notes[5] = ' (3,Sig_Max,S,V)'
        textcolors[4] = 'green'
        textcolors[5] = 'red'
    else:
        min_sig = 'green'
        max_sig = 'red'
        notes[4] = ' (3,Sig_Min,B,InV)'
        notes[5] = ' (3,Sig_Max,S,InV)'
        textcolors[4] = 'magenta'
        textcolors[5] = 'magenta'

    linecolors = [b, s, min_, max_, min_sig, max_sig]
    lim_max = df['Close'].max() * config.limt_multiplier
    delta = df['Close'].max() / config.annotation_orient
    positions = [delta, 2.5 * delta, 3.5 * delta, 4.5 * delta, 5.5 * delta, 6.5 * delta]   

    return notes, dates, prices, rsi_values, pe_values, gains, linecolors, textcolors, styles, positions, lim_max


def annotation_executed_trades(backtested_ohlc_df):
    """
    Extracts executed trade events from the backtested DataFrame for annotation.
    An executed trade is detected when the 'Position' changes:
      - From 0 to >0: Buy event
      - From >0 to 0: Sell event
    
    Returns:
      - notes: List of trade type strings ("Buy" or "Sell")
      - dates: List of corresponding dates (indices)
      - prices: List of trade prices (Close price at event)
      - portvals: List of Portfolio Values at event
      - rsi_values: List of RSI values at event
      - gains: List of trade returns (for sells; None for buys)
      - styles: List of line styles (e.g., '--' for both)
      - linecolors: List of colors (green for buys, red for sells)
      - textcolors: List of text colors (green for buys, red for sells)
      - positions: List of y positions for the annotations (e.g., the close price)
      - lim_max: A y-limit value calculated from df (for plotting purposes)
    """
    notes = []
    dates = []
    prices = []
    portvals = []
    rsi_values = []
    gains = []
    styles = []
    linecolors = []
    textcolors = []
    positions = []

    prev_position = 0  # initialize previous position
    
    for idx, row in backtested_ohlc_df.iterrows():
        curr_position = row.get('Position', 0)
        # Executed Buy: position goes from 0 to >0
        if curr_position > 0 and prev_position == 0:
            notes.append("Buy")
            dates.append(idx)
            prices.append(row['Close'])
            portvals.append(row['Portfolio_Value'])
            rsi_values.append(row['RSI'])
            gains.append(None)  # No trade return for a buy
            styles.append('--')
            linecolors.append('green')
            textcolors.append('green')
            positions.append(row['Close'])
        # Executed Sell: position goes from >0 to 0
        elif curr_position == 0 and prev_position > 0:
            notes.append("Sell")
            dates.append(idx)
            prices.append(row['Close'])
            portvals.append(row['Portfolio_Value'])
            rsi_values.append(row['RSI'])
            # Compute trade return assuming an initial cash of 10000 for simplicity.
            # (You may want to adjust this if you store per-trade buy prices.)
            trade_return = ((row['Portfolio_Value'] - 10000) / 10000 * 100)
            gains.append(trade_return)
            styles.append('--')
            linecolors.append('red')
            textcolors.append('red')
            positions.append(row['Close'])
        prev_position = curr_position

    lim_max = backtested_ohlc_df['Close'].max() * config.limt_multiplier
    return notes, dates, prices, rsi_values, portvals, gains, styles, linecolors, textcolors, positions, lim_max
