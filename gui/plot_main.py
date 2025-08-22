
# ===========================================================
# plot_main.py
# Plotting functions for the main chart
# ===========================================================

import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import gui.annotation_utils as utils
import support_files.config as config
import gui.getportfolio as portfolio

portfolio_details = portfolio.load_portfolio_from_json("portfolio.json")

def plot_main(ax,volume_ax, scrip, df, df_scrip_gain,backtested_ohlc_df):
    ax.clear()
    #volume_ax = ax.twinx() if config.volume_twin_enabled else None   
    if volume_ax is not None:
        volume_ax.clear()

    if config.candlestick_enabled and {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
                
        # Resample data based on candlestick_tf selection.
        if config.candlestick_tf == "Daily":
            df_candle = df.copy()
        elif config.candlestick_tf == "Weekly":
            df_candle = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
        elif config.candlestick_tf == "Monthly":
            df_candle = df.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
        # Reset index so that Date becomes a column.
        df_candle = df_candle.reset_index()[['Date','Open','High','Low','Close']].copy()
        df_candle['Date'] = df_candle['Date'].map(mdates.date2num)
        candlestick_ohlc(ax,
                         df_candle[['Date','Open','High','Low','Close']].values,
                         width=0.6,
                         colorup='green',
                         colordown='red',
                         alpha=0.8)
    else:
        # Fallback to a line plot of the Close price.
        if config.close_price_enabled:
            ax.plot(df.index, df['Close'], label='Close Price', color='navy', alpha=0.8, linestyle='-')    
    
    # Plot price and indicators.
    if config.close_price_enabled:
        ax.plot(df.index, df['Close'], label='Close Price', color='navy', alpha=0.8, linestyle='-')
    if config.bollinger_enabled:
        ax.plot(df.index, df['Bollinger_Upper'], label='Bollinger Upper', color='orange', alpha=0.6, linestyle='--')
        ax.plot(df.index, df['Bollinger_Middle'], label='Bollinger Middle', color='gray', alpha=0.5, linestyle='-.')
        ax.plot(df.index, df['Bollinger_Lower'], label='Bollinger Lower', color='purple', alpha=0.6, linestyle=':')
    if config.ema_9_enabled:
        ax.plot(df.index, df['ema_9'], label='EMA 9', color='lime', alpha=0.7, linestyle='-')
    if config.ema_20_enabled:
        ax.plot(df.index, df['ema_20'], label='EMA 20', color='cyan', alpha=0.7, linestyle='--')
    if config.ema_50_enabled:
        ax.plot(df.index, df['ema_50'], label='EMA 50', color='gold', alpha=0.7, linestyle='-')
    if config.ema_100_enabled:
        ax.plot(df.index, df['ema_100'], label='EMA 100', color='mediumvioletred', alpha=0.7, linestyle='--')
    if config.pe_enabled and 'P/E' in df.columns:
        ax.plot(df.index, df['P/E'], label='P/E', color='magenta', alpha=0.6, linestyle='-.')
        
    # Technical signal annotations.
    notes, dates, prices, rsi_values, pe_values, gains, linecolors, textcolors, styles, positions, lim_max = utils.annotation(df, df_scrip_gain)
    for note, date, price, rsi_val, pe_val, gain, lcolor, tcolor, style, pos in zip(
            notes, dates, prices, rsi_values, pe_values, gains, linecolors, textcolors, styles, positions):
        annotation_text = f"  {note}\n  {date}\n   Price: {price}\n   RSI: {rsi_val}\n   PE: {pe_val}\n   Gain: {gain}"
        if config.axv_line_enabled:
            ax.axvline(x=date, color=lcolor, linestyle=style, linewidth=1)
        if config.annotation_text_enabled:
            ann = ax.annotate(annotation_text,
                              xy=(date, pos),
                              xytext=(10, 0),  # 10 pixels right offset
                              textcoords='offset points',
                              ha='left',
                              fontsize=config.annotation_fontsize,
                              color=tcolor,
                              bbox=dict(facecolor='white', alpha=0.5))
            ann.draggable()  # Make annotation draggable.
    
    # Portfolio annotations.
    if config.portfolio_enabled and scrip in portfolio_details:
        for trade in portfolio_details[scrip]:
            invest_date = trade["Date Invested"]
            latest_date = df.index.max()
            cmp_val = df.loc[latest_date, 'Close']
            current_value = trade["Quantity"] * cmp_val
            gain_loss = current_value - trade["Amount Invested"]
            pct_gain_loss = (gain_loss / trade["Amount Invested"]) * 100
            days_holded = (latest_date - invest_date).days

            port_text = (
                r"$\mathbf{Portfolio}$" "\n" +
                r"$\mathbf{Scrip:}$" f" {scrip}" "\n" +
                r"$\mathbf{Side:}$" f" {trade['Side']}" "\n" +
                r"$\mathbf{Date Invested:}$" f" {invest_date.strftime('%d-%m-%Y')}" "\n" +
                r"$\mathbf{Quantity:}$" f" {trade['Quantity']}" "\n" +
                r"$\mathbf{Invested Price:}$" f" {trade['Invested Price']}" "\n" +
                r"$\mathbf{Amount Invested:}$" f" {trade['Amount Invested']}" "\n" +
                r"$\mathbf{CMP:}$" f" {cmp_val:.2f}" "\n" +
                r"$\mathbf{Current Value:}$" f" {current_value:.2f}" "\n" +
                r"$\mathbf{Gain/Loss:}$" f" {gain_loss:.2f}" "\n" +
                r"$\mathbf{PctGain/Loss:}$" f" {pct_gain_loss:.2f}" "\n" +
                r"$\mathbf{Days Holded:}$" f" {days_holded}"
            )
            ax.axvline(x=invest_date, color='blue', linestyle='--', linewidth=1.5)
            y_pos = df['Close'].max() * 0.8  # Adjust base y position as needed.
            ann = ax.annotate(port_text,
                        xy=(invest_date, y_pos),
                        xytext=(10, -10),  # Offset: 10 pixels right, 10 pixels down.
                        textcoords='offset points',
                        ha='left',
                        fontsize=config.annotation_fontsize,
                        color='black',
                        bbox=dict(facecolor='wheat', edgecolor='gray', alpha=0.7))
            ann.draggable()  # Make portfolio annotation draggable.
    
        # Backtest Annotations: Plot vertical lines and detailed annotations for buy and sell events.


    # Example snippet in plot_main():
    if config.backtest_annotation_enabled:
        (notes, dates, prices, rsi_values, portvals, gains,
         styles, linecolors, textcolors, positions, lim_max) = utils.annotation_executed_trades(backtested_ohlc_df)
    
        for note, date, price, rsi_val, portval, gain, style, lcolor, tcolor, pos in zip(
                notes, dates, prices, rsi_values, portvals, gains, styles, linecolors, textcolors, positions):
            annotation_text = f"{note} @ {price:.2f}\nPortVal: {portval:.2f}\nRSI: {rsi_val:.2f}"
            if note == "Sell" and gain is not None:
                annotation_text += f"\nRet: {gain:.2f}%"
            if config.axv_line_enabled:
                ax.axvline(x=date, color=lcolor, linestyle=style, linewidth=1)
            if config.annotation_text_enabled:
                ann = ax.annotate(annotation_text,
                                  xy=(date, pos),
                                  xytext=(10, 0),  # offset
                                  textcoords='offset points',
                                  ha='left',
                                  fontsize=config.annotation_fontsize,
                                  color=tcolor,
                                  bbox=dict(facecolor='white', alpha=0.5))
                ann.draggable()  # Make annotation draggable.




    ax.set_title(scrip)
    ax.set_ylabel('Price', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    if config.legend_visible:
     ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_ylim(0, lim_max)
    
    # Volume plotting.
    if volume_ax is not None and config.volume_twin_enabled:
        volume_ax.set_xticklabels([])
        volume = df['Volume']
        close_prices = df['Close']
        colors = ['g' if close_prices.iloc[i] > close_prices.iloc[i-1] else 'r' for i in range(1, len(close_prices))]
        colors.insert(0, 'g')
        volume_ax.bar(df.index, volume, color=colors, alpha=0.4)
        volume_ax.plot(df.index, df['volume_ema_20'], label='Volume EMA 20', color='blue', alpha=0.5)
        volume_ax.set_ylabel('Volume', color='black')
        volume_ax.yaxis.tick_right()
        volume_ax.tick_params(axis='y', labelcolor='black')
        volume_max = volume.max() * 2
        volume_ax.set_ylim(0, volume_max)
        volume_ax.set_autoscale_on(False)
        volume_ax.set_navigate(False)


# =====================================
# New independent volume plot function
# =====================================
def plot_volume_independent(ax, df):
    ax.clear()
    volume = df['Volume']
    close_prices = df['Close']
    colors = ['g' if close_prices.iloc[i] > close_prices.iloc[i-1] else 'r' for i in range(1, len(close_prices))]
    colors.insert(0, 'g')
    
    ax.bar(df.index, volume, color=colors, alpha=0.4)
    ax.plot(df.index, df['volume_ema_20'], label='Volume EMA 20', color='blue', alpha=0.5)
    ax.set_ylabel('Volume', color='black')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=0, labelsize=config.xaxis_label_fontsize)
    volume_max = volume.max() * 2
    ax.set_ylim(0, volume_max)
