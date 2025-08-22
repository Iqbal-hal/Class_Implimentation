#================================================================
#indicators.py 
#contains the functions to plot the indicators like MACD, RSI, ATR
#================================================================


import matplotlib.dates as mdates
import gui.annotation_utils as utils
import support_files.config as config



def plot_macd(ax, scrip, df, df_scrip_gain):
    hist_diff = df['Hist'].diff()
    def set_hist_dif_alpha(row, diff):
        if row.Hist > 0:
            return 0.9 if diff > 0 else 0.5
        else:
            return 0.4 if diff > 0 else 0.8
    df['Hist_dif_alpha'] = [set_hist_dif_alpha(row, diff) for row, diff in zip(df.itertuples(), hist_diff)]
    colors = ['g' if val >= 0 else 'r' for val in df['Hist']]
    alphas = df['Hist_dif_alpha']
    ax.clear()
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['Signal'], label='Signal', color='orange')
    if config.axv_line_enabled:
        notes, dates, _, _, _, _, linecolors, _, styles, _, _ = utils.annotation(df, df_scrip_gain)
        for date, lcolor, style in zip(dates, linecolors, styles):
            ax.axvline(x=date, color=lcolor, linestyle=style, linewidth=1)
    bars = ax.bar(df.index, df['Hist'], color=colors)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax.bar(df.index, df['Hist'], color=colors, label='Histogram')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=0, labelsize=config.xaxis_label_fontsize)
    lim_min = df['MACD'].min() * config.limt_multiplier
    lim_max = df['MACD'].max() * config.limt_multiplier
    ax.set_ylim(lim_min, lim_max)
def plot_rsi(ax, scrip, df):
    ax.clear()
    ax.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax.axhline(30, color='green', linestyle='--', linewidth=1)
    ax.axhline(70, color='red', linestyle='--', linewidth=1)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=0, labelsize=config.xaxis_label_fontsize)
    lim_max = df['RSI'].max() * config.limt_multiplier
    ax.set_ylim(0, lim_max)

def plot_atr(ax, scrip, df):
    ax.clear()
    ax.plot(df.index, df['ATR'], label='ATR', color='brown')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=0, labelsize=config.xaxis_label_fontsize)
    lim_max = df['ATR'].max() * config.limt_multiplier
    ax.set_ylim(0, lim_max)
