# ===========================================================
# gui_tech_issue1rev4.py
# ===========================================================

import json
import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg
# (mplcursors removed; using our custom tooltip)
from gui.plot_indicators import plot_macd, plot_rsi, plot_atr
from gui.plot_main import plot_main, plot_volume_independent
import support_files.config as config 
import gui.scroll_handler as scroll
from gui.fundamentals_gui import FundamentalsWindow

def initialize_gui(filtered_combined_df):
    # Unpack dataframes from the input tuple.
    filtered_ohlc_df = filtered_combined_df[0]
    filtered_gain_df = filtered_combined_df[1]
    backtested_ohlc_df = filtered_combined_df[2]

    try:
        stock_list = filtered_ohlc_df['Stock'].unique().tolist()
        print(f"\n#####################     Initialize gui         ################################################################")
        print(f" Filter status: {config.FILTER_ENABLED}\n Filter chosen: {config.ACTIVE_FILTER}\n Number of stocks: {len(stock_list)}")
        print(f" Stocks list:\n {stock_list}")
        print("#######################     END  Initialize gui     ##############################################################\n")
    except ValueError as e:
        print(f"Caught exception: {e}")

    current_stock_index = 0
    user_start_date = None
    user_end_date = None
    update_job = None

    # Global variables for the vertical line and hover tooltip.
    current_df = None       # DataFrame for the current stock (precomputed indicators)
    main_ax = None          # Main (first) axis in the figure
    vertical_line = None    # Persistent vertical line (created only once)
    hover_annotation = None # Custom annotation for hover tooltip

    root = tk.Tk()
    root.title("Stock Plot")
    root.geometry("1200x900")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    fundamentals_win = FundamentalsWindow(root, initial_data=None, initial_scrip="")

    plot_frame = tk.Frame(root)
    plot_frame.grid(row=0, column=0, sticky="nsew")
    control_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
    control_frame.grid(row=1, column=0, sticky="ew")

    main_canvas_frame = tk.Frame(plot_frame)
    main_canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vertical_slider_frame = tk.Frame(plot_frame, width=200, bg='lightgrey')
    vertical_slider_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
    vertical_slider_frame.pack_propagate(False)
    test_label = tk.Label(vertical_slider_frame, text="Test", bg="yellow")
    test_label.pack()
    header = tk.Label(vertical_slider_frame, text="Zoom Controls", bg='lightgrey', font=("Arial", 10, "bold"))
    header.pack(pady=5)
    
    fig = Figure(figsize=(16, 12), dpi=120)
    canvas = FigureCanvasTkAgg(fig, master=main_canvas_frame)

    class CustomToolbar(NavigationToolbar2Tk):
        def __init__(self, canvas, window):
            super().__init__(canvas, window)
            # Button for Options window.
            self.options_button = tk.Button(self, text="Options", command=self.open_options_window)
            self.options_button.pack(side=tk.LEFT, padx=5)
            sep2 = ttk.Separator(self, orient='vertical')
            sep2.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
            self.annotation_slider = tk.Scale(self, from_=5, to=30, orient=tk.HORIZONTAL, label="Annot.Font", command=self.update_annotation_fontsize)
            self.annotation_slider.set(config.annotation_fontsize)
            self.annotation_slider.pack(side=tk.LEFT, padx=5)
            self.xaxis_slider = tk.Scale(self, from_=5, to=30, orient=tk.HORIZONTAL, label="X-Axis Tick", command=self.update_xaxis_fontsize)
            self.xaxis_slider.set(config.xaxis_label_fontsize)
            self.xaxis_slider.pack(side=tk.LEFT, padx=5)
            sep3 = ttk.Separator(self, orient='vertical')
            sep3.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
            self.annotation_orientation_slider = tk.Scale(self, from_=4, to=9, orient=tk.HORIZONTAL, label="Annot. Orient", command=self.update_annotaion_orientation)
            self.annotation_orientation_slider.set(config.annotation_orient)
            self.annotation_orientation_slider.pack(side=tk.LEFT, padx=5)
            self.limit_multiplier_slider = tk.Scale(self, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Limit Mult", command=self.update_limit_multiplier)
            self.limit_multiplier_slider.set(config.limt_multiplier)
            self.limit_multiplier_slider.pack(side=tk.LEFT)
            sep4 = ttk.Separator(self, orient='vertical')
            sep4.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
            self.toggle_legend_button = tk.Button(self, text="Legend", command=self.toggle_legend, bg='green')
            self.toggle_legend_button.pack(side=tk.LEFT)

        def open_options_window(self):
            options_win = tk.Toplevel(self)
            options_win.title("Options")
            main_plot_var = tk.BooleanVar(value=config.main_plot_enabled)
            volume_twin_var = tk.BooleanVar(value=config.volume_twin_enabled)
            volume_independent_var = tk.BooleanVar(value=config.volume_independent_enabled)
            macd_var = tk.BooleanVar(value=config.macd_enabled)
            rsi_var = tk.BooleanVar(value=config.rsi_enabled)
            atr_var = tk.BooleanVar(value=config.atr_enabled)
            close_price_var = tk.BooleanVar(value=config.close_price_enabled)
            bollinger_var = tk.BooleanVar(value=config.bollinger_enabled)
            ema9_var = tk.BooleanVar(value=config.ema_9_enabled)
            ema20_var = tk.BooleanVar(value=config.ema_20_enabled)
            ema50_var = tk.BooleanVar(value=config.ema_50_enabled)
            ema100_var = tk.BooleanVar(value=config.ema_100_enabled)
            pe_var = tk.BooleanVar(value=config.pe_enabled)
            annotation_var = tk.BooleanVar(value=config.annotation_text_enabled)
            axvline_var = tk.BooleanVar(value=config.axv_line_enabled)
            candlestick_var = tk.BooleanVar(value=config.candlestick_enabled)
            portfolio_var = tk.BooleanVar(value=config.portfolio_enabled)
            candlestick_tf_var = tk.StringVar(value="Daily")
            tk.Checkbutton(options_win, text="Main Plot", variable=main_plot_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="Twin Volume", variable=volume_twin_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="Independent Volume", variable=volume_independent_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="MACD", variable=macd_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="RSI", variable=rsi_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="ATR", variable=atr_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="Close Price", variable=close_price_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="Bollinger Bands", variable=bollinger_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="EMA 9", variable=ema9_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="EMA 20", variable=ema20_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="EMA 50", variable=ema50_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="EMA 100", variable=ema100_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="PE Ratio", variable=pe_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="Annotation Text", variable=annotation_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="X-Line", variable=axvline_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="Candlestick", variable=candlestick_var).pack(anchor="w")
            tk.Checkbutton(options_win, text="Portfolio", variable=portfolio_var).pack(anchor="w")
            tk.Label(options_win, text="Candlestick Timeframe:").pack(anchor="w")
            tk.OptionMenu(options_win, candlestick_tf_var, "Daily", "Weekly", "Monthly").pack(anchor="w")
            def on_close():                
                config.main_plot_enabled = main_plot_var.get()
                config.volume_twin_enabled = volume_twin_var.get()
                config.volume_independent_enabled = volume_independent_var.get()
                config.macd_enabled = macd_var.get()
                config.rsi_enabled = rsi_var.get()
                config.atr_enabled = atr_var.get()
                config.close_price_enabled = close_price_var.get()
                config.bollinger_enabled = bollinger_var.get()
                config.ema_9_enabled = ema9_var.get()
                config.ema_20_enabled = ema20_var.get()
                config.ema_50_enabled = ema50_var.get()
                config.ema_100_enabled = ema100_var.get()
                config.pe_enabled = pe_var.get()
                config.annotation_text_enabled = annotation_var.get()
                config.axv_line_enabled = axvline_var.get()
                config.candlestick_enabled = candlestick_var.get()
                config.portfolio_enabled = portfolio_var.get()
                config.candlestick_tf = candlestick_tf_var.get()
                update_plot()
                options_win.destroy()
            tk.Button(options_win, text="Close", command=on_close).pack(pady=5)
            options_win.protocol("WM_DELETE_WINDOW", on_close)

        def toggle_legend(self):
            config.legend_visible = not config.legend_visible
            self.toggle_legend_button.config(bg='green' if config.legend_visible else 'red')
            update_plot()  
        
        def update_annotation_fontsize(self, val):             
             config.annotation_fontsize = int(val)
             update_plot()
        def update_xaxis_fontsize(self, val):            
             config.xaxis_label_fontsize = int(val)
             update_plot()
        def update_annotaion_orientation(self, val):             
             config.annotation_orient = float(val)
             update_plot()
        def update_limit_multiplier(self, val):            
             config.limt_multiplier = float(val)
             update_plot()
        def change_candlestick_tf(self, value):            
            config.candlestick_tf = value
            update_plot()
    
    toolbar = CustomToolbar(canvas, main_canvas_frame)
    toolbar.pack(side=tk.TOP, fill=tk.X)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    control_top_frame = tk.Frame(control_frame)
    control_top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    start_date_label = tk.Label(control_top_frame, text="Start Date (dd-mm-yyyy):")
    start_date_label.grid(row=0, column=0, padx=2)
    start_date_entry = tk.Entry(control_top_frame, width=12)
    start_date_entry.grid(row=0, column=1, padx=2)
    end_date_label = tk.Label(control_top_frame, text="End Date (dd-mm-yyyy):")
    end_date_label.grid(row=0, column=2, padx=2)
    end_date_entry = tk.Entry(control_top_frame, width=12)
    end_date_entry.grid(row=0, column=3, padx=2)
    set_date_button = tk.Button(control_top_frame, text="Set Date Range", command=lambda: set_date_range())
    set_date_button.grid(row=0, column=4, padx=2)
    clear_date_button = tk.Button(control_top_frame, text="Clear Date Range", command=lambda: clear_date_range())
    clear_date_button.grid(row=0, column=5, padx=2)
    ratio_label = tk.Label(control_top_frame, text="Plot Ratios (main,volume,macd,rsi,atr):")
    ratio_label.grid(row=0, column=6, padx=2)
    ratio_entry = tk.Entry(control_top_frame, width=15)
    ratio_entry.grid(row=0, column=7, padx=2)
    ratio_entry.insert(0, "1.5,0.7,1,1,1")
    set_ratio_button = tk.Button(control_top_frame, text="Set Ratios", command=lambda: set_ratios())
    set_ratio_button.grid(row=0, column=8, padx=2)
    reset_ratio_button = tk.Button(control_top_frame, text="Reset Ratios", command=lambda: reset_ratios())
    reset_ratio_button.grid(row=0, column=9, padx=2)

    time_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                           label="Time Scroll (Select 1-year window)", length=800,
                           command=lambda val: debounced_update(val))
    time_slider.set(0)
    time_slider.pack(side=tk.TOP, fill=tk.X, expand=True)

    # (mplcursors code removed; using our custom tooltip)
    cursor_list = []  

    def set_date_range():
        nonlocal user_start_date, user_end_date
        try:
            start_str = start_date_entry.get()
            end_str = end_date_entry.get()
            user_start_date = pd.to_datetime(start_str, format='%d-%m-%Y')
            user_end_date = pd.to_datetime(end_str, format='%d-%m-%Y')
            print("Set dates:", user_start_date, user_end_date)
            update_plot()
        except Exception as e:
            print("Invalid date format. Please use dd-mm-yyyy", e)

    def clear_date_range():
        nonlocal user_start_date, user_end_date
        user_start_date = None
        user_end_date = None
        print("Cleared date range")
        update_plot()

    def set_ratios():
        try:
            parts = ratio_entry.get().split(',')
            if len(parts) != 5:
                print("Please enter 5 ratios, e.g. 1.5,0.7,1,1,1")
                return
            config.plot_ratios["main"] = float(parts[0].strip())
            config.plot_ratios["volume"] = float(parts[1].strip())
            config.plot_ratios["macd"] = float(parts[2].strip())
            config.plot_ratios["rsi"]  = float(parts[3].strip())
            config.plot_ratios["atr"]  = float(parts[4].strip())
            print("Set ratios:", config.plot_ratios)
            update_plot()
        except Exception as e:
            print("Error setting ratios:", e)

    def reset_ratios():
        config.plot_ratios = {"main": 1.5, "volume": 0.7, "macd": 1.0, "rsi": 1.0, "atr": 1.0}
        ratio_entry.delete(0, tk.END)
        ratio_entry.insert(0, "1.5,0.7,1,1,1")
        print("Ratios reset")
        update_plot()

    def debounced_update(val):
        nonlocal update_job
        if update_job is not None:
            root.after_cancel(update_job)
        update_job = root.after(150, update_plot)

    def update_vertical(ax, orig, scale):
        y0, y1 = orig
        center = (y0 + y1) / 2
        half_range = (y1 - y0) * scale / 2
        ax.set_ylim(center - half_range, center + half_range)
        canvas.draw_idle()

    def update_plot():
        nonlocal current_stock_index, user_start_date, user_end_date, current_df, main_ax, vertical_line, hover_annotation
        current_stock = stock_list[current_stock_index]
        root.title(f"Stock Plot - {current_stock} -- Filter:{config.FILTER_ENABLED}/{config.ACTIVE_FILTER}")
        # Ensure the DataFrame index is datetime and tz-naive.
        df_scrip_ohlc = filtered_ohlc_df[filtered_ohlc_df['Stock'] == current_stock].copy()
        df_scrip_ohlc.index = pd.to_datetime(df_scrip_ohlc.index).tz_localize(None)
        df_scrip_gain = filtered_gain_df.loc[current_stock, :]
        df = df_scrip_ohlc.copy()
        current_df = df  # Store for tooltip lookup

        fundamentals_win.update_data(df_scrip_ohlc, current_stock)
        
        plots = []
        if config.main_plot_enabled:
            plots.append('main')
        if config.volume_independent_enabled:
            plots.append('volume')
        if config.macd_enabled:
            plots.append('macd')
        if config.rsi_enabled:
            plots.append('rsi')
        if config.atr_enabled:
            plots.append('atr')

        if not plots:
            return

        ratios = [config.plot_ratios.get(p, 1) for p in plots]
        fig.clear()
        gs = fig.add_gridspec(len(plots), 1, hspace=0.05, height_ratios=ratios)
        axes_list = []
        first_ax = None
        for i, plot_key in enumerate(plots):
            if i == 0:
                ax = fig.add_subplot(gs[i, 0])
                first_ax = ax
            else:
                ax = fig.add_subplot(gs[i, 0], sharex=first_ax)
            axes_list.append(ax)
            if plot_key == 'main':
                volume_ax = ax.twinx() if config.volume_twin_enabled else None
                plot_main(ax, volume_ax, current_stock, df, df_scrip_gain, backtested_ohlc_df)
            elif plot_key == 'volume':
                plot_volume_independent(ax, df)
            elif plot_key == 'macd':
                plot_macd(ax, current_stock, df, df_scrip_gain)
            elif plot_key == 'rsi':
                plot_rsi(ax, current_stock, df)
            elif plot_key == 'atr':
                plot_atr(ax, current_stock, df)
       
        if user_start_date is not None and user_end_date is not None:
            first_ax.set_xlim(user_start_date, user_end_date)
        else:
            start_date_all = df.index.min()
            end_date_all   = df.index.max()
            max_start = end_date_all - pd.DateOffset(years=1)
            if start_date_all < max_start:
                total_shift_days = (max_start - start_date_all).days
                shift_days = int((time_slider.get() / 100.0) * total_shift_days)
                new_start = start_date_all + pd.Timedelta(days=shift_days)
                new_end = new_start + pd.DateOffset(years=1)
                first_ax.set_xlim(new_start, new_end)
            else:
                first_ax.set_xlim(start_date_all, end_date_all)
        bottom_ax = axes_list[-1]
        bottom_ax.xaxis.set_major_locator(MaxNLocator(5))
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        bottom_ax.tick_params(axis='x', rotation=0, labelsize=config.xaxis_label_fontsize)
        for ax in axes_list[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
        canvas.draw_idle()
        
        def custom_format_coord(x, y):
            try:
                dt = mdates.num2date(x)
                return f"Date: {dt.strftime('%d-%m-%Y')}, Value: {y:.2f}"
            except Exception:
                return f"X: {x:.2f}, Y: {y:.2f}"
        for ax in axes_list:
            ax.format_coord = custom_format_coord

        # Rebuild zoom controls.
        for widget in vertical_slider_frame.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget("text") == "Zoom Controls":
                continue
            widget.destroy()
        for i, ax in enumerate(axes_list):
            orig_ylim = ax.get_ylim()
            frame = tk.Frame(vertical_slider_frame, bd=2, relief=tk.GROOVE, bg='lightgrey')
            frame.pack(pady=2, fill=tk.X, padx=5)
            lbl = tk.Label(frame, text=f"Y-Zoom Plot {i+1}", bg='lightgrey')
            lbl.pack(side=tk.TOP)
            slider = tk.Scale(frame, from_=0.1, to=3.0, resolution=0.1,
                              orient=tk.HORIZONTAL, showvalue=True,
                              command=lambda val, ax=ax, orig=orig_ylim: update_vertical(ax, orig, float(val)))
            slider.set(1.0)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            btn_frame = tk.Frame(frame, bg='lightgrey')
            btn_frame.pack(side=tk.LEFT, padx=5)
            plus_btn = tk.Button(btn_frame, text="+", command=lambda s=slider: s.set(round(s.get()+0.1, 1)))
            plus_btn.pack(side=tk.TOP)
            minus_btn = tk.Button(btn_frame, text="-", command=lambda s=slider: s.set(round(s.get()-0.1, 1)))
            minus_btn.pack(side=tk.BOTTOM)

        # --- Create or update the vertical line on the main axis ---
        main_ax = axes_list[0]
        if vertical_line is None:
            vertical_line = main_ax.axvline(x=main_ax.get_xlim()[0], color='lime', linestyle='--', zorder=10)
        else:
            vertical_line.set_xdata([main_ax.get_xlim()[0], main_ax.get_xlim()[0]])

        # --- Create or update the custom hover tooltip ---
        if hover_annotation is None:
            hover_annotation = main_ax.annotate(
                "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                arrowprops=dict(arrowstyle="->"),
                zorder=15
            )
            hover_annotation.set_visible(False)
        else:
            hover_annotation.set_visible(False)

    def on_key(event):
        nonlocal current_stock_index
        if event.keysym == 'Right':
            current_stock_index = (current_stock_index + 1) % len(stock_list)
            update_plot()
        elif event.keysym == 'Left':
            current_stock_index = (current_stock_index - 1) % len(stock_list)
            update_plot()
    root.bind('<Key>', on_key)

    # --- Motion Event: update vertical line and custom tooltip ---
    def on_motion(event):
        nonlocal hover_annotation, main_ax, vertical_line, current_df
        if event.xdata is None:
            if hover_annotation is not None:
                hover_annotation.set_visible(False)
            canvas.draw_idle()
            return

        # Use main_ax as the effective axis even if event comes from a twin axis.
        effective_ax = main_ax if main_ax is not None else event.inaxes

        if vertical_line is not None and effective_ax is not None:
            vertical_line.set_xdata([event.xdata, event.xdata])
        if current_df is not None and effective_ax is not None:
            try:
                dt = mdates.num2date(event.xdata)
                # Remove timezone information using replace
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                dt_ts = pd.Timestamp(dt)
                pos = current_df.index.searchsorted(dt_ts)
                if pos == 0:
                    nearest_date = current_df.index[0]
                elif pos == len(current_df.index):
                    nearest_date = current_df.index[-1]
                else:
                    before = current_df.index[pos-1]
                    after = current_df.index[pos]
                    if abs((dt_ts - before).total_seconds()) <= abs((after - dt_ts).total_seconds()):
                        nearest_date = before
                    else:
                        nearest_date = after
                nearest_close = current_df.loc[nearest_date, 'Close']
                hover_annotation.set_text(f"Date: {nearest_date.strftime('%d-%m-%Y')}\nClose: {nearest_close:.2f}")
                hover_annotation.xy = (mdates.date2num(nearest_date), nearest_close)
                hover_annotation.set_visible(True)
            except Exception as e:
                print("Lookup error:", e)
                hover_annotation.set_visible(False)
        else:
            hover_annotation.set_visible(False)
        canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("scroll_event", lambda event: scroll.on_scroll(event, fig))
    update_plot()
    tk.mainloop()

if __name__ == '__main__':
    initialize_gui()
