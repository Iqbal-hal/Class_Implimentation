
#===================================================================================================
# Description: This file contains the event handler for the scroll event in matplotlib.
# scroll_handler.py
# ===================================================================================================
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

def on_scroll(event, fig):
    """
    Scroll event handler that supports three cases:
      1. Both Shift and Control: Vertical-only zoom on the current axes.
      2. Only Shift: Vertical zoom on the main (first) axes.
      3. Only Control: Full (XY) zoom on the current axes.
    
    Parameters:
      event: the scroll event from matplotlib.
      fig: the current Figure object.
    """
    # Ensure we have a valid GUI event.
    if event.guiEvent is None:
        return
    state = event.guiEvent.state

    # Get the main (price) plot: assume it's always the first axes.
    if fig.axes:
        ax_main = fig.axes[0]
    else:
        return

    # Get the current axis under the mouse.
    ax = event.inaxes

    # Determine zoom factor.
    if event.button == 'up':
        scale_factor = 1 / 1.1
    elif event.button == 'down':
        scale_factor = 1.1
    else:
        return

    # Case 1: Both Shift and Control pressed -> vertical zoom on current axes.
    if (state & 0x0001) and (state & 0x0004) and (ax is not None):
        if event.ydata is None:
            return
        cur_ylim = ax.get_ylim()
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        # Use event.ydata as the reference for the vertical zoom.
        rel = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_ylim(event.ydata - new_height * (1 - rel),
                    event.ydata + new_height * rel)
        event.canvas.draw_idle()
        return

    # Case 2: Only Shift pressed -> vertical zoom on main axes.
    elif (state & 0x0001) and (ax_main is not None):
        # Convert display coordinates to data coordinates.
        try:
            x_main, y_main = ax_main.transData.inverted().transform((event.x, event.y))
        except Exception:
            return
        cur_ylim = ax_main.get_ylim()
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        rel = (cur_ylim[1] - y_main) / (cur_ylim[1] - cur_ylim[0])
        ax_main.set_ylim(y_main - new_height * (1 - rel),
                         y_main + new_height * rel)
        event.canvas.draw_idle()
        return

    # Case 3: Only Control pressed -> full (XY) zoom on the current axes.
    elif (state & 0x0004) and (ax is not None):
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        if event.xdata is None or event.ydata is None:
            return
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim(event.xdata - new_width * (1 - relx),
                    event.xdata + new_width * relx)
        ax.set_ylim(event.ydata - new_height * (1 - rely),
                    event.ydata + new_height * rely)
        event.canvas.draw_idle()
        return
