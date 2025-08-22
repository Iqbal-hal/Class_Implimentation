import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_ema(prices, period):
    """
    Calculate Exponential Moving Average (EMA) for given prices and period
    
    Args:
        prices (pd.Series): Series of stock prices
        period (int): Period for EMA calculation (e.g., 12, 26, 50, 200)
    
    Returns:
        pd.Series: EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()

def get_stock_data(symbol, period="1y"):
    """
    Fetch stock data using yfinance
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        pd.DataFrame: Stock data with OHLCV information
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def plot_stock_with_ema(data, symbol, ema_periods=[12, 26, 50]):
    """
    Plot stock price with EMA lines
    
    Args:
        data (pd.DataFrame): Stock data
        symbol (str): Stock symbol for title
        ema_periods (list): List of EMA periods to plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot closing price
    plt.plot(data.index, data['Close'], label=f'{symbol} Close Price', linewidth=2, color='black')
    
    # Plot EMAs
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, period in enumerate(ema_periods):
        ema = calculate_ema(data['Close'], period)
        plt.plot(data.index, ema, label=f'EMA-{period}', 
                linewidth=1.5, color=colors[i % len(colors)])
    
    plt.title(f'{symbol} Stock Price with Exponential Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_ema_signals(data, short_period=12, long_period=26):
    """
    Analyze EMA crossover signals for trading
    
    Args:
        data (pd.DataFrame): Stock data
        short_period (int): Short-term EMA period
        long_period (int): Long-term EMA period
    
    Returns:
        pd.DataFrame: Data with EMA signals
    """
    df = data.copy()
    
    # Calculate EMAs
    df[f'EMA_{short_period}'] = calculate_ema(df['Close'], short_period)
    df[f'EMA_{long_period}'] = calculate_ema(df['Close'], long_period)
    
    # Generate signals
    df['Signal'] = 0
    df['Signal'][short_period:] = np.where(
        df[f'EMA_{short_period}'][short_period:] > df[f'EMA_{long_period}'][short_period:], 1, 0
    )
    
    # Find crossover points
    df['Position'] = df['Signal'].diff()
    
    return df

def main():
    """
    Main function to demonstrate EMA calculation and analysis
    """
    # Stock symbol to analyze
    symbol = "AAPL"  # You can change this to any stock symbol
    
    print(f"Fetching data for {symbol}...")
    
    # Get stock data
    stock_data = get_stock_data(symbol, period="1y")
    
    if stock_data is None:
        print("Failed to fetch stock data")
        return
    
    print(f"Data fetched successfully! {len(stock_data)} trading days")
    print(f"Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
    
    # Calculate different EMAs
    ema_periods = [12, 26, 50, 200]
    
    print("\nCalculating EMAs...")
    for period in ema_periods:
        ema = calculate_ema(stock_data['Close'], period)
        current_ema = ema.iloc[-1]
        print(f"EMA-{period}: ${current_ema:.2f}")
    
    # Analyze crossover signals
    print("\nAnalyzing EMA crossover signals (12/26)...")
    signals_data = analyze_ema_signals(stock_data, 12, 26)
    
    # Find recent crossovers
    recent_crossovers = signals_data[signals_data['Position'] != 0].tail(5)
    if not recent_crossovers.empty:
        print("Recent crossover signals:")
        for date, row in recent_crossovers.iterrows():
            signal_type = "BULLISH" if row['Position'] > 0 else "BEARISH"
            print(f"  {date.date()}: {signal_type} crossover at ${row['Close']:.2f}")
    
    # Plot the results
    plot_stock_with_ema(stock_data, symbol, ema_periods)
    
    # Display current statistics
    current_price = stock_data['Close'].iloc[-1]
    print(f"\nCurrent Analysis for {symbol}:")
    print(f"Current Price: ${current_price:.2f}")
    
    for period in ema_periods:
        ema_value = calculate_ema(stock_data['Close'], period).iloc[-1]
        percentage_diff = ((current_price - ema_value) / ema_value) * 100
        position = "above" if percentage_diff > 0 else "below"
        print(f"Price is {abs(percentage_diff):.2f}% {position} EMA-{period}")

# Example usage with custom data
def ema_from_custom_data():
    """
    Example of calculating EMA from custom price data
    """
    # Sample price data
    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
    dates = pd.date_range(start='2024-01-01', periods=len(prices), freq='D')
    
    # Create DataFrame
    df = pd.DataFrame({'Close': prices}, index=dates)
    
    # Calculate EMA
    ema_12 = calculate_ema(df['Close'], 12)
    
    print("Custom Data EMA Example:")
    print("Date\t\tPrice\tEMA-12")
    print("-" * 35)
    for date, price, ema in zip(df.index, df['Close'], ema_12):
        print(f"{date.date()}\t${price:.2f}\t${ema:.2f}")

if __name__ == "__main__":
    # Run main analysis
    main()
    
    print("\n" + "="*50)
    print("CUSTOM DATA EXAMPLE")
    print("="*50)
    
    # Run custom data example
    ema_from_custom_data()
