import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
def compute_ma(data_path, prices, short_ma_period=16, long_ma_period=64, plot=False):
    """
    Compute and optionally plot the short and long EMAs for a given dataset.

    Parameters:
    data_path (str): Path to the CSV file containing the data.
    backadjusted_price_col (str): Name of the column with backadjusted prices.
    short_ema_period (int): Window period for the short EMA.
    long_ema_period (int): Window period for the long EMA.
    plot (bool): Whether to plot the price and EMAs.

    Returns:
    tuple: A tuple containing the pandas Series for the short MA and the long MA.
    """

    # Calculate EMAs
    short_ma = vbt.MA.run(prices, window=short_ma_period)
    long_ma = vbt.MA.run(prices, window=long_ma_period)

    # Plotting
    if plot:
        plt.figure(figsize=(14, 7))
        plt.plot(prices, label='Backadjusted Price', color='gray', alpha=0.5)
        plt.plot(short_ma.ma, label=f'{short_ma_period}-day MA', color='red')
        plt.plot(long_ma.ma, label=f'{long_ma_period}-day MA', color='blue')
        plt.title('Price and  Moving Averages')
        plt.legend()
        plt.show()

    return short_ma, long_ma

def compute_bbands(data_path, prices, window=20, std_dev=2, plot=False):

    bbands = vbt.BBANDS.run(prices, window=window, alpha=std_dev)

    if plot:
        plt.figure(figsize=(14,7))
        plt.plot(prices, label='Backadjusted Price', color='gray', alpha=0.5)
        plt.plot(bbands.upper, label=f'{window}-day upper BBANDS', color='green')
        plt.plot(bbands.lower, label=f'{window}-day lower BBANDS', color='green')
        plt.title('Price and BBANDS')
        plt.legend()
        plt.show()

    return bbands

def compute_rsi(data_path, prices, window=14, plot=False):
    rsi = vbt.RSI.run(prices, window=window)

    if plot:
        plt.figure(figsize=(14, 7))
        plt.plot(rsi.rsi, label=f'{window}-day RSI', color='blue', alpha=0.5)

        # Add horizontal lines at 70 and 30
        plt.axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')

        plt.title('RSI')
        plt.legend()
        plt.show()

    return rsi