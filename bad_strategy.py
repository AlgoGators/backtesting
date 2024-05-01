#this strategy sucks, and I'm looking to create a flushed out strategy class in the future
import indicators as ind
import pandas as pd
import vectorbt as vbt
import vbt_test as test

trend_data = test.get_trend_table('ES')
backadjusted_prices = trend_data['Backadjusted']
short_ma, long_ma = ind.compute_ma(trend_data, backadjusted_prices, plot=False)
bbands = ind.compute_bbands(trend_data, backadjusted_prices, plot=True)
rsi = ind.compute_rsi(trend_data, backadjusted_prices, plot=False)

def strategy(bbands, prices, rsi, holding_period) -> vbt.Portfolio:
    entries = (prices < bbands.lower) & (rsi.rsi<=30)
    exits = entries.shift(holding_period)
    exits = exits.fillna(False)
    portfolio = vbt.Portfolio.from_signals(prices, entries, exits, freq='D')
    return portfolio

portfolio = strategy(bbands,backadjusted_prices, rsi,8)

portfolio.plot_trades().show()

print(portfolio.stats())