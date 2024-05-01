#this strategy sucks, and I'm looking to create a flushed out strategy class in the future
import indicators as ind
import pandas as pd
import vectorbt as vbt
import vbt_test as test

trend_data = test.get_trend_table('ES')

backadjusted_prices = trend_data['Backadjusted']
high = trend_data['High']
low = trend_data['Low']
close = trend_data['Close']

short_ma, long_ma = ind.compute_ma(backadjusted_prices)
bbands = ind.compute_bbands(backadjusted_prices)
rsi = ind.compute_rsi(backadjusted_prices, window=11)
atr_20 = ind.compute_ATR(high,low,close, plot=True)
atr_5 = ind.compute_ATR(high,low,close, window=5)

def strategy(bbands, prices, rsi, atr_20, atr_5, holding_period, capital=10000) -> vbt.Portfolio:
    entries = (prices < bbands.lower) & (rsi.rsi < 30)

    exits = entries.shift(holding_period)
    exits = exits.fillna(False)

    portfolio = vbt.Portfolio.from_signals(close=prices,
                                           entries=entries,
                                           exits=exits,
                                           init_cash=capital,
                                           sl_stop=0.05, #stop loss at 5% below price
                                           sl_trail=True, #sets stop loss to trailing
                                           freq='D')
    return portfolio

portfolio = strategy(bbands,backadjusted_prices, rsi, atr_20, atr_5,8)

portfolio.plot_trades().show()

print(portfolio.stats())