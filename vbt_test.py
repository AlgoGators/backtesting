import pandas as pd
import sqlalchemy
import vectorbt as vbt
import tomllib as tl
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from plotly.io import to_image
from data_engine.manager import Manager, HistoricalManager

# Define your symbols and file paths at the beginning for easy reference
SYMBOLS = ['ES', 'NQ', 'ZN']
POSITIONS_FILE_PATH = r'optimized_positions.csv'
CONFIG_FILE_PATH = r'config.toml'
DATABASE_TABLE_NAME = 'close'


def load_config(config_file_path: str) -> Dict[str, Any]:
    with open(config_file_path, 'rb') as f:
        return tl.load(f)


def create_engine(config: Dict[str, Any]) -> sqlalchemy.engine.Engine:
    db_params = config['database']
    server_params = config['server']
    database_url = (f"postgresql://{server_params['user']}:{server_params['password']}@{server_params['ip']}:"
                    f"{db_params['port']}/{db_params['db_trend']}")
    return sqlalchemy.create_engine(database_url)


def fetch_data(engine: sqlalchemy.engine.Engine, table_name: str, symbols: list = None) -> pd.DataFrame:
    query = f"SELECT * FROM \"{table_name}_data\""
    df = pd.read_sql(query, engine, index_col='Date')
    return df['Close']


def combine_contract_prices(engine: sqlalchemy.engine.Engine) -> pd.DataFrame:
    price_data_list = []
    for symbol in SYMBOLS:
        # Fetch the data
        temp_df = fetch_data(engine, symbol)

        if not temp_df.empty:
            # Convert Series to DataFrame
            temp_df = temp_df.to_frame(name=f'{symbol}_Close')

            # Ensure the index is of the right type (DateTime) and in the format YYYY-MM-DD
            temp_df.index = pd.to_datetime(temp_df.index).strftime('%Y-%m-%d')

            price_data_list.append(temp_df)
        else:
            print(f"No data found for symbol: {symbol}")

    # Concatenate all dataframes along the columns
    price_data = pd.concat(price_data_list, axis=1)

    # Ensure the index is of the right type (DateTime) after concatenation
    price_data.index = pd.to_datetime(price_data.index)

    # Find the latest start date across all securities
    latest_start_date = max([df.index.min() for df in price_data_list])

    # Create a date range starting from the latest start date across all dataframes
    all_dates = pd.date_range(start=latest_start_date,end=price_data.index.max(), freq='D')

    # Reindex the dataframe with the complete date range, forward-filling missing values
    price_data = price_data.reindex(all_dates).dropna()

    return price_data


def load_and_prepare_positions(file_path: str) -> pd.DataFrame:
    positions_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    positions_df.index = pd.to_datetime(positions_df.index)
    return positions_df.round()


def process_symbol(df: pd.DataFrame, symbol: str, positions_df: pd.DataFrame, capital: float = 10000.0) -> vbt.Portfolio:
    # Prepare the symbol's price data
    symbol_df = df[[f'{symbol}_Close']].dropna().loc[~df.index.duplicated(keep='first')]

    # Prepare the positions DataFrame
    symbol_positions = positions_df[[symbol]].dropna()

    # Ensure indices match between price data and positions data
    combined_df = symbol_df.join(symbol_positions, how='inner')
    combined_df.dropna(inplace=True)

    # Calculate differences to find entries (buys) and exits (sells)
    position_diff = symbol_positions.diff().fillna(0)

    # Entries occur when the position difference is positive (buy)
    entries = position_diff > 0
    entries.rename(columns={symbol: f'{symbol}_Entries'}, inplace=True)

    # Exits occur when the position difference is negative (sell)
    exits = position_diff < 0
    exits.rename(columns={symbol: f'{symbol}_Exits'}, inplace=True)

    # The trade size is the absolute value of the difference
    trade_sizes = position_diff.abs()
    trade_sizes.rename(columns={symbol: f'{symbol}_Trade_Size'}, inplace=True)

    # Create a DataFrame after processing entries and exits
    entry_exit_trade_df = pd.concat([entries, exits, trade_sizes], axis=1)

    # Ensure indices match between symbol data and positions data
    combined_df = symbol_df.join(entry_exit_trade_df, how='inner')
    combined_df.dropna(inplace=True)

    # Create the portfolio using the entries, exits, and trade sizes
    portfolio = vbt.Portfolio.from_signals(
        close=combined_df[f'{symbol}_Close'],
        entries=combined_df[f'{symbol}_Entries'],
        exits=combined_df[f'{symbol}_Exits'],
        size=combined_df[f'{symbol}_Trade_Size'],
        fees=0.001,
        freq='D',
        init_cash=capital
    )

    return portfolio, combined_df #return combined_df to scrape for data in jumbo_portfolio

#takes daily returns as a percent and compounds it
def find_portfolio_pnl(portfolio: vbt.Portfolio, initial_capital):
    return(1 + portfolio.daily_returns()).cumprod() * initial_capital - initial_capital
    #To get portfolio value do find_portfolio_pnl(parameters) + initial capital

def get_trend_table(SYMBOLS, start_date = '2023-8-4', end_date = '2024-2-7'):

    manage = Manager()
    for symbol in SYMBOLS:
        df = manage.get_trend_table(f'{symbol}')
        filtered_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)]
        print(filtered_df)

def get_carry_table(SYMBOLS, start_date = '2023-8-4', end_date = '2024-2-7'):

    manage = Manager()
    for symbol in SYMBOLS:
        df = manage.get_carry_table(f'{symbol}') #still getting worked out
        filtered_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)]
        print(f'Trend Table for {symbol}')
        print(filtered_df)
def plot_plt(jumbo_portfolio, initial_capital = 10000):
    i = True
    # Here is another method of basic plotting that has worked well and doesn't require writing to an html file
    plt.figure(figsize=(14, 10))

    for symbol, portfolio in jumbo_portfolio.items():
        plt.plot(find_portfolio_pnl(portfolio[0], initial_capital), label=f'{symbol} Cumulative Money Value')
        if i is True:
            jumbo_portfolio_value = find_portfolio_pnl(portfolio[0], initial_capital)
            i = False
        else:
            jumbo_portfolio_value += find_portfolio_pnl(portfolio[0], initial_capital)

    plt.plot(jumbo_portfolio_value, label='Combined Cumulative Money Value', linestyle='--')

    plt.title('Cumulative Returns in Money Value')
    plt.xlabel('Date')
    plt.ylabel('Money Value ($)')
    plt.legend()
    plt.show()

def plot_vbt(jumbo_portfolio, initial_capital = 10000):
    cumulative_returns_list = []
    i = True
    for symbol, portfolio in jumbo_portfolio.items():
        # get and store cumulative returns from portfolio
        if i == True:
            cumulative_returns = portfolio[0].cumulative_returns() * initial_capital
            i = False
        else:
            cumulative_returns += portfolio[0].cumulative_returns() * initial_capital
        cumulative_returns_series = portfolio[0].cumulative_returns() * initial_capital
        cumulative_returns_list.append(cumulative_returns_series)

    # Combine all cumulative return series into a single DataFrame
    cumulative_returns_df = pd.concat(cumulative_returns_list, axis=1)

    # Ensure the DataFrame has appropriate column names (e.g., the symbols)
    cumulative_returns_df.columns = SYMBOLS
    cumulative_returns_df['Aggregate_Return'] = cumulative_returns

    # Now, plot all cumulative returns together using vectorbt
    fig = cumulative_returns_df.vbt.plot()

    # Save the plot to HTML
    fig.write_html('vbt_plot_returns.html')

def print_stats(jumbo_portfolio):
    for symbol, portfolio in jumbo_portfolio.items():
        print(f"Stats for {symbol}:")
        print(portfolio[0].stats())


#set initial capital
capital = 10000

# Load configuration and create database engine
config_data = load_config(CONFIG_FILE_PATH)
engine = create_engine(config_data)

# Load and prepare price data
price_data = combine_contract_prices(engine)
#price_data['Combined_Close'] = price_data.sum(axis=1)
print(price_data.head(100))

# Load positions data
positions_data = load_and_prepare_positions(POSITIONS_FILE_PATH)

#holds Symbol as key, and portfolio, combined_df as definition; Might make a class to hold any information/commands instead
jumbo_portfolio = {symbol: process_symbol(price_data, symbol, positions_data, capital) for symbol in SYMBOLS}

plot_vbt(jumbo_portfolio)
print_stats(jumbo_portfolio)
SYMBOLS = ['ES', 'ZN']
get_trend_table(SYMBOLS)
plot_plt(jumbo_portfolio)
#get_carry_table(SYMBOLS) still getting worked out

'''
#This is a failed attempt at combining portfolios in VBT
#Part of this failure is that the value of the portfolio is changing relative to the combined close, rather than just the portfolios currently held
def combine_portfolios(jumbo_portfolio):
    portfolios = []
    SYMBOLS = []
    for symbol, portfolio in jumbo_portfolio.items():
        portfolios.append(portfolio[1])
        SYMBOLS.append(symbol)
    combined_df = pd.concat(portfolios).groupby(level=0).sum()
    combined_df['Combined_Close'] = combined_df.sum(axis=1)
    combined_df['Combined_Money_Up'] = sum(
    combined_df[f'{symbol}_Close'] * combined_df[f'{symbol}_Trade_Size'] * combined_df[f'{symbol}_Entries']
    for symbol in SYMBOLS
)
    combined_df['Combined_Money_Down'] = sum(
    combined_df[f'{symbol}_Close'] * combined_df[f'{symbol}_Trade_Size'] * combined_df[f'{symbol}_Exits']
    for symbol in SYMBOLS
)
    combined_df['Combined_Money'] = (combined_df['Combined_Money_Up'] - combined_df['Combined_Money_Down']).abs()
    combined_df['Combined_Entries'] = np.where(combined_df['Combined_Money_Up'] > combined_df['Combined_Money_Down'], True, False)
    combined_df['Combined_Exits'] = np.where(combined_df['Combined_Money_Up'] < combined_df['Combined_Money_Down'], True, False)
    combined_df['Combined_Trade_Size'] = combined_df['Combined_Money']/combined_df['Combined_Close']

    print(combined_df)
    portfolio = vbt.Portfolio.from_signals(
        close=combined_df['Combined_Close'].astype(float),
        entries=combined_df['Combined_Entries'],
        exits=combined_df['Combined_Exits'],
        size=combined_df['Combined_Trade_Size'].astype(float),
        fees=0.0,
        freq='D',
        init_cash=10000
    )
    combined_df['port_pnl'] = find_portfolio_pnl(portfolio, 10000)
    combined_df.to_csv('mo_money.csv', index=True)
    #doesn't work as when unrelated symbol_closes change, the combined_close changes
    return portfolio

#Here is an attempt to implement it.
port2 = combine_portfolios(jumbo_portfolio)
#to plot VBT we need to write to html
fig = port2.cumulative_returns().vbt.plot()
fig.write_html('figure.html')
print(port2.stats())
'''

"""
#plot from vbt instead of pyplot
port = process_symbol(price_data, 'ES', positions_data)
tot_ret = port.cumulative_returns()*10000
print(port.stats())
fig = tot_ret.vbt.plot()
fig.write_html('figure.html')

#port.orders.records_readable.to_csv("portfolio_orders.csv", index=False)

"""
"""
# Process each symbol independently and store portfolios in a dictionary
jumbo_portfolio = {symbol: process_symbol(price_data, symbol, positions_data) for symbol in SYMBOLS}

i = True
#Here is another method of basic plotting that has worked well and doesn't require writing to an html file
plt.figure(figsize=(14, 10))

for symbol, portfolio in jumbo_portfolio.items():
    print(f"Stats for {symbol}:")
    print(portfolio.stats(), "\n")
    plt.plot(find_portfolio_pnl(portfolio, initial_capital), label=f'{symbol} Cumulative Money Value')
    if i is True:
        jumbo_portfolio_value = find_portfolio_pnl(portfolio, initial_capital)
        i = False
    else:
        jumbo_portfolio_value += find_portfolio_pnl(portfolio, initial_capital)
print(jumbo_portfolio_value.stats())

plt.plot(jumbo_portfolio_value, label='Combined Cumulative Money Value', linestyle='--')

plt.title('Cumulative Returns in Money Value')
plt.xlabel('Date')
plt.ylabel('Money Value ($)')
plt.legend()
plt.show()
"""
# Process each symbol independently and store portfolios in a dict

"""
# TO-DO: Figure out how to combine individual portfolios into a single 
            portfolio and plot the combined portfolio's total value.


# Process each symbol independently and store portfolios in a dict
capital = 10000  # Your desired capital
portfolios = {symbol: process_symbol(price_data, symbol, positions_data, capital) for symbol in SYMBOLS}

#All 40 contracts

# Combine portfolios
combined_portfolio = vbt.Portfolio.from_portfolios(list(portfolios.values()), freq='D')

# Print combined stats
print("Combined Portfolio Stats:")
print(combined_portfolio.stats())

# Plot the combined portfolio's total value
combined_portfolio.total_value().vbt.plot().show()
"""