import pandas as pd
import sqlalchemy
import vectorbt as vbt
import tomllib as tl
from typing import Dict, Any

# Define your symbols and file paths at the beginning for easy reference
SYMBOLS = ['ES', 'NQ', 'ZN']
CONFIG_FILE_PATH = r'backtesting\configv2.toml'
POSITIONS_FILE_PATH = r'backtesting\optimized_positions.csv'
DATABASE_TABLE_NAME = 'close'


def load_config(config_file_path: str) -> Dict[str, Any]:
    with open(config_file_path, 'rb') as f:
        return tl.load(f)


def create_engine(config: Dict[str, Any]) -> sqlalchemy.engine.Engine:
    db_params = config['database']
    server_params = config['server']
    database_url = (f"postgresql://{server_params['user']}:{server_params['password']}@{server_params['ip']}:"
                    f"{server_params['port']}/{db_params['db_trend']}")
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
    all_dates = pd.date_range(start=latest_start_date, end=price_data.index.max(), freq='D')

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

    return portfolio


# Load configuration and create database engine
config_data = load_config(CONFIG_FILE_PATH)
engine = create_engine(config_data)

# Load and prepare price data
price_data = combine_contract_prices(engine)

print(price_data.head(100))

# Load positions data
positions_data = load_and_prepare_positions(POSITIONS_FILE_PATH)

# Process each symbol independently and store portfolios in a dict
portfolios = {symbol: process_symbol(price_data, symbol, positions_data) for symbol in SYMBOLS}

# Aggregate stats from each portfolio
for symbol, portfolio in portfolios.items():
    print(f"Stats for {symbol}:")
    print(portfolio.stats(), "\n")

# Process each symbol independently and store portfolios in a dict
capital = 10000  # Your desired capital
portfolios = {symbol: process_symbol(price_data, symbol, positions_data, capital) for symbol in SYMBOLS}

# Combine portfolios
combined_portfolio = vbt.Portfolio.from_portfolios(list(portfolios.values()), freq='D')

# Print combined stats
print("Combined Portfolio Stats:")
print(combined_portfolio.stats())

# Plot the combined portfolio's total value
combined_portfolio.total_value().vbt.plot().show()

"""
Figure out how to combine individual portfolios into a single portfolio and plot the combined portfolio's total value.
"""