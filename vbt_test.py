import pandas as pd
import sqlalchemy
from vectorbt import Portfolio
import tomllib as tl
from typing import Dict, Any
import numpy as np
import vectorbt as vbt
import plotly.io as pio


# Function to load the configuration data
def load_config():
    with open('../configv2.toml', 'rb') as f:
        return tl.load(f)


# Function to create a database engine
def create_engine(config):
    db_params = config['database']
    server_params = config['server']
    DATABASE_URL = (f"postgresql://{server_params['user']}:{server_params['password']}@{server_params['ip']}:"
                    f"{server_params['port']}/{db_params['db_test_struct']}")
    return sqlalchemy.create_engine(DATABASE_URL)


# Function to fetch data from the database
def fetch_data(engine, symbols):
    query = f"SELECT * FROM close ORDER BY date"
    df = pd.read_sql(query, engine, index_col='date')
    return df[symbols]


# Function to load and round positions from a CSV file
def load_and_round_positions(file_path):
    positions_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    return positions_df.round()


# Function to generate entries and exits from positions
def generate_entries_exits(positions_df):
    positions_change_df = positions_df.diff().fillna(0)
    entries = positions_change_df > 0
    exits = positions_change_df < 0
    return entries.astype(bool), exits.astype(bool)


# Function to align all dataframes
def align_all(df, entries, exits, trade_sizes):
    aligned_df = df.align(entries, join='inner')[0]
    aligned_entries = aligned_df.align(entries, join='inner')[1]
    aligned_exits = aligned_df.align(exits, join='inner')[1]
    aligned_trade_sizes = aligned_df.align(trade_sizes, join='inner')[1]
    return aligned_df, aligned_entries, aligned_exits, aligned_trade_sizes


# Load configuration data
config_data = load_config()

# Create a database engine
engine = create_engine(config_data)

# Fetch price data
price_data = fetch_data(engine, ['ES', 'MNQ', 'ZN'])

# Load and round positions data
positions_data = load_and_round_positions('../optimized_positions.csv')

# Generate entries and exits
entries, exits = generate_entries_exits(positions_data)

# The size of trades should be the absolute value of the change in positions
trade_sizes = positions_data.diff().abs()

# Align all dataframes
aligned_price_data, aligned_entries, aligned_exits, aligned_trade_sizes = align_all(price_data, entries, exits,
                                                                                    trade_sizes)

# Ensure that all dataframes have the same index and shape
assert aligned_price_data.shape == aligned_entries.shape == aligned_exits.shape == aligned_trade_sizes.shape, "Dataframes are not aligned"

# Create the portfolio from signals
portfolio = Portfolio.from_signals(
    aligned_price_data,
    aligned_entries,
    aligned_exits,
    size=aligned_trade_sizes,
    fees=0.001,
    freq='D'
)

# Plot cumulative returns
cumulative_returns = portfolio.total_return()
cumulative_returns.vbt.plot(title='Cumulative Returns').show()

# Print statistics and records
print(portfolio.stats())
print(portfolio.trades.records_readable.head())
print(portfolio.positions.records_readable.head())
