import random
import pandas as pd
import yfinance as yf
from typing import List, Tuple


def make_ticker_pool(n: int, ticker_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates a pool of tickers by downloading their stock data using yfinance, ensuring all selected tickers
    have sufficient data within the provided date range. If any tickers fail to download or have insufficient data,
    additional tickers are sampled.

    Args:
        n (int): The number of tickers to be selected and downloaded.
        ticker_list (List[str]): A list of potential ticker symbols to select from.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the DataFrame with adjusted close prices of successfully downloaded tickers,
        and a list of tickers that were successfully processed and have sufficient data.
    """
    # Initialize variables to track selected and error-prone tickers
    selected_tickers = set()
    error_tickers = set()

    while len(selected_tickers) < n:
        # Calculate remaining tickers needed
        remaining_tickers_needed = n - len(selected_tickers)
        # Sample tickers
        temp_tickers = set(random.sample(ticker_list, remaining_tickers_needed))
        # Attempt to download data
        try:
            _df = yf.download(list(temp_tickers))['Adj Close']
            _df.ffill(inplace=True)
            _df.dropna(axis=1, how='any', inplace=True)
            selected_tickers.update(_df.columns)
        except Exception as e:
            # Track tickers causing errors
            error_tickers.update(temp_tickers)

        # Update the list of tickers by removing successfully added and error-prone tickers
        ticker_list = [ticker for ticker in ticker_list if
                       ticker not in selected_tickers and ticker not in error_tickers]

    # Download and concatenate data for final selected tickers
    final_df = yf.download(list(selected_tickers))['Adj Close']

    return final_df, list(selected_tickers - error_tickers)
