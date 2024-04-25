import pandas as pd
import numpy as np
import ticker_list
import random
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt, CLA
from datetime import timedelta, datetime


# Funktionen für Portfolio-Metriken
def calculate_metrics(portfolio_returns, benchmark_returns):
    metrics = {}
    # Max Drawdown
    metrics['max_drawdown'] = np.min(portfolio_returns / np.maximum.accumulate(portfolio_returns) - 1)
    # Durchschnittlicher Gewinn
    metrics['average_return'] = np.mean(portfolio_returns)
    # Volatilität
    metrics['volatility'] = np.std(portfolio_returns)
    # Sharpe-Ratio
    rf_rate = 0.01  # Risikofreie Rate, angenommen als 1%
    metrics['sharpe_ratio'] = (np.mean(portfolio_returns) - rf_rate) / np.std(portfolio_returns)
    # Sortino-Ratio
    negative_returns = portfolio_returns[portfolio_returns < 0]
    metrics['sortino_ratio'] = (np.mean(portfolio_returns) - rf_rate) / np.std(negative_returns)
    # Beta und Alpha
    covariance = np.cov(portfolio_returns, benchmark_returns)
    market_var = np.var(benchmark_returns)
    metrics['beta'] = covariance[0, 1] / market_var
    metrics['alpha'] = np.mean(portfolio_returns) - metrics['beta'] * np.mean(benchmark_returns)
    return metrics


def generate_time_windows(start_year, end_date, granularities, count_per_granularity):
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    windows = []
    durations = {'30 days': 30, '90 days': 90, '200 days': 200, '2 years': 365 * 2, '4 years': 365 * 4}

    for granularity, days in durations.items():
        for _ in range(count_per_granularity):
            start = datetime(start_year, 1, 1) + timedelta(
                days=random.randint(0, (end_datetime - datetime(start_year, 1, 1)).days - days))
            end = start + timedelta(days=days)
            windows.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return windows


N = 1_000  # Anzahl der Experimente für statistische Validität
random.seed(4242)

# Zeitfensterdefinitionen
time_windows = generate_time_windows(2006, "2023-12-31", ['30 days', '90 days', '200 days', '2 years', '4 years'], 10)

results = []

for n in range(N):
    for number_of_selections in [5, 10, 20]:
        tickers = random.sample(ticker_list, number_of_selections)
        for start, end in time_windows:
            data = yf.download(tickers, start=start, end=end)['Adj Close']
            if data.isna().any().any():
                continue  # FIXME: replace nan by last value, then drop nan

            # Berechnung der erwarteten Renditen und Risikomatrix
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)

            # Portfolio-Optimierungsmethoden
            ef = EfficientFrontier(mu, S)
            weights_ef = ef.max_sharpe()
            cleaned_weights_ef = ef.clean_weights()

            hrp = HRPOpt(data)
            weights_hrp = hrp.optimize()
            cleaned_weights_hrp = hrp.clean_weights()

            cla = CLA(mu, S)
            weights_cla = cla.max_sharpe()
            cleaned_weights_cla = cla.clean_weights()

            # Backtesting bis heute
            backtest_start = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
            backtest_end = datetime.today().strftime("%Y-%m-%d")
            backtest_data = yf.download(tickers, start=backtest_start, end=backtest_end)['Adj Close']

            # Berechnung der Metriken und Speicherung der Ergebnisse
            # Dies ist eine vereinfachte Annahme, dass alle Gewichtungen gleich bleiben
            sp500_data = yf.download("^GSPC", start=backtest_start, end=backtest_end)['Adj Close']
            portfolio_returns = np.dot(backtest_data.pct_change().fillna(0), list(cleaned_weights_ef.values()))
            sp500_returns = sp500_data.pct_change().fillna(0)

            metrics = calculate_metrics(portfolio_returns, sp500_returns)
            results.append({
                'start_opt': start,
                'end_opt': end,
                'stock_set_batch': tickers,
                'start_back': backtest_start.strftime("%Y-%m-%d"),
                'end_back': backtest_end,
                'metrics': metrics,
                'num_tickers': len(tickers),
                'opt_days': (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days,
                'backtest_days': (datetime.strptime(backtest_end, "%Y-%m-%d") - backtest_start).days
            })

df = pd.DataFrame(results)
