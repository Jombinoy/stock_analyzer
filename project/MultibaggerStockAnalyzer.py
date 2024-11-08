import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import os
from tqdm import tqdm  # For progress bar

warnings.filterwarnings('ignore')

class MultibaggerStockAnalyzer:
    def __init__(self, investment_amount: float = 100000, max_workers: int = 10):
        self.investment_amount = investment_amount
        self.today = datetime.now()
        self.start_date = self.today - timedelta(days=365 * 3)
        self.scaler = StandardScaler()
        self.results_df = pd.DataFrame()
        self.max_workers = max_workers

    def get_nifty_500_symbols(self) -> list:
        try:
            nifty_500_df = pd.read_csv('ind_nifty500list.csv')
            nifty_500_df['Symbol'] = nifty_500_df['Symbol'].apply(
                lambda x: f"{x.strip()}.NS" if isinstance(x, str) else x
            )
            return nifty_500_df['Symbol'].tolist()
        except Exception as e:
            print(f"Error loading NIFTY 500 symbols: {str(e)}")
            return []

    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single stock"""
        try:
            stock = yf.Ticker(symbol)
            # Fetch historical data
            df = stock.history(start=self.start_date, end=self.today)

            if df.empty:
                print(f"No historical data for {symbol}.")
                return None

            # Fetch fundamental data
            info = stock.info

            # Add fundamental data to DataFrame
            df['MarketCap'] = info.get('marketCap', np.nan)
            df['Revenue_Growth'] = info.get('revenueGrowth', np.nan)
            df['Earnings_Growth'] = info.get('earningsGrowth', np.nan)
            df['ROE'] = info.get('returnOnEquity', np.nan)
            df['Operating_Margins'] = info.get('operatingMargins', np.nan)
            df['Debt_To_Equity'] = info.get('debtToEquity', np.nan)

            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return None
        try:
            df['Returns_1Y'] = df['Close'].pct_change(periods=252)
            df['Returns_2Y'] = df['Close'].pct_change(periods=504)
            df['Returns_3Y'] = df['Close'].pct_change(periods=756)
            df['Volume_MA50'] = df['Volume'].rolling(window=50).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA50']
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            df['Price_Level'] = df['Close'] / df['Close'].rolling(window=252).max()
            df['Higher_Highs'] = (
                df['High'].rolling(window=20).max() > df['High'].rolling(window=20).max().shift(20)
            )
            return df
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return None

    def analyze_stock(self, symbol: str) -> dict:
        df = self.fetch_stock_data(symbol)
        if df is None or len(df) == 0:
            return None

        df = self.calculate_advanced_indicators(df)
        if df is None:
            return None

        latest = df.iloc[-1]
        multibagger_score = self._calculate_multibagger_score(df, latest)

        analysis = {
            'Symbol': symbol.replace('.NS', ''),
            'Current_Price': latest['Close'],
            'Market_Cap_Cr': latest['MarketCap'] / 10000000 if not pd.isna(latest['MarketCap']) else None,
            'Revenue_Growth': latest['Revenue_Growth'],
            'Earnings_Growth': latest['Earnings_Growth'],
            'ROE': latest['ROE'],
            'Operating_Margins': latest['Operating_Margins'],
            'Debt_To_Equity': latest['Debt_To_Equity'],
            'Returns_1Y': latest['Returns_1Y'],
            'Returns_3Y': latest['Returns_3Y'],
            'Volume_Ratio': latest['Volume_Ratio'],
            'Price_Level': latest['Price_Level'],
            'Volatility': latest['Volatility'],
            'Multibagger_Score': multibagger_score,
            'Potential_Category': self._categorize_potential(multibagger_score)
        }

        # Save individual stock data to CSV
        stock_data_dir = 'stock_data'
        if not os.path.exists(stock_data_dir):
            os.makedirs(stock_data_dir)
        df.to_csv(os.path.join(stock_data_dir, f'{symbol.replace(".NS", "")}_data.csv'))

        return analysis

    def _calculate_multibagger_score(self, df: pd.DataFrame, latest) -> float:
        score = 0
        if not pd.isna(latest['Revenue_Growth']):
            score += min(20, latest['Revenue_Growth'] * 100)
        if not pd.isna(latest['Earnings_Growth']):
            score += min(20, latest['Earnings_Growth'] * 100)
        if not pd.isna(latest['ROE']) and latest['ROE'] > 0.15:
            score += 10
        if not pd.isna(latest['Operating_Margins']) and latest['Operating_Margins'] > 0.15:
            score += 10
        if not pd.isna(latest['Debt_To_Equity']) and latest['Debt_To_Equity'] < 1:
            score += 10
        if not pd.isna(latest['Returns_1Y']) and latest['Returns_1Y'] > 0:
            score += 10
        if not pd.isna(latest['Volume_Ratio']) and latest['Volume_Ratio'] > 1.5:
            score += 10
        if latest['Higher_Highs']:
            score += 10
        return min(100, score)

    def _categorize_potential(self, score: float) -> str:
        if score >= 80:
            return 'Very High Potential'
        elif score >= 60:
            return 'High Potential'
        elif score >= 40:
            return 'Moderate Potential'
        else:
            return 'Low Potential'

    def analyze_all_stocks(self):
        # Create directory for storing individual stock data
        stock_data_dir = 'stock_data'
        if not os.path.exists(stock_data_dir):
            os.makedirs(stock_data_dir)

        results = []
        symbols = self.get_nifty_500_symbols()
        if not symbols:
            print("No symbols to analyze.")
            return self.results_df

        print("Analyzing stocks for multibagger potential...")

        # Using ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a dictionary to map futures to symbols
            future_to_symbol = {executor.submit(self.analyze_stock, symbol): symbol for symbol in symbols}

            # Use tqdm for progress bar
            for future in tqdm(as_completed(future_to_symbol), total=len(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    analysis = future.result()
                    if analysis:
                        results.append(analysis)
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")

        self.results_df = pd.DataFrame(results)
        if not self.results_df.empty:
            self.results_df = self.results_df.sort_values('Multibagger_Score', ascending=False)
        else:
            print("No valid stock analyses were completed.")

        return self.results_df
    def save_results(self, filename: str = 'multibagger_stocks_analysis.csv'):
        if not self.results_df.empty:
            # Remove duplicates based on 'Symbol' column
            self.results_df = self.results_df.drop_duplicates(subset='Symbol', keep='first')
            self.results_df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        else:
            print("No results to save. Please run analyze_all_stocks() first.")


    def visualize_results(self):
        if not self.results_df.empty:
            plt.figure(figsize=(12, 8))
            top_10 = self.results_df.head(10)
            plt.bar(top_10['Symbol'], top_10['Multibagger_Score'], color='teal')
            plt.title("Top 10 Multibagger Potential Stocks")
            plt.xlabel("Stock Symbol")
            plt.ylabel("Multibagger Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No results to visualize. Please run analyze_all_stocks() first.")

# Example Usage
if __name__ == "__main__":
    analyzer = MultibaggerStockAnalyzer(investment_amount=100000, max_workers=10)
    analyzer.analyze_all_stocks()  # Start analysis
    analyzer.save_results()        # Save results to CSV
    analyzer.visualize_results()   # Visualize the top results
