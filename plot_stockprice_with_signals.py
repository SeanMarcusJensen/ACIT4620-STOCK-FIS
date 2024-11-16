import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from utils import download_history


# Stock class to fetch data
class Stock:
    def __init__(self, ticker, start=None, end=None, period="1d", interval="1m"):
        self.__ticker = ticker
        self.__data = download_history(
            self.__ticker, start=start, end=end, period=period, interval=interval
        )

    def get_data(self):
        return self.__data


# Load the signal events CSV
signal_events = pd.read_csv("signal_events.csv", parse_dates=["Time"])
print("Signal Events Data Loaded:")
print(signal_events.head())  # Debugging line

# Fetch stock data using the Stock class
AAPL = Stock("AAPL")
stock_data = AAPL.get_data()

# Ensure 'Time' column is present and formatted correctly
if "Time" not in stock_data.columns:
    print("Error: 'Time' column not found in stock data.")
else:
    stock_data["Time"] = pd.to_datetime(stock_data["Time"])  # Ensure datetime format
    stock_data = stock_data.set_index("Time")  # Set 'Time' as the index

# Debugging: Print first few rows of stock_data
print("Stock Data:")
print(stock_data.head())

# Align stock data with signal events
stock_data = stock_data.sort_index()
signal_events = signal_events.sort_values("Time")

# Remove rows with missing or NaN values in the Signal column
signal_events = signal_events.dropna(subset=["Signal"])
print("Filtered Signal Events (NaN removed):")
print(signal_events.head())

# Assign colors and markers for each indicator and signal type
indicator_colors = {
    "RSI": "blue",
    "MACD": "green",
    "SO": "orange",
    "OBV": "red",
}

signal_markers = {
    "Buy": "^",  # Upward triangle for buy
    "Sell": "v",  # Downward triangle for sell
}

# Plot stock prices
plt.figure(figsize=(14, 7))
plt.plot(
    stock_data.index,
    stock_data["Close"],
    label="Stock Price",
    color="black",
    linewidth=1,
)

# Plot buy/sell signals
for _, row in signal_events.iterrows():
    indicator = row["Indicator"]
    signal_type = row["Signal"]
    time = row["Time"]

    # Skip rows with NaN or invalid signal types
    if pd.isna(signal_type) or signal_type not in signal_markers:
        continue

    # Extract price at the signal time
    price_at_signal = (
        stock_data.loc[time, "Close"] if time in stock_data.index else None
    )

    if price_at_signal is not None:
        plt.scatter(
            time,
            price_at_signal,
            color=indicator_colors[indicator],
            marker=signal_markers[signal_type],
            s=100,  # Size of marker
            label=f"{indicator} - {signal_type}"
            if f"{indicator} - {signal_type}"
            not in plt.gca().get_legend_handles_labels()[1]
            else "",
        )

# Beautify plot
plt.title("Stock Prices with Buy/Sell Signals", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.grid(alpha=0.3)
plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
