import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from techical_indicators import RSI, MACD, OBV, StochasticOscillator
from models import Stock, StockParams


def plot_stock_indicators(stock: Stock, indicators: dict) -> None:
    # Create subplots - one for the closing price and one per indicator
    fig, axs = plt.subplots(
        len(indicators) + 1,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})

    fig.suptitle(f"{stock.name} Stock Price and Technical Indicators")

    # Plot the stock closing price in the first subplot
    axs[0].plot(stock['Close'], label="Close Price", color="blue")
    axs[0].set_ylabel("Close Price")
    axs[0].grid()

    for index, (name, indicator) in enumerate(indicators.items()):
        axs[index + 1].plot(indicator, label=name)
        axs[index + 1].set_ylabel(name)
        axs[index + 1].grid()

    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])  # type: ignore

    plt.show()


if __name__ == "__main__":
    STOCK = StockParams \
        .from_command_line() \
        .get_stock()

    indicators = [RSI(), MACD(), OBV(), StochasticOscillator()]

    plot_stock_indicators(STOCK, {n.name: n(STOCK) for n in indicators})
