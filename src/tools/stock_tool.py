"""
Stock Tool — Fetches real-time and historical stock data via yfinance and Alpha Vantage.
"""
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from langchain_core.tools import BaseTool
from pydantic import Field

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
except ImportError:
    TimeSeries = None
    FundamentalData = None


class StockTool(BaseTool):
    """
    LangChain tool for fetching stock price data, volume, and key financial metrics.

    Tries yfinance first (no API key needed), falls back to Alpha Vantage,
    and returns mock data if neither is available.
    """

    name: str = "stock_data"
    description: str = """
    Fetches stock market data for a given ticker symbol.
    Input should be a stock ticker symbol (e.g., AAPL, GOOGL, TSLA).
    Returns current price, price change, volume, 52-week range, and key metrics.
    Useful for understanding stock performance alongside news sentiment.
    """

    alpha_vantage_key: Optional[str] = Field(default=None)
    days_back: int = Field(default=30)

    def model_post_init(self, __context: Any) -> None:
        """Resolve API key from environment if not provided."""
        if not self.alpha_vantage_key:
            object.__setattr__(self, "alpha_vantage_key", os.getenv("ALPHA_VANTAGE_KEY"))

    def _run(self, query: str) -> str:
        """
        Fetch stock data for the given symbol.

        Args:
            query: Stock ticker symbol (case-insensitive).

        Returns:
            Formatted string with stock data.
        """
        symbol = query.strip().upper()
        logger.info(f"Fetching stock data for {symbol}")

        # Try yfinance first (free, no key needed)
        if yf is not None:
            result = self._fetch_yfinance(symbol)
            if result:
                return result

        # Fall back to Alpha Vantage
        if TimeSeries is not None and self.alpha_vantage_key:
            result = self._fetch_alpha_vantage(symbol)
            if result:
                return result

        # Graceful fallback to mock data
        logger.warning(f"No live data available for {symbol} — returning mock data.")
        return self._mock_data(symbol)

    async def _arun(self, query: str) -> str:
        return self._run(query)

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, symbol: str) -> Optional[str]:
        """Fetch data using yfinance (no API key required)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or info.get("regularMarketPrice") is None:
                return None

            # Historical close prices for period change calculation
            hist = ticker.history(period=f"{self.days_back}d")
            period_start_price = float(hist["Close"].iloc[0]) if not hist.empty else None
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")

            period_change_pct = None
            if period_start_price and current_price:
                period_change_pct = ((current_price - period_start_price) / period_start_price) * 100

            day_change = info.get("regularMarketChangePercent", 0.0)
            volume = info.get("regularMarketVolume", 0)
            avg_volume = info.get("averageVolume", 0)
            volume_ratio = volume / avg_volume if avg_volume else None

            week_52_low = info.get("fiftyTwoWeekLow")
            week_52_high = info.get("fiftyTwoWeekHigh")
            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE")
            beta = info.get("beta")

            lines = [
                f"Stock Data for {symbol} (via yfinance)",
                f"  Current Price      : ${current_price:,.2f}" if current_price else "  Current Price      : N/A",
                f"  Day Change         : {day_change:+.2f}%" if day_change else "  Day Change         : N/A",
            ]
            if period_change_pct is not None:
                lines.append(f"  {self.days_back}-Day Change       : {period_change_pct:+.2f}%")
            if volume:
                lines.append(f"  Volume             : {volume:,}")
            if volume_ratio:
                lines.append(f"  Volume vs Average  : {volume_ratio:.2f}x")
            if week_52_low and week_52_high:
                lines.append(f"  52-Week Range      : ${week_52_low:,.2f} – ${week_52_high:,.2f}")
            if market_cap:
                lines.append(f"  Market Cap         : ${market_cap / 1e9:,.1f}B")
            if pe_ratio:
                lines.append(f"  P/E Ratio (TTM)    : {pe_ratio:.1f}")
            if beta:
                lines.append(f"  Beta               : {beta:.2f}")

            lines.append(f"\n  Data as of: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _fetch_alpha_vantage(self, symbol: str) -> Optional[str]:
        """Fetch data using Alpha Vantage API."""
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format="pandas")
            data, meta = ts.get_daily(symbol=symbol, outputsize="compact")

            if data is None or data.empty:
                return None

            latest = data.iloc[0]
            prev = data.iloc[1] if len(data) > 1 else latest

            close = float(latest["4. close"])
            prev_close = float(prev["4. close"])
            change_pct = ((close - prev_close) / prev_close) * 100
            volume = int(latest["5. volume"])

            period_data = data.head(self.days_back)
            period_start = float(period_data.iloc[-1]["4. close"])
            period_change = ((close - period_start) / period_start) * 100

            return (
                f"Stock Data for {symbol} (via Alpha Vantage)\n"
                f"  Current Price    : ${close:,.2f}\n"
                f"  Day Change       : {change_pct:+.2f}%\n"
                f"  {self.days_back}-Day Change    : {period_change:+.2f}%\n"
                f"  Volume           : {volume:,}\n"
                f"  Data as of: {latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else 'N/A'}"
            )

        except Exception as e:
            logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return None

    def _mock_data(self, symbol: str) -> str:
        """Return realistic-looking mock data for demos / tests."""
        import random
        random.seed(hash(symbol) % 1000)

        price = round(random.uniform(50, 500), 2)
        day_chg = round(random.uniform(-3.5, 3.5), 2)
        period_chg = round(random.uniform(-10, 15), 2)
        volume = random.randint(5_000_000, 80_000_000)

        return (
            f"Stock Data for {symbol} [DEMO — live data unavailable]\n"
            f"  Current Price    : ${price:,.2f}\n"
            f"  Day Change       : {day_chg:+.2f}%\n"
            f"  {self.days_back}-Day Change    : {period_chg:+.2f}%\n"
            f"  Volume           : {volume:,}\n"
            f"  Note: Install yfinance or set ALPHA_VANTAGE_KEY for live data.\n"
        )


class StockScreenerTool(BaseTool):
    """
    Tool for screening stocks based on financial criteria.

    Returns a list of stocks that match specified filters (sector, market cap, etc.)
    using yfinance under the hood.
    """

    name: str = "stock_screener"
    description: str = """
    Screens and compares multiple stocks by key financial metrics.
    Input should be a comma-separated list of stock symbols (e.g., 'AAPL,GOOGL,MSFT').
    Returns a comparison table with price, P/E, market cap, and performance metrics.
    """

    def _run(self, query: str) -> str:
        """
        Compare multiple stock symbols.

        Args:
            query: Comma-separated ticker symbols.

        Returns:
            Formatted comparison table.
        """
        symbols = [s.strip().upper() for s in query.split(",") if s.strip()]
        if not symbols:
            return "No symbols provided."

        logger.info(f"Screening {len(symbols)} symbols: {symbols}")

        rows: List[Dict[str, Any]] = []
        for symbol in symbols[:10]:  # cap at 10 to avoid rate limits
            row = self._get_summary(symbol)
            rows.append(row)

        if not rows:
            return "Could not retrieve data for any of the provided symbols."

        # Format as a table
        header = f"{'Symbol':<8} {'Price':>10} {'Day%':>8} {'MktCap(B)':>12} {'P/E':>8} {'Beta':>7}"
        sep = "─" * len(header)
        lines = [f"Stock Comparison ({datetime.now().strftime('%Y-%m-%d')})", sep, header, sep]

        for row in rows:
            lines.append(
                f"{row['symbol']:<8} "
                f"{row['price']:>10} "
                f"{row['day_change']:>8} "
                f"{row['market_cap']:>12} "
                f"{row['pe']:>8} "
                f"{row['beta']:>7}"
            )

        lines.append(sep)
        return "\n".join(lines)

    async def _arun(self, query: str) -> str:
        return self._run(query)

    def _get_summary(self, symbol: str) -> Dict[str, Any]:
        """Fetch a compact summary row for one symbol."""
        defaults = {
            "symbol": symbol,
            "price": "N/A",
            "day_change": "N/A",
            "market_cap": "N/A",
            "pe": "N/A",
            "beta": "N/A",
        }

        if yf is None:
            return defaults

        try:
            info = yf.Ticker(symbol).info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            day_chg = info.get("regularMarketChangePercent")
            mc = info.get("marketCap")
            pe = info.get("trailingPE")
            beta = info.get("beta")

            return {
                "symbol": symbol,
                "price": f"${price:,.2f}" if price else "N/A",
                "day_change": f"{day_chg:+.2f}%" if day_chg is not None else "N/A",
                "market_cap": f"{mc/1e9:,.1f}" if mc else "N/A",
                "pe": f"{pe:.1f}" if pe else "N/A",
                "beta": f"{beta:.2f}" if beta else "N/A",
            }
        except Exception as e:
            logger.warning(f"Summary fetch failed for {symbol}: {e}")
            return defaults


def create_stock_tools() -> List[BaseTool]:
    """Create and return all stock-related tools."""
    return [StockTool(), StockScreenerTool()]
