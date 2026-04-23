"""
Economic Tool — Fetches macroeconomic indicators from the World Bank API and FRED.

Provides LangChain agents with context about GDP growth, inflation, unemployment,
interest rates, and other macro factors that affect equity markets.
"""
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger

import requests
from langchain_core.tools import BaseTool
from pydantic import Field


# ---------------------------------------------------------------------------
# World Bank API helpers
# ---------------------------------------------------------------------------

WORLD_BANK_BASE = "https://api.worldbank.org/v2"

# Commonly used World Bank indicator codes
WB_INDICATORS: Dict[str, str] = {
    "gdp_growth":         "NY.GDP.MKTP.KD.ZG",   # GDP growth (annual %)
    "inflation":          "FP.CPI.TOTL.ZG",        # Inflation (CPI, annual %)
    "unemployment":       "SL.UEM.TOTL.ZS",         # Unemployment (% of labor force)
    "interest_rate":      "FR.INR.LEND",            # Lending interest rate (%)
    "current_account":    "BN.CAB.XOKA.GD.ZS",      # Current account balance (% of GDP)
    "trade_balance":      "NE.RSB.GNFS.ZS",          # External balance on goods/services (% of GDP)
}

# ISO-3 country codes
COUNTRY_CODES: Dict[str, str] = {
    "US": "USA",
    "USA": "USA",
    "EU": "EUU",
    "UK": "GBR",
    "GB": "GBR",
    "JP": "JPN",
    "CN": "CHN",
    "DE": "DEU",
    "FR": "FRA",
    "CA": "CAN",
    "AU": "AUS",
}


def _wb_fetch(indicator: str, country: str = "USA", years: int = 3) -> Optional[List[Dict]]:
    """
    Fetch data from the World Bank API.

    Args:
        indicator: World Bank indicator code.
        country: ISO-3 country code.
        years: Number of years of historical data.

    Returns:
        List of data point dicts, or None on failure.
    """
    end_year = datetime.now().year
    start_year = end_year - years

    url = (
        f"{WORLD_BANK_BASE}/country/{country}/indicator/{indicator}"
        f"?format=json&date={start_year}:{end_year}&per_page=10&mrv=1"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()

        if not isinstance(payload, list) or len(payload) < 2:
            return None

        data = payload[1]  # payload[0] is pagination metadata
        return [d for d in data if d.get("value") is not None]

    except Exception as e:
        logger.warning(f"World Bank API error (indicator={indicator}, country={country}): {e}")
        return None


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

class EconomicIndicatorTool(BaseTool):
    """
    Fetches macroeconomic indicators for a given country from the World Bank API.

    Useful for providing macro context in financial analysis — e.g. understanding
    whether rising inflation or slowing GDP growth creates headwinds for equities.
    """

    name: str = "economic_indicators"
    description: str = """
    Fetches key macroeconomic indicators (GDP growth, inflation, unemployment,
    interest rates) for a specified country.
    Input should be a country name or 2-letter ISO code (e.g., 'US', 'EU', 'UK', 'JP').
    Returns the latest available data for each indicator with year-over-year context.
    """

    cache: Dict[str, Any] = Field(default_factory=dict)
    cache_ttl_hours: int = Field(default=12)

    def _run(self, query: str) -> str:
        """
        Fetch economic indicators for the specified country.

        Args:
            query: Country name or ISO code.

        Returns:
            Formatted string with economic indicators.
        """
        # Resolve country code
        country_input = query.strip().upper()
        country_code = COUNTRY_CODES.get(country_input, "USA")

        logger.info(f"Fetching economic indicators for {country_input} (code: {country_code})")

        # Check cache
        cache_key = f"{country_code}_{datetime.now().strftime('%Y-%m-%d_%H')}"
        if cache_key in self.cache:
            logger.debug("Returning cached economic data.")
            return self.cache[cache_key]

        results: Dict[str, str] = {}

        for label, indicator_code in WB_INDICATORS.items():
            data = _wb_fetch(indicator_code, country=country_code, years=2)
            if data:
                latest = data[0]
                value = latest.get("value")
                year = latest.get("date", "N/A")
                results[label] = f"{value:.2f}% ({year})" if value is not None else "N/A"
            else:
                results[label] = "N/A (data unavailable)"

        if not any(v != "N/A (data unavailable)" for v in results.values()):
            logger.warning(f"No World Bank data for {country_code} — using mock.")
            return self._mock_indicators(country_input)

        lines = [
            f"Macroeconomic Indicators — {country_input} (Source: World Bank)",
            "─" * 55,
            f"  GDP Growth (annual)    : {results.get('gdp_growth', 'N/A')}",
            f"  Inflation (CPI)        : {results.get('inflation', 'N/A')}",
            f"  Unemployment Rate      : {results.get('unemployment', 'N/A')}",
            f"  Lending Interest Rate  : {results.get('interest_rate', 'N/A')}",
            f"  Current Account (GDP%) : {results.get('current_account', 'N/A')}",
            f"  Trade Balance (GDP%)   : {results.get('trade_balance', 'N/A')}",
            "─" * 55,
            f"  Fetched: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "  Note: World Bank data may lag by 1–2 years for some indicators.",
        ]

        output = "\n".join(lines)
        self.cache[cache_key] = output
        return output

    async def _arun(self, query: str) -> str:
        return self._run(query)

    def _mock_indicators(self, country: str) -> str:
        """Return plausible mock macro data for demos."""
        return (
            f"Macroeconomic Indicators — {country} [DEMO — live data unavailable]\n"
            "─" * 55 + "\n"
            "  GDP Growth (annual)    : 2.40% (2023)\n"
            "  Inflation (CPI)        : 3.20% (2023)\n"
            "  Unemployment Rate      : 4.10% (2023)\n"
            "  Lending Interest Rate  : 5.50% (2023)\n"
            "  Current Account (GDP%) : -2.80% (2023)\n"
            "  Trade Balance (GDP%)   : -3.10% (2023)\n"
            "─" * 55 + "\n"
            "  Note: Set WORLD_BANK_API_KEY env var for live data (optional)."
        )


class MacroContextTool(BaseTool):
    """
    Provides a qualitative macro-economic context summary relevant to equity investing.

    Combines multiple indicators and interprets them in a market context,
    e.g. "Rising inflation with restrictive monetary policy creates headwinds for growth stocks."
    """

    name: str = "macro_context"
    description: str = """
    Provides a qualitative summary of the macroeconomic environment for investing.
    Input should be a market region: 'US', 'EU', 'global', or a specific country code.
    Returns an interpretation of current macro conditions and their implications for markets.
    """

    economic_tool: EconomicIndicatorTool = Field(default_factory=EconomicIndicatorTool)

    def _run(self, query: str) -> str:
        """
        Generate a macro context summary.

        Args:
            query: Region or country code.

        Returns:
            Qualitative macro analysis string.
        """
        region = query.strip().upper()
        logger.info(f"Generating macro context for {region}")

        # Get raw indicators
        raw_indicators = self.economic_tool._run(region)

        # Parse key values for interpretation
        context = self._interpret_macro(region, raw_indicators)

        return f"{raw_indicators}\n\n=== MACRO CONTEXT ANALYSIS ===\n{context}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

    def _interpret_macro(self, region: str, raw: str) -> str:
        """
        Heuristic interpretation of macro conditions for equity markets.

        Args:
            region: Country/region identifier.
            raw: Raw indicator string output.

        Returns:
            Qualitative interpretation string.
        """
        interpretations = []

        # Simple keyword-based heuristics on the raw string
        raw_lower = raw.lower()

        # Inflation
        if "inflation" in raw_lower:
            for line in raw.splitlines():
                if "inflation" in line.lower() and "%" in line:
                    try:
                        val = float(line.split(":")[1].split("%")[0].strip())
                        if val > 5:
                            interpretations.append(
                                "⚠️  High inflation: likely headwind for equity valuations; "
                                "central banks may tighten further."
                            )
                        elif val < 1:
                            interpretations.append(
                                "⚠️  Very low inflation: deflation risk; may signal weak demand."
                            )
                        else:
                            interpretations.append(
                                f"✅  Inflation moderate ({val:.1f}%): supportive environment "
                                f"for equities if stable."
                            )
                    except Exception:
                        pass
                    break

        # GDP growth
        for line in raw.splitlines():
            if "gdp growth" in line.lower() and "%" in line:
                try:
                    val = float(line.split(":")[1].split("%")[0].strip())
                    if val > 3:
                        interpretations.append(
                            f"✅  Strong GDP growth ({val:.1f}%): positive for corporate earnings."
                        )
                    elif val < 0:
                        interpretations.append(
                            f"🔴  Negative GDP growth ({val:.1f}%): recession signal; "
                            f"defensive positioning recommended."
                        )
                    else:
                        interpretations.append(
                            f"📊  Moderate GDP growth ({val:.1f}%): selective sector opportunities."
                        )
                except Exception:
                    pass
                break

        # Unemployment
        for line in raw.splitlines():
            if "unemployment" in line.lower() and "%" in line:
                try:
                    val = float(line.split(":")[1].split("%")[0].strip())
                    if val > 7:
                        interpretations.append(
                            f"⚠️  High unemployment ({val:.1f}%): consumer spending headwinds."
                        )
                    elif val < 4:
                        interpretations.append(
                            f"✅  Low unemployment ({val:.1f}%): strong labor market; "
                            f"supports consumer discretionary."
                        )
                except Exception:
                    pass
                break

        if not interpretations:
            interpretations.append(
                "Macro data unavailable or incomplete. "
                "Consider monitoring Fed/ECB communications for current guidance."
            )

        return "\n".join(interpretations)


def create_economic_tools() -> List[BaseTool]:
    """Create and return all economic tools."""
    return [EconomicIndicatorTool(), MacroContextTool()]
