"""Screener-backed adapter for Indian stock research.

Scrapes https://www.screener.in to extract comprehensive fundamentals, ratios,
peer comparisons, pros/cons analysis, and historical financial statements.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from adapters.base_adapter import BaseStockAdapter

logger = logging.getLogger(__name__)


class ScreenerAdapter(BaseStockAdapter):
    """Scrapes stock fundamentals, statements, and sentiment from Screener.in."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def get_stock_info(self, stock_name: str) -> dict:
        """Fetch and scrape Screener.in for the given stock symbol.

        Returns a dictionary containing:
            - company_name: string
            - about: string description of company business
            - sector: string sector
            - industry: string industry
            - ratios: dict of key ratios (Market Cap, ROCE, ROE, PE, etc.)
            - pros: list of positive points
            - cons: list of negative points
            - tables: dict of financial tables (quarters, profit-loss, balance-sheet, cash-flow, ratios, shareholding)
        """
        # Clean symbol for Screener.in (e.g. RELIANCE.NS -> RELIANCE)
        symbol = stock_name.strip().upper()
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            symbol = symbol.rsplit(".", 1)[0]

        url = f"https://www.screener.in/company/{symbol}/"
        logger.info("Screener scrape start: url=%r", url)
        started = time.perf_counter()

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001 — surface any network / connection errors safely
            logger.warning(
                "Screener request failed for %r after %.0fms: %s",
                symbol,
                (time.perf_counter() - started) * 1000,
                e,
            )
            return {"error": f"Failed to fetch data from Screener.in: {e}"}

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 404:
            logger.info(
                "Screener symbol not found (404): %r after %.0fms",
                symbol,
                elapsed_ms,
            )
            return {"error": f"Symbol '{symbol}' not found on Screener.in"}
        elif response.status_code != 200:
            logger.warning(
                "Screener returned status %d for %r after %.0fms",
                response.status_code,
                symbol,
                elapsed_ms,
            )
            return {"error": f"Screener.in returned unexpected status code: {response.status_code}"}

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            # 1. Company Name
            title_el = soup.find("h1")
            company_name = title_el.text.strip() if title_el else None

            # 2. About section
            about_div = soup.find("div", class_="about")
            about = None
            if about_div:
                p_el = about_div.find("p")
                if p_el:
                    about = p_el.text.strip()

            # 3. Sector and Industry
            sector = None
            industry = None
            peers_sec = soup.find("section", id="peers")
            if peers_sec:
                sector_a = peers_sec.find("a", title="Sector")
                if sector_a:
                    sector = sector_a.text.strip()
                industry_a = peers_sec.find("a", title="Industry")
                if industry_a:
                    industry = industry_a.text.strip()

            # 4. Top Ratios
            ratios = {}
            ratios_ul = soup.find("ul", id="top-ratios")
            if ratios_ul:
                for li in ratios_ul.find_all("li"):
                    name_span = li.find("span", class_="name")
                    val_span = li.find("span", class_="number")
                    if name_span and val_span:
                        name = name_span.text.strip().replace("\n", "").replace("  ", " ")
                        val = val_span.text.strip().replace("\n", "").replace("  ", " ")
                        ratios[name] = val

            # 5. Pros & Cons (Analysis Section)
            pros = []
            cons = []
            analysis_sec = soup.find("section", id="analysis")
            if analysis_sec:
                pros = [li.text.strip() for li in analysis_sec.select(".pros li")]
                cons = [li.text.strip() for li in analysis_sec.select(".cons li")]

            # 6. Dynamic Tables (Quarters, Profit & Loss, Balance Sheet, Cash Flows, Ratios, Shareholding Pattern)
            tables = {}
            for sec in soup.find_all("section"):
                sec_id = sec.get("id")
                if not sec_id or sec_id in ("chart", "analysis", "peers", "documents"):
                    continue

                table_el = sec.find("table", class_="data-table")
                if table_el:
                    thead = table_el.find("thead")
                    tbody = table_el.find("tbody")
                    
                    headers = []
                    if thead:
                        headers = [th.text.strip() for th in thead.find_all("th") if th.text.strip()]
                    
                    rows = {}
                    if tbody:
                        for tr in tbody.find_all("tr"):
                            tds = [td.text.strip() for td in tr.find_all("td")]
                            if tds:
                                row_name = tds[0].replace("\n", "").replace("  ", " ").strip()
                                if row_name.endswith(" +"):
                                    row_name = row_name[:-2].strip()
                                values = [v.strip() for v in tds[1:]]
                                rows[row_name] = values
                    
                    tables[sec_id] = {
                        "headers": headers,
                        "rows": rows,
                    }

            logger.info(
                "Screener scrape ok: symbol=%r elapsed=%.0fms",
                symbol,
                elapsed_ms,
            )

            return {
                "symbol": symbol,
                "company_name": company_name,
                "about": about,
                "sector": sector,
                "industry": industry,
                "ratios": ratios,
                "pros": pros,
                "cons": cons,
                "tables": tables,
            }

        except Exception as e:  # noqa: BLE001 — catch any unexpected DOM parsing errors
            logger.warning("Screener parse failed for %r: %s", symbol, e)
            return {"error": f"Failed to parse data from Screener.in: {e}"}
