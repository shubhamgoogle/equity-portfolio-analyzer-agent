# Equity Portfolio Analyzer Agent

An agentic equity / stock / ETF / mutual fund portfolio analyzer that triangulates **four independent data sources** in a single tool call and synthesizes an opinionated **BUY / HOLD / SELL / WATCHLIST** verdict. Built on Google's [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) with a swappable LLM via [LiteLLM](https://docs.litellm.ai/).

The agent is designed to be fast and explainable: every number in the final answer carries a source label, and disagreements between sources are surfaced explicitly rather than papered over.

## Why this exists

Most "stock analysis" bots lean on a single data provider and inherit all of its blind spots — stale quotes, missing Indian coverage, no news context, or fundamentals that lag by a quarter. This agent fans out to four sources in parallel and cross-references them in one shot:

| Source       | Best for                                                    |
| ------------ | ----------------------------------------------------------- |
| Yahoo Finance | Global fundamentals, 5y statements, analyst targets, splits |
| Twelve Data   | Cross-check on quote + valuation, free RSI/MACD/EMA/BBands  |
| NSE / BSE     | Live INR quotes, sector P/E, VWAP, circuit limits, ASM/GSM  |
| Tavily        | Live news, regulatory events, sentiment                     |

The architecture and per-field coverage are catalogued in the **[data coverage audit canvas](.cursor/projects/Users-shubham-Documents-github-repos-equity-portfolio-analyzer-agent/canvases/data-coverage-audit.canvas.tsx)** — open it in Cursor to see exactly which of ~115 standard equity-research data points each adapter surfaces.

## Architecture

```
                ┌───────────────────────────────────────────────┐
                │    equity_portfolio_agent.agent.root_agent    │
                │  ADK Agent + LiteLLM(Llama-4-Maverick/Groq)   │
                └────────────────────┬──────────────────────────┘
                                     │ one tool call
                                     ▼
                     analyze_security(symbol: str)
                                     │
        ┌───────────────┬────────────┼────────────┬──────────────┐
        │ (parallel)    │            │            │              │
        ▼               ▼            ▼            ▼              ▼
┌──────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
│ YFinance     │ │ TwelveData │ │ NSE/BSE  │ │ Tavily   │
│ adapter      │ │ adapter    │ │ adapter  │ │ adapter  │
│  • info      │ │  • quote   │ │ nsepython│ │ web      │
│  • 5y stmts  │ │  • stats   │ │ nsetools │ │ search   │
│  • dividends │ │  • profile │ │ bsedata  │ │ + answer │
│  • splits    │ │  • RSI/    │ │   ↳ live │ │          │
│  • recs      │ │    MACD/   │ │     INR  │ │          │
│  • derived   │ │    EMA/    │ │     +VWAP│ │          │
│    ratios    │ │    BBands  │ │  +sector │ │          │
│              │ │  + TTL     │ │   P/E    │ │          │
│              │ │   cache    │ │  +ASM/   │ │          │
│              │ │            │ │   GSM    │ │          │
└──────────────┘ └────────────┘ └──────────┘ └──────────┘
        │              │             │             │
        └──────────────┴─────────────┴─────────────┘
                              │
                              ▼
              One combined JSON payload → LLM
                              │
                              ▼
          Synthesized verdict with cross-source citations
```

Key design choices:

- **One tool, one call.** The LLM gets the full picture in one shot — no multi-turn tool-loop bugs, no forgotten sources.
- **Concurrent fan-out.** All four adapters run on a `ThreadPoolExecutor` with a 45 s per-adapter timeout, so a slow NSE warm-up doesn't block Yahoo.
- **Graceful degradation.** Each adapter wraps every sub-call in try/except and returns `{"error": "..."}` instead of raising. The LLM treats any errored source as a missing source, not a failure.
- **TTL cache on TwelveData.** Repeated lookups for the same symbol within 5 minutes (configurable) are served from memory so you don't burn API credits.
- **Symbol normalization per source.** Yahoo wants `RELIANCE.NS`, NSE wants `RELIANCE`, TwelveData wants the bare ticker — the tool layer rewrites the symbol per adapter automatically.

## Quick start

### Prerequisites
- Python 3.11+ (3.13 verified)
- A few free API keys (see [Configuration](#configuration))

### Install

```bash
git clone <this-repo>
cd equity-portfolio-analyzer-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in the API keys you have
```

### Run

Three ways to interact with the agent:

**Interactive CLI**
```bash
python main.py
# or with a starter query:
python main.py "How is RELIANCE.NS doing?"
```

**ADK web UI** (browser, with chat sidebar and event timeline)
```bash
adk web
# defaults to http://localhost:8000
```

**ADK CLI** (single-shot, scriptable)
```bash
adk run equity_portfolio_agent
```

## Configuration

All configuration lives in `.env`. Copy `.env.example` to get started.

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | **Yes** | Powers the agent itself (Llama 4 Maverick via Groq). Free tier is generous enough for development. Get one at [console.groq.com/keys](https://console.groq.com/keys). |
| `TAVILY_API_KEY` | **Yes** | Live news / sentiment. Free tier covers ~1000 queries/month. Get one at [app.tavily.com](https://app.tavily.com/). |
| `TWELVE_API_KEY` | Optional | Cross-checks Yahoo's valuation + technical indicators. Free Basic plan is enough for US technicals; Indian-exchange coverage needs **Grow** plan or higher (see [Known limitations](#known-limitations--gotchas)). Get one at [twelvedata.com](https://twelvedata.com/account/api-keys). |
| `TWELVE_DATA_CACHE_TTL` | Optional | TTL (seconds) for the TwelveData symbol cache. Default `300` (5 min). Set `0` to disable caching. |
| `GOOGLE_API_KEY` *or* Vertex AI config | Optional | Only needed if you swap the agent's LLM to Gemini (see `equity_portfolio_agent/agent.py`). |

### Swapping the LLM provider

The agent routes through LiteLLM, so changing the model is a one-line edit in `equity_portfolio_agent/agent.py`:

```python
MODEL = LiteLlm(model="groq/meta-llama/llama-4-scout-17b-16e-instruct")
# or:
# MODEL = LiteLlm(model="openai/gpt-4o-mini")
# MODEL = LiteLlm(model="anthropic/claude-3-5-sonnet-latest")
# MODEL = LiteLlm(model="gemini/gemini-2.5-flash")
```

Make sure the matching API-key env var is set. See [LiteLLM's provider list](https://docs.litellm.ai/docs/providers) for the full menu.

## Usage examples

```text
You: How is RELIANCE.NS doing?

Agent: Snapshot
  Reliance Industries Ltd (NSE: RELIANCE / BSE: 500325) — Energy /
  Oil-Gas Refining & Marketing. ₹1364.0 (Yahoo) / ₹1363.6 (NSE).
  Day change −1.7%. F&O-listed, trading status Active, no
  ASM/GSM surveillance flag.

Fundamentals (cross-source)
  • P/E 22.85 (Yahoo) vs symbol-PE 19.65 (NSE) — slightly above
    sector P/E 19.65 (NSE), so trading roughly in line with peers.
  • ROCE 11.5% (Yahoo derived), interest coverage 5.55× —
    comfortable but not striking; capital intensity dragging.
  • Revenue CAGR 6.4% / 3y, profit CAGR 6.6% / 3y, EPS CAGR
    4.3% / 3y (Yahoo extras) — single-digit growth, decelerating.
  • Cash conversion (FCF/NI) 0.86 — healthy.
  • Beta 0.244 — low volatility.

Technicals
  RSI 71 (overbought), MACD bullish (8.2 > signal 6.3, histogram
  positive), price 294 above EMA(20) of 278, near upper Bollinger
  band — extended, vulnerable to a near-term pullback.

Recent context
  • Tavily: Q4 FY26 results beat estimates; retail JioMart
    profitability inflection; analyst targets ₹1696 mean (32 analysts).
  • Analyst trend: 7 strongBuy + 24 buy in last 3 months, no drift.

VERDICT: HOLD
REASONING:
  • Quality is good but not exceptional — ROCE ~11% and growth in
    high-single-digits.
  • Valuation is fair vs sector, not cheap.
  • Technicals are stretched (RSI 71, top of Bollinger).
  • Stronger entry likely on a 5-7% pullback to the EMA(20).

This is not personalized investment advice.
```

The agent accepts a wide variety of symbol formats — it figures out the right one per adapter:

- US:        `AAPL`, `MSFT`, `VFIAX`, `SPY`
- Indian:    `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`
- Yahoo style: `RELIANCE.NS`, `TCS.BO`
- BSE code:  `500325` (Reliance), `532540` (TCS)
- Free-form: `Reliance Industries` (the agent resolves it to `RELIANCE` before calling the tool)

## Data sources & coverage

The audit canvas tracks ~115 standard equity-research data points and labels each as **covered**, **partial / proxy**, or **missing**. Headline numbers after the latest enrichment pass:

| Bucket | Coverage |
| --- | --- |
| **Fundamentals** (valuation, profitability, returns, debt, cash flow, dividends) | Almost entirely covered. ROCE, interest coverage, working capital, capex, investing/financing CF, 3y revenue/profit/EPS CAGRs, dividend CAGR, full split history, FCF/NI cash conversion — all derived from yfinance. |
| **Technicals** | RSI, MACD (+ signal + histogram), EMA(20), Bollinger Bands via TwelveData free tier. SMA-50 and SMA-200 from both Yahoo and TwelveData. |
| **Indian-specific live quote** | VWAP, upper/lower circuit, sector P/E vs symbol P/E, face value, listing date, issued size, F&O availability, ASM/GSM surveillance flag, trading status — all from a single nsepython call. |
| **News & sentiment** | Tavily web search returning 3 results + an LLM-synthesized one-paragraph summary, plus analyst rating history (4 quarters of buy/hold/sell counts) from Yahoo. |
| **Indian shareholding splits** (FII / DII / promoter / retail) | Not covered — no free API. Roughly proxied by Yahoo's `heldPercentInsiders`/`heldPercentInstitutions`. |
| **F&O activity** (open interest, put-call ratio, bulk/block deals) | Not covered — would need separate NSE scraper calls. |
| **Composite scores** (Piotroski, Altman Z, F-score) | Not yet computed but derivable from the new statements block. Likely next addition. |

See the coverage audit canvas in this workspace for the full row-by-row breakdown.

## Known limitations & gotchas

- **TwelveData Indian coverage is plan-gated.** On the free Basic plan, even `/quote` for NSE tickers returns a "Grow plan required" error. The agent handles this gracefully (treats it as a missing source and leans on Yahoo + NSE/BSE for Indian symbols), but you won't get TwelveData's technicals layer for Indian tickers without an upgrade.
- **TwelveData fundamentals tier mismatch.** `/profile` is Grow-tier, `/statistics` is Pro-tier. If your key is on Basic those calls will fail and only quote-derived fields populate. The adapter still works; the `partial_errors` list will show what was skipped.
- **5-year CAGRs aren't computable.** Yahoo's free statement endpoints cap at 5 fiscal years, so 5y CAGRs (which need 6 datapoints) are mathematically impossible. The adapter computes 3y and 4y instead.
- **NSE delivery %, bulk deals, FII/DII flows.** NSE does publish these, but each requires a separate scraper endpoint we haven't wired in. The current NSE adapter sticks to fields available in the single `nse_eq()` response.
- **nsepython prints status notices to stdout.** The library `print()`s messages like "Please use nse_fno() function to reduce latency" — the adapter swallows them with `contextlib.redirect_stdout` so they don't leak into chat output.
- **Groq rate limits.** The free tier has a generous but bounded TPM. If you hit it, swap to `groq/llama-3.3-70b-versatile` or another LiteLLM provider in `equity_portfolio_agent/agent.py`.

## Project layout

```
equity-portfolio-analyzer-agent/
├── adapters/                       # one file per data source
│   ├── base_adapter.py             # BaseStockAdapter ABC
│   ├── yfinance_adapter.py         # Yahoo Finance + 5y statements + derived ratios
│   ├── twelvedata_adapter.py       # TwelveData quote + stats + profile + technicals (TTL-cached)
│   ├── nse_bse_adapter.py          # nsepython → nsetools → bsedata fallback chain
│   └── tavily_adapter.py           # Tavily web search for news / sentiment
├── equity_portfolio_agent/
│   ├── agent.py                    # root_agent: ADK Agent + LiteLLM model + INSTRUCTION
│   └── tools.py                    # analyze_security(): parallel fan-out + symbol normalization
├── main.py                         # interactive CLI driver
├── requirements.txt
├── .env.example
└── README.md
```

## Extending the agent

### Adding a new data source

1. Create `adapters/<source>_adapter.py` implementing `BaseStockAdapter.get_stock_info(name) -> dict`.
2. Register it in the fan-out inside `equity_portfolio_agent/tools.py::analyze_security`.
3. Add a section to the `INSTRUCTION` block in `equity_portfolio_agent/agent.py` describing the new key's shape so the LLM knows how to read it.
4. Add rows to the [coverage audit canvas](.cursor/projects/Users-shubham-Documents-github-repos-equity-portfolio-analyzer-agent/canvases/data-coverage-audit.canvas.tsx) to track what newly went from missing → covered.

### Tweaking the verdict style

The `INSTRUCTION` constant in `equity_portfolio_agent/agent.py` is the single source of truth for how the agent structures its reply (Snapshot → Fundamentals → Technicals → Recent context → Verdict). Edit it to change tone, section ordering, or what gets emphasized.

## License

This project is for personal / research use. Each underlying data API (Yahoo, Twelve Data, NSE/BSE, Tavily, Groq) has its own terms of service — make sure your usage complies.

---

Not personalized investment advice. The agent surfaces public data and a synthesized opinion; final decisions are yours.
