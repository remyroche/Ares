## Ares “No-Money” API Testing Plan

This plan verifies all exchange-facing flows used by Ares without risking funds. It leverages PAPER mode for full trading logic simulation and optional TESTNET for sandboxed, signed endpoint smoke tests.

### Scope
- **Public data paths**: ping/time, symbols, order book, trades/aggTrades, klines.
- **Private trading flows (simulated)**: order place/cancel/replace, fills, PnL, risk checks, position sizing, performance reporting.
- **Optional TESTNET**: minimal order lifecycle to validate signed request shapes and error handling against sandbox APIs.

### Safety rails (must follow)
- Set environment to paper mode for core tests:
  - `export TRADING_ENVIRONMENT=PAPER`
- Do not provide live keys while testing. Leave `*_API_KEY`/`*_API_SECRET` empty unless running sandbox TESTNET.
- Keep position sizes small; verify no withdrawal/fiat endpoints are present in runs.

### How modes are wired (for reference)

```startLine:27:endLine:35:src/config/environment.py
trading_environment: Literal["LIVE", "TESTNET", "PAPER"] = Field(
    default="PAPER",
    env="TRADING_ENVIRONMENT",
)
```

```startLine:69:endLine:76:src/supervisor/main.py
if env_settings.trading_environment == "PAPER":
    self.trader = PaperTrader(
        ...
    )
    self.logger.info("Paper Trader initialized for simulation.")
```

```startLine:163:endLine:172:src/tactician/enhanced_order_manager.py
self.paper_trading: bool = bool(config.get("paper_trading", True))
if not self.paper_trading and not self.exchange_client:
    ...
```

### Test matrix
- **Tier 1 – PAPER (no money, default)**
  - Core trading loop, order lifecycle, fills, PnL, risk checks.
  - Public data retrieval used by strategies.
  - Reporting artifacts written to `reports/paper_trading`.
- **Tier 2 – TESTNET (optional, sandbox keys)**
  - Signed endpoint smoke: place/cancel/query small orders on sandbox.
  - Ensures parameter shapes and error paths match expectations.

### Procedures

#### 1) PAPER end-to-end run (primary)
1. Ensure paper mode:
   - `export TRADING_ENVIRONMENT=PAPER`
2. Run a paper session:
   - `python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE`
3. Optional GUI for interactive checks:
   - `python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE`

Expected outcomes:
- No signed live requests; orders simulated internally.
- Trade list, fills, PnL, drawdown produced by `PaperTradingReporter` in `reports/paper_trading`.
- Logs show paper trader initialisation and simulated executions.

#### 2) Public endpoints (data-only)
- Run downloader/backtest steps that use only public APIs (no keys required). Examples:
  - Data loaders under `backtesting/` or training prep steps.
- Verify: responses succeed; data cached as expected; no auth errors.

#### 3) TESTNET smoke (optional)
Only if you want sandbox validation of signed endpoints.
1. Switch to TESTNET:
   - `export TRADING_ENVIRONMENT=TESTNET`
   - Provide sandbox keys for chosen exchange (e.g., Binance):
     - `export BINANCE_API_KEY=...`
     - `export BINANCE_API_SECRET=...`
2. Launch a minimal live pipeline (routes to sandbox in TESTNET):
   - `python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE`

Expected outcomes:
- Signed requests hit sandbox URLs; tiny orders placed/canceled.
- No impact to real funds; confirm order state transitions and error handling.

### Verification checklist
- **Logs**: PAPER mode selected; no production endpoints used. For TESTNET, sandbox base URLs observed; no permission errors with sandbox keys.
- **Trades**: Orders placed/canceled, fills simulated (PAPER) or acknowledged (TESTNET). No withdrawal calls.
- **PnL & Risk**: Realized/unrealized PnL, drawdown limits, position sizing applied.
- **Artifacts**: Reports generated in `reports/paper_trading` (JSON/CSV/HTML as configured).

### Failure handling
- PAPER run requires no keys; if missing-key errors appear, confirm `TRADING_ENVIRONMENT=PAPER` is exported.
- TESTNET auth failures: verify sandbox keys and that testnet toggles are enabled for the exchange.

### CI/CD (optional)
Add a CI job that runs in PAPER to guard trading flows without secrets:
- `export TRADING_ENVIRONMENT=PAPER`
- `python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE`
- Optionally run public data steps to validate data paths.

For a separate TESTNET job (opt-in): inject sandbox secrets; run a short lifecycle smoke with strict size/time limits.

### Deliverables
- Logs proving PAPER simulation coverage and, optionally, TESTNET smoke.
- Paper trading reports under `reports/paper_trading`.
- A short summary of pass/fail for: data retrieval, order lifecycle, PnL/risk checks, reporting.

### Appendix – Useful commands
- PAPER run: `TRADING_ENVIRONMENT=PAPER python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE`
- GUI (paper): `python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE`
- TESTNET live (sandbox):
  - `export TRADING_ENVIRONMENT=TESTNET`
  - `export BINANCE_API_KEY=... && export BINANCE_API_SECRET=...`
  - `python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE`

