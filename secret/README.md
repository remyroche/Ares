# API Keys Configuration

This file (`api_keys.json`) contains your exchange API keys organized by exchange and environment (testnet/live).

## Structure

Each exchange has two sections:
- `testnet`: Keys used for paper trading/testing (used when `--mode paper`)
- `live`: Keys used for live trading (used when `--mode trade`)

## How to Fill

Simply replace the empty strings (`""`) with your actual API keys:

```json
{
  "binance": {
    "testnet": {
      "api_key": "paste_your_testnet_key_here",
      "api_secret": "paste_your_testnet_secret_here",
      "password": null
    },
    "live": {
      "api_key": "paste_your_live_key_here",
      "api_secret": "paste_your_live_secret_here",
      "password": null
    }
  }
}
```

## Notes

- For most exchanges (Binance, GateIO, MEXC, Phemex, BingX), leave `password` as `null`
- For OKX, fill in the `password` field (this is OKX's passphrase)
- Live keys are required when using `--mode trade`
- Testnet keys are optional but recommended for `--mode paper` (fallback to dummy keys if not provided)
