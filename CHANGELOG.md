# Changelog

## Unreleased
- Store TradeManager state in non-executable formats: positions saved as Parquet and returns as JSON.
- Fix chained assignment in DataHandler to avoid pandas FutureWarning.
- Bump Requests dependency to >=2.32.4 to address CVE-2024-47081.
