# Changelog

## Unreleased
- Store TradeManager state in non-executable formats: positions saved as Parquet and returns as JSON.
- Fix chained assignment in DataHandler to avoid pandas FutureWarning.
- Raise minimum Requests version to 2.32.4 to mitigate CVE-2024-47081.
