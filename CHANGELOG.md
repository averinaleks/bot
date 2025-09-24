# Changelog

## Unreleased
- Store TradeManager state in non-executable formats: positions saved as Parquet and returns as JSON.
- Fix chained assignment in DataHandler to avoid pandas FutureWarning.
- Bump Requests dependency to >=2.32.4 to address CVE-2024-47081.
- Update linux-libc-dev to 6.8.0-76.76 to mitigate CVE-2025-21976.
- Update linux-libc-dev to a patched revision to close CVE-2025-21946.
- Replace Ray's commons-lang3 JAR with version 3.18.0 to address vulnerabilities.
- Explicitly upgrade libpam0g to mitigate CVE-2024-10963.
- Bundle upstream PAM fixes for CVE-2024-10041 alongside the CVE-2024-10963
  mitigation.
- Drop direct Ray and MLflow dependencies from default requirements to satisfy
  Trivy (CVE-2023-48022, CVE-2024-37052…37060) and provide a safe in-process
  Ray stub for local execution.
