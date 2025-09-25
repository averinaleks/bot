"""Update Debian changelog entry for patched PAM packages.

This helper mirrors the inline Python that previously lived
inside the Dockerfile. Keeping it as a standalone module makes
the Dockerfile compatible with older parsers used by Trivy 0.33.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path


ENTRY_BODY = (
    "  * Pull in upstream commits b3020da7 and b7b96362 to harden pam_unix\n"
    "    against CVE-2024-10041 and retain the pam_access nodns fix for\n"
    "    CVE-2024-10963.\n\n"
)

CHANGELOG_PATH = Path("debian/changelog")
ENTRY_TEMPLATE = (
    "pam ({version}) noble; urgency=medium\n\n"
    f"{ENTRY_BODY}"
    " -- Security Bot <security@example.com>  {timestamp}\n\n"
)


def main() -> None:
    original = CHANGELOG_PATH.read_text()

    if ENTRY_BODY in original:
        # The changelog was already updated by a previous run.
        return

    match = re.match(r"pam \(([^)]+)\)", original)
    if not match:
        raise RuntimeError("Unable to determine PAM base version from changelog")

    base_version = match.group(1)
    if "+bot" in base_version:
        target_version = base_version
    else:
        target_version = f"{base_version}+bot1"

    timestamp = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    entry = ENTRY_TEMPLATE.format(version=target_version, timestamp=timestamp)
    CHANGELOG_PATH.write_text(entry + original)


if __name__ == "__main__":
    main()
