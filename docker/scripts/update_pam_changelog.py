"""Update Debian changelog entry for patched PAM packages.

This helper mirrors the inline Python that previously lived
inside the Dockerfile. Keeping it as a standalone module makes
the Dockerfile compatible with older parsers used by Trivy 0.33.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

CHANGELOG_PATH = Path("debian/changelog")
ENTRY_TEMPLATE = (
    "pam (1.5.3-5ubuntu5.4+bot1) noble; urgency=medium\n\n"
    "  * Apply upstream fix 940747f to address CVE-2024-10963 in pam_access.\n\n"
    " -- Security Bot <security@example.com>  {timestamp}\n\n"
)


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    entry = ENTRY_TEMPLATE.format(timestamp=timestamp)
    CHANGELOG_PATH.write_text(entry + CHANGELOG_PATH.read_text())


if __name__ == "__main__":
    main()
