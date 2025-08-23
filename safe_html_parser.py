"""Safe HTML parser that mitigates CVE-2025-6069 by limiting input size.

This module provides :class:`SafeHTMLParser`, a drop-in replacement for
:class:`html.parser.HTMLParser` that rejects excessively large inputs. The
standard HTMLParser suffered from quadratic time complexity when processing
malformed documents. By capping the amount of data fed to the parser we avoid
potential denial-of-service attacks arising from this behavior.
"""

from __future__ import annotations

from html.parser import HTMLParser

DEFAULT_MAX_FEED = 1_000_000  # 1 MB in bytes


class SafeHTMLParser(HTMLParser):
    """HTMLParser subclass with a maximum cumulative feed size.

    Parameters
    ----------
    max_feed_size: int, optional
        Maximum total size in **bytes** of data that can be fed to the parser
        before a :class:`ValueError` is raised. Must be a positive integer and
        defaults to ``DEFAULT_MAX_FEED``.

    Notes
    -----
    Limiting the amount of data fed to the parser avoids the quadratic
    complexity issue described in CVE-2025-6069.
    """

    def __init__(self, *args, max_feed_size: int = DEFAULT_MAX_FEED, **kwargs):
        if max_feed_size <= 0:
            raise ValueError("max_feed_size must be positive")
        super().__init__(*args, **kwargs)
        self._max_feed_size = max_feed_size
        self._fed = 0

    @property
    def fed_bytes(self) -> int:
        return self._fed

    def feed(self, data: str) -> None:  # type: ignore[override]
        """Feed data to the parser, enforcing ``max_feed_size``.

        ``max_feed_size`` is interpreted as a limit on the number of **bytes**
        received, measured using UTF-8 encoding. This ensures multi-byte
        characters are accounted for correctly.

        Parameters
        ----------
        data: str
            Chunk of HTML data to process.
        """
        byte_length = len(data.encode("utf-8"))
        self._fed += byte_length
        if self._fed > self._max_feed_size:
            raise ValueError("HTML input exceeds maximum allowed size")
        super().feed(data)

    def close(self) -> None:  # type: ignore[override]
        """Finalize parsing and reset internal byte counter.

        ``HTMLParser.close`` resets the parser's internal state so that it can
        be reused. Without overriding it, :class:`SafeHTMLParser` would keep the
        previous ``_fed`` value and immediately raise ``ValueError`` on
        subsequent parses. Resetting the counter here ensures each new parse
        starts fresh.
        """
        super().close()
        self._fed = 0
