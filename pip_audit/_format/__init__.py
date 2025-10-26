"""Minimal output format module for pip_audit stub."""
from __future__ import annotations

import json
from typing import Dict, Iterable, List

from pip_audit._service.interface import ResolvedDependency, VulnerabilityResult


class JsonFormat:
    """Produce a stable JSON representation of audit results."""

    def __init__(self, output_desc: bool = False, output_aliases: bool = False):
        self.output_desc = output_desc
        self.output_aliases = output_aliases

    def format(
        self,
        result: Dict[ResolvedDependency, List[VulnerabilityResult]],
        _: Iterable[object],
    ) -> str:
        serializable: List[dict[str, object]] = []
        for spec, vulnerabilities in result.items():
            serializable.append(
                {
                    "name": spec.name,
                    "version": getattr(spec, "version", ""),
                    "vulnerabilities": [
                        {
                            "id": vuln.id,
                            "aliases": list(vuln.aliases),
                        }
                        for vuln in vulnerabilities
                    ],
                }
            )
        return json.dumps({"dependencies": serializable})
