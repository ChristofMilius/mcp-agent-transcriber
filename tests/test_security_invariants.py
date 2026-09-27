"""tests/test_security_invariants.py — keep a security dismissal honest.

Dependabot alert #1 (torch, CVE-2025-3000, GHSA-rrmf-rvhw-rf47) is dismissed
as `not_used`. The vulnerability lives in torch.jit.script, and this project only
ever calls torch.cuda.is_available() — so the vulnerable function is not on any
reachable path.

That dismissal is a claim about the code, not a fact about a fixed artifact, and
claims rot silently: the day someone adds a TorchScript path, "not_used" becomes
false and Dependabot stays quiet because it was told to stop looking. This test
is what makes the dismissal earned rather than merely documented.

It covers first-party code only. The same dismissal also asserts that
openai-whisper does not route inference through TorchScript, and nothing here
checks that — uv.lock does, by pinning the exact version, so any upgrade that
changed it shows up as a reviewable diff.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# The TorchScript surface: torch.jit, torch._jit, and anything under them.
_TORCHSCRIPT = re.compile(r"^torch\._?jit(\.|$)")


def _sources() -> list[Path]:
    if not _SRC.is_dir():
        raise AssertionError(
            f"no source tree at {_SRC}. Deliberately not skipped: a guard that "
            "silently no-ops is worse than no guard, because it reads as coverage."
        )
    return sorted(_SRC.rglob("*.py"))


def _dotted(node: ast.AST) -> str | None:
    """Rebuild a dotted name (torch.jit.script) from an attribute chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_no_torchscript_in_first_party_code():
    """CVE-2025-3000 is reachable only through TorchScript; nothing here uses it.

    Walks the AST rather than grepping text, so a comment or docstring explaining
    this very CVE cannot trip the guard — only real code can.
    """
    hits: dict[tuple[str, int], str] = {}

    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                target = _dotted(node)
            elif isinstance(node, ast.ImportFrom):
                target = node.module
            else:
                continue
            if target and _TORCHSCRIPT.match(target):
                hits.setdefault(
                    (str(path.relative_to(_SRC.parent)), node.lineno), target
                )

    assert not hits, (
        "TorchScript has appeared in first-party code, so Dependabot alert #1 "
        "(CVE-2025-3000) can no longer be dismissed as not_used. Re-open the "
        "alert and reassess reachability before this ships:\n  "
        + "\n  ".join(f"{loc}: {name}" for (loc, _), name in sorted(hits.items()))
    )
