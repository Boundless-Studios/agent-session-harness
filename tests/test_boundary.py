from __future__ import annotations

import ast
from pathlib import Path

from agent_session_harness import cli

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "agent_session_harness"
FORBIDDEN_IMPORT_ROOTS = {
    "agentic_pr_dash",
    "gaia",
    "gaia_private",
    "worktree_deck",
}
FORBIDDEN_STORAGE_FIELDS = {
    "prompt_text",
    "tool_input",
    "transcript_body",
    "user_samples",
}


def imports_with_roots(
    package_root: Path,
    forbidden_roots: set[str],
) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for path in package_root.rglob("*.py"):
        imported: list[str] = []
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".", 1)[0])
        forbidden = sorted(set(imported) & forbidden_roots)
        if forbidden:
            violations[str(path.relative_to(package_root))] = forbidden
    return violations


def test_package_has_no_host_imports() -> None:
    assert imports_with_roots(PACKAGE_ROOT, FORBIDDEN_IMPORT_ROOTS) == {}


def test_source_does_not_define_raw_prompt_storage_fields() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.rglob("*.py")
    )
    for name in FORBIDDEN_STORAGE_FIELDS:
        assert name not in source


def test_cli_smoke_requires_real_command_dispatch() -> None:
    try:
        exit_code = cli.main(["doctor", "--json"])
    except NotImplementedError:
        exit_code = None
    assert exit_code == 0, "doctor command dispatch is not implemented"
