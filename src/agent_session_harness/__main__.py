"""Allow ``python -m agent_session_harness`` to invoke the CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
