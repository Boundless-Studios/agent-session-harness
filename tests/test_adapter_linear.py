from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from agent_session_harness import cli
from agent_session_harness.adapters import linear
from agent_session_harness.adapters.command import sanitize_error

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"
SUCCESS = {
    "ok": True,
    "fingerprint": "a" * 64,
    "retryable": False,
    "error": None,
}


def _request(operation: str = "write") -> dict[str, object]:
    payload = json.loads(
        (FIXTURES / "checkpoint-request-v1.json").read_text(encoding="utf-8")
    )
    payload["operation"] = operation
    return payload


class FakeLinearTransport:
    def __init__(self) -> None:
        self.comments: list[dict[str, str]] = []
        self.payloads: list[dict[str, object]] = []
        self.lose_next_mutation_response = False
        self.page_info_override: dict[str, object] | None = None
        self.cursor_loop = False
        self.response_padding = ""

    def __call__(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        query = str(payload.get("query") or "")
        if "HarnessIssueComments" in query:
            variables = payload.get("variables")
            assert isinstance(variables, dict)
            after = variables.get("after")
            start = 0 if after is None else int(str(after))
            end = min(start + 100, len(self.comments))
            has_next_page = end < len(self.comments)
            page_info: dict[str, object] = {
                "hasNextPage": has_next_page,
                "endCursor": str(end) if has_next_page else None,
            }
            if self.cursor_loop:
                page_info = {"hasNextPage": True, "endCursor": str(start)}
            if self.page_info_override is not None:
                page_info = self.page_info_override
            response: dict[str, object] = {
                "data": {
                    "issue": {
                        "id": "linear-uuid-1",
                        "comments": {
                            "nodes": list(self.comments[start:end]),
                            "pageInfo": page_info,
                        },
                    }
                }
            }
            if self.response_padding:
                response["padding"] = self.response_padding
            return response
        if "HarnessCommentCreate" in query:
            variables = payload.get("variables")
            assert isinstance(variables, dict)
            mutation_input = variables.get("input")
            assert isinstance(mutation_input, dict)
            body = mutation_input.get("body")
            assert isinstance(body, str)
            self.comments.append(
                {"id": f"comment-{len(self.comments) + 1}", "body": body}
            )
            if self.lose_next_mutation_response:
                self.lose_next_mutation_response = False
                raise RuntimeError("Linear mutation response was lost")
            return {
                "data": {
                    "commentCreate": {
                        "success": True,
                        "comment": {"id": self.comments[-1]["id"]},
                    }
                }
            }
        raise AssertionError(f"unexpected GraphQL query: {query}")


def _mutations(transport: FakeLinearTransport) -> list[dict[str, object]]:
    return [
        payload
        for payload in transport.payloads
        if "HarnessCommentCreate" in str(payload["query"])
    ]


def test_write_read_and_acknowledge_are_idempotent() -> None:
    transport = FakeLinearTransport()

    first = linear.handle_request(_request("write"), transport)
    repeated = linear.handle_request(_request("write"), transport)
    read_back = linear.handle_request(_request("read"), transport)
    acknowledged = linear.handle_request(_request("acknowledge"), transport)
    repeated_ack = linear.handle_request(_request("acknowledge"), transport)

    assert first == repeated == read_back == acknowledged == repeated_ack == SUCCESS
    assert len(transport.comments) == 2
    checkpoint, acknowledgement = [comment["body"] for comment in transport.comments]
    assert "agent-session-harness checkpoint" in checkpoint
    assert "agent-session-harness acknowledgement" in acknowledgement
    assert "\n" in checkpoint
    assert "\\n" not in checkpoint
    assert all(
        payload["variables"]["input"]["issueId"] == "linear-uuid-1"
        for payload in _mutations(transport)
    )


def test_retry_after_lost_mutation_finds_checkpoint_on_later_page() -> None:
    transport = FakeLinearTransport()
    transport.comments = [
        {"id": f"comment-{index}", "body": f"historical comment {index}"}
        for index in range(100)
    ]
    transport.lose_next_mutation_response = True

    lost_response = linear.handle_request(_request("write"), transport)
    retried = linear.handle_request(_request("write"), transport)

    assert lost_response["ok"] is False
    assert lost_response["retryable"] is True
    assert retried == SUCCESS
    checkpoint_comments = [
        comment
        for comment in transport.comments
        if "agent-session-harness checkpoint" in comment["body"]
    ]
    assert len(checkpoint_comments) == 1
    query_payloads = [
        payload
        for payload in transport.payloads
        if "HarnessIssueComments" in str(payload["query"])
    ]
    assert any(payload["variables"].get("after") for payload in query_payloads)


@pytest.mark.parametrize("operation", ["read", "acknowledge"])
def test_finds_exact_markers_beyond_first_comment_page(operation: str) -> None:
    request = _request(operation)
    capsule = request["capsule"]
    idempotency_key = request["idempotency_key"]
    assert isinstance(capsule, dict)
    assert isinstance(idempotency_key, str)
    if operation == "acknowledge":
        marker = linear._acknowledgement_marker(capsule, idempotency_key)
    else:
        marker = linear._checkpoint_marker(capsule, idempotency_key)
    transport = FakeLinearTransport()
    transport.comments = [
        {"id": f"comment-{index}", "body": f"historical comment {index}"}
        for index in range(100)
    ] + [{"id": "matching-comment", "body": marker}]

    response = linear.handle_request(request, transport)

    assert response == SUCCESS
    assert _mutations(transport) == []


def test_rejects_malformed_pagination_without_mutating() -> None:
    transport = FakeLinearTransport()
    transport.comments = [{"id": "comment-1", "body": "historical comment"}]
    transport.page_info_override = {"hasNextPage": True, "endCursor": None}

    response = linear.handle_request(_request("write"), transport)

    assert response["ok"] is False
    assert response["retryable"] is False
    assert "pagination" in str(response["error"]).lower()
    assert _mutations(transport) == []


def test_cursor_loop_fails_closed_and_remains_retryable() -> None:
    transport = FakeLinearTransport()
    transport.comments = [{"id": "comment-1", "body": "historical comment"}]
    transport.cursor_loop = True

    response = linear.handle_request(_request("write"), transport)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert "cursor" in str(response["error"]).lower()
    assert _mutations(transport) == []


def test_comment_scan_limit_fails_closed_and_remains_retryable() -> None:
    transport = FakeLinearTransport()
    transport.comments = [
        {"id": f"comment-{index}", "body": f"historical comment {index}"}
        for index in range(1_001)
    ]

    response = linear.handle_request(_request("write"), transport)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert "limit" in str(response["error"]).lower()
    assert _mutations(transport) == []


def test_response_byte_limit_fails_closed_and_remains_retryable() -> None:
    transport = FakeLinearTransport()
    transport.response_padding = "x" * 4_200_000

    response = linear.handle_request(_request("write"), transport)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert "response byte limit" in str(response["error"]).lower()
    assert _mutations(transport) == []


def test_body_byte_limit_fails_closed_and_remains_retryable() -> None:
    transport = FakeLinearTransport()
    transport.comments = [{"id": "comment-1", "body": "x" * 1_100_000}]

    response = linear.handle_request(_request("write"), transport)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert "body byte limit" in str(response["error"]).lower()
    assert _mutations(transport) == []


def test_missing_credentials_or_transport_failure_is_retryable() -> None:
    def unavailable(_payload):
        raise RuntimeError("LINEAR_API_KEY=do-not-leak is unavailable")

    response = linear.handle_request(_request(), unavailable)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert "do-not-leak" not in response["error"]


def test_graphql_validation_errors_are_sanitized() -> None:
    def invalid(_payload):
        return {"errors": [{"message": "OPENAI_API_KEY=do-not-leak"}]}

    response = linear.handle_request(_request(), invalid)

    assert response["ok"] is False
    assert response["retryable"] is False
    assert "do-not-leak" not in response["error"]


def test_requires_an_issue_id() -> None:
    request = _request()
    request["capsule"]["task_ids"].pop("linear")
    transport = FakeLinearTransport()

    response = linear.handle_request(request, transport)

    assert response["ok"] is False
    assert response["retryable"] is False
    assert "Linear issue" in response["error"]
    assert transport.payloads == []


def test_credentials_are_injected_and_loaded_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def load_api_key() -> str:
        calls.append("loaded")
        return "linear-test-key"

    transport = linear.LinearGraphQLTransport(load_api_key=load_api_key)

    assert calls == []
    monkeypatch.setattr(
        "sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(_request()).encode()))
    )
    assert linear.main(fetch_graphql=FakeLinearTransport()) == 0
    assert calls == []
    assert transport.load_api_key is load_api_key


def test_environment_credential_provider_is_the_only_built_in_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_TEST_LINEAR_KEY", raising=False)
    provider = linear.environment_credential_provider("HARNESS_TEST_LINEAR_KEY")

    with pytest.raises(RuntimeError, match="HARNESS_TEST_LINEAR_KEY"):
        provider()

    monkeypatch.setenv("HARNESS_TEST_LINEAR_KEY", "linear-test-key")
    assert provider() == "linear-test-key"


def test_main_keeps_transport_overflow_retryable_and_redacted(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen: list[str] = []

    def oversized(_payload: dict[str, object]) -> dict[str, object]:
        seen.append(load_api_key())
        raise RuntimeError(
            "Linear GraphQL response byte limit exceeded secret=do-not-leak"
        )

    def load_api_key() -> str:
        return "linear-test-key"

    monkeypatch.setattr(
        "sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(_request()).encode()))
    )

    assert linear.main(fetch_graphql=oversized, load_api_key=load_api_key) == 0

    response = json.loads(capsys.readouterr().out)
    assert seen == ["linear-test-key"]
    assert response["ok"] is False
    assert response["retryable"] is True
    assert "do-not-leak" not in response["error"]


def test_adapter_does_not_import_a_host_credential_module() -> None:
    source = Path(linear.__file__).read_text(encoding="utf-8")

    assert "linear_compact_issues" not in source
    assert "import httpx" in source
    assert source.count("import httpx") == 1


def test_diagnostics_use_the_shared_credential_redaction() -> None:
    redacted = sanitize_error("transport failed with AWS_SECRET_ACCESS_KEY=do-not-leak")

    assert "do-not-leak" not in redacted
    assert "credential=[redacted]" in redacted


def test_cli_reports_a_bounded_failure_for_an_invalid_request(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"not json")))

    assert cli.main(["adapter", "linear"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["retryable"] is False
