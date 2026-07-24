"""Mirrored checkpoint adapter for Linear issues.

Credentials are always injected. The adapter never reaches into a host
project's secret store: callers supply a `CredentialProvider` (a zero-argument
callable returning the API key), or a fully built `FetchGraphQL` transport.
The default provider reads a single environment variable, which is the only
mechanism this package is willing to assume.

The HTTP transport needs the `linear` extra (`httpx`); `handle_request` itself
has no network dependency and is fully exercisable with a fake transport.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .checkpoint_records import (
    bounded_line,
    emit_response,
    failure,
    read_stdin_request,
    success,
    validate_checkpoint_request,
)

COMMENT_PAGE_SIZE = 100
MAX_COMMENT_PAGES = 10
MAX_COMMENTS = COMMENT_PAGE_SIZE * MAX_COMMENT_PAGES
MAX_COMMENT_BODY_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4_194_304
MAX_CURSOR_BYTES = 1_024
DEFAULT_ENDPOINT = "https://api.linear.app/graphql"
DEFAULT_CREDENTIAL_VARIABLE = "LINEAR_API_KEY"

FetchGraphQL = Callable[[dict[str, Any]], dict[str, Any]]
CredentialProvider = Callable[[], str]

_ISSUE_COMMENTS_QUERY = """
query HarnessIssueComments($issueId: String!, $after: String) {
  issue(id: $issueId) {
    id
    comments(first: 100, after: $after) {
      nodes { id body }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

_COMMENT_CREATE_MUTATION = """
mutation HarnessCommentCreate($input: CommentCreateInput!) {
  commentCreate(input: $input) {
    success
    comment { id }
  }
}
"""


class GraphQLResponseError(RuntimeError):
    """A permanent, schema-level Linear response failure."""


class PaginationScanError(RuntimeError):
    """A retryable failure to exhaustively scan bounded Linear comments."""


def environment_credential_provider(
    variable: str = DEFAULT_CREDENTIAL_VARIABLE,
) -> CredentialProvider:
    """Return a provider reading the API key from one environment variable."""

    def load() -> str:
        value = os.environ.get(variable)
        if not value:
            raise RuntimeError(f"{variable} is unavailable")
        return value

    return load


@dataclass(frozen=True)
class LinearGraphQLTransport:
    """Bounded HTTPS transport built from an injected credential provider."""

    load_api_key: CredentialProvider = field(
        default_factory=environment_credential_provider
    )
    endpoint: str = DEFAULT_ENDPOINT
    timeout_seconds: float = 20.0
    max_response_bytes: int = MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        if not callable(self.load_api_key):
            raise ValueError("credential provider must be callable")
        if self.timeout_seconds <= 0:
            raise ValueError("transport timeout must be positive")
        if self.max_response_bytes <= 0:
            raise ValueError("transport response bound must be positive")

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "the Linear transport requires the 'linear' extra"
            ) from exc

        api_key = self.load_api_key()
        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST", self.endpoint, json=payload, headers=headers
            ) as response:
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > self.max_response_bytes:
                        raise PaginationScanError(
                            "Linear GraphQL response byte limit exceeded"
                        )
                    chunks.append(chunk)
                status_code = response.status_code
        if status_code >= 500:
            raise PaginationScanError(f"Linear transport failed: {status_code}")
        if status_code >= 400:
            raise GraphQLResponseError(f"Linear transport rejected: {status_code}")
        try:
            decoded = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GraphQLResponseError("Linear returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise GraphQLResponseError("Linear response must be a JSON object")
        return decoded


def handle_request(
    request: Mapping[str, object], fetch_graphql: FetchGraphQL
) -> dict[str, object]:
    """Mirror one checkpoint operation; transport outages remain retryable."""

    try:
        parsed = validate_checkpoint_request(
            request,
            task_id_keys=("linear_issue_id", "linear"),
            task_id_label="Linear issue ID",
        )
        issue = _fetch_issue(fetch_graphql, parsed.task_id)
        marker = _checkpoint_marker(parsed.capsule, parsed.idempotency_key)
        if parsed.operation == "read":
            return (
                success(parsed.fingerprint)
                if _has_marker(issue, marker)
                else failure("Linear checkpoint marker was not found", retryable=True)
            )
        if parsed.operation == "acknowledge":
            marker = _acknowledgement_marker(parsed.capsule, parsed.idempotency_key)
            body = marker
        else:
            body = _checkpoint_body(parsed.capsule, marker)
        if _has_marker(issue, marker):
            return success(parsed.fingerprint)
        _create_comment(fetch_graphql, str(issue["id"]), body)
        verified = _fetch_issue(fetch_graphql, parsed.task_id)
        if not _has_marker(verified, marker):
            return failure("Linear comment read-back failed", retryable=True)
        return success(parsed.fingerprint)
    except GraphQLResponseError as exc:
        return failure(str(exc), retryable=False)
    except (KeyError, TypeError, ValueError) as exc:
        return failure(str(exc), retryable=False)
    except RuntimeError as exc:
        return failure(str(exc), retryable=True)


def _fetch_issue(fetch_graphql: FetchGraphQL, issue_id: str) -> dict[str, Any]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    issue_uuid: str | None = None
    all_comments: list[dict[str, str]] = []
    total_body_bytes = 0
    total_response_bytes = 0

    for page_number in range(MAX_COMMENT_PAGES):
        response = fetch_graphql(
            {
                "query": _ISSUE_COMMENTS_QUERY,
                "variables": {"issueId": issue_id, "after": cursor},
            }
        )
        total_response_bytes += _json_size_bytes(response)
        if total_response_bytes > MAX_RESPONSE_BYTES:
            raise PaginationScanError(
                "Linear issue comments response byte limit exceeded"
            )
        _raise_graphql_errors(response, "issue comments query")
        data = response.get("data")
        issue = data.get("issue") if isinstance(data, dict) else None
        current_issue_uuid = issue.get("id") if isinstance(issue, dict) else None
        if not isinstance(current_issue_uuid, str) or not current_issue_uuid:
            raise GraphQLResponseError("Linear issue was not found")
        if issue_uuid is None:
            issue_uuid = current_issue_uuid
        elif current_issue_uuid != issue_uuid:
            raise GraphQLResponseError("Linear issue pagination response was invalid")

        connection = issue.get("comments")
        if not isinstance(connection, dict):
            raise GraphQLResponseError("Linear issue comments response was invalid")
        nodes = connection.get("nodes")
        page_info = connection.get("pageInfo")
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            raise GraphQLResponseError("Linear issue pagination response was invalid")
        if len(nodes) > COMMENT_PAGE_SIZE:
            raise GraphQLResponseError("Linear issue comments response was invalid")
        if len(all_comments) + len(nodes) > MAX_COMMENTS:
            raise PaginationScanError("Linear issue comment scan limit exceeded")

        for comment in nodes:
            if not isinstance(comment, dict):
                raise GraphQLResponseError("Linear issue comments response was invalid")
            comment_id = comment.get("id")
            body = comment.get("body")
            if not isinstance(comment_id, str) or not isinstance(body, str):
                raise GraphQLResponseError("Linear issue comments response was invalid")
            total_body_bytes += len(body.encode("utf-8"))
            if total_body_bytes > MAX_COMMENT_BODY_BYTES:
                raise PaginationScanError(
                    "Linear issue comment body byte limit exceeded"
                )
            all_comments.append({"id": comment_id, "body": body})

        has_next_page = page_info.get("hasNextPage")
        if not isinstance(has_next_page, bool):
            raise GraphQLResponseError("Linear issue pagination response was invalid")
        if not has_next_page:
            return {"id": issue_uuid, "comments": {"nodes": all_comments}}

        end_cursor = page_info.get("endCursor")
        if (
            not isinstance(end_cursor, str)
            or not end_cursor
            or len(end_cursor.encode("utf-8")) > MAX_CURSOR_BYTES
        ):
            raise GraphQLResponseError("Linear issue pagination response was invalid")
        if end_cursor == cursor or end_cursor in seen_cursors:
            raise PaginationScanError("Linear issue comment cursor loop detected")
        if page_number + 1 >= MAX_COMMENT_PAGES:
            raise PaginationScanError("Linear issue comment scan limit exceeded")
        seen_cursors.add(end_cursor)
        cursor = end_cursor

    raise PaginationScanError("Linear issue comment scan limit exceeded")


def _json_size_bytes(response: object) -> int:
    try:
        encoded = json.dumps(
            response, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GraphQLResponseError(
            "Linear issue comments returned invalid JSON"
        ) from exc
    return len(encoded)


def _create_comment(fetch_graphql: FetchGraphQL, issue_id: str, body: str) -> None:
    response = fetch_graphql(
        {
            "query": _COMMENT_CREATE_MUTATION,
            "variables": {"input": {"issueId": issue_id, "body": body}},
        }
    )
    _raise_graphql_errors(response, "commentCreate mutation")
    payload = response.get("data", {}).get("commentCreate")
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise GraphQLResponseError("Linear commentCreate mutation failed")
    comment = payload.get("comment")
    if not isinstance(comment, dict) or not comment.get("id"):
        raise GraphQLResponseError("Linear commentCreate response was invalid")


def _raise_graphql_errors(response: object, operation: str) -> None:
    if not isinstance(response, dict):
        raise GraphQLResponseError(f"Linear {operation} returned invalid JSON")
    errors = response.get("errors")
    if not errors:
        return
    if isinstance(errors, list):
        messages = [
            str(error.get("message", error)) if isinstance(error, dict) else str(error)
            for error in errors
        ]
    else:
        messages = [str(errors)]
    raise GraphQLResponseError(
        f"Linear GraphQL error during {operation}: {'; '.join(messages)}"
    )


def _has_marker(issue: Mapping[str, object], marker: str) -> bool:
    comments = issue.get("comments")
    nodes = comments.get("nodes") if isinstance(comments, dict) else []
    return any(
        isinstance(comment, dict)
        and isinstance(comment.get("body"), str)
        and marker in comment["body"]
        for comment in nodes
    )


def _checkpoint_marker(capsule: dict[str, Any], idempotency_key: str) -> str:
    return (
        "<!-- agent-session-harness checkpoint "
        f"chain={bounded_line(capsule['chain_id'], 'chain_id', 160)} "
        f"generation={capsule['target_generation']} "
        f"fingerprint={capsule['fingerprint']} "
        f"idempotency={idempotency_key} -->"
    )


def _acknowledgement_marker(capsule: dict[str, Any], idempotency_key: str) -> str:
    return (
        "<!-- agent-session-harness acknowledgement "
        f"chain={bounded_line(capsule['chain_id'], 'chain_id', 160)} "
        f"generation={capsule['target_generation']} "
        f"fingerprint={capsule['fingerprint']} "
        f"idempotency={idempotency_key} -->"
    )


def _checkpoint_body(capsule: dict[str, Any], marker: str) -> str:
    return "\n".join(
        (
            marker,
            "**Agent session handoff checkpoint**",
            "",
            f"- Objective: {bounded_line(capsule['objective'], 'objective', 4000)}",
            "- Next: "
            + bounded_line(capsule["exact_next_action"], "exact_next_action", 4000),
            f"- Head: `{bounded_line(capsule['head'], 'head', 128)}`",
        )
    )


def main(
    *,
    load_api_key: CredentialProvider | None = None,
    fetch_graphql: FetchGraphQL | None = None,
    credential_variable: str = DEFAULT_CREDENTIAL_VARIABLE,
) -> int:
    """Read one checkpoint request from stdin and mirror it into Linear.

    Hosts with their own secret store pass `load_api_key` (or a complete
    `fetch_graphql` transport); nothing here imports host code.
    """

    try:
        request = read_stdin_request()
        transport = fetch_graphql or LinearGraphQLTransport(
            load_api_key=load_api_key
            or environment_credential_provider(credential_variable)
        )
        response = handle_request(request, transport)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        response = failure(str(exc), retryable=False)
    except RuntimeError as exc:
        response = failure(str(exc), retryable=True)
    emit_response(response)
    return 0
