"""Idempotent guardian bake reporting restricted to the existing BOU-2704 issue."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..guardian_bake_spool import GuardianBakeSpool, GuardianBakeSpoolRecord

TARGET_ISSUE = "BOU-2704"
MAX_COMMENT_PAGES = 10
MAX_COMMENTS = 1000
MAX_COMMENT_BODY_BYTES = 1_048_576

FetchGraphQL = Callable[[dict[str, Any]], dict[str, Any]]

_COMMENTS_QUERY = """
query GuardianBakeComments($issueId: String!, $after: String) {
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
mutation GuardianBakeCommentCreate($input: CommentCreateInput!) {
  commentCreate(input: $input) { success comment { id } }
}
"""


class GuardianBakeLinearSink:
    """Drain durable reports to one statically fixed Linear issue."""

    def __init__(self, fetch_graphql: FetchGraphQL):
        self.fetch_graphql = fetch_graphql

    def drain(
        self,
        spool: GuardianBakeSpool,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        delivered: list[str] = []
        for record in spool.pending():
            issue_id, comments = self._comments()
            marker = _marker(record)
            if not any(marker in body for body in comments):
                self._create_comment(issue_id, _format_report(record))
                _verified_issue, verified_comments = self._comments()
                if not any(marker in body for body in verified_comments):
                    raise RuntimeError("guardian bake Linear read-back failed")
            spool.acknowledge(record.record_id, now=now)
            delivered.append(record.record_id)
        return delivered

    def _comments(self) -> tuple[str, list[str]]:
        cursor: str | None = None
        issue_id: str | None = None
        comments: list[str] = []
        total_body_bytes = 0
        for _page in range(MAX_COMMENT_PAGES):
            response = self.fetch_graphql(
                {
                    "query": _COMMENTS_QUERY,
                    "variables": {"issueId": TARGET_ISSUE, "after": cursor},
                }
            )
            data = response.get("data")
            issue = data.get("issue") if isinstance(data, dict) else None
            current_id = issue.get("id") if isinstance(issue, dict) else None
            connection = issue.get("comments") if isinstance(issue, dict) else None
            if not isinstance(current_id, str) or not isinstance(connection, dict):
                raise RuntimeError("guardian bake Linear response is invalid")
            if issue_id is None:
                issue_id = current_id
            elif issue_id != current_id:
                raise RuntimeError(
                    "guardian bake Linear issue changed during pagination"
                )
            nodes = connection.get("nodes")
            page_info = connection.get("pageInfo")
            if not isinstance(nodes, list) or not isinstance(page_info, dict):
                raise RuntimeError("guardian bake Linear comments response is invalid")
            for node in nodes:
                body = node.get("body") if isinstance(node, dict) else None
                if not isinstance(body, str):
                    raise RuntimeError("guardian bake Linear comment is invalid")
                total_body_bytes += len(body.encode("utf-8"))
                if total_body_bytes > MAX_COMMENT_BODY_BYTES:
                    raise RuntimeError("guardian bake Linear comment bound exceeded")
                comments.append(body)
                if len(comments) > MAX_COMMENTS:
                    raise RuntimeError("guardian bake Linear comment count exceeded")
            has_next = page_info.get("hasNextPage")
            if has_next is False:
                return issue_id, comments
            next_cursor = page_info.get("endCursor")
            if (
                has_next is not True
                or not isinstance(next_cursor, str)
                or not next_cursor
            ):
                raise RuntimeError("guardian bake Linear pagination is invalid")
            if next_cursor == cursor:
                raise RuntimeError("guardian bake Linear pagination did not advance")
            cursor = next_cursor
        raise RuntimeError("guardian bake Linear page bound exceeded")

    def _create_comment(self, issue_id: str, body: str) -> None:
        response = self.fetch_graphql(
            {
                "query": _COMMENT_CREATE_MUTATION,
                "variables": {"input": {"issueId": issue_id, "body": body}},
            }
        )
        data = response.get("data")
        result = data.get("commentCreate") if isinstance(data, dict) else None
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError("guardian bake Linear comment creation failed")


def _marker(record: GuardianBakeSpoolRecord) -> str:
    return f"guardian-bake:{record.report.deduplication_key}"


def _format_report(record: GuardianBakeSpoolRecord) -> str:
    report = record.report
    usage = report.high_water_marks.usage
    resources = report.high_water_marks.resources
    return "\n".join(
        [
            f"<!-- {_marker(record)} -->",
            "### Guardian bake observation",
            f"- Platform/version: `{report.platform}` / `{report.guardian_version}`",
            f"- Seen: {record.repeat_count} time(s)",
            (
                "- Resources: "
                f"{resources.observed} observed, {resources.managed} managed, "
                f"{resources.ambiguous} ambiguous"
            ),
            f"- High-water usage: {usage.memory_bytes} bytes, {usage.cpu_percent:.2f}% CPU",
            f"- Reap decisions: {len(report.reap_decisions)}",
            f"- Refused decisions: {len(report.refused_decisions)}",
            f"- Errors: {len(report.errors)}",
        ]
    )
