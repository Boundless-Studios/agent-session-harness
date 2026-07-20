"""Fail-closed project-quiescence probe for host repositories.

This is the reference implementation of the `JsonSafetyCommand` protocol in
`agent_session_harness.safety`: it answers "is this checkout in the middle of
something a fresh-session rotation would corrupt?" using only signals the
harness can verify — a real Git index lock, harness-owned critical-section
leases, and sibling processes still occupying the runtime process group.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Literal, Mapping

from ..safety import ProjectSafetyObservation, ProjectSafetyStatus
from ..secure_files import UnsafePathError, read_private_text


MAX_INPUT_BYTES = 64 * 1024
MAX_GITDIR_BYTES = 4096
MAX_MARKER_BYTES = 8192
MAX_MARKERS = 32
_MARKER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}\Z")
_PROCESS_IDENTITY = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_KEYS = {
    "schema_version",
    "operation",
    "cwd",
    "chain_id",
    "generation",
    "process_group_id",
}
_MARKER_KEYS = {
    "schema_version",
    "name",
    "pid",
    "process_identity",
    "created_at",
}

PidStatus = Literal["alive", "dead", "unknown"]


def _observation(
    status: ProjectSafetyStatus,
    *,
    critical: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> ProjectSafetyObservation:
    return ProjectSafetyObservation(
        status=status,
        active_critical_sections=critical,
        warnings=warnings,
    )


@dataclass(frozen=True)
class CriticalMarker:
    name: str
    pid: int
    created_at: datetime
    process_identity: str


class _DarwinProcessInfo(ctypes.Structure):
    """Subset of macOS ``proc_bsdinfo`` containing process birth time."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def resolve_git_dir(worktree: str | os.PathLike[str]) -> Path:
    """Resolve a normal checkout or linked-worktree `.git` pointer safely."""

    root = Path(worktree).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("worktree is not a directory")
    dot_git = root / ".git"
    metadata = dot_git.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("git metadata path is a symlink")
    if stat.S_ISDIR(metadata.st_mode):
        return dot_git.resolve(strict=True)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("git metadata path is invalid")
    decoded = read_private_text(dot_git, max_bytes=MAX_GITDIR_BYTES).strip()
    prefix = "gitdir:"
    if not decoded.startswith(prefix):
        raise ValueError("worktree gitdir pointer is invalid")
    raw_target = decoded[len(prefix) :].strip()
    if not raw_target or "\x00" in raw_target or "\n" in raw_target:
        raise ValueError("worktree gitdir pointer is invalid")
    target = Path(raw_target).expanduser()
    if not target.is_absolute():
        target = dot_git.parent / target
    resolved = target.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("worktree gitdir target is invalid")
    return resolved


def probe_git_index_lock(git_dir: Path) -> ProjectSafetyObservation:
    """Treat any real Git index lock as an active critical section."""

    lock_path = git_dir / "index.lock"
    try:
        metadata = lock_path.lstat()
    except FileNotFoundError:
        return _observation(ProjectSafetyStatus.QUIESCENT)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("invalid-git-index-lock",),
        )
    return _observation(ProjectSafetyStatus.BUSY, critical=("git-index-lock",))


def read_critical_markers(
    worktree: str | os.PathLike[str],
    *,
    marker_dir: str | os.PathLike[str] | None = None,
    now: datetime | None = None,
) -> ProjectSafetyObservation:
    """Read only bounded, worktree-contained critical-section leases."""

    root = Path(worktree).expanduser().resolve(strict=True)
    candidate = (
        root / ".agent-session-harness" / "critical-sections"
        if marker_dir is None
        else Path(marker_dir).expanduser()
    )
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(root)
        _reject_symlink_components(root, candidate)
    except (OSError, ValueError):
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("unsafe-critical-marker-path",),
        )

    try:
        directory_metadata = candidate.lstat()
    except FileNotFoundError:
        return _observation(ProjectSafetyStatus.QUIESCENT)
    if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(
        directory_metadata.st_mode
    ):
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("unsafe-critical-marker-path",),
        )

    try:
        entries = sorted(candidate.iterdir(), key=lambda path: path.name)
    except OSError:
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("unreadable-critical-markers",),
        )
    if len(entries) > MAX_MARKERS:
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("too-many-critical-markers",),
        )

    active: list[str] = []
    warnings: list[str] = []
    observed_at = now or datetime.now(tz=timezone.utc)
    for path in entries:
        marker = _read_marker(path, observed_at)
        if marker is None:
            return _observation(
                ProjectSafetyStatus.UNKNOWN,
                warnings=("invalid-critical-marker",),
            )
        owner = pid_status(marker.pid)
        if owner == "unknown":
            return _observation(
                ProjectSafetyStatus.UNKNOWN,
                warnings=("critical-marker-owner-unknown",),
            )
        if owner == "dead":
            warnings.append(f"stale-critical-marker:{marker.name}")
            continue
        current_identity = process_identity(marker.pid)
        if current_identity is None:
            return _observation(
                ProjectSafetyStatus.UNKNOWN,
                warnings=("critical-marker-identity-unknown",),
            )
        if current_identity != marker.process_identity:
            warnings.append(f"stale-critical-marker:{marker.name}")
        else:
            active.append(marker.name)

    return _observation(
        ProjectSafetyStatus.BUSY if active else ProjectSafetyStatus.QUIESCENT,
        critical=tuple(active),
        warnings=tuple(warnings),
    )


def probe_project_safety(
    worktree: str | os.PathLike[str],
    *,
    process_group_id: int | None = None,
    marker_dir: str | os.PathLike[str] | None = None,
    now: datetime | None = None,
) -> ProjectSafetyObservation:
    """Merge Git, lease, and process-group probes with unknown precedence."""

    try:
        root = Path(worktree).expanduser().resolve(strict=True)
        git_result = probe_git_index_lock(resolve_git_dir(root))
        marker_result = read_critical_markers(root, marker_dir=marker_dir, now=now)
    except (OSError, ValueError, UnsafePathError):
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("invalid-worktree-state",),
        )

    results = (git_result, marker_result)
    if process_group_id is not None:
        results = (*results, probe_runtime_process_group(process_group_id))
    if any(result.status is ProjectSafetyStatus.UNKNOWN for result in results):
        status = ProjectSafetyStatus.UNKNOWN
    elif any(result.status is ProjectSafetyStatus.BUSY for result in results):
        status = ProjectSafetyStatus.BUSY
    else:
        status = ProjectSafetyStatus.QUIESCENT
    active = tuple(
        dict.fromkeys(
            section for result in results for section in result.active_critical_sections
        )
    )
    warnings = tuple(
        dict.fromkeys(warning for result in results for warning in result.warnings)
    )
    return _observation(status, critical=active, warnings=warnings)


def runtime_process_group_members(process_group_id: int) -> set[int] | None:
    """Read verified runtime-group membership from the installed harness."""

    from ..guardian import verified_process_group_members

    return verified_process_group_members(process_group_id)


def probe_runtime_process_group(process_group_id: int) -> ProjectSafetyObservation:
    """Block rotation while sibling runtime hooks still occupy the group."""

    if isinstance(process_group_id, bool) or process_group_id <= 0:
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("invalid-runtime-process-group",),
        )
    members = runtime_process_group_members(process_group_id)
    if members is None or process_group_id not in members:
        return _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("runtime-process-group-unknown",),
        )
    if members - {process_group_id}:
        return _observation(
            ProjectSafetyStatus.BUSY,
            critical=("runtime-child-processes",),
        )
    return _observation(ProjectSafetyStatus.QUIESCENT)


def pid_status(pid: int) -> PidStatus:
    """Return a tri-state owner check without enumerating unrelated processes."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except (PermissionError, OSError):
        return "unknown"
    return "alive"


def process_identity(pid: int) -> str | None:
    """Bind a PID to its kernel process birth so PID reuse is detectable."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    birth = _kernel_process_birth(pid)
    if birth is None:
        return None
    return hashlib.sha256(f"{pid}:{birth}".encode()).hexdigest()


def _kernel_process_birth(pid: int) -> str | None:
    if sys.platform.startswith("linux"):
        try:
            decoded = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        command_end = decoded.rfind(")")
        if command_end < 0:
            return None
        fields = decoded[command_end + 2 :].split()
        return f"linux:{fields[19]}" if len(fields) > 19 else None
    if sys.platform == "darwin":
        return _darwin_process_birth(pid)
    return None


def _darwin_process_birth(pid: int) -> str | None:
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = library.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        info = _DarwinProcessInfo()
        size = ctypes.sizeof(info)
        result = proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
    except (AttributeError, OSError):
        return None
    if result != size or info.pbi_pid != pid or info.pbi_start_tvsec == 0:
        return None
    return f"darwin:{info.pbi_start_tvsec}:{info.pbi_start_tvusec}"


@dataclass(frozen=True)
class ProjectSafetyProbe:
    """In-process `JsonSafetyCommand` binding for this repository probe."""

    name: str = "project-safety"

    def execute(self, payload: Mapping[str, object]) -> dict[str, object]:
        request = _validate_request(payload)
        observation = probe_project_safety(
            str(request["cwd"]),
            process_group_id=int(request["process_group_id"]),  # type: ignore[arg-type]
        )
        return observation.model_dump(mode="json")


def main(*, stdin_text: str | None = None) -> int:
    """Read one probe request from stdin and write one bounded observation."""

    decoded = stdin_text
    if decoded is None:
        decoded = sys.stdin.read(MAX_INPUT_BYTES + 1)
    if len(decoded.encode("utf-8")) > MAX_INPUT_BYTES:
        observation = _observation(
            ProjectSafetyStatus.UNKNOWN,
            warnings=("invalid-request",),
        )
    else:
        try:
            payload = ProjectSafetyProbe().execute(json.loads(decoded))
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = _observation(
                ProjectSafetyStatus.UNKNOWN,
                warnings=("invalid-request",),
            ).model_dump(mode="json")
        observation = ProjectSafetyObservation.model_validate(payload)
    sys.stdout.write(
        json.dumps(
            observation.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    sys.stdout.write("\n")
    return 0


def _validate_request(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != _REQUEST_KEYS:
        raise ValueError("invalid request shape")
    if payload.get("schema_version") != 1 or payload.get("operation") != "probe":
        raise ValueError("invalid request version or operation")
    cwd = payload.get("cwd")
    chain_id = payload.get("chain_id")
    generation = payload.get("generation")
    process_group_id = payload.get("process_group_id")
    if not isinstance(cwd, str) or not 0 < len(cwd) <= 4096:
        raise ValueError("invalid worktree path")
    if not Path(cwd).is_absolute():
        raise ValueError("worktree path must be absolute")
    if not isinstance(chain_id, str) or not 0 < len(chain_id) <= 160:
        raise ValueError("invalid chain identity")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("invalid generation")
    if (
        isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id <= 0
    ):
        raise ValueError("invalid process group")
    return payload


def _read_marker(path: Path, now: datetime) -> CriticalMarker | None:
    try:
        if path.suffix != ".json":
            return None
        payload = json.loads(read_private_text(path, max_bytes=MAX_MARKER_BYTES))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        UnsafePathError,
    ):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != 2 or set(payload) != _MARKER_KEYS:
        return None
    name = payload.get("name")
    pid = payload.get("pid")
    created_at = payload.get("created_at")
    marker_identity = payload.get("process_identity")
    if not isinstance(name, str) or _MARKER_NAME.fullmatch(name) is None:
        return None
    if name != path.stem:
        return None
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if (
        not isinstance(marker_identity, str)
        or _PROCESS_IDENTITY.fullmatch(marker_identity) is None
    ):
        return None
    if not isinstance(created_at, str) or len(created_at) > 64:
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None or created.utcoffset() is None:
        return None
    if created.astimezone(timezone.utc) > now.astimezone(timezone.utc) + timedelta(
        minutes=5
    ):
        return None
    return CriticalMarker(
        name=name,
        pid=pid,
        created_at=created.astimezone(timezone.utc),
        process_identity=marker_identity,
    )


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    relative = candidate.relative_to(root)
    current = root
    for component in relative.parts:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("critical marker path contains a symlink")
