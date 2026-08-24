from __future__ import annotations

import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from shutil import rmtree
from unittest.mock import Mock

import pytest

from agent_session_harness.daemon_controller import (
    DaemonControllerClient,
    DaemonControllerServer,
    DaemonOperation,
    DaemonRequest,
    ensure_controller,
)
from agent_session_harness.daemon_lifecycle import (
    DaemonDefinition,
    DaemonLifecyclePhase,
)


def _definition(tmp_path: Path) -> DaemonDefinition:
    return DaemonDefinition(
        daemon_key="test-daemon",
        argv=(sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=tmp_path,
    )


@pytest.fixture
def controller():
    root = Path(tempfile.mkdtemp(prefix="ash-ctrl-", dir="/tmp")).resolve()
    server = DaemonControllerServer(
        socket_path=root / "controller.sock",
        state_directory=root / "state",
        startup_probe_seconds=0.01,
        stop_timeout=0.5,
        kill_timeout=0.5,
    )
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.wait_until_ready(timeout=1)
    try:
        yield server, DaemonControllerClient(server.socket_path)
    finally:
        server.close()
        thread.join(timeout=1)
        rmtree(root)


def test_separate_clients_share_controller_ownership(controller, tmp_path: Path):
    server, first_client = controller
    definition = _definition(tmp_path)

    running = first_client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    assert running.record.phase is DaemonLifecyclePhase.RUNNING

    second_client = DaemonControllerClient(server.socket_path)
    stopped = second_client.request(
        DaemonRequest(operation=DaemonOperation.STOP, definition=definition)
    )
    assert stopped.record.phase is DaemonLifecyclePhase.STOPPED
    assert server.socket_path.exists()


def test_controller_adopts_changed_definition_after_failed_start(
    controller, tmp_path: Path
):
    _, client = controller
    definition = _definition(tmp_path)
    failed_definition = definition.model_copy(
        update={"argv": (str(tmp_path / "missing-daemon"),)}
    )
    with pytest.raises(RuntimeError, match="process creation failed"):
        client.request(
            DaemonRequest(
                operation=DaemonOperation.START,
                definition=failed_definition,
            )
        )

    changed = definition.model_copy(
        update={"argv": (sys.executable, "-c", "import time; time.sleep(61)")}
    )
    running = client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=changed)
    )

    assert running.record.phase is DaemonLifecyclePhase.RUNNING
    assert controller[0]._definitions[definition.daemon_key] == changed
    client.request(DaemonRequest(operation=DaemonOperation.STOP, definition=changed))


def test_changed_definition_status_preserves_live_process_identity(
    controller, tmp_path: Path
):
    _, client = controller
    definition = _definition(tmp_path)
    running = client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    changed = definition.model_copy(
        update={"argv": (sys.executable, "-c", "import time; time.sleep(61)")}
    )

    status = client.request(
        DaemonRequest(operation=DaemonOperation.STATUS, definition=changed)
    )

    assert status.record.phase is DaemonLifecyclePhase.RUNNING
    assert status.record.process_identity == running.record.process_identity
    client.request(DaemonRequest(operation=DaemonOperation.STOP, definition=changed))


def test_changed_cwd_status_preserves_live_process_identity(controller, tmp_path: Path):
    _, client = controller
    definition = _definition(tmp_path)
    running = client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    changed_cwd = tmp_path / "changed-cwd"
    changed_cwd.mkdir()
    changed = definition.model_copy(update={"cwd": changed_cwd})

    status = client.request(
        DaemonRequest(operation=DaemonOperation.STATUS, definition=changed)
    )

    assert status.record.phase is DaemonLifecyclePhase.RUNNING
    assert status.record.process_identity == running.record.process_identity
    client.request(DaemonRequest(operation=DaemonOperation.STOP, definition=changed))


def test_changed_definition_retains_failed_supervisor_with_live_process_group(
    controller, tmp_path: Path, monkeypatch
):
    server, client = controller
    definition = _definition(tmp_path)
    running = client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    supervisor = server._supervisors[definition.daemon_key]
    failed = running.record.model_copy(
        update={
            "phase": DaemonLifecyclePhase.FAILED,
            "detail": "daemon leader exited while its process group remains",
        }
    )
    monkeypatch.setattr(supervisor, "status", lambda: failed)
    changed = definition.model_copy(
        update={"argv": (sys.executable, "-c", "import time; time.sleep(61)")}
    )

    drifted = client.request(
        DaemonRequest(operation=DaemonOperation.STATUS, definition=changed)
    )

    assert drifted.record.phase is DaemonLifecyclePhase.FAILED
    assert drifted.record.process_identity == running.record.process_identity
    assert server._definitions[definition.daemon_key] == definition
    assert server._supervisors[definition.daemon_key] is supervisor
    client.request(DaemonRequest(operation=DaemonOperation.STOP, definition=changed))


@pytest.mark.parametrize("operation", [DaemonOperation.START, DaemonOperation.RESTART])
def test_changed_definition_restart_uses_new_definition(
    controller, tmp_path: Path, operation: DaemonOperation
):
    _, client = controller
    definition = _definition(tmp_path)
    client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    changed_cwd = tmp_path / "changed-cwd"
    changed_cwd.mkdir()
    marker = changed_cwd / "started.txt"
    changed = definition.model_copy(
        update={
            "argv": (
                sys.executable,
                "-c",
                "from pathlib import Path; import time; Path('started.txt').touch(); time.sleep(60)",
            ),
            "cwd": changed_cwd,
        }
    )

    restarted = client.request(DaemonRequest(operation=operation, definition=changed))

    assert restarted.record.phase is DaemonLifecyclePhase.RUNNING
    for _ in range(50):
        if marker.exists():
            break
        time.sleep(0.01)
    assert marker.exists()
    assert controller[0]._definitions[definition.daemon_key] == changed
    client.request(DaemonRequest(operation=DaemonOperation.STOP, definition=changed))


def test_changed_definition_restart_restores_old_process_after_failed_start(
    controller, tmp_path: Path
):
    _, client = controller
    definition = _definition(tmp_path)
    running = client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    changed = definition.model_copy(
        update={"argv": (str(tmp_path / "missing-daemon"),)}
    )

    with pytest.raises(RuntimeError, match="process creation failed"):
        client.request(
            DaemonRequest(operation=DaemonOperation.RESTART, definition=changed)
        )

    status = client.request(
        DaemonRequest(operation=DaemonOperation.STATUS, definition=changed)
    )

    assert status.record.phase is DaemonLifecyclePhase.RUNNING
    assert status.record.generation > running.record.generation
    assert controller[0]._definitions[definition.daemon_key] == definition
    client.request(DaemonRequest(operation=DaemonOperation.STOP, definition=changed))


def test_changed_definition_restart_reports_failed_rollback(controller, tmp_path: Path):
    _, client = controller
    old_cwd = tmp_path / "old-cwd"
    old_cwd.mkdir()
    definition = _definition(old_cwd)
    client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    changed = definition.model_copy(
        update={"argv": (str(tmp_path / "missing-daemon"),)}
    )
    old_cwd.rmdir()

    with pytest.raises(
        RuntimeError,
        match="process creation failed.*rollback failed.*DaemonLaunchError",
    ):
        client.request(
            DaemonRequest(operation=DaemonOperation.RESTART, definition=changed)
        )


def test_changed_definition_restart_retains_failed_replacement_with_identity(
    controller, tmp_path: Path, monkeypatch
):
    server, client = controller
    definition = _definition(tmp_path)
    running = client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    old_supervisor = server._supervisors[definition.daemon_key]
    changed = definition.model_copy(
        update={"argv": (sys.executable, "-c", "import time; time.sleep(61)")}
    )
    replacement = Mock()
    replacement.start.side_effect = RuntimeError("replacement launch failed")
    replacement.status.return_value = running.record.model_copy(
        update={
            "phase": DaemonLifecyclePhase.FAILED,
            "detail": "cleanup timed out with descendants alive",
        }
    )
    monkeypatch.setattr(server, "_new_supervisor", lambda _definition: replacement)

    with pytest.raises(RuntimeError, match="replacement launch failed"):
        client.request(
            DaemonRequest(operation=DaemonOperation.RESTART, definition=changed)
        )

    assert server._definitions[definition.daemon_key] == changed
    assert server._supervisors[definition.daemon_key] is replacement
    replacement.status.assert_called_once_with()
    assert old_supervisor.status().phase is DaemonLifecyclePhase.STOPPED


def test_changed_definition_restart_reports_launch_and_recovery_status_errors(
    controller, tmp_path: Path, monkeypatch
):
    server, client = controller
    definition = _definition(tmp_path)
    client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    changed = definition.model_copy(
        update={"argv": (sys.executable, "-c", "import time; time.sleep(61)")}
    )
    replacement = Mock()
    replacement.start.side_effect = RuntimeError("replacement launch failed")
    replacement.status.side_effect = OSError("replacement status failed")
    monkeypatch.setattr(server, "_new_supervisor", lambda _definition: replacement)

    with pytest.raises(
        RuntimeError,
        match="replacement launch failed.*recovery status failed.*OSError.*replacement status failed",
    ):
        client.request(
            DaemonRequest(operation=DaemonOperation.RESTART, definition=changed)
        )


def test_status_from_separate_client_preserves_controller_ownership(
    controller, tmp_path: Path
):
    server, client = controller
    definition = _definition(tmp_path)
    client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )

    status = DaemonControllerClient(server.socket_path).request(
        DaemonRequest(operation=DaemonOperation.STATUS, definition=definition)
    )

    assert status.record.phase is DaemonLifecyclePhase.RUNNING


def test_socket_is_private(controller):
    server, _ = controller
    assert server.socket_path.stat().st_mode & 0o777 == 0o600


def test_second_controller_cannot_replace_live_socket(controller):
    server, _ = controller
    contender = DaemonControllerServer(
        socket_path=server.socket_path,
        state_directory=server.state_directory,
    )

    with pytest.raises(TimeoutError, match="lock is busy"):
        contender.serve()


def test_controller_rejects_public_existing_socket_parent(tmp_path: Path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    server = DaemonControllerServer(
        socket_path=public / "controller.sock",
        state_directory=public / "state",
    )

    with pytest.raises(RuntimeError, match="must be private"):
        server.serve()

    assert public.stat().st_mode & 0o777 == 0o755


def test_disconnected_malformed_client_does_not_stop_controller(
    controller, tmp_path: Path
):
    server, client = controller
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(server.socket_path))
        connection.sendall(b"{")

    response = client.request(
        DaemonRequest(
            operation=DaemonOperation.START,
            definition=_definition(tmp_path),
        )
    )
    assert response.record.phase is DaemonLifecyclePhase.RUNNING


def test_request_payload_is_bounded(tmp_path: Path):
    client = DaemonControllerClient(tmp_path / "missing.sock", max_message_bytes=32)
    request = DaemonRequest(
        operation=DaemonOperation.START,
        definition=_definition(tmp_path),
    )

    with pytest.raises(ValueError, match="exceeds"):
        client.request(request)


def test_ensure_controller_launches_detached_server(monkeypatch, tmp_path: Path):
    availability = iter((False, False, True))
    launches = []
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller._controller_available",
        lambda _path, _state: next(availability),
    )
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller.subprocess.Popen",
        lambda argv, **options: launches.append((argv, options)),
    )

    ensure_controller(tmp_path / "controller.sock", tmp_path / "state")

    assert len(launches) == 1
    argv, options = launches[0]
    assert argv[2:4] == ["agent_session_harness.daemon_controller", "serve"]
    assert options["start_new_session"] is True


def test_ensure_controller_passes_takeover_to_detached_server(
    monkeypatch, tmp_path: Path
):
    availability = iter((False, True))
    launches = []
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller._controller_available",
        lambda _path, _state: next(availability),
    )
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller.subprocess.Popen",
        lambda argv, **options: launches.append((argv, options)),
    )

    ensure_controller(
        tmp_path / "controller.sock",
        tmp_path / "state",
        allow_process_takeover=True,
    )

    assert "--takeover" in launches[0][0]


def test_ensure_controller_does_not_launch_when_available(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller._controller_available",
        lambda _path, _state: True,
    )
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller.subprocess.Popen",
        lambda *_args, **_options: pytest.fail("controller should not launch"),
    )

    ensure_controller(tmp_path / "controller.sock", tmp_path / "state")


def test_ensure_controller_rejects_live_state_directory_mismatch(controller, tmp_path):
    server, _ = controller

    with pytest.raises(RuntimeError, match="state directory mismatch"):
        ensure_controller(server.socket_path, tmp_path / "different-state")
