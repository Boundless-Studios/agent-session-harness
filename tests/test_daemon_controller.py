from __future__ import annotations

import socket
import sys
import tempfile
import threading
from pathlib import Path
from shutil import rmtree

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
    for _ in range(100):
        if not server.socket_path.exists():
            break
        threading.Event().wait(0.01)
    assert not server.socket_path.exists()


def test_controller_rejects_changed_definition(controller, tmp_path: Path):
    _, client = controller
    definition = _definition(tmp_path)
    client.request(
        DaemonRequest(operation=DaemonOperation.START, definition=definition)
    )
    changed = definition.model_copy(update={"argv": (sys.executable, "-V")})

    with pytest.raises(RuntimeError, match="definition changed"):
        client.request(
            DaemonRequest(operation=DaemonOperation.STOP, definition=changed)
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
        lambda _path: next(availability),
    )
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller.subprocess.Popen",
        lambda argv, **options: launches.append((argv, options)),
    )

    ensure_controller(tmp_path / "controller.sock", tmp_path / "state")

    assert len(launches) == 1
    argv, options = launches[0]
    assert argv[2:4] == ("agent_session_harness.daemon_controller", "serve")
    assert options["start_new_session"] is True


def test_ensure_controller_does_not_launch_when_available(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller._controller_available",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "agent_session_harness.daemon_controller.subprocess.Popen",
        lambda *_args, **_options: pytest.fail("controller should not launch"),
    )

    ensure_controller(tmp_path / "controller.sock", tmp_path / "state")
