from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
from pathlib import Path

import pytest


def _read_until(fd: int, marker: bytes, *, timeout: float = 8.0) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while marker not in output and time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.1)
        if not readable:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


@pytest.mark.skipif(not hasattr(pty, "fork"), reason="POSIX PTY support required")
def test_guarded_runtime_owns_and_restores_the_interactive_terminal(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "interactive_runtime.py"
    runtime.write_text(
        "import json, os, signal, sys\n"
        "signal.signal(signal.SIGINT, lambda *_: print('RUNTIME_SIGINT', flush=True))\n"
        "signal.signal(signal.SIGCONT, lambda *_: print('RUNTIME_CONTINUED', flush=True))\n"
        "signal.signal(signal.SIGWINCH, lambda *_: print('RUNTIME_RESIZED', flush=True))\n"
        "foreground = os.tcgetpgrp(0) == os.getpgrp()\n"
        "print('RUNTIME_READY', flush=True)\n"
        "line = sys.stdin.readline().strip()\n"
        "print('RUNTIME_RESULT ' + json.dumps({\n"
        "    'foreground': foreground, 'isatty': os.isatty(0), 'line': line\n"
        "}, sort_keys=True), flush=True)\n",
        encoding="utf-8",
    )
    supervisor = tmp_path / "pty_supervisor.py"
    supervisor.write_text(
        "import json, os, pathlib, sys, time\n"
        "from agent_session_harness.process import LaunchRequest, PosixProcessDriver\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "runtime = pathlib.Path(sys.argv[2])\n"
        "driver = PosixProcessDriver(root / 'process-state')\n"
        "managed = driver.start_fresh(LaunchRequest(\n"
        "    runtime='codex', chain_id='interactive-chain', generation=0,\n"
        "    cwd=root, executable=sys.executable, runtime_args=(str(runtime),),\n"
        "))\n"
        "deadline = time.monotonic() + 8\n"
        "while driver.is_alive(managed) and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "status = driver.exit_status(managed)\n"
        "print('SUPERVISOR_DONE ' + json.dumps({\n"
        "    'restored': os.tcgetpgrp(0) == os.getpgrp(),\n"
        "    'return_code': None if status is None else status.return_code,\n"
        "    'reason': None if status is None else status.reason.value,\n"
        "}, sort_keys=True), flush=True)\n"
        "driver.clear_exit_status(managed)\n",
        encoding="utf-8",
    )

    child_pid, master_fd = pty.fork()
    if child_pid == 0:
        os.execve(
            sys.executable,
            [sys.executable, str(supervisor), str(tmp_path), str(runtime)],
            os.environ.copy(),
        )

    waited = False
    try:
        before_input = _read_until(master_fd, b"RUNTIME_READY")
        assert b"RUNTIME_READY" in before_input, before_input.decode(
            "utf-8", errors="replace"
        )
        os.write(master_fd, b"\x03")
        after_interrupt = _read_until(master_fd, b"RUNTIME_SIGINT")
        assert b"RUNTIME_SIGINT" in after_interrupt
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 100, 0, 0))
        after_resize = _read_until(master_fd, b"RUNTIME_RESIZED")
        assert b"RUNTIME_RESIZED" in after_resize
        os.write(master_fd, b"\x1a")
        after_suspend = _read_until(master_fd, b"RUNTIME_CONTINUED")
        assert b"RUNTIME_CONTINUED" in after_suspend
        os.write(master_fd, b"hello-from-pty\n")
        after_input = _read_until(master_fd, b"SUPERVISOR_DONE")
        output = (
            before_input + after_interrupt + after_resize + after_suspend + after_input
        ).decode("utf-8", errors="replace")
        assert '"foreground": true' in output
        assert '"isatty": true' in output
        assert '"line": "hello-from-pty"' in output
        assert '"restored": true' in output
        assert '"return_code": 0' in output
        assert '"reason": "natural"' in output
        waited_pid, status = os.waitpid(child_pid, 0)
        waited = True
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        os.close(master_fd)
        if not waited:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(child_pid, 0)
