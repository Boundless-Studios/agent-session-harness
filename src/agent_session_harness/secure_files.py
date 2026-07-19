"""Symlink-safe, private file operations for harness-controlled state."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import stat
from typing import Iterator


class UnsafePathError(RuntimeError):
    """A state path could redirect harness I/O outside its declared location."""


def lexical_absolute(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path without resolving symlinks."""

    expanded = Path(path).expanduser()
    return Path(os.path.abspath(expanded))


def _flags(*values: int) -> int:
    result = 0
    for value in values:
        result |= value
    return result


def _nofollow_flag() -> int:
    return getattr(os, "O_NOFOLLOW", 0)


def _cloexec_flag() -> int:
    return getattr(os, "O_CLOEXEC", 0)


def _directory_flag() -> int:
    return getattr(os, "O_DIRECTORY", 0)


def _ensure_directory_tree(path: Path) -> None:
    absolute = lexical_absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafePathError(f"state parent is a symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnsafePathError(f"state parent is not a directory: {current}")


@contextmanager
def _parent_descriptor(path: Path) -> Iterator[tuple[Path, int]]:
    target = lexical_absolute(path)
    _ensure_directory_tree(target.parent)
    descriptor = os.open(
        target.parent,
        _flags(
            os.O_RDONLY,
            _directory_flag(),
            _nofollow_flag(),
            _cloexec_flag(),
        ),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnsafePathError(f"state parent is not a directory: {target.parent}")
        yield target, descriptor
    finally:
        os.close(descriptor)


def _target_metadata(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise UnsafePathError(f"state target is a symlink: {name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise UnsafePathError(f"state target is not a regular file: {name}")
    return metadata


def private_exists(path: str | os.PathLike[str]) -> bool:
    with _parent_descriptor(Path(path)) as (target, parent_descriptor):
        return _target_metadata(parent_descriptor, target.name) is not None


def _open_private(
    path: Path,
    flags: int,
    *,
    create: bool = False,
    mode: int = 0o600,
) -> int:
    with _parent_descriptor(path) as (target, parent_descriptor):
        _target_metadata(parent_descriptor, target.name)
        open_flags = _flags(flags, _nofollow_flag(), _cloexec_flag())
        if create:
            open_flags |= os.O_CREAT
        descriptor = os.open(
            target.name,
            open_flags,
            mode,
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise UnsafePathError(f"state target is not a regular file: {target}")
            if create or flags & os.O_ACCMODE != os.O_RDONLY:
                os.fchmod(descriptor, mode)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor


def read_private_text(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
    max_bytes: int | None = None,
) -> str:
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("private read byte bound must be positive")
    descriptor = _open_private(Path(path), os.O_RDONLY)
    with os.fdopen(descriptor, "r", encoding=encoding) as handle:
        if max_bytes is None:
            return handle.read()
        encoded = handle.buffer.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            raise ValueError(f"private file exceeds {max_bytes} bytes")
        return encoded.decode(encoding)


def private_file_mode(path: str | os.PathLike[str]) -> int:
    descriptor = _open_private(Path(path), os.O_RDONLY)
    try:
        return stat.S_IMODE(os.fstat(descriptor).st_mode)
    finally:
        os.close(descriptor)


def append_private_text(
    path: str | os.PathLike[str],
    value: str,
    *,
    encoding: str = "utf-8",
) -> None:
    target = lexical_absolute(path)
    existed = private_exists(target)
    descriptor = _open_private(target, os.O_WRONLY | os.O_APPEND, create=True)
    with os.fdopen(descriptor, "a", encoding=encoding) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    if not existed:
        fsync_private_directory(target.parent)


def atomic_write_private_text(
    path: str | os.PathLike[str],
    value: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
) -> None:
    if mode < 0 or mode > 0o777:
        raise ValueError("private file mode must be between 000 and 777")
    with _parent_descriptor(Path(path)) as (target, parent_descriptor):
        _target_metadata(parent_descriptor, target.name)
        temporary_name = f".{target.name}.{secrets.token_hex(12)}.tmp"
        descriptor = os.open(
            temporary_name,
            _flags(
                os.O_WRONLY,
                os.O_CREAT,
                os.O_EXCL,
                _nofollow_flag(),
                _cloexec_flag(),
            ),
            mode,
            dir_fd=parent_descriptor,
        )
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "w", encoding=encoding) as handle:
                descriptor = -1
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def private_unlink(path: str | os.PathLike[str]) -> bool:
    with _parent_descriptor(Path(path)) as (target, parent_descriptor):
        if _target_metadata(parent_descriptor, target.name) is None:
            return False
        os.unlink(target.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        return True


def fsync_private_directory(path: str | os.PathLike[str]) -> None:
    directory = lexical_absolute(path)
    _ensure_directory_tree(directory)
    descriptor = os.open(
        directory,
        _flags(
            os.O_RDONLY,
            _directory_flag(),
            _nofollow_flag(),
            _cloexec_flag(),
        ),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_lock(path: str | os.PathLike[str]) -> Iterator[None]:
    descriptor = _open_private(Path(path), os.O_RDWR, create=True)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError("exclusive file locking is unavailable") from exc
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
