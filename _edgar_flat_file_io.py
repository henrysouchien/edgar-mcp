"""Descriptor-relative reads and atomic writes for source-owned flat files."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import stat


@dataclass(frozen=True, slots=True)
class FlatFileIdentity:
    root_path: str
    root_device: int
    root_inode: int
    filename: str
    file_device: int
    file_inode: int


@dataclass(frozen=True, slots=True)
class FlatFileSnapshot:
    identity: FlatFileIdentity
    content: bytes


def validate_flat_filename(filename: str) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("filename must be a flat contained filename")
    return filename


def _required_os_flag(name: str) -> int:
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise RuntimeError(f"{name} is required for safe flat-file I/O")
    return value


def _open_directory(
    root_path: Path,
    *,
    child_dir: str | None = None,
    create_child: bool = False,
) -> tuple[Path, int, os.stat_result]:
    resolved_root = Path(root_path).resolve(strict=True)
    flags = os.O_RDONLY
    flags |= _required_os_flag("O_DIRECTORY")
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= _required_os_flag("O_NOFOLLOW")
    root_fd = os.open(str(resolved_root), flags)
    if child_dir is None:
        return resolved_root, root_fd, os.fstat(root_fd)

    normalized_child = validate_flat_filename(child_dir)
    try:
        if create_child:
            try:
                os.mkdir(normalized_child, 0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
        child_fd = os.open(normalized_child, flags, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    child_stat = os.fstat(child_fd)
    if not stat.S_ISDIR(child_stat.st_mode):
        os.close(child_fd)
        raise ValueError("child directory is not a directory")
    return resolved_root / normalized_child, child_fd, child_stat


def read_flat_file(
    root_path: Path,
    filename: str,
    *,
    max_bytes: int | None = None,
    child_dir: str | None = None,
) -> FlatFileSnapshot:
    """Read one exact regular, single-link child without following symlinks."""
    normalized_filename = validate_flat_filename(filename)
    resolved_root, root_fd, root_stat = _open_directory(
        root_path,
        child_dir=child_dir,
    )
    file_fd = -1
    try:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= _required_os_flag("O_NOFOLLOW")
        flags |= getattr(os, "O_NONBLOCK", 0)
        file_fd = os.open(normalized_filename, flags, dir_fd=root_fd)
        initial_stat = os.fstat(file_fd)
        if not stat.S_ISREG(initial_stat.st_mode) or initial_stat.st_nlink != 1:
            raise ValueError("file must be a regular single-link file")
        if max_bytes is not None and initial_stat.st_size > max_bytes:
            raise ValueError("file exceeds the maximum supported size")

        read_limit = initial_stat.st_size if max_bytes is None else max_bytes
        chunks: list[bytes] = []
        remaining = read_limit + 1
        while remaining > 0:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        final_stat = os.fstat(file_fd)
        if len(content) > read_limit:
            raise ValueError("file exceeds the maximum supported size")
        if (
            len(content) != initial_stat.st_size
            or final_stat.st_dev != initial_stat.st_dev
            or final_stat.st_ino != initial_stat.st_ino
            or final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
            or final_stat.st_mode != initial_stat.st_mode
            or final_stat.st_nlink != initial_stat.st_nlink
        ):
            raise ValueError("file changed while being read")
        identity = FlatFileIdentity(
            root_path=str(resolved_root),
            root_device=root_stat.st_dev,
            root_inode=root_stat.st_ino,
            filename=normalized_filename,
            file_device=initial_stat.st_dev,
            file_inode=initial_stat.st_ino,
        )
        return FlatFileSnapshot(identity=identity, content=content)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


def atomic_write_flat_file(
    root_path: Path,
    filename: str,
    content: bytes,
    *,
    child_dir: str | None = None,
) -> FlatFileIdentity:
    """Atomically replace one flat child with an O_NOFOLLOW 0600 regular file."""
    normalized_filename = validate_flat_filename(filename)
    resolved_root, root_fd, root_stat = _open_directory(
        root_path,
        child_dir=child_dir,
        create_child=child_dir is not None,
    )
    temp_name = f".{normalized_filename}.{secrets.token_hex(12)}.tmp"
    temp_created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | _required_os_flag("O_NOFOLLOW")
        file_fd = os.open(temp_name, flags, 0o600, dir_fd=root_fd)
        temp_created = True
        try:
            remaining = memoryview(content)
            while remaining:
                written = os.write(file_fd, remaining)
                if written <= 0:
                    raise OSError("failed to write output file")
                remaining = remaining[written:]
            os.fsync(file_fd)
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise ValueError("output is not a regular single-link file")
            if file_stat.st_size != len(content):
                raise ValueError("output size did not match producer bytes")
            identity = FlatFileIdentity(
                root_path=str(resolved_root),
                root_device=root_stat.st_dev,
                root_inode=root_stat.st_ino,
                filename=normalized_filename,
                file_device=file_stat.st_dev,
                file_inode=file_stat.st_ino,
            )
        finally:
            os.close(file_fd)
        os.replace(
            temp_name,
            normalized_filename,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        temp_created = False
        os.fsync(root_fd)
        return identity
    finally:
        if temp_created:
            try:
                os.unlink(temp_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        os.close(root_fd)


__all__ = [
    "FlatFileIdentity",
    "FlatFileSnapshot",
    "atomic_write_flat_file",
    "read_flat_file",
    "validate_flat_filename",
]
