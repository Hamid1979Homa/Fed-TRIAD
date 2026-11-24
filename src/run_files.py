"""File integrity and output locking for the experiment runner."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, content):
    with atomic_output(path) as tmp:
        tmp.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@contextmanager
def directory_lock(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".writer.lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is writing to {path}") from exc
        yield
