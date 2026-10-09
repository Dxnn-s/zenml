#  Copyright (c) ZenML GmbH 2026. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Tests for the database backup engines used during migrations."""

import io
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from zenml.zen_stores.migrations.backup.mydumper import (
    MyDumperDatabaseBackupEngine,
)
from zenml.zen_stores.migrations.backup.sqlalchemy import (
    DBCloneDatabaseBackupEngine,
    FileDatabaseBackupEngine,
)
from zenml.zen_stores.sql_zen_store import SqlZenStoreConfiguration


def _sqlite_config(tmp_path: Path) -> SqlZenStoreConfiguration:
    """Create a store configuration backed by a small sqlite database.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        The store configuration.
    """
    db_path = tmp_path / "zenml.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany("INSERT INTO t (name) VALUES (?)", [("a",), ("b",)])
    backup_directory = tmp_path / "backups"
    backup_directory.mkdir()
    return SqlZenStoreConfiguration(
        url=f"sqlite:///{db_path.as_posix()}",
        backup_directory=str(backup_directory),
    )


def _mysql_config(tmp_path: Path) -> SqlZenStoreConfiguration:
    """Create a MySQL store configuration that is never connected to.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        The store configuration.
    """
    backup_directory = tmp_path / "backups"
    backup_directory.mkdir()
    return SqlZenStoreConfiguration(
        url="mysql://user:secret@db.example.com:3306/zenml",
        backup_directory=str(backup_directory),
    )


def _temp_location(engine: Any) -> str:
    """The temporary location a backup engine writes to before renaming.

    Args:
        engine: The backup engine.

    Returns:
        The temporary location.
    """
    return f"{engine.backup_location}.tmp"


def test_file_backup_sqlite_writes_a_complete_copy(tmp_path: Path) -> None:
    """A sqlite backup is a byte-for-byte copy with no temporary leftovers."""
    config = _sqlite_config(tmp_path)
    engine = FileDatabaseBackupEngine(config)

    engine.backup_database()

    assert config.database is not None
    assert (
        Path(engine.backup_location).read_bytes()
        == Path(config.database).read_bytes()
    )
    assert not os.path.exists(_temp_location(engine))


def test_file_backup_sqlite_failure_leaves_nothing_to_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy that fails halfway leaves no file at the final location."""
    config = _sqlite_config(tmp_path)
    engine = FileDatabaseBackupEngine(config)

    real_copyfile = shutil.copyfile
    destinations: list[str] = []

    def flaky_copyfile(src: str, dst: str, *args: Any, **kwargs: Any) -> Any:
        destinations.append(dst)
        if len(destinations) == 1:
            Path(dst).write_bytes(b"partial")
            raise OSError("disk full")
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", flaky_copyfile)

    with pytest.raises(OSError, match="disk full"):
        engine.backup_database()

    assert not os.path.exists(engine.backup_location)
    assert not os.path.exists(_temp_location(engine))

    # The next backup must not reuse anything from the failed attempt.
    engine.backup_database(overwrite=False)

    assert destinations == [_temp_location(engine)] * 2
    assert config.database is not None
    assert (
        Path(engine.backup_location).read_bytes()
        == Path(config.database).read_bytes()
    )
    assert not os.path.exists(_temp_location(engine))


def test_file_backup_dump_failure_leaves_nothing_to_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON dump that fails halfway leaves no file at the final location."""
    engine = FileDatabaseBackupEngine(_mysql_config(tmp_path))
    dump_targets: list[str] = []

    def flaky_dump(**kwargs: Any) -> None:
        dump_file = kwargs["dump_file"]
        dump_targets.append(dump_file.name)
        dump_file.write('{"table": "t", "data": [')
        if len(dump_targets) == 1:
            raise RuntimeError("connection lost")
        dump_file.write("]}\n")

    monkeypatch.setattr(engine, "backup_database_to_storage", flaky_dump)

    with pytest.raises(RuntimeError, match="connection lost"):
        engine.backup_database()

    assert not os.path.exists(engine.backup_location)
    assert not os.path.exists(_temp_location(engine))

    # The next backup must not reuse anything from the failed attempt.
    engine.backup_database(overwrite=False)

    assert dump_targets == [_temp_location(engine)] * 2
    assert (
        Path(engine.backup_location).read_text()
        == '{"table": "t", "data": []}\n'
    )
    assert not os.path.exists(_temp_location(engine))


def test_file_backup_reuses_a_complete_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing complete backup is still reused when overwrite is off."""
    engine = FileDatabaseBackupEngine(_mysql_config(tmp_path))
    Path(engine.backup_location).write_text("complete\n")
    dump = MagicMock()
    monkeypatch.setattr(engine, "backup_database_to_storage", dump)

    engine.backup_database(overwrite=False)

    dump.assert_not_called()
    assert Path(engine.backup_location).read_text() == "complete\n"


class _FakeMyDumperProcess:
    """Stands in for the mydumper subprocess.

    Writes a file into the requested output directory, like a dump that got
    partway (or all the way) through, and exits with the given return code.
    """

    def __init__(self, cmd: list[str], return_code: int) -> None:
        """Initialize the fake process.

        Args:
            cmd: The mydumper command line.
            return_code: The return code to exit with.
        """
        self.stdout = io.StringIO("")
        self._return_code = return_code
        output_dir = cmd[cmd.index("--outputdir") + 1]
        os.makedirs(output_dir, exist_ok=True)
        Path(output_dir, "zenml.t.00000.sql").write_text("INSERT ...\n")

    def wait(self) -> int:
        """Wait for the process to finish.

        Returns:
            The return code.
        """
        return self._return_code


def _mydumper_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> MyDumperDatabaseBackupEngine:
    """Create a mydumper engine without mydumper being installed.

    Args:
        tmp_path: The pytest temporary directory.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The backup engine.
    """
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    return MyDumperDatabaseBackupEngine(_mysql_config(tmp_path))


def test_mydumper_backup_failure_leaves_nothing_to_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mydumper run that fails leaves no directory at the final location."""
    engine = _mydumper_engine(tmp_path, monkeypatch)
    output_dirs: list[str] = []
    return_codes = iter([1, 0])

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeMyDumperProcess:
        output_dirs.append(cmd[cmd.index("--outputdir") + 1])
        return _FakeMyDumperProcess(cmd, next(return_codes))

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="return code 1"):
        engine.backup_database()

    assert not os.path.exists(engine.backup_location)
    assert not os.path.exists(_temp_location(engine))

    # The next backup must not reuse anything from the failed attempt.
    engine.backup_database(overwrite=False)

    assert output_dirs == [_temp_location(engine)] * 2
    assert os.path.isfile(
        os.path.join(engine.backup_location, "zenml.t.00000.sql")
    )
    assert not os.path.exists(_temp_location(engine))


def test_mydumper_backup_reuses_a_complete_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing complete backup directory is still reused."""
    engine = _mydumper_engine(tmp_path, monkeypatch)
    os.makedirs(engine.backup_location)
    popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", popen)

    engine.backup_database(overwrite=False)

    popen.assert_not_called()


def test_db_clone_backup_drops_partial_database_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone that fails halfway drops the partial backup database."""
    engine = DBCloneDatabaseBackupEngine(
        _mysql_config(tmp_path), location="zenml-backup"
    )
    monkeypatch.setattr(engine, "database_exists", lambda database=None: False)
    monkeypatch.setattr(engine, "create_database", MagicMock())
    monkeypatch.setattr(engine, "create_engine", MagicMock())
    monkeypatch.setattr(
        engine,
        "_copy_database",
        MagicMock(side_effect=RuntimeError("connection lost")),
    )
    cleanup = MagicMock()
    monkeypatch.setattr(engine, "cleanup_database_backup", cleanup)

    with pytest.raises(RuntimeError, match="connection lost"):
        engine.backup_database()

    cleanup.assert_called_once()
