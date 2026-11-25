# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite storage for compressed user embeddings."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path

EmbeddingRecord = tuple[bytes, int]


class EmbeddingCache:
    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        self._connection = connection
        self._path = path

    @classmethod
    @contextmanager
    def open(cls, path: Path) -> Iterator[EmbeddingCache]:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "key TEXT PRIMARY KEY, data BLOB NOT NULL CHECK(length(data) = 132), "
                "token_count INTEGER NOT NULL CHECK(token_count >= 0))"
            )
            yield cls(connection, path)

    def _read(self, key: str) -> EmbeddingRecord | None:
        row = self._connection.execute(
            "SELECT data, token_count FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
        if row is not None and (
            not isinstance(row[0], bytes)
            or len(row[0]) != 132
            or not isinstance(row[1], int)
            or row[1] < 0
        ):
            raise ValueError(f"Invalid embedding cache entry in {self._path}")
        return row

    def get_many(self, keys: Iterable[str]) -> dict[str, EmbeddingRecord]:
        return {key: row for key in keys if (row := self._read(key)) is not None}

    def put_many(self, entries: Iterable[tuple[str, bytes, int]]) -> dict[str, EmbeddingRecord]:
        """Commit one batch and return the stored winner for each key."""
        saved = {}
        with self._connection:
            for key, data, count in entries:
                self._connection.execute(
                    "INSERT INTO embeddings VALUES (?, ?, ?) ON CONFLICT(key) DO NOTHING",
                    (key, data, count),
                )
                # Another run may have stored this input while we encoded it.
                row = self._read(key)
                assert row is not None
                saved[key] = row
        return saved
