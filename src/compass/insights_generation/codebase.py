# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only, repository-confined tools for code-aware Problem validation."""

import base64
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from threading import Timer
from typing import Annotated

from nooa.agentdoc import hidden, spec
from nooa.skill import Skill

_BLOCKED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_BLOCKED_FILENAMES = frozenset({".npmrc", ".pypirc", "credentials.json"})
_BLOCKED_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_MAX_FILE_BYTES = 1_000_000
_MAX_LISTED_FILES = 200
_MAX_MATCHES = 100
_MAX_READ_LINES = 400
_SEARCH_TIMEOUT_SECONDS = 30


class CodebaseTools(Skill):
    """Search and read files inside one codebase without modifying it."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        if not self._root.is_dir():
            raise ValueError(f"code_base must be an existing directory: {root}")

    @hidden
    def _is_blocked(self, path: Path) -> bool:
        relative = path.relative_to(self._root)
        if any(part in _BLOCKED_DIRECTORIES for part in relative.parts):
            return True
        name = path.name.lower()
        return (
            name == ".env"
            or name.startswith(".env.")
            or name in _BLOCKED_FILENAMES
            or path.suffix.lower() in _BLOCKED_SUFFIXES
        )

    @hidden
    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"path escapes the configured codebase: {path}")
        if self._is_blocked(resolved):
            raise ValueError(f"path is not available to code validation: {path}")
        return resolved

    @hidden
    def _run_rg(self, args: list[str], scope: str = ".") -> Iterator[bytes]:
        executable = shutil.which("rg")
        if executable is None:
            raise RuntimeError("Code search requires ripgrep (rg) on PATH; install ripgrep.")
        exclusions = [
            arg for name in sorted(_BLOCKED_DIRECTORIES) for arg in ("--glob", f"!**/{name}/**")
        ]
        for name in [
            ".env",
            ".env.*",
            *sorted(_BLOCKED_FILENAMES),
            *(f"*{suffix}" for suffix in sorted(_BLOCKED_SUFFIXES)),
        ]:
            exclusions.extend(("--iglob", f"!**/{name}"))
        # Ignore repository/user ignore files deliberately: preserve Compass coverage.
        command = [
            executable,
            "--no-config",
            "--hidden",
            "--no-ignore",
            *args,
            *exclusions,
            "--",
            scope,
        ]
        with (
            tempfile.TemporaryFile() as errors,
            subprocess.Popen(
                command, cwd=self._root, stdout=subprocess.PIPE, stderr=errors
            ) as process,
        ):
            timer = Timer(_SEARCH_TIMEOUT_SECONDS, process.kill)
            timer.start()
            try:
                assert process.stdout is not None
                stream = process.stdout
                pending = b""
                separator = b"\0" if "--null" in args else b"\n"
                for block in iter(lambda: stream.read(65536), b""):
                    pieces = (pending + block).split(separator)
                    pending = pieces.pop()
                    yield from pieces
                if pending:
                    yield pending
                if process.wait() not in (0, 1):
                    errors.seek(0)
                    raise RuntimeError(
                        "ripgrep failed or exceeded 30 seconds: "
                        + errors.read(2000).decode("utf-8", errors="replace")
                    )
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                process.wait()

    @hidden
    def _read_text(self, path: Path) -> str | None:
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                return None
            with path.open("rb") as source:
                data = source.read(_MAX_FILE_BYTES + 1)
        except OSError:
            return None
        if len(data) > _MAX_FILE_BYTES or b"\0" in data:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def list_files(
        self,
        pattern: Annotated[
            str,
            spec(description="Optional repository-relative glob, such as src/**/*.py"),
        ] = "**/*",
    ) -> list[str]:
        """List up to 200 paths; narrow the glob if 200 return. Discovery skips symlinks."""

        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("pattern must stay inside the codebase")
        matches = []
        with closing(self._run_rg(["--files", "--null", "--glob", pattern])) as lines:
            for line in lines:
                path = line.decode("utf-8", errors="surrogateescape")
                try:
                    resolved = self._resolve(path)
                except ValueError:
                    continue
                matches.append(str(resolved.relative_to(self._root)))
                if len(matches) == _MAX_LISTED_FILES:
                    break
        return sorted(matches)

    def search_code(
        self,
        query: Annotated[str, spec(description="Literal text to search for")],
        path: Annotated[
            str,
            spec(description="Repository-relative file or directory to search"),
        ] = ".",
    ) -> list[dict[str, str | int]]:
        """Search UTF-8 files with ripgrep's case-insensitive literal matching.

        Scope to a known file/directory first. At most 100 matching lines return;
        narrow the path/query if the limit is reached. Ignored/hidden source is included.
        Regex syntax is literal. Each directory search can use ripgrep worker threads.
        """

        if not query:
            raise ValueError("query must not be empty")
        if query.splitlines() != [query]:
            return []
        scope = self._resolve(path)
        if not scope.exists():
            raise FileNotFoundError(path)
        if scope.is_file() and scope.stat().st_size > _MAX_FILE_BYTES:
            return []
        matches: list[dict[str, str | int]] = []
        checked: dict[str, bool] = {}
        args = [
            "--json",
            "--fixed-strings",
            "--ignore-case",
            "--encoding",
            "utf-8",
            "--max-filesize",
            str(_MAX_FILE_BYTES),
            "--regexp",
            query,
        ]
        with closing(self._run_rg(args, str(scope.relative_to(self._root)))) as lines:
            for line in lines:
                event = json.loads(line)
                if event["type"] != "match":
                    continue
                data = event["data"]
                name = data["path"].get("text")
                if name is None:
                    name = base64.b64decode(data["path"]["bytes"]).decode(
                        "utf-8", errors="surrogateescape"
                    )
                if name not in checked:
                    try:
                        checked[name] = self._read_text(self._resolve(name)) is not None
                    except ValueError:
                        checked[name] = False
                if not checked[name]:
                    continue
                text = data["lines"].get("text")
                if text is None:
                    continue
                matches.append(
                    {
                        "path": str(Path(name)),
                        "line": data["line_number"],
                        "text": text.rstrip("\r\n")[:500],
                    }
                )
                if len(matches) == _MAX_MATCHES:
                    break
        return sorted(matches, key=lambda item: (item["path"], item["line"]))

    def read_file(
        self,
        path: Annotated[str, spec(description="Repository-relative file path")],
        start_line: Annotated[int, spec(description="First line to read, starting at 1")] = 1,
        end_line: Annotated[
            int | None,
            spec(description="Last line to read; defaults to at most 400 lines"),
        ] = None,
    ) -> str:
        """Read a numbered excerpt from one UTF-8 text file."""

        file_path = self._resolve(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"file does not exist: {path}")
        if start_line < 1:
            raise ValueError("start_line must be at least 1")
        if end_line is not None and end_line < start_line:
            raise ValueError("end_line must not be before start_line")

        text = self._read_text(file_path)
        if text is None:
            raise ValueError(f"file is binary, non-UTF-8, or larger than {_MAX_FILE_BYTES} bytes")
        lines = text.splitlines()
        final_line = min(end_line or start_line + _MAX_READ_LINES - 1, len(lines))
        final_line = min(final_line, start_line + _MAX_READ_LINES - 1)
        width = len(str(final_line))
        return "\n".join(
            f"{line_number:>{width}}| {lines[line_number - 1]}"
            for line_number in range(start_line, final_line + 1)
        )


__all__ = ["CodebaseTools"]
