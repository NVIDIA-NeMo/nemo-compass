# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nooa.unifiedllm import create_tool_from_callable

from compass.insights_generation.codebase import CodebaseTools


def test_codebase_tools_search_and_read_source(tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("def answer():\n    return 'hello'\n", encoding="utf-8")
    tools = CodebaseTools(tmp_path)

    assert tools.list_files("**/*.py") == ["src/agent.py"]
    assert tools.search_code("RETURN", "src") == [
        {"path": "src/agent.py", "line": 2, "text": "    return 'hello'"}
    ]
    assert tools.read_file("src/agent.py", 1, 2) == "1| def answer():\n2|     return 'hello'"

    schemas = {
        method.__name__: create_tool_from_callable(method).get_parameter_schema()
        for method in (tools.list_files, tools.search_code, tools.read_file)
    }
    assert schemas["search_code"]["required"] == ["query"]
    assert schemas["read_file"]["required"] == ["path"]


def test_codebase_tools_reject_sensitive_and_external_paths(tmp_path: Path) -> None:
    codebase = tmp_path / "repo"
    codebase.mkdir()
    (codebase / ".env").write_text("TOKEN=secret", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True", encoding="utf-8")
    (codebase / "outside.py").symlink_to(outside)
    tools = CodebaseTools(codebase)

    with pytest.raises(ValueError, match="not available"):
        tools.read_file(".env")
    with pytest.raises(ValueError, match="escapes"):
        tools.read_file("outside.py")


def test_cache_reuses_reads_but_tracks_source_changes(tmp_path, monkeypatch):
    source = tmp_path / "agent.py"
    source.write_text("hello\n")
    tools = CodebaseTools(tmp_path)
    assert tools.search_code("hello")
    read_bytes = tools._cache_stats["bytes_read"]
    assert tools.read_file("agent.py") == "1| hello"
    assert tools._cache_stats["bytes_read"] == read_bytes
    source.write_text("changed\n")
    assert not tools.search_code("hello")
    assert tools.read_file("agent.py") == "1| changed"
    (tmp_path / "new.py").write_text("hello again\n")
    assert tools.search_code("hello")[0]["path"] == "new.py"
    source.unlink()
    assert tools.list_files() == ["new.py"]


def test_cache_does_not_authorize_a_replaced_symlink(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "agent.py"
    source.write_text("inside")
    tools = CodebaseTools(repo)
    assert tools.search_code("inside")
    outside = tmp_path / "outside.py"
    outside.write_text("outside secret")
    source.unlink()
    source.symlink_to(outside)
    assert tools.list_files() == []
    assert tools.search_code("secret") == []
    with pytest.raises(ValueError, match="escapes"):
        tools.read_file("agent.py")


def test_source_reads_and_cache_memory_are_bounded(tmp_path, monkeypatch):
    from compass.insights_generation import codebase

    monkeypatch.setattr(codebase, "_MAX_CACHE_BYTES", 128)
    large = tmp_path / "large"
    large.write_bytes(b"x" * (codebase._MAX_FILE_BYTES + 1))
    tools = CodebaseTools(tmp_path)
    assert tools.search_code("x") == []
    assert tools._cache_stats["bytes_read"] == 0
    for i in range(3):
        (tmp_path / str(i)).write_text("small text")
        tools.read_file(str(i))
    assert tools._cache_stats["cache_bytes"] <= 128


def test_source_operation_telemetry_preserves_tool_outputs(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from compass import telemetry_phases

    records = []

    @contextmanager
    def phase(name, inputs):
        yield SimpleNamespace(set_output=lambda output: records.append((name, inputs, output)))

    monkeypatch.setattr(telemetry_phases, "_phase", phase)
    (tmp_path / "agent.py").write_text("hello")
    assert CodebaseTools(tmp_path).read_file("agent.py") == "1| hello"
    assert records[0][0] == "codebase.read_file"
    assert records[0][1]["args"] == ("agent.py",)
    assert records[0][2]["cache_totals"]["bytes_read"] == 5


def test_concurrent_source_paths_belong_to_their_investigation(tmp_path):
    import asyncio

    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(name)
    tools = CodebaseTools(tmp_path)

    async def read(name):
        with tools.capture_reads() as paths:
            await asyncio.sleep(0)
            tools.read_file(name)
            await asyncio.sleep(0)
        return paths

    async def run():
        return await asyncio.gather(read("a.py"), read("b.py"))

    assert asyncio.run(run()) == [{"a.py"}, {"b.py"}]
