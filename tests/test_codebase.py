# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
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


@pytest.mark.parametrize(
    "name",
    [
        "ignored.py",
        ".hidden.py",
        "nested/file.py",
        "with space.py",
        "café.py",
        "line\nbreak.py",
        "-leading.py",
    ],
)
def test_discovery_and_search_include_source_despite_ignore_rules(tmp_path, monkeypatch, name):
    source = tmp_path / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("needle\n", encoding="utf-8")
    for ignore in (".gitignore", ".ignore", ".rgignore"):
        (tmp_path / ignore).write_text("*.py\nnested/\n")
    config = tmp_path / "rg-config"
    config.write_text("--glob=!*.py\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    tools = CodebaseTools(tmp_path)
    assert name in tools.list_files()
    assert tools.search_code("NEEDLE") == [{"path": name, "line": 1, "text": "needle"}]
    assert tools.search_code("needle", name)[0]["path"] == name


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".ENV.prod",
        "credentials.json",
        ".npmrc",
        "key.PEM",
        "node_modules/x.py",
        ".git/x.py",
        "build/x.py",
    ],
)
def test_exclusions_apply_to_discovery_search_and_explicit_paths(tmp_path, name):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("needle")
    tools = CodebaseTools(tmp_path)
    assert name not in tools.list_files()
    assert tools.search_code("needle") == []
    with pytest.raises(ValueError):
        tools.search_code("needle", name)


def test_search_limits_and_glob_scope_in_large_tree(tmp_path):
    for i in range(250):
        p = tmp_path / f"area{i:03}" / "agent.py"
        p.parent.mkdir()
        p.write_text("needle\n" * 5)
    tools = CodebaseTools(tmp_path)
    assert len(tools.list_files()) == 200
    assert len(tools.search_code("needle")) == 100
    assert tools.list_files("area249/*.py") == ["area249/agent.py"]
    assert len(tools.search_code("needle", "area249")) == 5
    assert len(tools.search_code("needle", "area249/agent.py")) == 5
    (tmp_path / "area249/agent.py").unlink()
    assert tools.list_files("area249/*.py") == []


def test_search_is_literal_and_preserves_line_numbers(tmp_path):
    (tmp_path / "code.py").write_bytes("first\r\na|b -needle café Straße\r\nlast\r\n".encode())
    tools = CodebaseTools(tmp_path)
    for query in ("a|b", "-needle", "CAFÉ"):
        assert tools.search_code(query)[0] == {
            "path": "code.py",
            "line": 2,
            "text": "a|b -needle café Straße",
        }
    assert tools.search_code("a.*b") == []
    assert tools.search_code("STRASSE") == []  # Native Unicode matching, documented.
    assert tools.search_code("first\na|b") == []
    with pytest.raises(ValueError):
        tools.search_code("")


def test_unreadable_text_and_oversize_are_not_returned(tmp_path):
    (tmp_path / "binary").write_bytes(b"needle\0tail")
    (tmp_path / "invalid").write_bytes(b"needle\xff")
    (tmp_path / "huge").write_bytes(b"needle" + b"x" * 1_000_000)
    (tmp_path / "valid").write_text("needle")
    tools = CodebaseTools(tmp_path)
    assert tools.search_code("needle") == [{"path": "valid", "line": 1, "text": "needle"}]
    for name in ("binary", "invalid", "huge"):
        assert tools.search_code("needle", name) == []
        with pytest.raises(ValueError):
            tools.read_file(name)


def test_symlinks_cannot_expand_discovery_or_escape_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_text("needle")
    (repo / "alias.py").symlink_to(repo / "source.py")
    (tmp_path / "outside.py").write_text("needle")
    (repo / "external.py").symlink_to(tmp_path / "outside.py")
    (repo / "loop").symlink_to(repo, target_is_directory=True)
    tools = CodebaseTools(repo)
    assert tools.list_files() == ["source.py"]
    assert len(tools.search_code("needle")) == 1
    assert tools.search_code("needle", "alias.py")[0]["path"] == "source.py"
    with pytest.raises(ValueError):
        tools.search_code("needle", "external.py")
    with pytest.raises(ValueError):
        tools.list_files("../*.py")
    with pytest.raises(ValueError):
        tools.search_code("needle", "../outside.py")


def test_empty_repository_missing_scope_and_missing_executable(tmp_path, monkeypatch):
    from compass.insights_generation import codebase

    tools = CodebaseTools(tmp_path)
    assert tools.list_files() == []
    assert tools.search_code("absent") == []
    with pytest.raises(FileNotFoundError):
        tools.search_code("absent", "missing")
    monkeypatch.setattr(codebase.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires ripgrep"):
        tools.search_code("absent")


@pytest.mark.parametrize("body", ["exit 2", "exec sleep 2"])
def test_backend_errors_and_timeouts_are_not_empty_results(tmp_path, monkeypatch, body):
    from compass.insights_generation import codebase

    executable = tmp_path / "fake-rg"
    executable.write_text("#!/bin/sh\n" + body + "\n")
    executable.chmod(0o755)
    monkeypatch.setattr(codebase.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(codebase, "_SEARCH_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(RuntimeError, match="ripgrep failed"):
        CodebaseTools(tmp_path).search_code("needle")


@pytest.mark.skipif(
    sys.platform != "linux", reason="Non-UTF-8 filenames require a supporting filesystem"
)
def test_non_utf8_filename_and_long_lines(tmp_path):
    name = "nonutf-\udcff.py"
    (tmp_path / name).write_text("needle " + "x" * 700)
    tools = CodebaseTools(tmp_path)
    assert tools.list_files() == [name]
    found = tools.search_code("needle")
    assert found[0]["path"] == name
    assert found[0]["text"] == "needle " + "x" * 493


def test_read_file_line_bounds_and_file_changes(tmp_path):
    p = tmp_path / "file.py"
    p.write_text("line\n" * 500)
    tools = CodebaseTools(tmp_path)
    assert len(tools.read_file("file.py").splitlines()) == 400
    assert len(tools.read_file("file.py", 495, 500).splitlines()) == 6
    p.write_text("changed")
    assert tools.read_file("file.py") == "1| changed"
    assert tools.search_code("line") == []
    assert tools.search_code("changed")[0]["line"] == 1


def test_long_lines_are_clipped(tmp_path):
    (tmp_path / "long.py").write_text("needle " + "x" * 700)
    assert CodebaseTools(tmp_path).search_code("needle")[0]["text"] == "needle " + "x" * 493
