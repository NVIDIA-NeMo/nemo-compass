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
