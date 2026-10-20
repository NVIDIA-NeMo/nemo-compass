# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent source link validation and local file locations."""

from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import AfterValidator


def validate_source_url(value: str) -> str:
    parsed = urlsplit(value)
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("source URL must not contain whitespace or control characters")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URL must not contain credentials")
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        return value
    if parsed.scheme == "file" and not parsed.netloc and parsed.path.startswith("/"):
        return value
    raise ValueError("source URL must be an absolute HTTP(S) URL or local file URI")


SourceURL = Annotated[str, AfterValidator(validate_source_url)]


def http_source_url(value: object) -> str | None:
    """Ignore absent or unusable provider links without failing trace ingestion."""
    if not isinstance(value, str):
        return None
    try:
        validated = validate_source_url(value)
        return validated if urlsplit(validated).scheme in {"http", "https"} else None
    except ValueError:
        return None


def file_source_url(path: Path, line_number: int | None = None) -> str:
    url = path.resolve().as_uri()
    return f"{url}#L{line_number}" if line_number is not None else url
