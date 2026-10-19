#!/usr/bin/env -S uv run
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and publish a Compass release candidate to NVIDIA Artifactory."""

from __future__ import annotations

import argparse
import base64
import email
import hashlib
import os
import re
import shlex
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"

ENV_PYPI_URL = "ARTIFACTORY_PYPI_URL"
ENV_TOKEN = "ARTIFACTORY_TOKEN"

_RELEASE_PACKAGE = "compass"
_ALLOWED_HOSTS = {"artifactory.nvidia.com", "urm.nvidia.com"}
_FORBIDDEN_ENV_FILES = {".env", ".env.local", ".env.example"}


class PublishConfigurationError(ValueError):
    """Raised when local Artifactory publishing is not configured safely."""


@dataclass(frozen=True)
class Wheel:
    """Metadata needed to publish and verify one built wheel."""

    path: Path
    package_name: str
    version: str


def _load_artifactory_env(path: Path) -> None:
    """Load only the two Artifactory settings from a dotenv file."""
    if not path.is_file():
        return

    allowed = {ENV_PYPI_URL, ENV_TOKEN}
    for key, value in dotenv_values(path).items():
        if key not in allowed or key in os.environ:
            continue
        if value:
            os.environ[key] = value


def _required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise PublishConfigurationError(f"{name} must be set in .env or the environment")
    return value


def _validate_publish_url(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme != "https":
        raise PublishConfigurationError(f"{ENV_PYPI_URL} must use HTTPS")
    if parsed.hostname not in _ALLOWED_HOSTS:
        allowed = ", ".join(sorted(_ALLOWED_HOSTS))
        raise PublishConfigurationError(f"{ENV_PYPI_URL} host must be one of: {allowed}")
    if parsed.username or parsed.password:
        raise PublishConfigurationError(
            f"{ENV_PYPI_URL} must not contain credentials; use {ENV_TOKEN}"
        )
    if parsed.query or parsed.fragment:
        raise PublishConfigurationError(f"{ENV_PYPI_URL} must not contain a query or fragment")
    _, separator, repository = parsed.path.partition("/api/pypi/")
    if not separator or not repository or "/" in repository:
        raise PublishConfigurationError(
            f"{ENV_PYPI_URL} must end with an Artifactory PyPI repository path"
        )
    return urlunsplit(parsed)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"+ {shlex.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _build_wheel() -> Wheel:
    build_env = os.environ.copy()
    for name in (ENV_TOKEN, "UV_PUBLISH_PASSWORD", "UV_PUBLISH_TOKEN"):
        build_env.pop(name, None)
    _run(["uv", "build", "--wheel", "--no-sources", "--clear"], env=build_env)
    wheels = list(DIST_DIR.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one Compass wheel in {DIST_DIR}, found {wheels}")
    return _inspect_wheel(wheels[0])


def _inspect_wheel(wheel: Path) -> Wheel:
    """Read identity metadata and reject environment files in a wheel."""
    with zipfile.ZipFile(wheel) as package:
        members = package.namelist()
        forbidden = sorted(
            member for member in members if PurePosixPath(member).name in _FORBIDDEN_ENV_FILES
        )
        if forbidden:
            raise RuntimeError(f"refusing to publish {wheel}: contains {forbidden}")

        metadata_paths = [member for member in members if member.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError(f"expected one METADATA file in {wheel}, found {metadata_paths}")
        metadata = email.message_from_bytes(package.read(metadata_paths[0]))

    package_name = metadata.get("Name", "").strip()
    version = metadata.get("Version", "").strip()
    if package_name != _RELEASE_PACKAGE:
        raise RuntimeError(f"unexpected wheel package name: {package_name!r}")
    if not version:
        raise RuntimeError(f"wheel has no version metadata: {wheel}")
    return Wheel(path=wheel, package_name=package_name, version=version)


def _direct_artifact_url(publish_url: str, wheel: Wheel) -> str:
    base, separator, repository = publish_url.partition("/api/pypi/")
    if not separator or not repository or "/" in repository:
        raise PublishConfigurationError(
            f"cannot derive an artifact URL from {ENV_PYPI_URL}={publish_url!r}"
        )
    return "/".join(
        (
            base,
            quote(repository, safe="-_."),
            quote(wheel.package_name, safe="-_."),
            quote(wheel.version, safe="-_.+"),
            quote(wheel.path.name, safe="-_.+"),
        )
    )


def _publish(wheel: Wheel, publish_url: str, token: str, *, dry_run: bool) -> None:
    publish_env = os.environ.copy()
    publish_env.pop("UV_PUBLISH_TOKEN", None)
    publish_env.update(
        {
            # JFrog requires an empty username when an access token is the
            # password. UV_PUBLISH_TOKEN is unsuitable because uv substitutes
            # the PyPI-specific username "__token__".
            "UV_PUBLISH_USERNAME": "",
            "UV_PUBLISH_PASSWORD": token,
        }
    )

    command = [
        "uv",
        "publish",
        "--trusted-publishing",
        "never",
        "--publish-url",
        publish_url,
        "--check-url",
        f"{publish_url}/simple",
    ]
    if dry_run:
        command.append("--dry-run")
    command.append(str(wheel.path))
    _run(command, env=publish_env)


def _authorization_header(token: str) -> str:
    credentials = base64.b64encode(f":{token}".encode()).decode("ascii")
    return f"Basic {credentials}"


def _download_artifact(url: str, token: str) -> bytes | None:
    """Download an artifact, returning ``None`` only when it does not exist."""
    request = Request(url, headers={"Authorization": _authorization_header(token)})
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _publish_checksums(url: str, contents: bytes, token: str) -> None:
    """Submit client checksums for an artifact already uploaded by uv."""
    checksums = {
        "sha1": hashlib.sha1(contents, usedforsecurity=False).hexdigest(),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }
    headers = {
        "Authorization": _authorization_header(token),
        "Content-Type": "text/plain",
    }

    print("Publishing client checksums (SHA-1 and SHA-256)")
    for algorithm, digest in checksums.items():
        request = Request(
            f"{url}.{algorithm}",
            data=digest.encode("ascii"),
            headers=headers,
            method="PUT",
        )
        with urlopen(request, timeout=60) as response:
            response.read()


def _verify_artifact(url: str, wheel: Wheel, token: str) -> None:
    expected_digest = hashlib.sha256(wheel.path.read_bytes()).hexdigest()
    contents = _download_artifact(url, token)
    if contents is None:
        raise RuntimeError(f"published artifact disappeared from {url}")
    if hashlib.sha256(contents).hexdigest() != expected_digest:
        raise RuntimeError(f"published artifact checksum does not match {wheel.path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate without uploading the wheel",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_artifactory_env(REPO_ROOT / ".env")

    try:
        publish_url = _validate_publish_url(_required_setting(ENV_PYPI_URL))
        token = _required_setting(ENV_TOKEN)
        wheel = _build_wheel()
        if re.search(r"rc\d+(?:[.+]|$)", wheel.version) is None:
            raise PublishConfigurationError(
                f"refusing to publish non-RC compass version {wheel.version!r}; set an rcN version"
            )
    except PublishConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    artifact_url = _direct_artifact_url(publish_url, wheel)
    if args.dry_run:
        print(f"Checking {wheel.path.name} as {wheel.package_name}=={wheel.version}")
        _publish(wheel, publish_url, token, dry_run=True)
        print(f"Dry run complete. Artifact URL would be:\n{artifact_url}")
        return 0

    remote_contents = _download_artifact(artifact_url, token)
    local_contents = wheel.path.read_bytes()
    if remote_contents is not None:
        remote_digest = hashlib.sha256(remote_contents).hexdigest()
        local_digest = hashlib.sha256(local_contents).hexdigest()
        if remote_digest != local_digest:
            print(
                f"error: {wheel.package_name}=={wheel.version} already exists with different "
                "contents; bump the package version instead of overwriting it",
                file=sys.stderr,
            )
            return 2
        print(f"Matching artifact already exists; skipping wheel upload:\n{artifact_url}")
    else:
        print(f"Publishing {wheel.path.name} as {wheel.package_name}=={wheel.version}")
        _publish(wheel, publish_url, token, dry_run=False)

    _publish_checksums(artifact_url, local_contents, token)
    _verify_artifact(artifact_url, wheel, token)
    print(f"Published and checksum-verified:\n{artifact_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
