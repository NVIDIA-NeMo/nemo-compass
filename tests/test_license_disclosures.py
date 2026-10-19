# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise archive collection and failures that could silently lose attribution."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collector = load_tool("collect_license_texts")
generator = load_tool("generate_third_party_licenses")
distribution = load_tool("check_distribution_licenses")


def test_distribution_identity_and_dependencies():
    metadata = (
        b"Name: compass\nVersion: 0.1.0\n"
        b"License-Expression: Apache-2.0\nLicense-File: LICENSE\n"
        b"Requires-Dist: pydantic>=2\n"
    )
    distribution.check_metadata(metadata, {"LICENSE": b"license"}, "compass", "0.1.0")
    for invalid, error in [
        (metadata.replace(b"Name: compass", b"Name: other"), "Package name differs"),
        (metadata.replace(b"Version: 0.1.0", b"Version: 9.0.0"), "Package version differs"),
        (
            metadata + b'Requires-Dist: Trace_Ingest[langsmith]>=0.1; extra == "langsmith"\n',
            "must not depend on a separate trace-ingest",
        ),
    ]:
        with pytest.raises(AssertionError, match=error):
            distribution.check_metadata(invalid, {"LICENSE": b"license"}, "compass", "0.1.0")


def test_inventory_preserves_platform_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "_load_overrides", lambda: {})
    scan = tmp_path / "scan.json"
    scan.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "packages": [
                            {
                                "package": {"name": "torch", "version": version},
                                "licenses": ["BSD-3-Clause"],
                            }
                            for version in ["2.13.0", "2.13.0+cpu"]
                        ]
                    }
                ]
            }
        )
    )
    assert [(r["name"], r["version"]) for r in generator._license_records(scan)] == [
        ("torch", "2.13.0"),
        ("torch", "2.13.0+cpu"),
    ]


def wheel(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return output.getvalue()


def cached_artifact(tmp_path, name, data):
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / digest).write_bytes(data)
    return {"url": f"https://example.org/{name}", "hash": "sha256:" + digest}


def test_collects_nested_notices_and_declared_nonstandard_names():
    files = {
        "pkg.dist-info/METADATA": "License-File: attribution.custom\n\n",
        "pkg.dist-info/licenses/LICENSE": "Original copyright and license\n",
        "pkg.dist-info/licenses/attribution.custom": "Extra required attribution\n",
        "vendor/THIRD_PARTY_NOTICES.md": "Bundled notices\n",
        "vendor/COPYING": "Bundled license\n",
        "check_license.py": "not a notice",
        "License.rtf": "{\\rtf1 formatted copy}\0",
    }
    result = collector._archive_documents(wheel(files), "package.whl")
    assert set(result) == set(files) - {"pkg.dist-info/METADATA", "check_license.py", "License.rtf"}
    assert result["vendor/THIRD_PARTY_NOTICES.md"] == b"Bundled notices\n"


def test_missing_declared_file_fails():
    data = wheel({"pkg.dist-info/METADATA": "License-File: missing.txt\n\n"})
    with pytest.raises(RuntimeError, match="declared license file missing"):
        collector._archive_documents(data, "package.whl")


def test_reads_tar_license_symlink_without_extracting(tmp_path):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        member = tarfile.TarInfo("pkg/LICENSE")
        member.size = 8
        archive.addfile(member, io.BytesIO(b"License\n"))
        link = tarfile.TarInfo("pkg/sub/LICENSE")
        link.type = tarfile.SYMTYPE
        link.linkname = "../LICENSE"
        archive.addfile(link)
    assert collector._archive_documents(output.getvalue(), "pkg.tar.gz") == {
        "pkg/LICENSE": b"License\n",
        "pkg/sub/LICENSE": b"License\n",
    }
    assert list(tmp_path.iterdir()) == []


def test_checks_cached_archive_hash(tmp_path):
    artifact = cached_artifact(tmp_path, "pkg.whl", b"original")
    (tmp_path / artifact["hash"].split(":")[1]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        collector._download(artifact, tmp_path)


def test_universal_wheel_selection_ignores_input_order(tmp_path):
    universal = cached_artifact(tmp_path, "pkg-py3-none-any.whl", wheel({"LICENSE": "License\n"}))
    binary = {"url": "https://example.org/pkg-linux.whl", "hash": "sha256:" + "0" * 64}
    for wheels in [[binary, universal], [universal, binary]]:
        documents = collector.collect({"name": "pkg", "version": "1", "wheels": wheels}, tmp_path)
        assert documents[0]["source"] == universal["url"]


def test_no_license_fails_unless_supplemented(tmp_path):
    artifact = cached_artifact(tmp_path, "pkg-py3-none-any.whl", wheel({"AUTHORS": "Author\n"}))
    package = {"name": "pkg", "version": "1", "wheels": [artifact]}
    with pytest.raises(RuntimeError, match="No license text"):
        collector.collect(package, tmp_path)
    supplement = cached_artifact(tmp_path, "LICENSE", b"Copyright Example. License terms.\n")
    assert collector.collect(package, tmp_path, [supplement])[0]["text"].startswith("Copyright")


def test_empty_license_fails(tmp_path):
    artifact = cached_artifact(tmp_path, "pkg-py3-none-any.whl", wheel({"LICENSE": ""}))
    with pytest.raises(RuntimeError, match="Empty license"):
        collector.collect({"name": "pkg", "version": "1", "wheels": [artifact]}, tmp_path)


def test_deduplication_retains_all_original_paths():
    docs = [
        {"source": "archive", "sha256": "hash", "path": path, "text": "Copyright and terms\n"}
        for path in ["LICENSE", "vendor/LICENSE"]
    ]
    result = collector.render_texts([("pkg", "1", docs)])
    assert result.count("Copyright and terms") == 1
    assert "File: LICENSE" in result and "File: vendor/LICENSE" in result


def test_stale_exception_requires_review(tmp_path, monkeypatch):
    path = tmp_path / "exceptions.yaml"
    path.write_text("pkg:\n  version: old\n  documents: []\n")
    monkeypatch.setattr(generator, "EXCEPTIONS_PATH", path)
    with pytest.raises(RuntimeError, match="stale license exception"):
        generator._supplements([{"name": "pkg", "version": "new"}])
