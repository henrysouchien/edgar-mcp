from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest


_PRODUCER = "get_filing_sections"
_CONTENT = "# Filing\n\n## SECTION: Item 1. Business\nTrusted disclosure.\n"


def _module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from edgar_mcp import server

    module = importlib.reload(server)
    monkeypatch.setattr(module, "FILE_OUTPUT_DIR", tmp_path.resolve())
    return module


def _issue(module, path: Path, content: str = _CONTENT, **kwargs):
    receipt = module._write_trusted_artifact(path, content)
    return module._issue_artifact_handle(
        receipt,
        producer=_PRODUCER,
        kind=module._FILING_ARTIFACT_KIND,
        media_type=module._FILING_ARTIFACT_MEDIA_TYPE,
        **kwargs,
    )


def _resolve(module, handle: str):
    return module._resolve_artifact_handle(
        handle,
        expected_producer=_PRODUCER,
        expected_kind=module._FILING_ARTIFACT_KIND,
        expected_media_type=module._FILING_ARTIFACT_MEDIA_TYPE,
    )


def _managed_root(module, tmp_path: Path) -> Path:
    return tmp_path / module._MANAGED_ARTIFACT_DIRNAME


def _database(module, tmp_path: Path) -> Path:
    return _managed_root(module, tmp_path) / module._ARTIFACT_DATABASE_FILENAME


def _backing(record) -> Path:
    return Path(record.root_path) / record.backing_filename


def _rows(module, tmp_path: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(_database(module, tmp_path))
    connection.row_factory = sqlite3.Row
    try:
        return list(connection.execute("SELECT * FROM artifact_capabilities"))
    finally:
        connection.close()


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["EDGAR_MCP_OUTPUT_DIR"] = str(tmp_path)
    return env


def _run_subprocess(script: str, tmp_path: Path, *args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=Path(__file__).parents[1],
        env=_subprocess_env(tmp_path),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout.strip())


def test_public_tool_schema_has_bearer_handle_without_transport_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module(monkeypatch, tmp_path)

    tools = {tool.name: tool for tool in asyncio.run(module.mcp.list_tools())}
    extract_schema = tools["extract_filing_file"].parameters
    sections_schema = tools["get_filing_sections"].parameters

    assert "artifact_handle" in extract_schema["properties"]
    assert "artifact_handle" in extract_schema["required"]
    assert "file_path" not in extract_schema["properties"]
    assert "ctx" not in extract_schema["properties"]
    assert "ctx" not in sections_schema["properties"]
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_ARTIFACT_RECORDS: dict" not in source
    assert "_artifact_binding" not in source
    assert "__artifact_session_id" not in source


def test_product_file_routes_use_shared_atomic_writer() -> None:
    from edgar_mcp import server

    for proxy in (server._proxy_get_financials, server._proxy_get_filing_sections):
        source = inspect.getsource(proxy)
        assert ".write_text(" not in source
        assert "atomic_write_flat_file(" in source


def test_store_persists_only_bearer_digest_and_private_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module(monkeypatch, tmp_path)
    public_path = tmp_path / "trusted.md"
    record = _issue(module, public_path)
    rows = _rows(module, tmp_path)

    assert rows[0]["handle_sha256"] == module._artifact_handle_digest(record.handle)
    assert record.handle.encode() not in _database(module, tmp_path).read_bytes()
    assert _CONTENT.encode() not in _database(module, tmp_path).read_bytes()
    assert public_path.read_text(encoding="utf-8") == _CONTENT
    assert _managed_root(module, tmp_path).stat().st_mode & 0o777 == 0o700
    assert _database(module, tmp_path).stat().st_mode & 0o777 == 0o600
    assert _backing(record).stat().st_mode & 0o777 == 0o600


def test_reissue_keeps_both_snapshots_after_public_copy_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module(monkeypatch, tmp_path)
    public_path = tmp_path / "same-name.md"
    first = _issue(module, public_path, "first snapshot")
    second = _issue(module, public_path, "second snapshot")
    public_path.write_text("user-edited convenience copy", encoding="utf-8")

    first_record, first_bytes = _resolve(module, first.handle)
    second_record, second_bytes = _resolve(module, second.handle)

    assert first_record.backing_filename != second_record.backing_filename
    assert first_bytes == b"first snapshot"
    assert second_bytes == b"second snapshot"
    assert public_path.read_text(encoding="utf-8") == "user-edited convenience copy"


@pytest.mark.parametrize("tamper", ["content", "same_content_replacement"])
def test_backing_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    module = _module(monkeypatch, tmp_path)
    record = _issue(module, tmp_path / "trusted.md")
    backing = _backing(record)
    if tamper == "content":
        backing.write_text("tampered", encoding="utf-8")
        expected = "content changed"
    else:
        backing.unlink()
        backing.write_bytes(_CONTENT.encode())
        expected = "file identity changed"

    with pytest.raises(ValueError, match=expected):
        _resolve(module, record.handle)


@pytest.mark.parametrize(
    ("count_cap", "byte_cap", "contents"),
    [
        (2, 1024, ("first", "second", "newcomer")),
        (64, 10, ("aaaaaa", "bbbb", "c")),
    ],
)
def test_capacity_rejects_newcomer_without_evicting_live_handles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    count_cap: int,
    byte_cap: int,
    contents: tuple[str, ...],
) -> None:
    module = _module(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_MAX_ARTIFACT_RECORDS", count_cap)
    monkeypatch.setattr(module, "_MAX_ARTIFACT_REGISTRY_BYTES", byte_cap)
    accepted = [
        _issue(module, tmp_path / f"accepted-{index}.md", content)
        for index, content in enumerate(contents[:-1])
    ]
    newcomer = module._write_trusted_artifact(
        tmp_path / "newcomer.md",
        contents[-1],
    )

    with pytest.raises(module._ArtifactCapacityError, match="capacity is full"):
        module._issue_artifact_handle(
            newcomer,
            producer=_PRODUCER,
            kind=module._FILING_ARTIFACT_KIND,
            media_type=module._FILING_ARTIFACT_MEDIA_TYPE,
        )

    assert [_resolve(module, item.handle)[1] for item in accepted] == [
        content.encode() for content in contents[:-1]
    ]
    assert len(_rows(module, tmp_path)) == len(accepted)


def test_sections_tool_returns_stable_capacity_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_MAX_ARTIFACT_RECORDS", 1)
    existing = _issue(module, tmp_path / "existing.md", "existing snapshot")
    monkeypatch.setattr(
        module,
        "_call_api",
        lambda *args, **kwargs: {
            "status": "success",
            "filing_type": "10-K",
            "sections": {
                "item_1": {
                    "header": "Item 1. Business",
                    "word_count": 2,
                    "text": "New disclosure.",
                    "tables": [],
                }
            },
        },
    )

    result = asyncio.run(
        module.get_filing_sections(
            ticker="AAPL",
            year=2025,
            quarter=4,
            sections=["item_1"],
            output="file",
        )
    )

    assert result == {
        "status": "error",
        "error_type": "artifact_capacity_exceeded",
        "message": (
            "Artifact capability capacity is full; retry after an existing "
            "handle expires"
        ),
    }
    assert _resolve(module, existing.handle)[1] == b"existing snapshot"
    assert len(_rows(module, tmp_path)) == 1


def test_sections_timeout_before_tables_write_does_not_issue_hidden_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "_call_api",
        lambda *args, **kwargs: {
            "status": "success",
            "filing_type": "10-K",
            "sections": {
                "item_1": {
                    "header": "Item 1. Business",
                    "word_count": 2,
                    "text": "New disclosure.",
                    "tables": ["| A | B |"],
                }
            },
            "tables_structured": {"item_1": [{"label": "Revenue"}]},
        },
    )
    decisions = iter((False, True))
    monkeypatch.setattr(module, "_deadline_expired", lambda _args: next(decisions))

    result = asyncio.run(
        module.get_filing_sections(
            ticker="AAPL",
            year=2025,
            quarter=4,
            sections=["item_1"],
            include_tables=True,
            output="file",
        )
    )

    assert result == {
        "status": "error",
        "message": "Request timed out before structured tables could be written",
    }
    assert not _database(module, tmp_path).exists()
    assert list(tmp_path.rglob(".artifact-*.bin")) == []
    assert list(tmp_path.glob("*_tables.json")) == []


def test_corrupt_database_fails_closed_and_preserves_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module(monkeypatch, tmp_path)
    public_path = tmp_path / "trusted.md"
    record = _issue(module, public_path)
    backing = _backing(record)
    _database(module, tmp_path).write_bytes(b"not a sqlite database")

    result = module._proxy_extract_filing_file(
        {"artifact_handle": record.handle, "schema_name": "risk_factors"}
    )

    assert result["status"] == "error"
    assert result["error_type"] == "invalid_artifact_handle"
    assert public_path.exists()
    assert backing.read_bytes() == _CONTENT.encode()


def test_separate_process_restart_resolves_unexpired_handle(tmp_path: Path) -> None:
    issue_script = r"""
import json, os
from pathlib import Path
from edgar_mcp import server as module
root = Path(os.environ["EDGAR_MCP_OUTPUT_DIR"]).resolve()
module.FILE_OUTPUT_DIR = root
receipt = module._write_trusted_artifact(root / "restart.md", "restart snapshot")
record = module._issue_artifact_handle(
    receipt,
    producer="get_filing_sections",
    kind=module._FILING_ARTIFACT_KIND,
    media_type=module._FILING_ARTIFACT_MEDIA_TYPE,
)
print(json.dumps({"handle": record.handle, "expires_at": record.expires_at}))
"""
    resolve_script = r"""
import json, os, sys
from pathlib import Path
from edgar_mcp import server as module
module.FILE_OUTPUT_DIR = Path(os.environ["EDGAR_MCP_OUTPUT_DIR"]).resolve()
record, content = module._resolve_artifact_handle(
    sys.argv[1],
    expected_producer="get_filing_sections",
    expected_kind=module._FILING_ARTIFACT_KIND,
    expected_media_type=module._FILING_ARTIFACT_MEDIA_TYPE,
)
print(json.dumps({"handle": record.handle, "content": content.decode()}))
"""

    issued = _run_subprocess(issue_script, tmp_path)
    resolved = _run_subprocess(resolve_script, tmp_path, issued["handle"])

    assert issued["expires_at"] > time.time()
    assert resolved == {
        "handle": issued["handle"],
        "content": "restart snapshot",
    }


def test_public_reconnect_journey_uses_private_snapshot(tmp_path: Path) -> None:
    issue_script = r"""
import json, os
from pathlib import Path
from edgar_mcp import server as module
module.FILE_OUTPUT_DIR = Path(os.environ["EDGAR_MCP_OUTPUT_DIR"]).resolve()
module._call_api = lambda *args, **kwargs: {
    "status": "success",
    "filing_type": "10-K",
    "sections": {"item_1": {
        "header": "Item 1. Business", "word_count": 2,
        "text": "Restart disclosure.", "tables": [],
    }},
}
print(json.dumps(module._proxy_get_filing_sections({
    "ticker": "AAPL", "year": 2025, "quarter": 4,
    "sections": ["item_1"], "output": "file",
})))
"""
    resolve_script = r"""
import json, os, sys
from pathlib import Path
from edgar_mcp import server as module
module.FILE_OUTPUT_DIR = Path(os.environ["EDGAR_MCP_OUTPUT_DIR"]).resolve()
posted = []
def post(path, payload, timeout=300):
    posted.append(payload)
    return {"status": "success", "extractions_by_schema": {"risk_factors": []}}
module._post_api = post
result = module._proxy_extract_filing_file({
    "artifact_handle": sys.argv[1], "schema_name": "risk_factors",
})
print(json.dumps({"result": result, "content": posted[0]["content"]}))
"""

    issued = _run_subprocess(issue_script, tmp_path)
    Path(issued["file_path"]).write_text("edited visible copy", encoding="utf-8")
    resolved = _run_subprocess(resolve_script, tmp_path, issued["artifact_handle"])

    assert resolved["result"]["status"] == "ok"
    assert resolved["result"]["artifact_handle"] == issued["artifact_handle"]
    assert "Restart disclosure." in resolved["content"]
    assert "edited visible copy" not in resolved["content"]


def test_concurrent_process_issuers_preserve_every_handle(tmp_path: Path) -> None:
    script = r"""
import json, os, sys
from pathlib import Path
from edgar_mcp import server as module
root = Path(os.environ["EDGAR_MCP_OUTPUT_DIR"]).resolve()
module.FILE_OUTPUT_DIR = root
index = sys.argv[1]
content = f"concurrent snapshot {index}"
receipt = module._write_trusted_artifact(root / f"concurrent-{index}.md", content)
record = module._issue_artifact_handle(
    receipt,
    producer="get_filing_sections",
    kind=module._FILING_ARTIFACT_KIND,
    media_type=module._FILING_ARTIFACT_MEDIA_TYPE,
)
print(json.dumps({"handle": record.handle, "content": content}))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(index)],
            cwd=Path(__file__).parents[1],
            env=_subprocess_env(tmp_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(8)
    ]
    issued = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=45)
        assert process.returncode == 0, stderr
        issued.append(json.loads(stdout.strip()))

    from edgar_mcp import server

    module = importlib.reload(server)
    module.FILE_OUTPUT_DIR = tmp_path.resolve()
    assert len(_rows(module, tmp_path)) == len(issued)
    assert len({item["handle"] for item in issued}) == len(issued)
    for item in issued:
        assert _resolve(module, item["handle"])[1].decode() == item["content"]
