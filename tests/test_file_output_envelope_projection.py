from __future__ import annotations

from copy import deepcopy

import pytest

from edgar_mcp import server


_ENVELOPE_KEYS = ("semantic_status", "coverage_status", "coverage_warning")


def _financials_payload(envelope: dict[str, object]) -> dict[str, object]:
    return {
        "status": "success",
        "facts": [{"tag": "us-gaap:Revenue", "current_value": 100, "prior_value": 90}],
        "metadata": {"source": {"filing_type": "10-K", "accession": "example"}},
        **envelope,
    }


def _sections_payload(envelope: dict[str, object]) -> dict[str, object]:
    return {
        "status": "success",
        "filing_type": "10-K",
        "sections": {
            "item_1": {
                "header": "Item 1. Business",
                "state": "body",
                "text": "Business description.",
                "tables": [],
                "word_count": 2,
            }
        },
        **envelope,
    }


@pytest.mark.parametrize(
    "envelope",
    [
        {
            "semantic_status": "partial",
            "coverage_status": "narrow",
            "coverage_warning": "Only Item 1 was requested.",
        },
        {
            "semantic_status": None,
            "coverage_status": None,
            "coverage_warning": None,
        },
        {},
    ],
    ids=["producer-values", "explicit-null", "omitted"],
)
@pytest.mark.parametrize("projection", ["financials", "sections"])
def test_file_output_preserves_producer_envelope_presence_and_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    projection: str,
    envelope: dict[str, object],
) -> None:
    monkeypatch.setattr(server, "FILE_OUTPUT_DIR", tmp_path)
    if projection == "financials":
        producer = _financials_payload(envelope)
        arguments = {
            "ticker": "AAPL",
            "year": 2025,
            "quarter": 4,
            "output": "file",
        }
        proxy = server._proxy_get_financials
    else:
        producer = _sections_payload(envelope)
        arguments = {
            "ticker": "AAPL",
            "year": 2025,
            "quarter": 4,
            "output": "file",
            "__artifact_owner_id": "owner-1",
            "__artifact_session_id": "session-1",
        }
        proxy = server._proxy_get_filing_sections

    monkeypatch.setattr(
        server, "_call_api", lambda *_args, **_kwargs: deepcopy(producer)
    )

    result = proxy(arguments)

    assert result["status"] == "success"
    for key in _ENVELOPE_KEYS:
        if key in envelope:
            assert key in result
            assert result[key] == envelope[key]
        else:
            assert key not in result
