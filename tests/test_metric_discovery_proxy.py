from edgar_mcp import server


def test_proxy_list_metrics_forwards_to_http_endpoint(monkeypatch):
    calls = []

    def fake_call_api(path, params, timeout=300):
        calls.append((path, params, timeout))
        return {
            "status": "success",
            "ticker": "PCTY",
            "date_type_filter": "FY",
            "total_candidates": 1,
            "returned_candidates": 1,
            "metrics": [
                {
                    "tag": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                    "metric_name": "RevenueFromContractWithCustomerExcludingAssessedTax",
                }
            ],
        }

    monkeypatch.setattr(server, "_call_api", fake_call_api)

    result = server._proxy_list_metrics(
        {
            "ticker": "pcty",
            "year": 2025,
            "quarter": 4,
            "date_type": "FY",
            "limit": 5000,
            "include_values": False,
        }
    )

    assert result["status"] == "success"
    assert calls == [
        (
            "/api/financials/list_metrics",
            {
                "ticker": "pcty",
                "year": 2025,
                "quarter": 4,
                "full_year_mode": "true",
                "source": "auto",
                "limit": 1000,
                "include_values": "false",
                "date_type": "FY",
            },
            300,
        )
    ]
    assert result["metrics"][0]["metric_name"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_proxy_search_metrics_forwards_to_http_endpoint(monkeypatch):
    calls = []

    def fake_call_api(path, params, timeout=300):
        calls.append((path, params, timeout))
        return {
            "status": "success",
            "ticker": "PCTY",
            "query": params["query"],
            "total_matches": 1,
            "matches": [
                {
                    "tag": "us-gaap:LongTermDebtCurrent",
                    "metric_name": "LongTermDebtCurrent",
                    "match_score": 100.0,
                },
            ],
            "low_confidence": False,
            "confidence_reason": None,
        }

    monkeypatch.setattr(server, "_call_api", fake_call_api)

    result = server._proxy_search_metrics(
        {
            "ticker": "PCTY",
            "year": 2025,
            "quarter": 3,
            "query": " debt ",
            "source": "8k",
            "date_type": "Q",
            "role": ["cash-flow", "balance_sheet"],
            "limit": 150,
            "include_values": False,
        }
    )

    assert result["status"] == "success"
    assert calls == [
        (
            "/api/financials/search_metrics",
            {
                "ticker": "PCTY",
                "year": 2025,
                "quarter": 3,
                "query": "debt",
                "full_year_mode": "false",
                "source": "8k",
                "limit": 100,
                "include_values": "false",
                "date_type": "Q",
                "role": "balance_sheet,cash_flow",
            },
            300,
        )
    ]
    assert result["matches"][0]["metric_name"] == "LongTermDebtCurrent"


def test_proxy_search_metrics_rejects_blank_query_without_api_call(monkeypatch):
    def fake_call_api(*args, **kwargs):
        raise AssertionError("_call_api should not be called")

    monkeypatch.setattr(server, "_call_api", fake_call_api)

    assert server._proxy_search_metrics(
        {
            "ticker": "PCTY",
            "year": 2025,
            "quarter": 4,
            "query": " ",
        }
    ) == {"status": "error", "message": "Missing required parameter: query"}


def test_proxy_search_metrics_rejects_unknown_role_without_api_call(monkeypatch):
    def fake_call_api(*args, **kwargs):
        raise AssertionError("_call_api should not be called")

    monkeypatch.setattr(server, "_call_api", fake_call_api)

    assert server._proxy_search_metrics(
        {
            "ticker": "PCTY",
            "year": 2025,
            "quarter": 4,
            "query": "revenue",
            "role": ["cashflows"],
        }
    ) == {"status": "error", "message": "Unknown statement role: cashflows"}


def test_proxy_list_metrics_rejects_invalid_limit_without_api_call(monkeypatch):
    def fake_call_api(*args, **kwargs):
        raise AssertionError("_call_api should not be called")

    monkeypatch.setattr(server, "_call_api", fake_call_api)

    assert server._proxy_list_metrics(
        {
            "ticker": "PCTY",
            "year": 2025,
            "quarter": 4,
            "limit": "many",
        }
    ) == {"status": "error", "message": "limit must be an integer between 1 and 1000"}
