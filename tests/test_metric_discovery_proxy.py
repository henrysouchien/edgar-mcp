from edgar_mcp import server


def test_proxy_list_metrics_forwards_to_http_endpoint(monkeypatch):
    calls = []

    def fake_call_api(path, params, timeout=300):
        calls.append((path, params, timeout))
        return {
            "status": "success",
            "metadata": {"source": {"filing_type": "10-K"}},
            "facts": [
                {
                    "tag": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                    "concept_label": "Revenue from Contract with Customer, Excluding Assessed Tax",
                    "date_type": "FY",
                    "current_period_value": 100.0,
                    "prior_period_value": 90.0,
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
            "/api/financials",
            {
                "ticker": "pcty",
                "year": 2025,
                "quarter": 4,
                "full_year_mode": "true",
                "source": "auto",
            },
            300,
        )
    ]
    assert result["metrics"][0]["metric_name"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert "current_value" not in result["metrics"][0]


def test_proxy_search_metrics_forwards_to_http_endpoint(monkeypatch):
    calls = []

    def fake_call_api(path, params, timeout=300):
        calls.append((path, params, timeout))
        return {
            "status": "success",
            "metadata": {"source": {"filing_type": "10-Q"}},
            "facts": [
                {
                    "tag": "us-gaap:LongTermDebtCurrent",
                    "concept_label": "Long-Term Debt, Current",
                    "date_type": "Q",
                    "current_period_value": 10.0,
                    "prior_period_value": 8.0,
                    "presentation_role": "CONSOLIDATED BALANCE SHEETS",
                },
                {
                    "tag": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                    "concept_label": "Revenue",
                    "date_type": "Q",
                    "current_period_value": 100.0,
                    "prior_period_value": 90.0,
                    "presentation_role": "CONSOLIDATED STATEMENTS OF OPERATIONS",
                },
            ],
        }

    monkeypatch.setattr(server, "_call_api", fake_call_api)

    result = server._proxy_search_metrics(
        {
            "ticker": "PCTY",
            "year": 2025,
            "quarter": 3,
            "query": " debt ",
            "source": "8k",
            "date_type": "bad-value",
            "role": ["cash-flow", "balance_sheet"],
            "limit": 150,
            "include_values": "false",
        }
    )

    assert result["status"] == "success"
    assert calls == [
        (
            "/api/financials",
            {
                "ticker": "PCTY",
                "year": 2025,
                "quarter": 3,
                "full_year_mode": "false",
                "source": "8k",
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
