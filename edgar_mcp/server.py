#!/usr/bin/env python3
"""
MCP Server for EDGAR Financial Data.

Proxies all tools to the remote EDGAR API (edgarparser.com).
Self-contained client helpers only; it does not run the local filing pipeline.
"""

# CRITICAL: Redirect stdout to stderr before imports.
# MCP uses stdout for JSON-RPC traffic.
import sys

_real_stdout = sys.stdout
sys.stdout = sys.stderr

import asyncio
import json
import os
import re
import time
from contextlib import redirect_stdout
from difflib import SequenceMatcher
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Annotated, Any, Literal

import requests
from fastmcp import FastMCP
from pydantic import Field

# Restore stdout for MCP transport.
sys.stdout = _real_stdout


ROW_CLASS_OPERATIONAL_KPI = "operational_kpi"
ROW_CLASS_MARGIN_RATE = "margin_or_rate"
ROW_CLASS_GROWTH_METRIC = "growth_metric"
ROW_CLASS_SEGMENT_MEMBER = "segment_member"
_ROW_CLASS_SUBTOTAL = "subtotal_or_total"
_ROW_CLASS_UNKNOWN = "unknown"

_ROLE_DISCLOSURE_REJECT = (
    "details",
    "detail",
    "additional",
    "schedule",
    "computation",
    "narrative",
    "supplemental",
    "disclosure",
)
_ROLE_BS_RE = re.compile(
    r"(balancesheets?)"
    r"|(statements?of.*(balance|financialposition|financialcondition|condition|capitalization))"
    r"|(financialposition|financialcondition)"
)
_ROLE_IS_RE = re.compile(
    r"(incomestatements?)"
    r"|((?!.*comprehensive)statements?of.*(income|earnings|operations))"
)
_ROLE_CI_RE = re.compile(r"(statements?of.*comprehensive)|(comprehensive(income|loss))")
_ROLE_EQ_RE = re.compile(
    r"((stockholders|shareholders|shareowners|partners|member)(equity|deficit|investment|capital))"
    r"|(statements?of.*(equity|capital|investment|deficit))"
    r"|(changesin.*(equity|capital|investment))"
)
_ROLE_CF_RE = re.compile(r"(cashflow(s)?statement(s)?)|(statements?of.*cashflow)|(cashflow)")
_ROLE_BUCKETS = (
    ("BS", _ROLE_BS_RE),
    ("IS", _ROLE_IS_RE),
    ("CI", _ROLE_CI_RE),
    ("EQ", _ROLE_EQ_RE),
    ("CF", _ROLE_CF_RE),
    ("DISC", None),
)
_FRIENDLY_ROLE_NAMES = (
    "balance_sheet",
    "income_statement",
    "comprehensive_income",
    "equity",
    "cash_flow",
    "other",
)
_FRIENDLY_ROLE_BY_BUCKET = {
    "BS": "balance_sheet",
    "IS": "income_statement",
    "CI": "comprehensive_income",
    "EQ": "equity",
    "CF": "cash_flow",
    "DISC": "other",
}

_RATIO_LABEL_KEYWORDS = ("margin", "rate", "ratio", "yield", "penetration", "take rate", "mix")
_PRICE_KEYWORDS = (
    "price",
    "pricing",
    "fee",
    "fees",
    "arpu",
    "average selling price",
    "average revenue per",
    "average monthly revenue",
    "revenue per",
)
_USER_KEYWORDS = ("active rider", "mapc", "member", "subscriber", "cardholder", "rider", "user", "customer", "consumer")
_FOOTPRINT_KEYWORDS = ("warehouse", "store", "facility", "headcount", "employee", "location")
_RETENTION_KEYWORDS = ("retention", "renewal", "churn", "penetration", "net retention")
_PIPELINE_KEYWORDS = ("pipeline", "phase 1", "phase 2", "phase 3", "trial")
_COMP_KEYWORDS = ("comparable", "same-store", "same store", "comp")
_VOLUME_KEYWORDS = (
    "annual recurring revenue",
    "annual run-rate revenue",
    "arr expansion rate",
    "booking",
    "booked",
    "gov",
    "gross booking value",
    "gross order value",
    "ride",
    "trip",
    "order",
    "shipment",
    "volume",
    "unit",
    "utilization",
)


def _role_canonicalize(role: str) -> str:
    return str(role or "").lower().replace("-", "").replace("_", "").replace(" ", "")


def _role_bucket(role: str) -> str:
    canon = _role_canonicalize(role)
    if not canon:
        return "DISC"
    if any(marker in canon for marker in _ROLE_DISCLOSURE_REJECT):
        return "DISC"
    for name, pattern in _ROLE_BUCKETS:
        if pattern is not None and pattern.search(canon):
            return name
    return "DISC"


def _friendly_role(bucket_name: str) -> str:
    return _FRIENDLY_ROLE_BY_BUCKET.get(bucket_name, "other")


def normalize_role_filter(raw: str | None) -> tuple[str, ...]:
    """Canonicalize comma-separated statement role filters for MCP requests."""
    if raw is None:
        return ()
    values = {
        token.strip().lower().replace("-", "_")
        for token in raw.split(",")
        if token.strip()
    }
    unknown = sorted(values - set(_FRIENDLY_ROLE_NAMES))
    if unknown:
        raise ValueError(f"Unknown statement role: {unknown[0]}")
    return tuple(sorted(values))


def enrich_match_metadata(fact: dict) -> dict[str, Any]:
    """Return statement-role metadata derived from one API fact."""
    raw_role = fact.get("presentation_role")
    roles = []
    if raw_role is not None:
        roles = [role.strip() for role in str(raw_role).split("|") if role.strip()]

    presentation_roles = sorted(set(roles))
    statement_roles = sorted({_friendly_role(_role_bucket(role)) for role in roles})
    metadata: dict[str, Any] = {
        "presentation_roles": presentation_roles,
        "statement_roles": statement_roles,
        "statement_position": (
            "aggregate"
            if fact.get("axis_key") in (None, "", "__NONE__")
            else "dimensional"
        ),
    }
    if len(statement_roles) == 1:
        metadata["statement_role"] = statement_roles[0]
    return metadata


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def classify_operating_metric_label(label: str) -> tuple[str, str | None]:
    """Classify a filing-native operational label without importing parser internals."""
    lowered = re.sub(r"[^a-z0-9]+", " ", str(label or "").lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if (
        _contains_any(lowered, ("revenue", "sales"))
        and _contains_any(lowered, ("segment", "product", "geograph", "region", "category", "channel", "brand"))
    ):
        return ROW_CLASS_SEGMENT_MEMBER, "volume_metric"
    if _contains_any(lowered, _PRICE_KEYWORDS):
        return ROW_CLASS_OPERATIONAL_KPI, "pricing_metric"
    if _contains_any(lowered, _RETENTION_KEYWORDS):
        return ROW_CLASS_OPERATIONAL_KPI, "retention_metric"
    if _contains_any(lowered, _USER_KEYWORDS):
        return ROW_CLASS_OPERATIONAL_KPI, "user_metric"
    if _contains_any(lowered, _FOOTPRINT_KEYWORDS):
        return ROW_CLASS_OPERATIONAL_KPI, "footprint_metric"
    if _contains_any(lowered, _PIPELINE_KEYWORDS):
        return ROW_CLASS_OPERATIONAL_KPI, "pipeline_metric"
    if _contains_any(lowered, _COMP_KEYWORDS):
        return ROW_CLASS_GROWTH_METRIC, "comp_metric"
    if _contains_any(lowered, _VOLUME_KEYWORDS):
        return ROW_CLASS_OPERATIONAL_KPI, "volume_metric"
    if _contains_any(lowered, _RATIO_LABEL_KEYWORDS):
        return ROW_CLASS_MARGIN_RATE, "pricing_metric"
    if _contains_any(lowered, ("total", "subtotal")):
        return _ROW_CLASS_SUBTOTAL, None
    return _ROW_CLASS_UNKNOWN, None


def _get_output_dir() -> Path:
    env_val = os.getenv("EDGAR_MCP_OUTPUT_DIR", "").strip()
    if env_val:
        output_dir = Path(env_val)
    else:
        output_dir = Path.cwd() / "exports" / "file_output"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_test"
        probe.touch()
        probe.unlink()
    except OSError:
        output_dir = Path.home() / ".cache" / "edgar-mcp" / "file_output"
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


FILE_OUTPUT_DIR = _get_output_dir()

ExtractionPeriod = Annotated[
    str,
    Field(
        description=(
            "Inclusive period bound in YYYY-Qn or YYYY-FY format, for example "
            "2024-Q4 or 2024-FY. ISO dates such as 2024-12-31 are not accepted."
        )
    ),
]

ConceptPeriod = Annotated[
    str,
    Field(
        description=(
            "Inclusive concept period bound in FYyyyy, YYYY-FY, or YYYY-Qn format, "
            "for example FY2024, 2024-FY, or 2024-Q4. ISO dates are not accepted."
        )
    ),
]

mcp = FastMCP(
    "edgar-parser-mcp",
    instructions=(
        "Parser-backed SEC EDGAR filing and financial data tools. Use get_filings for filing metadata, "
        "get_financials for full facts, get_metric for specific facts, "
        "get_metric_series for multi-period metric time series, warm_metric_cache "
        "to enqueue async cache warming, warm_metric_cache_status to poll warm jobs, "
        "get_concept for registry-backed concept values from cached financials, "
        "compare_concept for caller-ordered cross-filer concept comparison, "
        "concept_trend for cache-only concept time series without restatement fields, "
        "get_statement for deterministic template-backed structured statements, "
        "list_metrics/search_metrics "
        "for discovery, describe_filing to inspect cached layer availability, "
        "search_filing_text for same-filing cached markdown text search, "
        "get_operational_kpi_drivers for structured operational KPI values and driver rates, "
        "get_filing_evidence to plan and retrieve same-filing evidence in one call, "
        "get_filing_cover_facts for exact cover-page DEI facts such as shares outstanding, "
        "get_filing_sections for narrative/table sections, "
        "get_filing_document for readable sectioned markdown with pagination, "
        "get_filing_tables for structured table lookup, and "
        "get_filing_extractions for cache-or-extract filing langextract spans, "
        "search_extractions for read-only cross-filing structured span search, "
        "get_extraction_series for time-series counts over cached langextract spans, "
        "search_filing_tables for cross-filing table metadata search, "
        "compare_filing_tables for caller-ordered cross-filer table comparison, "
        "extract_filing_file for ad-hoc local markdown extraction, and "
        "list_extraction_schemas for schema discovery."
    ),
)

# ---------------------------------------------------------------------------
# Remote API helpers
# ---------------------------------------------------------------------------

def _get_api_config():
    base_url = os.getenv("EDGAR_API_URL", "https://www.edgarparser.com").rstrip("/")
    api_key = os.getenv("EDGAR_API_KEY", "")
    return base_url, api_key


def _call_api(path: str, params: dict, timeout: int = 300) -> dict:
    """HTTP GET to the remote EDGAR API. Returns parsed JSON or error dict."""
    base_url, api_key = _get_api_config()
    if not api_key:
        return {"status": "error", "message": "EDGAR_API_KEY is not configured"}

    url = f"{base_url}{path}"
    payload = dict(params)
    payload["key"] = api_key

    t0 = time.time()
    try:
        resp = requests.get(url, params=payload, timeout=timeout)
    except requests.RequestException as exc:
        return {"status": "error", "message": f"EDGAR API request failed after {time.time()-t0:.1f}s: {exc}"}

    try:
        data = resp.json()
    except ValueError:
        return {"status": "error", "message": f"Invalid JSON from EDGAR API (HTTP {resp.status_code})"}

    if resp.status_code != 200:
        if isinstance(data, dict) and data:
            return data
        return {"status": "error", "message": f"EDGAR API error (HTTP {resp.status_code})"}

    return data


def _post_api(path: str, payload: dict, timeout: int = 300) -> dict:
    """HTTP POST to the remote EDGAR API. Returns parsed JSON or error dict."""
    base_url, api_key = _get_api_config()
    if not api_key:
        return {"status": "error", "message": "EDGAR_API_KEY is not configured"}

    url = f"{base_url}{path}"
    params = {"key": api_key}

    t0 = time.time()
    try:
        resp = requests.post(url, params=params, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return {"status": "error", "message": f"EDGAR API request failed after {time.time()-t0:.1f}s: {exc}"}

    try:
        data = resp.json()
    except ValueError:
        return {"status": "error", "message": f"Invalid JSON from EDGAR API (HTTP {resp.status_code})"}

    if resp.status_code != 200:
        if isinstance(data, dict) and data:
            return data
        return {"status": "error", "message": f"EDGAR API error (HTTP {resp.status_code})"}

    return data


def _allowed_output_roots() -> list[Path]:
    raw_roots: list[Path] = []

    env_root = os.getenv("EDGAR_MCP_OUTPUT_DIR", "").strip()
    if env_root:
        raw_roots.append(Path(env_root).expanduser())

    api_root = Path(
        os.getenv("EDGAR_API_ROOT", str(Path(__file__).resolve().parent))
    ).expanduser()
    document_service_filings_dir = Path(
        os.getenv(
            "DOCUMENT_SERVICE_FILINGS_DIR",
            str(api_root / "data" / "filings"),
        )
    ).expanduser()

    raw_roots.append(Path.cwd() / "exports" / "file_output")
    raw_roots.append(Path("~/.cache/edgar-mcp/file_output").expanduser())
    raw_roots.append(Path.cwd() / "data" / "filings")
    raw_roots.append(document_service_filings_dir)

    roots: list[Path] = []
    seen: set[str] = set()
    for root in raw_roots:
        resolved = root.resolve(strict=False)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def validate_file_path(file_path: str) -> str:
    """Resolve a filing path and ensure it is under an allowed EDGAR output root."""
    candidate = Path(file_path).expanduser()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")

    for root in _allowed_output_roots():
        try:
            resolved.relative_to(root)
            return str(resolved)
        except ValueError:
            continue

    allowed = ", ".join(str(root) for root in _allowed_output_roots())
    raise ValueError(f"File path is outside allowed EDGAR output roots: {resolved}. Allowed roots: {allowed}")


def _safe_filename_part(value: str, fallback: str) -> str:
    """Normalize untrusted text into a filesystem-safe filename segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-_")
    return cleaned or fallback


def _pick_metric_values(fact: dict) -> tuple[object, object]:
    """Mirror /api/metric value precedence for a fact record."""
    current = fact.get("current_value")
    if current is None:
        current = fact.get("visual_current_value")
    if current is None:
        current = fact.get("current_period_value")

    prior = fact.get("prior_value")
    if prior is None:
        prior = fact.get("visual_prior_value")
    if prior is None:
        prior = fact.get("prior_period_value")

    return current, prior


def _split_identifier_tokens(value: object) -> list[str]:
    """
    Tokenize metric/tag text for fuzzy matching.
    Handles namespace separators, camel/Pascal case, and punctuation.
    """
    if value is None:
        return []

    text = str(value)
    text = text.replace(":", " ").replace("/", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return [token for token in text.lower().split() if token]


_SEARCH_QUERY_PHRASE_ALIASES = {
    "eps": ["earnings", "per", "share"],
    "diluted eps": ["earnings", "per", "share", "diluted"],
    "basic eps": ["earnings", "per", "share", "basic"],
    "revenue": ["revenue", "from", "contract", "with", "customer"],
    "capex": ["capital", "expenditures", "payments", "to", "acquire", "property", "plant", "and", "equipment"],
    "d a": ["depreciation", "and", "amortization"],
    "da": ["depreciation", "and", "amortization"],
    "sg a": ["selling", "general", "and", "administrative"],
    "sga": ["selling", "general", "and", "administrative"],
    "r d": ["research", "and", "development"],
    "cogs": ["cost", "of", "goods", "sold", "cost", "of", "revenue"],
    "fcf": ["free", "cash", "flow"],
    "cfo": ["operating", "cash", "flow"],
    "ocf": ["operating", "cash", "flow", "net", "cash", "provided", "by", "operating", "activities"],
    "ppe": ["property", "plant", "and", "equipment"],
    "goodwill": ["goodwill"],
    "shares outstanding": ["common", "stock", "shares", "outstanding"],
}

_SEARCH_QUERY_TOKEN_ALIASES = {
    "eps": ["earnings", "per", "share"],
    "rev": ["revenue"],
    "capex": ["capital", "expenditures"],
    "ocf": ["operating", "cash", "flow"],
    "fcf": ["free", "cash", "flow"],
    "cogs": ["cost", "of", "revenue"],
    "ppe": ["property", "plant", "and", "equipment"],
    "sga": ["selling", "general", "and", "administrative"],
    "da": ["depreciation", "and", "amortization"],
    "cfo": ["operating", "cash", "flow"],
}

_METRIC_SEARCH_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "including",
    "excluding",
    "of",
    "on",
    "the",
    "to",
    "with",
}

_METRIC_SEARCH_FAMILY_BASE_TOKENS = {
    "assets_total": {"asset", "assets", "total"},
    "assets_current": {"asset", "assets", "current"},
    "assets_noncurrent": {"asset", "assets", "non", "noncurrent"},
    "cash_and_cash_equivalents": {"cash", "equivalent", "equivalents"},
    "cash_restricted_total": {"cash", "equivalent", "equivalents", "restricted", "total"},
    "operating_cash_flow": {
        "activities",
        "activity",
        "cash",
        "cfo",
        "flow",
        "net",
        "ocf",
        "operating",
        "provided",
        "used",
    },
    "debt_total": {
        "borrowing",
        "borrowings",
        "capital",
        "debt",
        "lease",
        "leases",
        "long",
        "obligation",
        "obligations",
        "term",
        "total",
    },
    "debt_current": {"borrowing", "borrowings", "current", "debt", "lease", "leases"},
    "debt_noncurrent": {"borrowing", "borrowings", "debt", "lease", "leases", "non", "noncurrent"},
    "equity_total": {"equity", "shareholder", "shareholders", "stockholder", "stockholders", "total"},
    "liabilities_and_equity": {
        "equity",
        "liabilities",
        "liability",
        "shareholder",
        "shareholders",
        "stockholder",
        "stockholders",
        "total",
    },
    "liabilities_total": {"liabilities", "liability", "total"},
    "liabilities_current": {"current", "liabilities", "liability"},
    "liabilities_noncurrent": {"liabilities", "liability", "non", "noncurrent"},
    "operating_income": {"income", "loss", "operating"},
    "revenue": {
        "assessed",
        "contract",
        "customer",
        "customers",
        "net",
        "revenue",
        "revenues",
        "sale",
        "sales",
        "tax",
        "total",
    },
}

_METRIC_SEARCH_OPERATIONAL_REVENUE_TOKENS = {
    "account",
    "accounts",
    "customer",
    "customers",
    "member",
    "members",
    "membership",
    "memberships",
    "subscriber",
    "subscribers",
    "subscription",
    "subscriptions",
    "user",
    "users",
}

_METRIC_SEARCH_REQUIRED_TOKEN_ALIASES = {
    "account": {"account", "accounts"},
    "accounts": {"account", "accounts"},
    "customer": {"customer", "customers"},
    "customers": {"customer", "customers"},
    "member": {"member", "members", "membership", "memberships"},
    "members": {"member", "members", "membership", "memberships"},
    "membership": {"member", "members", "membership", "memberships"},
    "memberships": {"member", "members", "membership", "memberships"},
    "subscriber": {"subscriber", "subscribers", "subscription", "subscriptions"},
    "subscribers": {"subscriber", "subscribers", "subscription", "subscriptions"},
    "subscription": {"subscriber", "subscribers", "subscription", "subscriptions"},
    "subscriptions": {"subscriber", "subscribers", "subscription", "subscriptions"},
    "user": {"user", "users"},
    "users": {"user", "users"},
    "continuing": {"continuing", "continued"},
    "continued": {"continuing", "continued"},
    "discontinued": {"discontinued"},
    "refinance": {"instrument", "principal", "unsecured"},
    "refinanceable": {"instrument", "principal", "unsecured"},
    "refinanced": {"instrument", "principal", "unsecured"},
    "refinancing": {"instrument", "principal", "unsecured"},
    "tranche": {"instrument", "principal", "unsecured"},
    "tranches": {"instrument", "principal", "unsecured"},
    "operation": {"operation", "operations"},
    "operations": {"operation", "operations"},
}


def _expand_query_variants(query: str) -> list[list[str]]:
    """Return tokenized query variants for robust matching."""
    base_tokens = _split_identifier_tokens(query)
    if not base_tokens:
        return []

    variants = {tuple(base_tokens)}

    base_phrase = " ".join(base_tokens)
    phrase_alias = _SEARCH_QUERY_PHRASE_ALIASES.get(base_phrase)
    if phrase_alias:
        variants.add(tuple(phrase_alias))

    for i, token in enumerate(base_tokens):
        replacement = _SEARCH_QUERY_TOKEN_ALIASES.get(token)
        if replacement:
            expanded = base_tokens[:i] + replacement + base_tokens[i + 1 :]
            variants.add(tuple(expanded))
            if token == "eps" and "diluted" in base_tokens:
                variants.add(tuple(["earnings", "per", "share", "diluted"]))
            if token == "eps" and "basic" in base_tokens:
                variants.add(tuple(["earnings", "per", "share", "basic"]))

    return [list(variant) for variant in variants]


def _meaningful_metric_search_tokens(tokens: list[str]) -> set[str]:
    return {token for token in tokens if token not in _METRIC_SEARCH_STOP_TOKENS}


def _metric_search_tokens(metric: dict) -> list[str]:
    metric_tokens = _split_identifier_tokens(metric.get("metric_name", ""))
    metric_tokens += _split_identifier_tokens(metric.get("tag", ""))
    metric_tokens += _split_identifier_tokens(metric.get("concept_label", ""))
    metric_tokens += _split_identifier_tokens(metric.get("debt_component_kind", ""))
    date_type = _normalize_date_type(metric.get("date_type"))
    if date_type:
        metric_tokens.append(date_type.lower())
    return metric_tokens


def _metric_search_token_sets(metric: dict) -> tuple[set[str], set[str]]:
    metric_tokens = _metric_search_tokens(metric)
    dimension_tokens = set()
    for dim in metric.get("dimensions") or []:
        if not isinstance(dim, dict):
            continue
        for field in (dim.get("axis_label", ""), dim.get("member_label", "")):
            tokens = _split_identifier_tokens(field)
            dimension_tokens.update(tokens)
            if tokens:
                dimension_tokens.add("".join(tokens))

    return set(metric_tokens), dimension_tokens


def _metric_search_query_profile(query: str, query_family: str | None) -> dict:
    tokens = _meaningful_metric_search_tokens(_split_identifier_tokens(query))
    family_tokens = _METRIC_SEARCH_FAMILY_BASE_TOKENS.get(query_family or "", set())
    if query_family == "operating_cash_flow" and not (tokens & {"continuing", "continued", "discontinued"}):
        family_tokens = family_tokens | {"operation", "operations"}
    required_modifiers = set()
    if query_family:
        required_modifiers = {
            token
            for token in tokens
            if token not in family_tokens
        }
    is_debt_instrument_basis_query = (
        query_family == "debt_total"
        and bool(tokens & _DEBT_QUERY_TOKENS)
        and bool(tokens & _DEBT_INSTRUMENT_BASIS_QUERY_TOKENS)
    )
    if is_debt_instrument_basis_query:
        required_modifiers -= _DEBT_SCENARIO_QUERY_TOKENS

    is_operational_revenue_kpi = (
        query_family == "revenue"
        and bool(tokens & _METRIC_SEARCH_OPERATIONAL_REVENUE_TOKENS)
        and bool(tokens & {"average", "per", "arpu", "arm"})
    )
    if is_operational_revenue_kpi:
        required_modifiers.update(tokens & _METRIC_SEARCH_OPERATIONAL_REVENUE_TOKENS)
        required_modifiers.update(tokens & {"average", "per", "arpu", "arm"})

    return {
        "tokens": tokens,
        "required_modifiers": required_modifiers,
        "is_debt_instrument_basis_query": is_debt_instrument_basis_query,
        "is_operational_revenue_kpi": is_operational_revenue_kpi,
    }


def _metric_search_required_modifier_evidence(query_profile: dict, metric: dict) -> dict:
    required_modifiers = set(query_profile.get("required_modifiers") or [])
    if not required_modifiers:
        return {"matched": set(), "unmatched": set()}

    metric_tokens, dimension_tokens = _metric_search_token_sets(metric)
    candidate_tokens = metric_tokens | dimension_tokens
    matched = set()
    for token in required_modifiers:
        aliases = _METRIC_SEARCH_REQUIRED_TOKEN_ALIASES.get(token, {token})
        if aliases & candidate_tokens:
            matched.add(token)

    return {
        "matched": matched,
        "unmatched": required_modifiers - matched,
    }


def _metric_search_apply_modifier_gate(
    score: float,
    *,
    query_profile: dict,
    modifier_evidence: dict,
    semantic_relation: str | None,
) -> float:
    required_modifiers = set(query_profile.get("required_modifiers") or [])
    if not required_modifiers or not score:
        return score

    matched = set(modifier_evidence.get("matched") or [])
    unmatched = set(modifier_evidence.get("unmatched") or [])
    if not unmatched:
        return score

    if not matched:
        if semantic_relation == "exact":
            return min(score, 60.0)
        if semantic_relation == "related":
            return min(score, 54.0)
        return 0.0 if score < 90.0 else min(score, 72.0)

    modifier_coverage = len(matched) / max(len(required_modifiers), 1)
    return min(score, 64.0 + (modifier_coverage * 16.0))


def _metric_search_confidence(query_profile: dict, ranked: list[dict]) -> tuple[bool, str | None]:
    required_modifiers = set(query_profile.get("required_modifiers") or [])
    if not ranked:
        return True, "No XBRL metrics matched the query."

    if required_modifiers:
        strong_match = any(
            not match.get("unmatched_query_modifiers")
            and float(match.get("match_score") or 0) >= 75.0
            for match in ranked
        )
        if not strong_match:
            missing = sorted(required_modifiers)
            reason = (
                "No strong XBRL metric matched required query modifiers: "
                + ", ".join(missing)
                + "."
            )
            if query_profile.get("is_operational_revenue_kpi"):
                reason += " This looks like an operational KPI that may live in narrative tables, not tagged XBRL."
            return True, reason

    top_scores = {match.get("match_score") for match in ranked[: min(len(ranked), 5)]}
    top_score = float(ranked[0].get("match_score") or 0)
    if len(top_scores) == 1 and len(ranked) > 1 and top_score < 75.0:
        return True, "Top candidates have uniform low scores; validate manually before using a match."

    return False, None


def _normalize_date_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in {"Q", "YTD", "FY"} else None


def _truthy_bool_arg(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _effective_full_year_mode_for_metric_request(
    *,
    quarter: object,
    full_year_mode: object,
    date_type: object,
) -> bool:
    if _truthy_bool_arg(full_year_mode):
        return True
    try:
        requested_quarter = int(quarter)
    except (TypeError, ValueError):
        return False
    return requested_quarter == 4 and _normalize_date_type(date_type) == "FY"


def _role_query_param(role: object) -> str | None:
    if role is None:
        return None
    if isinstance(role, str):
        raw = role
    else:
        try:
            items = list(role)  # type: ignore[arg-type]
        except TypeError:
            raw = str(role)
        else:
            if not items:
                return None
            raw = ",".join(str(item) for item in items)

    normalized = normalize_role_filter(raw)
    return ",".join(normalized) if normalized else None


def _normalize_tag_local_name(value: object) -> str:
    raw = str(value or "")
    local = raw.split(":", 1)[1] if ":" in raw else raw
    return re.sub(r"[^a-z0-9]", "", local.lower())


_BALANCE_SHEET_LOCAL_NAMES = {
    "assets",
    "assetscurrent",
    "assetsnoncurrent",
    "cashandcashequivalentsatcarryingvalue",
    "cashcashequivalentsrestrictedcashandrestrictedcashequivalents",
    "liabilities",
    "liabilitiescurrent",
    "liabilitiesnoncurrent",
    "liabilitiesandstockholdersequity",
    "liabilitiesandstockholdersequityincludingportionattributabletononcontrollinginterest",
    "debtandcapitalleaseobligations",
    "debtcurrent",
    "financeleaseliability",
    "financeleaseliabilitycurrent",
    "longtermdebtandcapitalleaseobligations",
    "longtermdebtcurrent",
    "longtermdebtnoncurrent",
    "othernotespayable",
    "othernotespayablecurrent",
    "stockholdersequity",
    "stockholdersequityincludingportionattributabletononcontrollinginterest",
    "unsecureddebt",
    "unsecureddebtcurrent",
}


def _dimension_tokens(fact: dict) -> set[str]:
    tokens: set[str] = set()
    axis_key = _normalize_tag_local_name(fact.get("axis_key"))
    if axis_key:
        tokens.add(axis_key)

    dimensions = fact.get("dimensions")
    if isinstance(dimensions, list):
        for dimension in dimensions:
            if not isinstance(dimension, dict):
                continue
            for field in (
                "axis",
                "axis_label",
                "member",
                "member_label",
                "dimension",
                "axis_name",
                "member_name",
            ):
                normalized = _normalize_tag_local_name(dimension.get(field))
                if normalized:
                    tokens.add(normalized)
    return tokens


def _debt_component_kind(fact: dict) -> str | None:
    local_name = _normalize_tag_local_name(fact.get("tag"))
    if not local_name:
        return None

    if local_name == "debtinstrumentinterestratestatedpercentage":
        return "coupon_rate"
    if local_name.startswith("financeleaseliability"):
        return "finance_lease"
    if local_name.startswith("othernotespayable"):
        return "other_notes"
    if local_name in {
        "debtandcapitalleaseobligations",
        "longtermdebtandcapitalleaseobligations",
    }:
        return "total_debt_rollup"
    if local_name in {"debtcurrent", "longtermdebtcurrent", "unsecureddebtcurrent"}:
        return "current_debt"
    if local_name in {"longtermdebt", "longtermdebtnoncurrent"}:
        return "debt_carrying_amount"
    if local_name.startswith("longtermdebtmaturitiesrepaymentsofprincipal"):
        return "debt_maturity_bucket"
    if local_name == "unsecureddebt":
        dimensions = _dimension_tokens(fact)
        if any("debtinstrumentaxis" in token for token in dimensions):
            return "instrument_principal"
        return "unsecured_debt"

    return None


_METRIC_SEMANTIC_FAMILIES_BY_LOCAL_NAME = {
    "assets": "assets_total",
    "assetscurrent": "assets_current",
    "assetsnoncurrent": "assets_noncurrent",
    "cashandcashequivalentsatcarryingvalue": "cash_and_cash_equivalents",
    "cashcashequivalentsrestrictedcashandrestrictedcashequivalents": "cash_restricted_total",
    "liabilities": "liabilities_total",
    "liabilitiescurrent": "liabilities_current",
    "liabilitiesnoncurrent": "liabilities_noncurrent",
    "liabilitiesandpartnerscapital": "liabilities_and_equity",
    "liabilitiesandstockholdersequity": "liabilities_and_equity",
    "liabilitiesandstockholdersequityincludingportionattributabletononcontrollinginterest": "liabilities_and_equity",
    "debtandcapitalleaseobligations": "debt_total",
    "debtcurrent": "debt_current",
    "longtermdebtandcapitalleaseobligations": "debt_total",
    "longtermdebtcurrent": "debt_current",
    "longtermdebt": "debt_carrying_amount",
    "longtermdebtnoncurrent": "debt_noncurrent",
    "netcashprovidedbyusedinoperatingactivities": "operating_cash_flow",
    "netcashprovidedbyusedinoperatingactivitiescontinuingoperations": "operating_cash_flow",
    "operatingincomeloss": "operating_income",
    "revenue": "revenue",
    "revenues": "revenue",
    "revenuefromcontractwithcustomerexcludingassessedtax": "revenue",
    "revenuefromcontractwithcustomerincludingassessedtax": "revenue",
    "salesrevenuegoodsnet": "revenue",
    "salesrevenuenet": "revenue",
    "salesrevenueservicesnet": "revenue",
    "stockholdersequity": "equity_total",
    "stockholdersequityincludingportionattributabletononcontrollinginterest": "equity_total",
    "totalnetsales": "revenue",
}

_LIABILITY_QUERY_TOKENS = {"liability", "liabilities"}
_EQUITY_QUERY_TOKENS = {
    "equity",
    "stockholder",
    "stockholders",
    "shareholder",
    "shareholders",
}
_ASSET_QUERY_TOKENS = {"asset", "assets"}
_DEBT_QUERY_TOKENS = {"debt", "borrowings", "borrowing"}
_DEBT_INSTRUMENT_BASIS_QUERY_TOKENS = {
    "bearing",
    "refinance",
    "refinanceable",
    "refinanced",
    "refinancing",
    "tranche",
    "tranches",
}
_DEBT_SCENARIO_QUERY_TOKENS = _DEBT_INSTRUMENT_BASIS_QUERY_TOKENS | {
    "higher",
    "interest",
    "rate",
    "rates",
}
_OPERATING_CASH_FLOW_QUERY_TOKENS = {
    "activities",
    "activity",
    "cfo",
    "ocf",
    "operating",
    "operation",
    "operations",
}
_NON_OPERATING_CASH_FLOW_QUERY_TOKENS = {
    "financing",
    "investing",
}
_FLOW_QUERY_TOKENS = {
    "issuance",
    "issued",
    "proceeds",
    "repayment",
    "repayments",
    "payment",
    "payments",
}

_RELATED_METRIC_FAMILIES = {
    "assets_total": {"assets_current", "assets_noncurrent"},
    "assets_current": {"assets_total"},
    "assets_noncurrent": {"assets_total"},
    "cash_and_cash_equivalents": {"cash_restricted_total"},
    "debt_total": {
        "debt_carrying_amount",
        "debt_coupon_rate",
        "debt_current",
        "debt_finance_lease",
        "debt_instrument_principal",
        "debt_noncurrent",
        "debt_other_notes",
    },
    "debt_current": {"debt_total"},
    "debt_noncurrent": {"debt_total"},
    "liabilities_and_equity": {"equity_total", "liabilities_total"},
    "liabilities_total": {"liabilities_current", "liabilities_noncurrent"},
    "liabilities_current": {"liabilities_total"},
    "liabilities_noncurrent": {"liabilities_total"},
}

_INCOMPATIBLE_METRIC_FAMILIES = {
    "equity_total": {"liabilities_and_equity"},
    "liabilities_total": {"liabilities_and_equity"},
    "liabilities_current": {"liabilities_and_equity"},
    "liabilities_noncurrent": {"liabilities_and_equity"},
}


def _role_looks_balance_sheet(value: object) -> bool:
    role = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    if not role:
        return False
    return (
        "balancesheet" in role
        or "balancesheets" in role
        or "statementoffinancialposition" in role
        or "statementsoffinancialposition" in role
    )


def _metric_looks_balance_sheet_snapshot(fact: dict) -> bool:
    if _normalize_tag_local_name(fact.get("tag")) in _BALANCE_SHEET_LOCAL_NAMES:
        return True
    if _role_looks_balance_sheet(fact.get("presentation_role")):
        return True

    hierarchy = fact.get("presentation_hierarchy")
    if isinstance(hierarchy, list):
        for entry in hierarchy:
            if isinstance(entry, dict) and _role_looks_balance_sheet(entry.get("role")):
                return True
    return False


_BALANCE_SHEET_QUERY_TOKENS = {
    "asset",
    "assets",
    "cash",
    "equivalent",
    "equivalents",
    "restricted",
    "debt",
    "borrowings",
    "borrowing",
    "lease",
    "leases",
    "liability",
    "liabilities",
    "equity",
    "stockholders",
    "shareholders",
    "payable",
    "payables",
    "receivable",
    "receivables",
    "inventory",
    "inventories",
    "balance",
    "sheet",
    "current",
    "noncurrent",
    "non",
}


def _query_looks_balance_sheet_metric(query: str) -> bool:
    tokens = set(_split_identifier_tokens(query))
    if not tokens:
        return False
    return bool(tokens & _BALANCE_SHEET_QUERY_TOKENS)


def _query_looks_operating_cash_flow_metric(tokens: set[str]) -> bool:
    if tokens & {"cfo", "ocf"}:
        return True
    if {"cash", "flow"} <= tokens and not (tokens & _NON_OPERATING_CASH_FLOW_QUERY_TOKENS):
        return bool(tokens & _OPERATING_CASH_FLOW_QUERY_TOKENS)
    return False


def _infer_metric_query_family(query: str) -> str | None:
    tokens_list = _split_identifier_tokens(query)
    tokens = set(tokens_list)
    if not tokens:
        return None

    has_liability = bool(tokens & _LIABILITY_QUERY_TOKENS)
    has_equity = bool(tokens & _EQUITY_QUERY_TOKENS)
    if has_liability and has_equity:
        return "liabilities_and_equity"
    if has_equity:
        return "equity_total"
    if has_liability:
        if "current" in tokens and "total" not in tokens:
            return "liabilities_current"
        if "noncurrent" in tokens or ("non" in tokens and "current" in tokens):
            return "liabilities_noncurrent"
        return "liabilities_total"

    if tokens & _ASSET_QUERY_TOKENS:
        if "current" in tokens and "total" not in tokens:
            return "assets_current"
        if "noncurrent" in tokens or ("non" in tokens and "current" in tokens):
            return "assets_noncurrent"
        return "assets_total"

    if _query_looks_operating_cash_flow_metric(tokens):
        return "operating_cash_flow"

    if "cash" in tokens:
        if "restricted" in tokens:
            return "cash_restricted_total"
        return "cash_and_cash_equivalents"

    if tokens & _DEBT_QUERY_TOKENS:
        if tokens & _FLOW_QUERY_TOKENS:
            return None
        if "current" in tokens and "total" not in tokens:
            return "debt_current"
        if "noncurrent" in tokens or ("non" in tokens and "current" in tokens):
            return "debt_noncurrent"
        return "debt_total"

    phrase = " ".join(tokens_list)
    if "cost" not in tokens and (
        "revenue" in tokens
        or "revenues" in tokens
        or phrase in {"net sales", "total net sales", "sales revenue", "sales revenues"}
    ):
        return "revenue"

    if {"operating", "income"} <= tokens:
        return "operating_income"

    return None


def _metric_semantic_family(metric: dict) -> str | None:
    debt_component_kind = metric.get("debt_component_kind")
    if debt_component_kind == "instrument_principal":
        return "debt_instrument_principal"
    if debt_component_kind == "coupon_rate":
        return "debt_coupon_rate"
    if debt_component_kind == "finance_lease":
        return "debt_finance_lease"
    if debt_component_kind == "other_notes":
        return "debt_other_notes"
    if debt_component_kind == "total_debt_rollup":
        return "debt_total"
    if debt_component_kind == "current_debt":
        return "debt_current"
    if debt_component_kind == "debt_carrying_amount":
        return "debt_carrying_amount"

    for field in (metric.get("tag"), metric.get("metric_name")):
        local_name = _normalize_tag_local_name(field)
        family = _METRIC_SEMANTIC_FAMILIES_BY_LOCAL_NAME.get(local_name)
        if family:
            return family

        if local_name.endswith(("liabilitiescurrent", "liabilitycurrent")):
            return "liabilities_current"
        if local_name.endswith(("liabilitiesnoncurrent", "liabilitynoncurrent")):
            return "liabilities_noncurrent"

    label_compact = _normalize_tag_local_name(metric.get("concept_label"))
    if label_compact in {
        "liabilitiesandequity",
        "liabilitiesandstockholdersequity",
        "liabilitiesandstockholdersequityincludingportionattributabletononcontrollinginterest",
    }:
        return "liabilities_and_equity"
    if label_compact in {"netsales", "totalnetsales", "netrevenue", "netrevenues", "revenue", "revenues"}:
        return "revenue"
    if label_compact in {
        "cashandcashequivalents",
        "cashandcashequivalentsatcarryingvalue",
    }:
        return "cash_and_cash_equivalents"
    if label_compact in {
        "cashcashequivalentsrestrictedcashandrestrictedcashequivalents",
        "cashcashequivalentsrestrictedcashandrestrictedcashequivalentsincludingdisposalgroupanddiscontinuedoperations",
    }:
        return "cash_restricted_total"
    if label_compact in {
        "netcashprovidedbyusedinoperatingactivities",
        "netcashprovidedbyusedinoperatingactivitiescontinuingoperations",
        "cashprovidedbyusedinoperatingactivityincludingdiscontinuedoperation",
        "cashprovidedbyusedinoperatingactivitycontinuingoperation",
    }:
        return "operating_cash_flow"
    if label_compact in {
        "equityattributabletoparent",
        "stockholdersequity",
        "stockholdersequityincludingportionattributabletononcontrollinginterest",
        "shareholdersequity",
    }:
        return "equity_total"
    if label_compact == "liabilities":
        return "liabilities_total"
    if label_compact in {"currentliabilities", "liabilitiescurrent"}:
        return "liabilities_current"
    if label_compact in {"noncurrentliabilities", "liabilitiesnoncurrent"}:
        return "liabilities_noncurrent"
    if label_compact == "assets":
        return "assets_total"
    if label_compact in {"currentassets", "assetscurrent"}:
        return "assets_current"
    if label_compact in {"noncurrentassets", "assetsnoncurrent"}:
        return "assets_noncurrent"
    return None


def _metric_family_relation(query_family: str | None, metric_family: str | None) -> str | None:
    if not query_family or not metric_family:
        return None
    if query_family == metric_family:
        return "exact"
    if metric_family in _INCOMPATIBLE_METRIC_FAMILIES.get(query_family, set()):
        return "incompatible"
    if metric_family in _RELATED_METRIC_FAMILIES.get(query_family, set()):
        return "related"
    return None


def _metric_semantic_score_floor(query_family: str | None, metric_family: str | None) -> float:
    relation = _metric_family_relation(query_family, metric_family)
    if relation == "exact":
        return 96.0
    if relation == "related":
        return 72.0
    return 0.0


def _metric_search_candidate_allowed(query_family: str | None, metric_family: str | None) -> bool:
    if query_family == "operating_cash_flow":
        return metric_family == "operating_cash_flow"
    return True


def _catalog_date_type_matches(
    fact: dict,
    *,
    fact_date_type: str | None,
    target_date_type: str | None,
    full_year_mode: bool,
) -> bool:
    if not target_date_type:
        return True
    if fact_date_type == target_date_type:
        return True
    return (
        full_year_mode
        and target_date_type == "FY"
        and fact_date_type == "Q"
        and _metric_looks_balance_sheet_snapshot(fact)
    )


def _normalize_sections_source(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower().replace("-", "")
    if not cleaned:
        return None
    return cleaned


_AGENT_SOURCES = {"auto", "8k", "proxy", "20f", "6k"}
_AGENT_SOURCE_ERROR = "source must be one of auto, 8k, proxy, 20f, 6k"
_CONCEPT_SOURCES = {"auto", "20f"}
_CONCEPT_SOURCE_ERROR = "source must be one of auto, 20f"
_FILING_FORM_TYPES = Literal["10-K", "10-Q", "8-K", "DEF 14A", "20-F", "6-K"]
_FILING_SOURCES = Literal["auto", "8k", "proxy", "20f", "6k"]
_FILING_LIST_SOURCES = Literal["auto", "8k", "20f", "6k"]
_CONCEPT_SOURCE_LITERAL = Literal["auto", "20f"]
_STATEMENT_TYPE_LITERAL = Literal[
    "income_statement",
    "balance_sheet",
    "cash_flow_statement",
]


def _normalize_agent_source_arg(value: object) -> str:
    source = str(value or "auto").strip().lower().replace("-", "")
    if source in {"", "auto"}:
        return "auto"
    return source


_OPERATIONAL_DRIVER_METRICS: tuple[dict[str, Any], ...] = (
    {
        "canonical": "Gross Bookings",
        "aliases": ("gross bookings", "gross booking", "bookings"),
        "unit": "USD millions",
    },
    {
        "canonical": "Revenue",
        "aliases": ("revenue", "revenues", "sales"),
        "unit": "USD millions",
    },
)

_DECREASE_TERMS = {
    "decrease",
    "decreased",
    "decline",
    "declined",
    "down",
    "fell",
    "drop",
    "dropped",
}

_GROWTH_CUE_TERMS = _DECREASE_TERMS | {
    "grow",
    "grew",
    "growth",
    "increase",
    "increased",
    "up",
}

_OPERATIONAL_TOPIC_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "basis",
    "bridge",
    "by",
    "change",
    "compared",
    "comparison",
    "constant",
    "currency",
    "decomposition",
    "driver",
    "drivers",
    "filing",
    "for",
    "from",
    "growth",
    "in",
    "into",
    "kpi",
    "kpis",
    "metric",
    "metrics",
    "of",
    "operating",
    "operational",
    "period",
    "rate",
    "rates",
    "reported",
    "segment",
    "segments",
    "the",
    "to",
    "trend",
    "trends",
    "value",
    "values",
    "versus",
    "volume",
    "vs",
    "with",
    "year",
    "yoy",
}

_OPERATIONAL_TABLE_LABEL_DENY_EXACT = {
    "",
    "$ change",
    "% change",
    "constant currency $ change",
    "constant currency % change",
    "costs and expenses",
    "year ended",
}

_OPERATIONAL_TABLE_LABEL_DENY_PREFIXES = (
    "cost of ",
    "costs of ",
    "income from ",
    "loss from ",
    "net income",
    "net loss",
)


def _clean_operational_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _select_seed_operational_driver_metrics(topic: object) -> list[dict[str, Any]]:
    query = _clean_operational_text(topic).lower()
    selected: list[dict[str, Any]] = []
    for metric in _OPERATIONAL_DRIVER_METRICS:
        aliases = metric.get("aliases") or ()
        if any(alias in query for alias in aliases):
            item = dict(metric)
            item["match_source"] = "registry_alias"
            selected.append(item)
    return selected or list(_OPERATIONAL_DRIVER_METRICS)


def _operational_metric_regex(metric_name: str) -> str:
    return r"\s+".join(re.escape(part) for part in metric_name.split())


def _operational_metric_aliases(metric: dict[str, Any]) -> list[str]:
    aliases = {str(metric.get("canonical") or "").strip()}
    aliases.update(str(alias).strip() for alias in metric.get("aliases") or ())
    expanded = set()
    for alias in aliases:
        if not alias:
            continue
        expanded.add(alias)
        if alias.lower().endswith("s") and len(alias) > 3:
            expanded.add(alias[:-1])
        elif len(alias) > 2:
            expanded.add(f"{alias}s")
    return sorted(expanded, key=len, reverse=True)


def _operational_text_tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _meaningful_operational_tokens(value: object) -> set[str]:
    return {
        token
        for token in _operational_text_tokens(value)
        if len(token) > 1 and token not in _OPERATIONAL_TOPIC_STOP_TOKENS
    }


def _make_operational_metric(
    label: str,
    *,
    match_source: str,
    score: int = 0,
    unit: str | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "canonical": label,
        "aliases": tuple(aliases or ()),
        "match_source": match_source,
        "match_score": score,
    }
    if unit:
        metric["unit"] = unit
    return metric


def _find_previous_sentence_boundary(text: str, pos: int) -> int:
    cursor = max(0, min(pos, len(text)))
    while True:
        period = text.rfind(".", 0, cursor)
        newline = text.rfind("\n", 0, cursor)
        boundary = max(period, newline)
        if boundary < 0:
            return 0
        if boundary == period and 0 < period < len(text) - 1:
            if text[period - 1].isdigit() and text[period + 1].isdigit():
                cursor = period
                continue
        return boundary + 1


def _find_next_sentence_boundary(text: str, pos: int) -> int:
    cursor = max(0, min(pos, len(text)))
    while cursor < len(text):
        period = text.find(".", cursor)
        newline = text.find("\n", cursor)
        candidates = [idx for idx in (period, newline) if idx >= 0]
        if not candidates:
            return len(text)
        boundary = min(candidates)
        if boundary == period and 0 < period < len(text) - 1:
            if text[period - 1].isdigit() and text[period + 1].isdigit():
                cursor = period + 1
                continue
        return boundary + 1 if boundary == period else boundary
    return len(text)


def _iter_operational_sentences(text: str, metric_names: list[str]) -> list[tuple[int, int, str]]:
    spans: dict[tuple[int, int], str] = {}
    for metric_name in metric_names:
        pattern = re.compile(_operational_metric_regex(metric_name), re.IGNORECASE)
        for match in pattern.finditer(text):
            start = _find_previous_sentence_boundary(text, match.start())
            end = _find_next_sentence_boundary(text, match.end())
            sentence = _clean_operational_text(text[start:end])
            if sentence and len(sentence) <= 1500 and "|" not in sentence:
                spans[(start, end)] = sentence
    return [(start, end, sentence) for (start, end), sentence in sorted(spans.items())]


def _source_from_operational_span(
    text: str,
    start: int,
    end: int,
    *,
    section: str = "item_7",
) -> dict[str, Any]:
    snippet = _clean_operational_text(text[max(0, start - 160): min(len(text), end + 160)])
    return {
        "source_tool": "get_filing_document",
        "section": section,
        "char_start": start,
        "char_end": end,
        "snippet": snippet[:900],
    }


def _operational_pipe_table_group_start(text: str, row_start: int) -> int:
    """Return the first character offset for the contiguous markdown table block."""
    line_start = text.rfind("\n", 0, row_start) + 1
    cursor = line_start
    while cursor > 0:
        previous_end = cursor - 1
        previous_start = text.rfind("\n", 0, max(previous_end, 0)) + 1
        previous_line = text[previous_start:previous_end]
        if "|" not in previous_line or not previous_line.strip():
            break
        cursor = previous_start
    return cursor


def _operational_source_group_id(prefix: str, offset: int) -> str:
    return f"{prefix}:{offset}"


def _operational_precision(raw: object, *, source_basis: str) -> str:
    text = str(raw or "")
    if "." in text:
        return "table_decimal" if source_basis == "table_reported" else "narrative_decimal"
    if source_basis == "table_reported":
        return "table_rounded"
    return "narrative_reported"


def _operational_scale_from_raw(raw: object) -> str | None:
    lowered = str(raw or "").lower()
    if "billion" in lowered:
        return "billions"
    if "million" in lowered:
        return "millions"
    if "thousand" in lowered:
        return "thousands"
    return None


def _operational_unit_from_raw(raw: object, fallback: str | None = None) -> str:
    if "$" in str(raw or ""):
        scale = _operational_scale_from_raw(raw)
        return f"USD {scale}" if scale else "USD"
    return fallback or "reported filing units"


def _parse_operational_percent(raw: object, verb: object = "") -> float | None:
    text = str(raw or "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    value = float(match.group(0))
    if "(" in text and ")" in text and value > 0:
        value = -value
    verb_text = str(verb or "").lower()
    if any(term in verb_text for term in _DECREASE_TERMS) and value > 0:
        value = -value
    return value


def _parse_operational_amount(raw: object) -> float | None:
    text = str(raw or "").strip()
    if not text or "%" in text:
        return None
    normalized = text.replace("$", "").replace(",", "").replace("\u2014", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    value = float(match.group(0))
    if "(" in normalized and ")" in normalized and value > 0:
        value = -value
    return value


def _parse_operational_amount_cell_values(cells: list[str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        stripped = cell.strip()
        if not stripped or stripped in {"$", "%"} or "%" in stripped:
            continue
        if index + 1 < len(cells) and cells[index + 1].strip() == "%":
            continue
        raw = f"${stripped}" if index > 0 and cells[index - 1].strip() == "$" else stripped
        value = _parse_operational_amount(raw)
        if value is not None:
            values.append(
                {
                    "value": value,
                    "value_raw": raw,
                    "precision": _operational_precision(raw, source_basis="table_reported"),
                }
            )
    return values


def _parse_operational_amount_cells(cells: list[str]) -> list[float]:
    return [item["value"] for item in _parse_operational_amount_cell_values(cells)]


def _clean_operational_metric_label(value: object) -> str:
    text = _clean_operational_text(value).strip("* ")
    text = re.sub(r"\(\s*\d+(?:\s*,\s*\d+)*\s*\)", "", text)
    text = re.sub(r"\s*,\s*\(\s*\d+\s*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" |:-,")


def _normalized_operational_table_label(value: object) -> str:
    text = _clean_operational_metric_label(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _table_label_matches_operational_metric(label: object, metric: dict[str, Any]) -> bool:
    normalized_label = _normalized_operational_table_label(label)
    return any(
        normalized_label == _normalized_operational_table_label(alias)
        for alias in _operational_metric_aliases(metric)
    )


def _find_operational_metric_match(sentence: str, metric: dict[str, Any]) -> re.Match[str] | None:
    for alias in _operational_metric_aliases(metric):
        pattern = _operational_metric_regex(alias)
        match = re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", sentence, re.IGNORECASE)
        if match:
            return match
    return None


def _parse_operational_percent_cell_values(cells: list[str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        raw = ""
        if "%" in cell:
            raw = cell
        elif index + 1 < len(cells) and cells[index + 1].strip() == "%":
            raw = f"{cell}%"
        if not raw:
            continue
        value = _parse_operational_percent(raw)
        if value is not None:
            values.append(
                {
                    "value": value,
                    "value_raw": raw,
                    "precision": _operational_precision(raw, source_basis="table_reported"),
                }
            )
    return values


def _parse_operational_percent_cells(cells: list[str]) -> list[float]:
    return [item["value"] for item in _parse_operational_percent_cell_values(cells)]


def _driver_period_label(year: int, quarter: int) -> str:
    if quarter == 4:
        return f"FY{year} vs FY{year - 1}"
    return f"FY{year} Q{quarter} vs prior-year period"


def _value_period_labels(year: int, quarter: int) -> tuple[str, str]:
    if quarter == 4:
        return f"FY{year - 1}", f"FY{year}"
    return f"FY{year - 1} Q{quarter}", f"FY{year} Q{quarter}"


def _driver_direction(value_percent: float) -> str:
    if value_percent < 0:
        return "decrease"
    if value_percent > 0:
        return "increase"
    return "flat"


def _driver_text_from_sentence(sentence: str) -> str | None:
    for marker in (" due to ", " primarily attributable to ", " driven by "):
        idx = sentence.lower().find(marker)
        if idx >= 0:
            return sentence[idx + len(marker):].strip(" .") or None
    return None


def _append_growth_row(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    segment: str,
    period: str,
    basis: str,
    value_percent: float,
    source: dict[str, Any],
    sentence: str | None = None,
    value_raw: str | None = None,
    source_basis: str = "narrative_reported",
    precision: str | None = None,
    source_group_id: str | None = None,
    recommended_for_calculation: bool = False,
) -> None:
    row: dict[str, Any] = {
        "kind": "growth_rate",
        "metric_name": metric_name,
        "segment": segment,
        "period": period,
        "basis": basis,
        "value": value_percent / 100.0,
        "value_percent": value_percent,
        "value_raw": value_raw or f"{value_percent:g}%",
        "value_semantic": "growth_rate",
        "source_basis": source_basis,
        "precision": precision or _operational_precision(value_raw or f"{value_percent:g}%", source_basis=source_basis),
        "recommended_for_calculation": recommended_for_calculation,
        "direction": _driver_direction(value_percent),
        "source": source,
    }
    if source_group_id:
        row["source_group_id"] = source_group_id
    if sentence:
        driver_text = _driver_text_from_sentence(sentence)
        if driver_text:
            row["driver_text"] = driver_text
    rows.append(row)


def _contains_operational_growth_cue(text: str) -> bool:
    tokens = set(_operational_text_tokens(text))
    return bool(tokens & _GROWTH_CUE_TERMS)


def _iter_operational_metric_mentions(sentence: str, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for metric in metrics:
        for alias in _operational_metric_aliases(metric):
            pattern = _operational_metric_regex(alias)
            for match in re.finditer(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", sentence, re.IGNORECASE):
                candidate = {
                    "start": match.start(),
                    "end": match.end(),
                    "metric": metric,
                    "metric_name": str(metric["canonical"]),
                    "text": match.group(0),
                }
                if _operational_metric_mention_allowed(sentence, candidate):
                    candidates.append(candidate)
    candidates.sort(key=lambda item: (int(item["start"]), -(int(item["end"]) - int(item["start"]))))
    mentions: list[dict[str, Any]] = []
    for candidate in candidates:
        start = int(candidate["start"])
        end = int(candidate["end"])
        if any(start < int(existing["end"]) and end > int(existing["start"]) for existing in mentions):
            continue
        mentions.append(candidate)
    return sorted(mentions, key=lambda item: int(item["start"]))


def _iter_operational_percent_mentions(sentence: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for match in re.finditer(r"\(?-?\d+(?:\.\d+)?\)?\s*%", sentence):
        mentions.append(
            {
                "start": match.start(),
                "end": match.end(),
                "raw": match.group(0),
            }
        )
    return mentions


def _operational_metric_mention_allowed(sentence: str, mention: dict[str, Any]) -> bool:
    """Filter exact alias matches that are embedded in a different metric phrase."""
    metric_name = _normalized_operational_table_label(mention.get("metric_name"))
    if metric_name != "revenue":
        return True

    start = int(mention["start"])
    before = sentence[max(0, start - 48):start]
    return not re.search(r"\b(?:cost|costs|expense|expenses)\s+of\s*$", before, re.IGNORECASE)


def _operational_segment_from_prefix(prefix: str, segment_vocabulary: dict[str, str] | None = None) -> str:
    prefix = prefix.strip(" ,;:-")
    if not prefix:
        return "Consolidated"
    tail = re.split(r"[,;:]", prefix)[-1].strip(" ,;:-")

    if segment_vocabulary:
        normalized_tail = _normalized_operational_table_label(tail)
        matches = [
            (normalized, label)
            for normalized, label in segment_vocabulary.items()
            if normalized_tail == normalized or normalized_tail.endswith(f" {normalized}")
        ]
        if matches:
            return max(matches, key=lambda item: len(item[0]))[1]

    candidate = " ".join(tail.split()[-4:]).strip(" ,;:-")
    lowered = candidate.lower()
    if (
        not candidate
        or lowered == "overall"
        or lowered in {"cost", "cost of", "costs", "costs of"}
        or lowered.endswith(("and", "or", "the"))
        or lowered.startswith(("same period", "period in "))
        or "%" in candidate
        or _contains_operational_growth_cue(candidate)
    ):
        return "Consolidated"
    return candidate


def _iter_operational_growth_clauses(sentence: str) -> list[str]:
    clauses: list[str] = []
    for semicolon_part in re.split(r"\s*;\s*", sentence):
        pieces = re.split(
            r",\s+and\s+(?=[A-Z][A-Za-z&/().'\-]*(?:\s+[A-Z][A-Za-z&/().'\-]*){0,4}\s+"
            r"(?:grew|increased|decreased|declined|was up|were up|was down|were down)\b)",
            semicolon_part,
        )
        for piece in pieces:
            clause = _clean_operational_text(piece)
            if clause:
                clauses.append(clause)
    return clauses or [sentence]


def _score_operational_percent_binding(
    sentence: str,
    percent: dict[str, Any],
    mention: dict[str, Any],
    mentions: list[dict[str, Any]],
) -> int:
    percent_start = int(percent["start"])
    percent_end = int(percent["end"])
    metric_start = int(mention["start"])
    metric_end = int(mention["end"])

    if percent_end <= metric_start:
        if any(percent_end <= int(other["start"]) < metric_start for other in mentions if other is not mention):
            return 0
        between = sentence[percent_end:metric_start]
        distance = metric_start - percent_end
        direct_pre_metric = re.search(
            r"\b(?:increase|increased|decrease|decreased|decline|declined|growth|grew|up|down)\s+"
            r"(?:in|of)\s+(?:the\s+)?(?:number\s+of\s+)?$",
            between,
            re.IGNORECASE,
        )
        if distance <= 140 and direct_pre_metric:
            return 1200 - distance
        return 0

    if percent_start >= metric_end:
        if any(metric_end < int(other["start"]) < percent_start for other in mentions if other is not mention):
            return 0
        between = sentence[metric_end:percent_start]
        distance = percent_start - metric_end
        if distance <= 220 and _contains_operational_growth_cue(between):
            return 900 - distance
    return 0


def _bind_operational_percents_to_metrics(
    sentence: str,
    percentages: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    bindings: dict[int, dict[str, Any]] = {}
    for index, percent in enumerate(percentages):
        best_score = 0
        best_mention: dict[str, Any] | None = None
        for mention in mentions:
            score = _score_operational_percent_binding(sentence, percent, mention, mentions)
            if score and int(percent["end"]) <= int(mention["start"]):
                if any(
                    other_index != index
                    and int(percent["end"]) <= int(other["start"]) < int(mention["start"])
                    for other_index, other in enumerate(percentages)
                ):
                    score = 0
            if score > best_score:
                best_score = score
                best_mention = mention
        if best_mention is not None:
            bindings[index] = best_mention
    return bindings


def _basis_for_operational_percent(
    sentence: str,
    percent_index: int,
    percentages: list[dict[str, Any]],
) -> str:
    percent = percentages[percent_index]
    next_start = (
        int(percentages[percent_index + 1]["start"])
        if percent_index + 1 < len(percentages)
        else min(len(sentence), int(percent["end"]) + 140)
    )
    local_after = sentence[int(percent["end"]):next_start].lower()
    local_before = sentence[max(0, int(percent["start"]) - 80): int(percent["start"])].lower()
    if "constant currency" in local_after or "constant currency" in local_before:
        return "constant_currency"
    return "reported"


def _verb_context_for_operational_percent(
    sentence: str,
    percent: dict[str, Any],
    mention: dict[str, Any],
) -> str:
    percent_start = int(percent["start"])
    percent_end = int(percent["end"])
    metric_start = int(mention["start"])
    metric_end = int(mention["end"])
    if percent_end <= metric_start:
        return sentence[percent_end:metric_start]
    return sentence[metric_end:percent_start]


def _narrative_change_amount_for_metric(
    sentence: str,
    mention: dict[str, Any],
    mentions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    metric_end = int(mention["end"])
    next_metric_start = min(
        [int(other["start"]) for other in mentions if int(other["start"]) > metric_end] or [len(sentence)]
    )
    local = sentence[metric_end:next_metric_start]
    match = re.search(
        r"\b(?:by|of)\s+(?P<raw>\$?\(?-?\d+(?:,\d{3})*(?:\.\d+)?\)?\s*(?:thousand|million|billion)?)",
        local,
        re.IGNORECASE,
    )
    if not match:
        return None
    raw = match.group("raw").strip()
    if "%" in raw or local[match.end("raw"):].lstrip().startswith("%"):
        return None
    if _operational_narrative_amount_role(local, match, growth_context=True) != "change_amount":
        return None
    value = _parse_operational_amount(raw)
    if value is None:
        return None
    return {
        "value": value,
        "value_raw": raw,
        "unit": _operational_unit_from_raw(raw),
        "scale": _operational_scale_from_raw(raw),
        "precision": _operational_precision(raw, source_basis="narrative_change_amount"),
    }


_OPERATIONAL_NARRATIVE_AMOUNT_YEAR_RE = re.compile(
    r"(?P<raw>\$?\(?-?\d+(?:,\d{3})*(?:\.\d+)?\)?\s*(?:thousand|million|billion)?)"
    r"\s+(?:for|in|during|as of|at)\s+(?:the\s+)?"
    r"(?:(?:fiscal|calendar)\s+)?(?:(?:year|quarter|period|twelve months)\s+ended\s+)?"
    r"(?:[A-Za-z]+\s+\d{1,2},?\s+)?(?P<year>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def _operational_narrative_amount_role(
    text: str,
    match: re.Match[str],
    *,
    growth_context: bool = False,
) -> str | None:
    raw = match.group("raw").strip()
    amount_start = match.start("raw")
    amount_end = match.end("raw")
    if "%" in raw or text[amount_end:].lstrip().startswith("%"):
        return None

    before = _clean_operational_text(text[max(0, amount_start - 140): amount_start]).lower()
    if re.search(
        r"\b(?:change|changed|increase|increased|decrease|decreased|decline|declined|growth|grew|up|down|rose|fell)"
        r"\s+(?:by|of)\s*$",
        before,
    ):
        return "change_amount"
    if re.search(r"\b(?:by|of)\s*$", before) and (
        growth_context or _contains_operational_growth_cue(before[-80:])
    ):
        return "change_amount"
    if re.search(
        r"\b(?:to|from|was|were|is|are|totaled|totalled|amounted\s+to|reached|reported|generated|delivered)\s*$",
        before,
    ):
        return "metric_value"
    return None


def _iter_operational_narrative_amount_years(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for match in _OPERATIONAL_NARRATIVE_AMOUNT_YEAR_RE.finditer(text):
        if _operational_narrative_amount_role(text, match) != "metric_value":
            continue
        raw = match.group("raw").strip()
        value = _parse_operational_amount(raw)
        if value is None:
            continue
        values.append(
            {
                "value": value,
                "value_raw": raw,
                "period": f"FY{int(match.group('year'))}",
                "unit": _operational_unit_from_raw(raw, fallback="reported filing units"),
                "scale": _operational_scale_from_raw(raw),
                "precision": _operational_precision(raw, source_basis="narrative_reported"),
                "amount_role": "metric_value",
                "start": match.start(),
                "end": match.end(),
            }
        )
    return values


def _operational_narrative_value_segment(prefix: str, metrics: list[dict[str, Any]]) -> str:
    if _iter_operational_metric_mentions(prefix, metrics):
        return "Consolidated"
    segment = _operational_segment_from_prefix(prefix)
    normalized = _normalized_operational_table_label(segment)
    if re.search(r"\b(?:19|20)\d{2}\b", segment) or normalized.startswith(("in ", "for ", "during ")):
        return "Consolidated"
    return segment


def _extract_operational_value_rows_from_prose(
    text: str,
    *,
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = sorted(
        {alias for metric in metrics for alias in _operational_metric_aliases(metric)},
        key=len,
        reverse=True,
    )
    for start, end, sentence in _iter_operational_sentences(text, metric_names):
        source = _source_from_operational_span(text, start, end)
        source_group_id = _operational_source_group_id("prose_sentence", start)
        mentions = _iter_operational_metric_mentions(sentence, metrics)
        if not mentions:
            continue
        for mention in mentions:
            metric_end = int(mention["end"])
            next_metric_start = min(
                [int(other["start"]) for other in mentions if int(other["start"]) > metric_end] or [len(sentence)]
            )
            local = sentence[metric_end:next_metric_start]
            if not _contains_operational_growth_cue(local):
                continue
            for amount in _iter_operational_narrative_amount_years(local):
                row: dict[str, Any] = {
                    "kind": "metric_value",
                    "metric_name": str(mention["metric_name"]),
                    "segment": _operational_narrative_value_segment(sentence[: int(mention["start"])], metrics),
                    "period": amount["period"],
                    "basis": "reported",
                    "value": amount["value"],
                    "value_raw": amount["value_raw"],
                    "value_semantic": "period_absolute_value",
                    "unit": amount["unit"],
                    "scale": amount["scale"],
                    "source_basis": "narrative_reported",
                    "precision": amount["precision"],
                    "recommended_for_calculation": False,
                    "source_group_id": source_group_id,
                    "source": source,
                }
                rows.append(row)
    return rows


def _extract_operational_growth_rows_from_prose(
    text: str,
    *,
    metrics: list[dict[str, Any]],
    year: int,
    quarter: int,
    filing_vocabulary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = sorted(
        {alias for metric in metrics for alias in _operational_metric_aliases(metric)},
        key=len,
        reverse=True,
    )
    period = _driver_period_label(year, quarter)
    segment_vocabulary = (filing_vocabulary or {}).get("segments") or {}
    for start, end, sentence in _iter_operational_sentences(text, metric_names):
        source = _source_from_operational_span(text, start, end)
        source_group_id = _operational_source_group_id("prose_sentence", start)
        for clause in _iter_operational_growth_clauses(sentence):
            mentions = _iter_operational_metric_mentions(clause, metrics)
            percentages = _iter_operational_percent_mentions(clause)
            if not mentions or not percentages:
                continue
            bindings = _bind_operational_percents_to_metrics(clause, percentages, mentions)
            if not bindings:
                continue

            bound_mentions: dict[int, dict[str, Any]] = {}
            for percent_index, mention in sorted(bindings.items(), key=lambda item: int(percentages[item[0]]["start"])):
                percent = percentages[percent_index]
                verb_context = _verb_context_for_operational_percent(clause, percent, mention)
                value_percent = _parse_operational_percent(percent["raw"], verb_context)
                if value_percent is None:
                    continue
                bound_mentions[id(mention)] = mention
                _append_growth_row(
                    rows,
                    metric_name=str(mention["metric_name"]),
                    segment=_operational_segment_from_prefix(
                        clause[: int(mention["start"])],
                        segment_vocabulary,
                    ),
                    period=period,
                    basis=_basis_for_operational_percent(clause, percent_index, percentages),
                    value_percent=value_percent,
                    value_raw=str(percent["raw"]),
                    source=source,
                    sentence=clause,
                    source_basis="narrative_reported",
                    source_group_id=source_group_id,
                    recommended_for_calculation=False,
                )

            for mention in bound_mentions.values():
                change_amount = _narrative_change_amount_for_metric(clause, mention, mentions)
                if change_amount is None:
                    continue
                rows.append(
                    {
                        "kind": "change_amount",
                        "metric_name": str(mention["metric_name"]),
                        "segment": _operational_segment_from_prefix(
                            clause[: int(mention["start"])],
                            segment_vocabulary,
                        ),
                        "period": period,
                        "basis": "reported",
                        "value": change_amount["value"],
                        "value_raw": change_amount["value_raw"],
                        "value_semantic": "change_amount",
                        "unit": change_amount["unit"],
                        "scale": change_amount["scale"],
                        "source_basis": "narrative_change_amount",
                        "precision": change_amount["precision"],
                        "recommended_for_calculation": False,
                        "source_group_id": source_group_id,
                        "source": source,
                    }
                )
    return rows


def _operational_pipe_cells(line: str, *, keep_empty: bool = False) -> list[str]:
    cells = [_clean_operational_text(cell).strip("* ") for cell in line.strip().strip("|").split("|")]
    return cells if keep_empty else [cell for cell in cells if cell]


def _operational_pipe_line_is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in _operational_pipe_cells(line, keep_empty=True) if cell.strip()]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _make_operational_pipe_table_block(items: list[tuple[int, int, str, bool]]) -> dict[str, Any] | None:
    rows: list[tuple[int, int, str, list[str], str]] = []
    for start, end, line, is_separator in items:
        if is_separator:
            continue
        cells = _operational_pipe_cells(line)
        if len(cells) < 3:
            continue
        rows.append((start, end, _clean_operational_metric_label(cells[0]), cells, line))
    if not rows:
        return None
    return {
        "start": items[0][0],
        "end": items[-1][1],
        "rows": rows,
        "lines": [line for _start, _end, line, _is_separator in items],
    }


def _iter_operational_pipe_table_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: list[tuple[int, int, str, bool]] = []
    current_has_separator = False
    previous_end: int | None = None

    def flush() -> None:
        nonlocal current, current_has_separator
        block = _make_operational_pipe_table_block(current)
        if block is not None:
            blocks.append(block)
        current = []
        current_has_separator = False

    for line_match in re.finditer(r"(?m)^.*\|.*$", text):
        if previous_end is not None:
            gap = text[previous_end: line_match.start()]
            if gap.count("\n") > 1 or gap.strip():
                flush()
        previous_end = line_match.end()

        line = line_match.group(0)
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        is_separator = _operational_pipe_line_is_separator(line)
        if is_separator and current_has_separator and current:
            header = current.pop()
            if current:
                flush()
            current = [header, (line_match.start(), line_match.end(), line, True)]
            current_has_separator = True
            continue

        current.append((line_match.start(), line_match.end(), line, is_separator))
        if is_separator:
            current_has_separator = True

    flush()
    return blocks


def _iter_operational_pipe_table_rows(text: str) -> list[tuple[int, int, str, list[str], str]]:
    rows: list[tuple[int, int, str, list[str], str]] = []
    for block in _iter_operational_pipe_table_blocks(text):
        rows.extend(block["rows"])
    return rows


def _operational_table_label_allowed(label: object) -> bool:
    normalized = _normalized_operational_table_label(label)
    if normalized in _OPERATIONAL_TABLE_LABEL_DENY_EXACT:
        return False
    if any(normalized.startswith(prefix) for prefix in _OPERATIONAL_TABLE_LABEL_DENY_PREFIXES):
        return False
    if not normalized or normalized.isdigit() or len(normalized) > 90:
        return False
    tokens = _meaningful_operational_tokens(normalized)
    return bool(tokens)


def _score_operational_label_against_topic(label: object, topic: object) -> int:
    normalized_label = _normalized_operational_table_label(label)
    normalized_topic = _normalized_operational_table_label(topic)
    if not normalized_label or not normalized_topic:
        return 0
    if normalized_label in normalized_topic:
        return 100 + len(normalized_label.split())
    label_tokens = _meaningful_operational_tokens(normalized_label)
    topic_tokens = _meaningful_operational_tokens(normalized_topic)
    if not label_tokens or not topic_tokens:
        return 0
    overlap = label_tokens & topic_tokens
    if not overlap:
        return 0
    score = 20 * len(overlap)
    if label_tokens <= topic_tokens:
        score += 50
    if any(len(token) >= 4 for token in overlap):
        score += 10
    return score


def _operational_label_classifier(label: object) -> tuple[str, str | None]:
    return classify_operating_metric_label(str(label or ""))


def _operational_label_has_metric_signal(label: object) -> bool:
    row_class, metric_family = _operational_label_classifier(label)
    return row_class in {
        ROW_CLASS_OPERATIONAL_KPI,
        ROW_CLASS_GROWTH_METRIC,
        ROW_CLASS_MARGIN_RATE,
        ROW_CLASS_SEGMENT_MEMBER,
    } or metric_family is not None


def _operational_label_matches_registry_metric(label: object) -> bool:
    return any(_table_label_matches_operational_metric(label, metric) for metric in _OPERATIONAL_DRIVER_METRICS)


def _operational_topic_requests_driver_surface(topic: object) -> bool:
    normalized = _normalized_operational_table_label(topic)
    if "take rate" in normalized:
        return True
    tokens = set(normalized.split())
    return bool(tokens & {"decomposition", "driver", "drivers", "kpi", "metrics", "operational", "volume"})


_OPERATIONAL_KPI_GET_METRIC_REDIRECT_ROW_CLASSES = {
    ROW_CLASS_OPERATIONAL_KPI,
    ROW_CLASS_SEGMENT_MEMBER,
}

_OPERATIONAL_KPI_GET_METRIC_REDIRECT_RATE_PHRASES = {
    "arr expansion rate",
    "dollar based net retention",
    "dollar-based net retention",
    "net retention",
    "renewal rate",
    "same store",
    "same-store",
    "take rate",
}

_OPERATIONAL_KPI_GET_METRIC_REDIRECT_TOKENS = {
    "active",
    "arr",
    "booked",
    "booking",
    "bookings",
    "cardholder",
    "customer",
    "customers",
    "dau",
    "experiences",
    "gbv",
    "gov",
    "mapc",
    "mau",
    "member",
    "members",
    "membership",
    "memberships",
    "nights",
    "nrr",
    "order",
    "orders",
    "retention",
    "rider",
    "riders",
    "subscriber",
    "subscribers",
    "subscription",
    "subscriptions",
    "take",
    "trip",
    "trips",
    "user",
    "users",
}

_OPERATIONAL_KPI_TOPIC_ACRONYMS = {
    "arr": "ARR",
    "arpu": "ARPU",
    "dau": "DAU",
    "gbv": "GBV",
    "gov": "GOV",
    "mapc": "MAPC",
    "mau": "MAU",
    "nrr": "NRR",
}


def _display_operational_kpi_topic(metric_name: object) -> str:
    tokens = _split_identifier_tokens(metric_name)
    if not tokens:
        return _clean_operational_text(metric_name)

    words: list[str] = []
    for index, token in enumerate(tokens):
        acronym = _OPERATIONAL_KPI_TOPIC_ACRONYMS.get(token)
        if acronym:
            words.append(acronym)
        elif index > 0 and token in {"and", "as", "by", "for", "from", "in", "of", "per", "to", "vs", "with"}:
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _operational_kpi_metric_redirect(args: dict) -> dict[str, Any] | None:
    topic = _display_operational_kpi_topic(args.get("metric_name"))
    if not topic:
        return None

    normalized = _normalized_operational_table_label(topic)
    tokens = set(normalized.split())
    row_class, metric_family = _operational_label_classifier(topic)
    has_operational_signal = bool(tokens & _OPERATIONAL_KPI_GET_METRIC_REDIRECT_TOKENS) or any(
        phrase in normalized
        for phrase in _OPERATIONAL_KPI_GET_METRIC_REDIRECT_RATE_PHRASES
    )
    registry_match = _operational_label_matches_registry_metric(topic)

    should_redirect = (
        (registry_match and has_operational_signal)
        or row_class in _OPERATIONAL_KPI_GET_METRIC_REDIRECT_ROW_CLASSES
        or (
            row_class in {ROW_CLASS_GROWTH_METRIC, ROW_CLASS_MARGIN_RATE}
            and has_operational_signal
        )
    )
    if not should_redirect:
        return None

    source = _normalize_agent_source_arg(args.get("source"))
    suggested_args = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "topic": topic,
        "source": source,
    }
    return {
        "tool": "get_operational_kpi_drivers",
        "arguments": suggested_args,
        "topic": topic,
        "row_class": row_class,
        "metric_family": metric_family,
        "match_basis": "shared_kpi_classifier" if not registry_match else "operational_driver_registry",
    }


def _metric_response_is_metric_not_found(response: dict) -> bool:
    if response.get("status") != "error":
        return False
    error_type = str(response.get("error_type") or "").strip().lower()
    if error_type == "metric_not_found":
        return True
    message = str(response.get("message") or "").lower()
    return "metric" in message and "not found" in message


def _annotate_operational_kpi_metric_miss(response: dict, args: dict) -> dict:
    if not _metric_response_is_metric_not_found(response):
        return response

    redirect = _operational_kpi_metric_redirect(args)
    if not redirect:
        return response

    enriched = dict(response)
    enriched["remediation_tool"] = redirect["tool"]
    enriched["remediation_hint"] = (
        "This metric name looks like a non-XBRL operational KPI. Use "
        "get_operational_kpi_drivers to extract filing-native table or MD&A values "
        "with source-basis metadata before using it in calculations."
    )
    enriched["operational_kpi_redirect"] = redirect
    return enriched


def _metric_response_details(response: dict) -> dict[str, Any]:
    details = response.get("details")
    return details if isinstance(details, dict) else {}


def _metric_discovery_args(args: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    date_type = _normalize_date_type(args.get("date_type"))
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    source = _normalize_agent_source_arg(args.get("source"))
    search_args: dict[str, Any] = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "query": args["metric_name"],
        "full_year_mode": effective_full_year_mode,
        "source": source,
    }
    list_args: dict[str, Any] = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "full_year_mode": effective_full_year_mode,
        "source": source,
    }
    if date_type:
        search_args["date_type"] = date_type
        list_args["date_type"] = date_type
    if args.get("role"):
        search_args["role"] = args["role"]
        list_args["role"] = args["role"]
    return search_args, list_args


def _annotate_metric_cache_miss(response: dict, args: dict) -> dict:
    if response.get("status") != "error":
        return response

    details = _metric_response_details(response)
    error_type = str(response.get("error_type") or "").strip().lower()
    message = str(response.get("message") or "")
    is_cache_miss = (
        error_type == "cache_miss"
        or bool(details.get("cache_miss"))
        or "no cached data" in message.lower()
        or "no cached annual data" in message.lower()
    )
    if not is_cache_miss:
        return response

    enriched = dict(response)
    for key in ("cache_miss", "warm_hint", "remediation_tool", "remediation_hint"):
        if key in details:
            enriched.setdefault(key, details[key])

    warm_hint = enriched.get("warm_hint")
    if isinstance(warm_hint, dict):
        enriched.setdefault("cache_miss", True)
        enriched.setdefault("remediation_tool", "warm_metric_cache")
        enriched.setdefault(
            "remediation_hint",
            "Call warm_metric_cache with warm_hint.body.items, poll "
            "warm_metric_cache_status until complete, then retry get_metric.",
        )
        if message and "warm_metric_cache" not in message:
            enriched["message"] = (
                f"{message} Call warm_metric_cache with warm_hint.body.items, "
                "poll warm_metric_cache_status until complete, then retry get_metric."
            )
    else:
        enriched.setdefault("cache_miss", True)
        enriched.setdefault("remediation_tool", "get_financials")
        enriched.setdefault(
            "remediation_hint",
            "Call get_financials for the same ticker/year/quarter/full_year_mode, "
            "then retry get_metric.",
        )
    return enriched


def _annotate_financial_metric_miss(response: dict, args: dict) -> dict:
    if not _metric_response_is_metric_not_found(response):
        return response
    if response.get("operational_kpi_redirect"):
        return response

    details = _metric_response_details(response)
    enriched = dict(response)
    for key in ("remediation_tool", "remediation_hint", "search_metrics_args", "list_metrics_args"):
        if key in details:
            enriched.setdefault(key, details[key])

    search_args, list_args = _metric_discovery_args(args)
    enriched.setdefault("remediation_tool", "search_metrics")
    enriched.setdefault(
        "remediation_hint",
        "Do not keep guessing metric_name. Call search_metrics with the same "
        "ticker/period filters to discover the exact metric_name, or list_metrics "
        "to inspect available tags, then retry get_metric.",
    )
    enriched.setdefault("search_metrics_args", search_args)
    enriched.setdefault("list_metrics_args", list_args)
    return enriched


def _operational_table_block_years(block: dict[str, Any]) -> list[int]:
    years: list[int] = []
    seen: set[int] = set()
    for _start, _end, _label, cells, _line in block.get("rows", [])[:4]:
        for cell in cells:
            for match in re.finditer(r"\b(?:19|20)\d{2}\b", str(cell)):
                year = int(match.group(0))
                if year in seen:
                    continue
                seen.add(year)
                years.append(year)
    return years


def _operational_total_metric_label_for_block(block: dict[str, Any]) -> str | None:
    for _start, _end, label, _cells, _line in block.get("rows", []):
        normalized = _normalized_operational_table_label(label)
        match = re.fullmatch(r"(?:total|consolidated)\s+(.+)", normalized)
        if match:
            return _clean_operational_metric_label(match.group(1)).title()
    return None


def _operational_row_is_dimension_member(label: object, block: dict[str, Any]) -> bool:
    if _operational_label_has_metric_signal(label) or _operational_label_matches_registry_metric(label):
        return False
    total_metric = _operational_total_metric_label_for_block(block)
    if not total_metric:
        return False
    label_tokens = _meaningful_operational_tokens(label)
    metric_tokens = _meaningful_operational_tokens(total_metric)
    return bool(label_tokens and metric_tokens and not (label_tokens & metric_tokens))


def _operational_table_block_has_operating_context(block: dict[str, Any]) -> bool:
    signal_count = 0
    for _start, _end, label, cells, _line in block.get("rows", []):
        if len(_parse_operational_amount_cells(cells[1:])) < 2:
            continue
        if _operational_label_has_metric_signal(label) or _operational_label_matches_registry_metric(label):
            signal_count += 1
    return signal_count >= 2


_OPERATIONAL_SEGMENT_LABEL_DENY_EXACT = {
    "",
    "business",
    "consolidated",
    "operating",
    "reportable",
    "segment",
    "segments",
    "total",
}


def _operational_segment_label_allowed(label: object) -> bool:
    normalized = _normalized_operational_table_label(label)
    if normalized in _OPERATIONAL_SEGMENT_LABEL_DENY_EXACT:
        return False
    if not normalized or normalized.isdigit():
        return False
    tokens = normalized.split()
    if len(tokens) > 6:
        return False
    if _operational_label_has_metric_signal(label) or _operational_label_matches_registry_metric(label):
        return False
    return any(re.search(r"[a-z]", token) for token in tokens)


def _add_operational_segment_label(vocabulary: dict[str, str], label: object) -> None:
    cleaned = _clean_operational_metric_label(label)
    cleaned = re.sub(r"\bsegments?\b$", "", cleaned, flags=re.IGNORECASE).strip(" ,;:-")
    if not _operational_segment_label_allowed(cleaned):
        return
    normalized = _normalized_operational_table_label(cleaned)
    vocabulary.setdefault(normalized, cleaned)


def _discover_operational_segment_vocabulary(text: str) -> dict[str, str]:
    segments: dict[str, str] = {}
    for block in _iter_operational_pipe_table_blocks(text):
        for _start, _end, label, _cells, _line in block.get("rows", []):
            if _operational_row_is_dimension_member(label, block):
                _add_operational_segment_label(segments, label)

    heading_pattern = re.compile(
        r"(?<![a-z0-9])(?P<label>[A-Z][A-Za-z0-9&/().'\- ]{1,80}?)\s+Segments?\b"
    )
    for match in heading_pattern.finditer(text):
        label = match.group("label")
        label = re.split(r"[.\n|]", label)[-1]
        _add_operational_segment_label(segments, label)
    return segments


def _operational_filing_vocabulary(text: str) -> dict[str, Any]:
    return {
        "segments": _discover_operational_segment_vocabulary(text),
    }


def _operational_table_block_relevant(block: dict[str, Any], metrics: list[dict[str, Any]]) -> bool:
    matched_metrics: set[str] = set()
    for _start, _end, label, cells, _line in block.get("rows", []):
        if _operational_row_is_dimension_member(label, block):
            continue
        if len(_parse_operational_amount_cells(cells[1:])) < 2:
            continue
        for metric in metrics:
            if _table_label_matches_operational_metric(label, metric):
                matched_metrics.add(str(metric["canonical"]))
    return len(matched_metrics) >= 2 or _operational_table_block_has_operating_context(block)


def _operational_table_label_is_discoverable(
    label: object,
    *,
    block: dict[str, Any],
    topic: object,
    score: int,
) -> bool:
    if _operational_row_is_dimension_member(label, block):
        return False
    if score > 0 and (_operational_label_has_metric_signal(label) or _operational_label_matches_registry_metric(label)):
        return True
    if score >= 70 and len(_meaningful_operational_tokens(label)) > 1:
        return True
    return _operational_topic_requests_driver_surface(topic) and _operational_table_block_has_operating_context(block) and (
        _operational_label_has_metric_signal(label) or _operational_label_matches_registry_metric(label)
    )


def _operational_period_amount_items_for_row(
    cells: list[str],
    *,
    block: dict[str, Any],
    year: int,
    quarter: int,
) -> list[tuple[str, dict[str, Any]]]:
    amount_values = _parse_operational_amount_cell_values(cells[1:])
    if len(amount_values) < 2:
        return []

    if quarter == 4:
        years = _operational_table_block_years(block)
        if years and len(amount_values) >= len(years):
            by_year = dict(zip(years, amount_values, strict=False))
            return [(f"FY{period_year}", by_year[period_year]) for period_year in years]

    prior_period, current_period = _value_period_labels(year, quarter)
    return [(prior_period, amount_values[0]), (current_period, amount_values[1])]


def _discover_table_operational_metrics(text: str, topic: object) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in _iter_operational_pipe_table_blocks(text):
        for _start, _end, label, cells, line in block.get("rows", []):
            if not _operational_table_label_allowed(label):
                continue
            amount_values = _parse_operational_amount_cells(cells[1:])
            if len(amount_values) < 2:
                continue
            score = _score_operational_label_against_topic(label, topic)
            if not _operational_table_label_is_discoverable(label, block=block, topic=topic, score=score):
                continue
            normalized = _normalized_operational_table_label(label)
            if normalized in seen:
                continue
            seen.add(normalized)
            discovered.append(
                _make_operational_metric(
                    label,
                    match_source="filing_table_label",
                    score=score or 40,
                    unit="USD millions" if "$" in line else "reported filing units",
                )
            )
    return sorted(discovered, key=lambda item: int(item.get("match_score", 0)), reverse=True)


def _discover_prose_operational_metrics(text: str, topic: object) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"(?P<label>[A-Za-z][A-Za-z0-9&/().,\- ']{1,80}?)\s+"
        r"(?:grew|increased|decreased|declined|was up|were up|was down|were down|up|down)\b"
        r"[^.\n]{0,180}?\d+(?:\.\d+)?\s*%",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        label = _clean_operational_metric_label(match.group("label"))
        label = re.sub(r"^(overall|the|our|total)\s+", "", label, flags=re.IGNORECASE).strip()
        if not _operational_table_label_allowed(label):
            continue
        if not (_operational_label_has_metric_signal(label) or _operational_label_matches_registry_metric(label)):
            continue
        score = _score_operational_label_against_topic(label, topic)
        if score <= 0:
            continue
        normalized = _normalized_operational_table_label(label)
        if normalized in seen:
            continue
        seen.add(normalized)
        discovered.append(_make_operational_metric(label, match_source="filing_prose", score=score))
    return sorted(discovered, key=lambda item: int(item.get("match_score", 0)), reverse=True)


def _discover_operational_driver_metrics(topic: object, text: str, *, limit: int = 12) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_token_sets: list[set[str]] = []

    def add(metric: dict[str, Any]) -> None:
        normalized = _normalized_operational_table_label(metric.get("canonical"))
        if not normalized or normalized in seen:
            return
        tokens = _meaningful_operational_tokens(normalized)
        if tokens and any(tokens <= existing or existing <= tokens for existing in seen_token_sets):
            return
        seen.add(normalized)
        if tokens:
            seen_token_sets.append(tokens)
        selected.append(metric)

    for metric in _select_seed_operational_driver_metrics(topic):
        if metric.get("match_source") == "registry_alias":
            add(metric)

    for metric in _discover_table_operational_metrics(text, topic):
        add(metric)
        if len(selected) >= limit:
            return selected

    for metric in _discover_prose_operational_metrics(text, topic):
        add(metric)
        if len(selected) >= limit:
            return selected

    if not selected:
        for metric in _select_seed_operational_driver_metrics(topic):
            fallback = dict(metric)
            fallback.setdefault("match_source", "registry_fallback")
            add(fallback)
            if len(selected) >= limit:
                break
    return selected


def _extract_operational_rows_from_pipe_tables(
    text: str,
    *,
    metrics: list[dict[str, Any]],
    year: int,
    quarter: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparison_period = _driver_period_label(year, quarter)
    for block in _iter_operational_pipe_table_blocks(text):
        if not _operational_table_block_relevant(block, metrics):
            continue
        source_group_id = _operational_source_group_id("markdown_table", int(block["start"]))
        for start, end, label, cells, line in block.get("rows", []):
            if _operational_row_is_dimension_member(label, block):
                continue
            matched_metric = next(
                (metric for metric in metrics if _table_label_matches_operational_metric(label, metric)),
                None,
            )
            if matched_metric is None:
                continue
            metric_name = str(matched_metric["canonical"])
            source = _source_from_operational_span(text, start, end)
            amount_values = _operational_period_amount_items_for_row(cells, block=block, year=year, quarter=quarter)
            for period, item in amount_values:
                rows.append(
                    {
                        "kind": "metric_value",
                        "metric_name": metric_name,
                        "segment": "Consolidated",
                        "period": period,
                        "basis": "reported",
                        "value": item["value"],
                        "value_raw": item["value_raw"],
                        "value_semantic": "period_absolute_value",
                        "unit": matched_metric.get("unit") or ("USD millions" if "$" in line else "reported filing units"),
                        "source_basis": "table_reported",
                        "precision": item["precision"],
                        "recommended_for_calculation": True,
                        "source_group_id": source_group_id,
                        "source": source,
                    }
                )
            percent_values = _parse_operational_percent_cell_values(cells[1:]) if amount_values else []
            if percent_values:
                _append_growth_row(
                    rows,
                    metric_name=metric_name,
                    segment="Consolidated",
                    period=comparison_period,
                    basis="reported",
                    value_percent=percent_values[0]["value"],
                    value_raw=percent_values[0]["value_raw"],
                    source=source,
                    source_basis="table_reported",
                    precision=percent_values[0]["precision"],
                    source_group_id=source_group_id,
                    recommended_for_calculation=True,
                )
                if len(percent_values) > 1:
                    _append_growth_row(
                        rows,
                        metric_name=metric_name,
                        segment="Consolidated",
                        period=comparison_period,
                        basis="constant_currency",
                        value_percent=percent_values[-1]["value"],
                        value_raw=percent_values[-1]["value_raw"],
                        source=source,
                        source_basis="table_reported",
                        precision=percent_values[-1]["precision"],
                        source_group_id=source_group_id,
                        recommended_for_calculation=True,
                    )
    return rows


def _operational_row_numeric_value(row: dict[str, Any]) -> float | None:
    metric_value = row.get("value_percent") if row.get("kind") == "growth_rate" else row.get("value")
    return metric_value if isinstance(metric_value, float | int) else None


def _operational_source_rank(row: dict[str, Any]) -> int:
    source_basis = row.get("source_basis")
    if source_basis == "table_reported":
        return 0
    if source_basis == "narrative_reported":
        return 1
    if source_basis == "narrative_change_amount":
        return 2
    return 3


def _dedupe_operational_driver_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in sorted(rows, key=_operational_source_rank):
        metric_value = _operational_row_numeric_value(row)
        if isinstance(metric_value, float | int):
            metric_value = round(metric_value, 6)
        key = (
            row.get("kind"),
            row.get("metric_name"),
            row.get("segment"),
            row.get("period"),
            row.get("basis"),
            metric_value,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _annotate_operational_basis_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_basis_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") not in {"growth_rate", "metric_value"}:
            continue
        key = (
            row.get("kind"),
            row.get("metric_name"),
            row.get("segment"),
            row.get("period"),
            row.get("basis"),
        )
        by_basis_key.setdefault(key, []).append(row)

    for grouped_rows in by_basis_key.values():
        bases = {row.get("source_basis") for row in grouped_rows}
        values = {
            round(value, 6)
            for row in grouped_rows
            for value in [_operational_row_numeric_value(row)]
            if value is not None
        }
        if "table_reported" not in bases or not any(str(basis).startswith("narrative") for basis in bases):
            continue
        if len(values) <= 1:
            continue
        table_values = [
            _operational_row_numeric_value(row)
            for row in grouped_rows
            if row.get("source_basis") == "table_reported"
        ]
        note = "Conflicting table and narrative values; prefer same-source table rows for calculations."
        for row in grouped_rows:
            row["basis_conflict"] = True
            row["basis_conflict_note"] = note
            if row.get("source_basis") != "table_reported":
                row["recommended_for_calculation"] = False
            if table_values and row.get("source_basis") != "table_reported":
                row["table_reported_value"] = table_values[0]
    return rows


def _operational_basis_conflict_warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if not row.get("basis_conflict"):
            continue
        key = (row.get("metric_name"), row.get("period"), row.get("basis"))
        if key in seen:
            continue
        seen.add(key)
        warnings.append(
            f"Basis conflict for {row.get('metric_name')} {row.get('period')} ({row.get('basis')}); "
            "prefer table_reported rows with recommended_for_calculation=true."
        )
    return warnings


def _extract_operational_driver_rows(
    text: str,
    *,
    metrics: list[dict[str, Any]],
    year: int,
    quarter: int,
    filing_vocabulary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = _extract_operational_rows_from_pipe_tables(text, metrics=metrics, year=year, quarter=quarter)
    rows.extend(_extract_operational_value_rows_from_prose(text, metrics=metrics))
    rows.extend(
        _extract_operational_growth_rows_from_prose(
            text,
            metrics=metrics,
            year=year,
            quarter=quarter,
            filing_vocabulary=filing_vocabulary,
        )
    )
    return _annotate_operational_basis_conflicts(_dedupe_operational_driver_rows(rows))


_SPECIAL_SECTION_ALIASES = {
    "earnings_release": "earnings_release",
    "earnings_press_release": "earnings_release",
    "proxy_statement": "proxy_statement",
    "annual_report": "annual_report",
    "form_20_f": "annual_report",
    "20_f": "annual_report",
    "foreign_report": "foreign_report",
    "foreign_issuer_report": "foreign_report",
    "form_6_k": "foreign_report",
    "6_k": "foreign_report",
}


def _normalize_requested_section_key(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        return ""

    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    slug = slug.replace("part_i_", "part1_").replace("part_ii_", "part2_")
    slug = re.sub(r"^part_?([12])_?", r"part\1_", slug)

    if slug in _SPECIAL_SECTION_ALIASES:
        return _SPECIAL_SECTION_ALIASES[slug]
    canonical_match = re.fullmatch(
        r"(?:(part[12])_)?item_?(\d{1,2})([a-z]?)(?:_notes)?",
        slug,
    )
    if canonical_match:
        part_prefix = f"{canonical_match.group(1)}_" if canonical_match.group(1) else ""
        item_num = canonical_match.group(2)
        item_letter = canonical_match.group(3) or ""
        notes_suffix = "_notes" if slug.endswith("_notes") else ""
        item_key = f"item{item_num}{item_letter}" if part_prefix else f"item_{item_num}{item_letter}"
        return f"{part_prefix}{item_key}{notes_suffix}"

    lowered = raw.lower()
    item_match = re.search(r"\bitem\s*\.?\s*(\d{1,2})\s*\.?\s*([a-z])?\b", lowered)
    if not item_match:
        return raw

    part_prefix = ""
    prefix = lowered[: item_match.start()]
    part_match = re.search(r"\bpart\s*(i|ii|1|2)\b", prefix)
    if part_match:
        part_token = part_match.group(1)
        part_prefix = "part1_" if part_token in {"i", "1"} else "part2_"

    item_num = item_match.group(1)
    item_letter = item_match.group(2) or ""
    item_key = f"item{item_num}{item_letter}" if part_prefix else f"item_{item_num}{item_letter}"
    return f"{part_prefix}{item_key}"


def _normalize_requested_sections(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        try:
            raw_items = list(value)  # type: ignore[arg-type]
        except TypeError:
            raw_items = [value]

    normalized = [
        section
        for item in raw_items
        if (section := _normalize_requested_section_key(item))
    ]
    return normalized or None


def _list_filing_section_headers(text: str) -> list[str]:
    return [
        match.group("header").strip()
        for match in re.finditer(r"^## SECTION: (?P<header>.+)$", text, flags=re.MULTILINE)
    ]


def _match_section_headers(headers: list[str], filters: list[str]) -> list[str]:
    compiled = [re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE) for value in filters]
    matched: list[str] = []
    seen: set[str] = set()
    for header in headers:
        if not any(pattern.search(header) for pattern in compiled):
            continue
        if header in seen:
            continue
        seen.add(header)
        matched.append(header)
    return matched


def _normalize_extraction_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, sort_keys=True, default=str)
        deduped[key] = item
    return sorted(
        deduped.values(),
        key=lambda item: (
            int(item.get("char_start", -1)),
            int(item.get("char_end", -1)),
            str(item.get("class", "")),
            str(item.get("text", "")),
        ),
    )


def _build_metric_catalog(
    financials_result: dict,
    date_type: str | None = None,
    *,
    full_year_mode: bool = False,
) -> list[dict]:
    """Create a deduplicated metric catalog from /api/financials facts."""
    facts = financials_result.get("facts")
    if not isinstance(facts, list):
        return []

    target_date_type = _normalize_date_type(date_type)
    catalog = {}

    for fact in facts:
        if not isinstance(fact, dict):
            continue
        raw_tag = fact.get("tag")
        if not raw_tag or not isinstance(raw_tag, str):
            continue

        fact_date_type = _normalize_date_type(fact.get("date_type"))
        if not _catalog_date_type_matches(
            fact,
            fact_date_type=fact_date_type,
            target_date_type=target_date_type,
            full_year_mode=full_year_mode,
        ):
            continue

        bare_tag = raw_tag.split(":", 1)[1] if ":" in raw_tag else raw_tag
        current, prior = _pick_metric_values(fact)
        axis_key = str(fact.get("axis_key") or "__NONE__")
        candidate = {
            "metric_name": bare_tag,
            "tag": raw_tag,
            "concept_label": fact.get("concept_label"),
            "date_type": fact_date_type,
            "axis_key": axis_key,
            "dimensions": fact.get("dimensions"),
            "source": fact.get("source"),
            "scale": fact.get("scale"),
            "current_value": current,
            "prior_value": prior,
            "has_value": current is not None or prior is not None,
        }
        debt_component_kind = _debt_component_kind(fact)
        if debt_component_kind:
            candidate["debt_component_kind"] = debt_component_kind
        if fact.get("scope_status"):
            candidate["scope_status"] = fact.get("scope_status")
        if fact.get("scope_warning"):
            candidate["scope_warning"] = fact.get("scope_warning")
        if fact.get("scope_bridge_ids"):
            candidate["scope_bridge_ids"] = fact.get("scope_bridge_ids")
        candidate.update(enrich_match_metadata(fact))

        key = (raw_tag.lower(), fact_date_type or "", axis_key)
        existing = catalog.get(key)
        if existing is None:
            catalog[key] = candidate
            continue

        # Prefer entries with usable values.
        if candidate["has_value"] and not existing["has_value"]:
            catalog[key] = candidate

    return sorted(
        catalog.values(),
        key=lambda item: (
            (item["metric_name"] or "").lower(),
            item["date_type"] or "",
            item["axis_key"] or "",
        ),
    )


def _score_metric_match(query: str, metric: dict) -> float:
    query_variants = _expand_query_variants(query)
    if not query_variants:
        return 0.0

    metric_tokens = _metric_search_tokens(metric)
    if not metric_tokens:
        return 0.0

    metric_text = " ".join(metric_tokens)
    metric_compact = "".join(metric_tokens)
    metric_token_set, dimension_token_set = _metric_search_token_sets(metric)

    best_score = 0.0
    for query_tokens in query_variants:
        query_text = " ".join(query_tokens)
        query_compact = "".join(query_tokens)
        score = 0.0

        if query_text == metric_text or query_compact == metric_compact:
            score = max(score, 100.0)

        # Phrase and compact containment handle spaces/hyphens/camel-case differences.
        if query_text and query_text in metric_text:
            score = max(score, 94.0)
        if query_compact and query_compact in metric_compact:
            score = max(score, 90.0)

        # Token coverage captures multi-word fuzzy matches.
        meaningful_tokens = _meaningful_metric_search_tokens(query_tokens)
        overlap_tokens = meaningful_tokens & metric_token_set
        if overlap_tokens and (len(meaningful_tokens) < 3 or len(overlap_tokens) >= 2):
            coverage = len(overlap_tokens) / max(len(meaningful_tokens), 1)
            score = max(score, 55.0 + (coverage * 30.0))

        # Final safety net for near matches.
        if query_compact and metric_compact:
            ratio = SequenceMatcher(None, query_compact, metric_compact).ratio()
            if ratio >= 0.55:
                score = max(score, 45.0 + (ratio * 35.0))

        if score > 0 and any(token in dimension_token_set for token in query_tokens):
            score += 0.1

        best_score = max(best_score, score)

    return round(best_score, 2)


def _metric_dimension_score_adjustment(query: str, metric: dict) -> float:
    query_tokens = set(_split_identifier_tokens(query))
    if not query_tokens:
        return 0.0

    dimension_token_set = set()
    for dim in metric.get("dimensions") or []:
        if not isinstance(dim, dict):
            continue
        for field in (dim.get("axis_label", ""), dim.get("member_label", "")):
            tokens = _split_identifier_tokens(field)
            dimension_token_set.update(tokens)
            if tokens:
                dimension_token_set.add("".join(tokens))

    return 0.1 if query_tokens & dimension_token_set else 0.0


def _metric_discovery_score_adjustment(
    query: str,
    metric: dict,
    *,
    date_type: str | None,
    full_year_mode: bool,
    query_family: str | None = None,
    metric_family: str | None = None,
) -> float:
    adjustment = 0.0
    query_tokens = set(_split_identifier_tokens(query))
    debt_component_kind = metric.get("debt_component_kind")

    relation = _metric_family_relation(query_family, metric_family)
    if relation == "exact":
        adjustment += 2.0
    elif relation == "related":
        adjustment -= 2.0
    elif relation == "incompatible":
        adjustment -= 35.0

    if (
        full_year_mode
        and date_type == "FY"
        and _normalize_date_type(metric.get("date_type")) == "Q"
        and _metric_looks_balance_sheet_snapshot(metric)
    ):
        adjustment += 0.5 if str(metric.get("axis_key") or "__NONE__") == "__NONE__" else 0.1
        if "total" in query_tokens:
            adjustment += 0.75

    is_debt_instrument_basis_query = (
        query_family == "debt_total"
        and bool(query_tokens & _DEBT_QUERY_TOKENS)
        and bool(query_tokens & _DEBT_INSTRUMENT_BASIS_QUERY_TOKENS)
    )
    if is_debt_instrument_basis_query:
        if debt_component_kind == "instrument_principal":
            adjustment += 34.0
        elif debt_component_kind == "coupon_rate":
            adjustment += 6.0
        elif debt_component_kind in {
            "current_debt",
            "debt_carrying_amount",
            "finance_lease",
            "other_notes",
            "total_debt_rollup",
        }:
            adjustment -= 24.0

    if query_tokens & {"coupon", "coupons", "stated"}:
        if debt_component_kind == "coupon_rate":
            adjustment += 18.0
        elif debt_component_kind == "instrument_principal":
            adjustment += 4.0
    if "finance" in query_tokens and debt_component_kind == "finance_lease":
        adjustment += 4.0
    if query_tokens & {"notes", "other"} and debt_component_kind == "other_notes":
        adjustment += 4.0

    return adjustment


def _deadline_expired(args: dict) -> bool:
    """Cooperative timeout check for worker-thread handlers."""
    deadline = args.get("__deadline_monotonic")
    if deadline is None:
        return False
    try:
        return time.monotonic() >= float(deadline)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Tool dispatch — remote API proxies
# ---------------------------------------------------------------------------

def _proxy_get_filings(args: dict) -> dict:
    params = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
    }
    source = _normalize_agent_source_arg(args.get("source"))
    if source == "proxy":
        return {
            "status": "error",
            "error_type": "unsupported_source_for_tool",
            "message": (
                "get_filings does not list proxy/DEF 14A filings. Use "
                "get_event_filings with form_types=['DEF 14A'] for proxy discovery, "
                "or use get_filing_document/get_filing_sections/get_filing_tables "
                "with source='proxy' and quarter=4 for proxy content."
            ),
            "remediation_tool": "get_event_filings",
            "remediation_hint": (
                "For proxy questions, do not retry get_filings with source='proxy'. "
                "DEF 14A metadata is event-filing discovery; proxy content tools use "
                "source='proxy' on the annual quarter only."
            ),
        }
    if source not in {"auto", "8k", "20f", "6k"}:
        return {"status": "error", "message": "source must be one of auto, 8k, 20f, 6k"}
    if source != "auto":
        params["source"] = source
    return _call_api("/api/filings", params)


def _proxy_get_event_filings(args: dict) -> dict:
    params: dict[str, Any] = {
        "limit": args.get("limit", 50),
        "sort_order": args.get("sort_order", "desc"),
    }
    for key in (
        "ticker",
        "cik",
        "filing_date_from",
        "filing_date_to",
        "query",
    ):
        value = args.get(key)
        if value is not None:
            params[key] = value
    for key in ("form_types", "related_tickers"):
        value = args.get(key)
        if value:
            params[key] = value
    return _call_api("/api/filings/events", params, timeout=60)


def _proxy_describe_filing(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}
    return _call_api(
        "/api/filing/describe",
        {
            "ticker": args["ticker"],
            "year": args["year"],
            "quarter": args["quarter"],
            "source": source,
        },
        timeout=30,
    )


def _proxy_get_financials(args: dict) -> dict:
    output_mode = args.get("output", "file")

    result = _call_api("/api/financials", {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "full_year_mode": str(args.get("full_year_mode", False)).lower(),
        "source": args.get("source", "auto"),
    })

    if result.get("status") != "success" or output_mode != "file":
        return result

    # Write full JSON to local file, return summary + file_path
    ticker = _safe_filename_part(str(args["ticker"]).upper(), "ticker")
    year = int(args["year"])
    quarter = int(args["quarter"])
    source_info = (result.get("metadata", {}).get("source") or result.get("source") or {})
    filing_type = source_info.get("filing_type", "")

    FILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{ticker}_{quarter}Q{year % 100:02d}_financials.json"
    file_path = (FILE_OUTPUT_DIR / filename).resolve()
    root_dir = FILE_OUTPUT_DIR.resolve()
    if not file_path.is_relative_to(root_dir):
        return {"status": "error", "message": "Invalid output path"}
    if _deadline_expired(args):
        return {"status": "error", "message": "Request timed out before file output could be written"}

    file_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    facts = result.get("facts", []) if isinstance(result.get("facts"), list) else []

    response = {
        "status": "success",
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "filing_type": filing_type,
        "output": "file",
        "file_path": str(file_path.resolve()),
        "hint": "Use Read tool with file_path. Use jq or Grep to search for specific metrics.",
        "metadata": {
            "total_facts": len(facts),
            "source": source_info,
        },
    }
    if result.get("scope_warnings"):
        response["scope_warnings"] = result.get("scope_warnings")
    if result.get("scope_bridges"):
        response["scope_bridges"] = result.get("scope_bridges")
    return response


def _proxy_get_metric(args: dict) -> dict:
    date_type = args.get("date_type") or ""
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    try:
        role_param = _role_query_param(args.get("role"))
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    params = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "metric_name": args["metric_name"],
        "full_year_mode": str(effective_full_year_mode).lower(),
        "source": args.get("source", "auto"),
        "date_type": date_type,
    }
    if role_param:
        params["role"] = role_param
    response = _call_api("/api/metric", params)
    response = _annotate_metric_cache_miss(response, args)
    response = _annotate_operational_kpi_metric_miss(response, args)
    return _annotate_financial_metric_miss(response, args)


def _proxy_get_concept(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _CONCEPT_SOURCES:
        return {"status": "error", "message": _CONCEPT_SOURCE_ERROR}

    date_type = args.get("date_type") or ""
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    params = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "concept_name": args["concept_name"],
        "full_year_mode": str(effective_full_year_mode).lower(),
        "source": source,
        "date_type": date_type,
    }
    return _call_api("/api/concept", params, timeout=300)


def _proxy_cite_concept(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _CONCEPT_SOURCES:
        return {"status": "error", "message": _CONCEPT_SOURCE_ERROR}

    date_type = args.get("date_type") or ""
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    payload: dict[str, Any] = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "concept_name": args["concept_name"],
        "full_year_mode": effective_full_year_mode,
        "source": source,
        "date_type": date_type,
        "include_extractions": bool(args.get("include_extractions", False)),
        "max_narrative_spans_per_source": args.get("max_narrative_spans_per_source", 5),
        "allow_stale_extractions": bool(args.get("allow_stale_extractions", False)),
    }
    if args.get("narrative_sources") is not None:
        payload["narrative_sources"] = args["narrative_sources"]
    return _post_api("/api/concept/cite", payload, timeout=600)


def _proxy_compare_concept(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _CONCEPT_SOURCES:
        return {"status": "error", "message": _CONCEPT_SOURCE_ERROR}

    tickers = args.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        return {"status": "error", "message": "tickers must be a non-empty list"}

    date_type = args.get("date_type") or ""
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    payload = {
        "concept_name": args["concept_name"],
        "tickers": tickers,
        "year": args["year"],
        "quarter": args["quarter"],
        "full_year_mode": effective_full_year_mode,
        "source": source,
        "date_type": date_type,
        "ticker_periods": args.get("ticker_periods"),
    }
    return _post_api("/api/concept/compare", payload, timeout=300)


def _proxy_concept_trend(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _CONCEPT_SOURCES:
        return {"status": "error", "message": _CONCEPT_SOURCE_ERROR}

    params = {
        "ticker": args["ticker"],
        "concept_name": args["concept_name"],
        "period_from": args["period_from"],
        "period_to": args["period_to"],
        "source": source,
        "date_type": args.get("date_type") or "",
    }
    return _call_api("/api/concept/trend", params, timeout=600)


def _proxy_get_statement(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _CONCEPT_SOURCES:
        return {"status": "error", "message": _CONCEPT_SOURCE_ERROR}

    params: dict[str, Any] = {
        "ticker": args["ticker"],
        "statement": args["statement"],
        "source": source,
        "date_type": args.get("date_type") or "",
    }
    if args.get("year") is not None:
        params["year"] = args["year"]
    if args.get("quarter") is not None:
        params["quarter"] = args["quarter"]
    if args.get("year") is not None or args.get("quarter") is not None:
        effective_full_year_mode = _effective_full_year_mode_for_metric_request(
            quarter=args.get("quarter"),
            full_year_mode=args.get("full_year_mode", False),
            date_type=params["date_type"],
        )
        params["full_year_mode"] = str(effective_full_year_mode).lower()
    elif args.get("full_year_mode") is not None:
        params["full_year_mode"] = str(_truthy_bool_arg(args.get("full_year_mode"))).lower()
    if args.get("period_from") is not None:
        params["period_from"] = args["period_from"]
    if args.get("period_to") is not None:
        params["period_to"] = args["period_to"]
    return _call_api("/api/statement", params, timeout=600)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _first_metric_series_period_error(result: dict[str, Any]) -> str | None:
    series = result.get("series")
    if not isinstance(series, list):
        return None
    for entry in series:
        if not isinstance(entry, dict):
            continue
        message = entry.get("error") or entry.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def _metric_series_error_message(result: dict[str, Any]) -> str:
    ticker = str(result.get("ticker") or "requested ticker")
    metric_name = str(result.get("metric_name") or "requested metric")
    requested = _safe_int(result.get("periods_requested"))
    returned = _safe_int(result.get("periods_returned") or result.get("periods_fetched"))
    uncached = _safe_int(result.get("periods_uncached"))
    missing = _safe_int(result.get("periods_missing"))
    failed = _safe_int(result.get("periods_failed"))
    total = requested or returned + uncached + missing + failed
    period_error = _first_metric_series_period_error(result)

    if result.get("cache_miss") or uncached:
        uncached_count = uncached or missing or total
        count = f"{uncached_count}/{total} requested periods" if total else "requested periods"
        return (
            f"No cached metric series data for {ticker} {metric_name}: {count} are uncached. "
            "Call warm_metric_cache with warm_hint.body.items, poll warm_metric_cache_status "
            "until complete, then retry get_metric_series."
        )
    if failed:
        count = f"{failed}/{total} requested periods" if total else f"{failed} period(s)"
        detail = f": {period_error}" if period_error else "."
        return f"Metric series failed for {count} for {ticker} {metric_name}{detail}"
    if missing:
        count = f"{missing}/{total} requested periods" if total else f"{missing} period(s)"
        detail = f": {period_error}" if period_error else "."
        return f"Metric series has no values for {count} for {ticker} {metric_name}{detail}"
    return (
        f"Metric series returned status=error for {ticker} {metric_name} with no top-level "
        "message; inspect series entries for period-level errors."
    )


def _enrich_metric_series_error(result: dict) -> dict:
    if result.get("status") != "error" or result.get("message"):
        return result
    enriched = dict(result)
    enriched["message"] = _metric_series_error_message(enriched)
    return enriched


def _proxy_get_metric_series(args: dict) -> dict:
    date_type = args.get("date_type") or ""
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("end_quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    try:
        role_param = _role_query_param(args.get("role"))
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    params = {
        "ticker": args["ticker"],
        "metric_name": args["metric_name"],
        "end_year": args["end_year"],
        "end_quarter": args["end_quarter"],
        "periods": args.get("periods", 8),
        "full_year_mode": str(effective_full_year_mode).lower(),
        "source": args.get("source", "auto"),
        "date_type": date_type,
        "cached_only": str(args.get("cached_only", False)).lower(),
    }
    if role_param:
        params["role"] = role_param
    axis_key = str(args.get("axis_key") or "").strip()
    if axis_key:
        params["axis_key"] = axis_key
    result = _call_api(
        "/api/metric/series",
        params,
        timeout=600,
    )
    return _enrich_metric_series_error(result)


def _proxy_warm_metric_cache(args: dict) -> dict:
    items = args.get("items")
    if not isinstance(items, list) or not items:
        return {"status": "error", "message": "items must be a non-empty list"}
    return _post_api("/api/warm", {"items": items}, timeout=300)


def _proxy_warm_metric_cache_status(args: dict) -> dict:
    job_id = str(args.get("job_id", "")).strip()
    if not job_id:
        return {"status": "error", "message": "job_id is required"}
    return _call_api(f"/api/warm/{job_id}", {}, timeout=60)


def _proxy_list_metrics(args: dict) -> dict:
    date_type = _normalize_date_type(args.get("date_type"))
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    limit = args.get("limit", 200)
    include_values = bool(args.get("include_values", True))
    try:
        limit = max(1, min(int(limit), 1000))
    except (TypeError, ValueError):
        return {"status": "error", "message": "limit must be an integer between 1 and 1000"}

    financials = _call_api("/api/financials", {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "full_year_mode": str(effective_full_year_mode).lower(),
        "source": args.get("source", "auto"),
    })
    if financials.get("status") != "success":
        return financials

    catalog = _build_metric_catalog(
        financials,
        date_type=date_type,
        full_year_mode=effective_full_year_mode,
    )
    total_candidates = len(catalog)
    catalog = catalog[:limit]

    if not include_values:
        for item in catalog:
            item.pop("current_value", None)
            item.pop("prior_value", None)

    metadata = financials.get("metadata", {}) if isinstance(financials.get("metadata"), dict) else {}
    response = {
        "status": "success",
        "ticker": str(args["ticker"]).upper(),
        "year": int(args["year"]),
        "quarter": int(args["quarter"]),
        "full_year_mode": effective_full_year_mode,
        "source": metadata.get("source", {}),
        "date_type_filter": date_type,
        "total_candidates": total_candidates,
        "returned_candidates": len(catalog),
        "metrics": catalog,
        "hint": "Pass metric_name from this list into get_metric.",
    }
    if financials.get("scope_warnings"):
        response["scope_warnings"] = financials.get("scope_warnings")
    if financials.get("scope_bridges"):
        response["scope_bridges"] = financials.get("scope_bridges")
    return response


def _proxy_search_metrics(args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"status": "error", "message": "Missing required parameter: query"}

    date_type = _normalize_date_type(args.get("date_type"))
    effective_full_year_mode = _effective_full_year_mode_for_metric_request(
        quarter=args.get("quarter"),
        full_year_mode=args.get("full_year_mode", False),
        date_type=date_type,
    )
    expand_annual_snapshots = (
        effective_full_year_mode
        and _query_looks_balance_sheet_metric(query)
    )
    try:
        role_param = _role_query_param(args.get("role"))
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    role_filter = set(normalize_role_filter(role_param))
    limit = args.get("limit", 20)
    include_values = bool(args.get("include_values", True))
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        return {"status": "error", "message": "limit must be an integer between 1 and 100"}

    financials = _call_api("/api/financials", {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "full_year_mode": str(effective_full_year_mode).lower(),
        "source": args.get("source", "auto"),
    })
    if financials.get("status") != "success":
        return financials

    catalog = _build_metric_catalog(
        financials,
        date_type=date_type,
        full_year_mode=expand_annual_snapshots,
    )
    if role_filter:
        catalog = [
            item
            for item in catalog
            if role_filter.intersection(item.get("statement_roles") or [])
        ]
    query_family = _infer_metric_query_family(query)
    query_profile = _metric_search_query_profile(query, query_family)
    ranked = []
    for item in catalog:
        metric_family = _metric_semantic_family(item)
        if not _metric_search_candidate_allowed(query_family, metric_family):
            continue
        semantic_relation = _metric_family_relation(query_family, metric_family)
        lexical_score = _score_metric_match(query, item)
        semantic_floor = _metric_semantic_score_floor(query_family, metric_family)
        score = max(lexical_score, semantic_floor)
        if semantic_floor >= lexical_score:
            score += _metric_dimension_score_adjustment(query, item)
        score += _metric_discovery_score_adjustment(
            query,
            item,
            date_type=date_type,
            full_year_mode=expand_annual_snapshots,
            query_family=query_family,
            metric_family=metric_family,
        )
        modifier_evidence = _metric_search_required_modifier_evidence(query_profile, item)
        score = _metric_search_apply_modifier_gate(
            score,
            query_profile=query_profile,
            modifier_evidence=modifier_evidence,
            semantic_relation=semantic_relation,
        )
        if score <= 0:
            continue
        match = {**item, "match_score": round(score, 2)}
        if metric_family:
            match["semantic_family"] = metric_family
        if semantic_relation:
            match["semantic_relation"] = semantic_relation
        if query_profile["required_modifiers"]:
            matched_modifiers = sorted(modifier_evidence["matched"])
            unmatched_modifiers = sorted(modifier_evidence["unmatched"])
            match["matched_query_modifiers"] = matched_modifiers
            match["unmatched_query_modifiers"] = unmatched_modifiers
            match["match_confidence"] = "fallback" if unmatched_modifiers else "strong"
        ranked.append(match)

    ranked.sort(key=lambda item: (-item["match_score"], item["metric_name"].lower(), item.get("date_type") or ""))
    ranked = ranked[:limit]
    low_confidence, confidence_reason = _metric_search_confidence(query_profile, ranked)

    if not include_values:
        for item in ranked:
            item.pop("current_value", None)
            item.pop("prior_value", None)

    metadata = financials.get("metadata", {}) if isinstance(financials.get("metadata"), dict) else {}
    response = {
        "status": "success",
        "ticker": str(args["ticker"]).upper(),
        "year": int(args["year"]),
        "quarter": int(args["quarter"]),
        "full_year_mode": effective_full_year_mode,
        "query": query,
        "query_intent": query_family,
        "required_query_modifiers": sorted(query_profile["required_modifiers"]),
        "date_type_filter": date_type,
        "source": metadata.get("source", {}),
        "total_matches": len(ranked),
        "matches": ranked,
        "low_confidence": low_confidence,
        "confidence_reason": confidence_reason,
        "hint": (
            "Low confidence: validate fallback matches before use; operational KPIs may require narrative/table tools."
            if low_confidence
            else "Use top match.metric_name with get_metric, then validate returned metric tag/value."
        ),
    }
    if query_profile.get("is_debt_instrument_basis_query"):
        response["debt_component_guidance"] = (
            "Structured XBRL debt components are surfaced without pre-summing. "
            "For refinancing or rate-shock analysis, consumers should choose the "
            "appropriate component basis, such as instrument_principal rows, and "
            "aggregate only after validating exclusions like finance_lease and other_notes."
        )
    if financials.get("scope_warnings"):
        response["scope_warnings"] = financials.get("scope_warnings")
    if financials.get("scope_bridges"):
        response["scope_bridges"] = financials.get("scope_bridges")
    return response


def _proxy_get_filing_sections(args: dict) -> dict:
    """Proxy filing sections to remote API, with local file-write for output='file'."""
    output_mode = args.get("output", "file")
    sections_list = _normalize_requested_sections(args.get("sections"))
    tables_only = args.get("tables_only", False)
    include_tables = bool(args.get("include_tables", False))
    fallback = bool(args.get("fallback", False))
    source_raw = args.get("source")
    source = _normalize_sections_source(source_raw)
    if source not in (None, "8k", "proxy", "20f", "6k"):
        return {
            "status": "error",
            "message": f"Invalid source: '{source_raw}'. Supported: '8k', 'proxy', '20f', '6k', or omit for 10-K/10-Q.",
        }

    # Build remote API params
    params = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "format": args.get("format", "summary"),
    }
    if source is not None:
        params["source"] = source
    if sections_list:
        params["sections"] = ",".join(sections_list)
    params["include_tables"] = "true" if include_tables else "false"
    if fallback:
        params["fallback"] = "true"

    if output_mode == "file":
        # Fetch full untruncated text for file output
        params["format"] = "full"
        params["max_words"] = "none"
    else:
        max_words = args.get("max_words", 3000)
        params["max_words"] = str(max_words) if max_words is not None else "none"

    result = _call_api("/api/sections", params)

    if result.get("status") != "success":
        return result

    # Normalize section-level table counts for both inline and file modes.
    for section in result.get("sections", {}).values():
        if not isinstance(section, dict):
            continue
        tables = section.get("tables", []) or []
        nonempty_tables = [t for t in tables if (t or "").strip()]
        section["table_count"] = len(nonempty_tables)
    total_tables = sum(
        s.get("table_count", 0)
        for s in result.get("sections", {}).values()
        if isinstance(s, dict)
    )
    if not isinstance(result.get("metadata"), dict):
        result["metadata"] = {}
    result["metadata"]["total_table_count"] = total_tables

    # Strip narrative text if tables_only requested
    if tables_only:
        for section in result.get("sections", {}).values():
            if not isinstance(section, dict):
                continue
            section.pop("text", None)
            table_words = 0
            for table in section.get("tables", []) or []:
                table_text = (table or "").strip()
                if table_text:
                    table_words += len(table_text.split())
            section["word_count"] = table_words

    if output_mode != "file":
        return result

    # Write sections to local markdown file
    ticker = _safe_filename_part(str(args["ticker"]).upper(), "ticker")
    year = int(args["year"])
    quarter = int(args["quarter"])
    filing_type = result.get("filing_type", "")
    sections_data = result.get("sections", {})

    FILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build filename
    if source == "8k":
        source_suffix = "_8k"
    elif source == "proxy":
        source_suffix = "_proxy"
    elif source == "20f":
        source_suffix = "_20f"
    elif source == "6k":
        source_suffix = "_6k"
    else:
        source_suffix = ""
    if sections_list:
        safe_keys = [_safe_filename_part(key, "section") for key in sorted(sections_list)]
        keys_part = "_".join(safe_keys)
        filename = f"{ticker}_{quarter}Q{year % 100:02d}{source_suffix}_{keys_part}.md"
    else:
        filename = f"{ticker}_{quarter}Q{year % 100:02d}{source_suffix}_sections.md"
    file_path = (FILE_OUTPUT_DIR / filename).resolve()
    root_dir = FILE_OUTPUT_DIR.resolve()
    if not file_path.is_relative_to(root_dir):
        return {"status": "error", "message": "Invalid output path"}
    if _deadline_expired(args):
        return {"status": "error", "message": "Request timed out before file output could be written"}

    # Build markdown
    total_words = sum(s.get("word_count", 0) for s in sections_data.values())
    section_keys = ", ".join(sections_data.keys()) if sections_data else "none"
    lines = [
        f"# {ticker} {filing_type} - Q{quarter} FY{year}: Filing Sections",
        f"> Sections: {section_keys} | Total words: {total_words:,} | Total tables: {total_tables:,}",
        "---",
    ]
    for section in sections_data.values():
        header = section.get("header", "Unknown Section")
        lines.append(f"## SECTION: {header}")
        state = section.get("state")
        if state:
            state_line = f"**State:** {state}"
            if state == "cross_reference" and section.get("cross_reference_target"):
                state_line += f" ({section.get('cross_reference_target')})"
            if section.get("declaration_type"):
                state_line += f" | **Declaration:** {section.get('declaration_type')}"
            lines.append(state_line)
        lines.append(f"**Word count:** {section.get('word_count', 0):,}")
        lines.append(f"**Table count:** {section.get('table_count', 0):,}")
        if not tables_only:
            text = section.get("text", "").strip()
            if text:
                lines.append(text)
        tables = section.get("tables", [])
        if tables:
            lines.append("### TABLES")
            for table in tables:
                table_text = (table or "").strip()
                if table_text:
                    lines.append(table_text)
        lines.append("---")
    if lines and lines[-1] == "---":
        lines.pop()

    file_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    tables_file_path: Path | None = None
    if include_tables:
        tables_structured = result.get("tables_structured", {}) or {}
        if file_path.name.endswith("_sections.md"):
            tables_filename = f"{file_path.name[:-len('_sections.md')]}_tables.json"
        else:
            tables_filename = f"{file_path.stem}_tables.json"
        tables_file_path = (FILE_OUTPUT_DIR / tables_filename).resolve()
        if not tables_file_path.is_relative_to(root_dir):
            return {"status": "error", "message": "Invalid structured tables output path"}
        if _deadline_expired(args):
            return {"status": "error", "message": "Request timed out before structured tables could be written"}
        tables_file_path.write_text(json.dumps(tables_structured, indent=2), encoding="utf-8")

    # Return summary (no full text inline) + file_path
    summary_sections = {
        key: {
            "header": s.get("header"),
            "state": s.get("state", "body"),
            "declaration_type": s.get("declaration_type"),
            "cross_reference_target": s.get("cross_reference_target"),
            "word_count": s.get("word_count", 0),
            "table_count": s.get("table_count", 0),
        }
        for key, s in sections_data.items()
    }
    aggregates = {
        "sections_found": result.get("sections_found", list(sections_data.keys())),
        "sections_absent": result.get("sections_absent", []),
        "sections_missing": result.get("sections_missing", []),
        "sections_unavailable": result.get("sections_unavailable", []),
    }
    response = {
        "status": "success",
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "filing_type": filing_type,
        "output": "file",
        "file_path": str(file_path.resolve()),
        "hint": "Use Read tool with file_path. Grep '^## SECTION:' for anchors.",
        "sections": summary_sections,
        **aggregates,
        "metadata": {
            "total_word_count": total_words,
            "total_words": total_words,
            "total_table_count": total_tables,
            "section_count": len(sections_data),
        },
    }
    if "result_status" in result:
        response["result_status"] = result["result_status"]
    if "result_message" in result:
        response["result_message"] = result["result_message"]
    if tables_file_path is not None:
        response["tables_file_path"] = str(tables_file_path)
    return response


def _proxy_get_filing_tables(args: dict) -> dict:
    """Proxy structured filing tables to the remote API."""
    source_raw = args.get("source")
    source = _normalize_sections_source(source_raw)
    if source not in (None, "8k", "proxy", "20f", "6k"):
        return {
            "status": "error",
            "message": f"Invalid source: '{source_raw}'. Supported: '8k', 'proxy', '20f', '6k', or omit for 10-K/10-Q.",
        }

    params = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
    }
    section = args.get("section")
    table_id = args.get("table_id")
    accession = args.get("accession")
    if section:
        params["section"] = section
    if table_id:
        params["table_id"] = table_id
    if accession:
        params["accession"] = accession
    if source is not None:
        params["source"] = source
    return _call_api("/api/tables", params, timeout=300)


def _proxy_get_filing_document(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    params: dict[str, Any] = {
        "source": source,
        "max_chars": args.get("max_chars", 200_000),
    }
    accession = args.get("accession")
    if accession:
        params["accession"] = accession
        for key in ("ticker", "year", "quarter", "cik", "form_type", "primary_document"):
            if args.get(key) is not None:
                params[key] = args[key]
    else:
        missing = [key for key in ("ticker", "year", "quarter") if args.get(key) is None]
        if missing:
            return {"status": "error", "message": f"{', '.join(missing)} required unless accession is provided"}
        params["ticker"] = args["ticker"]
        params["year"] = args["year"]
        params["quarter"] = args["quarter"]
    sections = args.get("sections")
    if isinstance(sections, list):
        sections_param = ",".join(str(item).strip() for item in sections if str(item).strip())
        if sections_param:
            params["sections"] = sections_param
    elif sections is not None and str(sections).strip():
        params["sections"] = str(sections)
    if args.get("char_start") is not None:
        params["char_start"] = args["char_start"]
    if args.get("char_end") is not None:
        params["char_end"] = args["char_end"]
    response = _call_api("/api/filing/document", params, timeout=120)
    return _annotate_filing_document_section_error(response)


def _annotate_filing_document_section_error(response: dict) -> dict:
    if response.get("status") != "error":
        return response
    message = str(response.get("message") or "")
    error_type = str(response.get("error_type") or "")
    section_error = error_type in {"unknown_section_name", "ambiguous_section_name"} or (
        message.startswith("Unknown section name:") or message.startswith("Ambiguous section name:")
    )
    if not section_error:
        return response

    enriched = dict(response)
    details = response.get("details")
    if isinstance(details, dict):
        available_sections = details.get("available_sections")
        matched_sections = details.get("matched_sections")
        if isinstance(available_sections, list):
            enriched.setdefault("available_sections", available_sections)
        if isinstance(matched_sections, list):
            enriched.setdefault("matched_sections", matched_sections)
    enriched.setdefault("error_type", "section_name_resolution_failed")
    enriched.setdefault("remediation_tool", "search_filing_text")
    enriched.setdefault(
        "remediation_hint",
        "Use sections with canonical keys or documented aliases. For topical requests such as debt terms, "
        "non-GAAP reconciliation, proxy voting rights, or director compensation, call search_filing_text first "
        "and then read the returned section key or char range with get_filing_document.",
    )
    return enriched


def _proxy_get_operational_kpi_drivers(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    topic = _clean_operational_text(args.get("topic") or args.get("query"))
    if not topic:
        return {"status": "error", "message": "topic is required"}

    sections = args.get("sections") or ["item_7"]
    document = _proxy_get_filing_document(
        {
            "ticker": args["ticker"],
            "year": args["year"],
            "quarter": args["quarter"],
            "source": source,
            "sections": sections,
            "max_chars": args.get("max_chars", 200_000),
        }
    )
    if document.get("status") != "success":
        return {
            "status": "error",
            "message": "Unable to retrieve filing document for operational KPI extraction",
            "document_status": document.get("status"),
            "document_message": document.get("message"),
            "remediation_hint": (
                "Warm or fetch the annual MD&A document first with get_filing_document "
                "using sections=['item_7'] and a large max_chars value, then retry "
                "get_operational_kpi_drivers."
            ),
        }

    markdown = document.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        return {
            "status": "error",
            "message": "get_filing_document returned no markdown content",
            "document_keys": sorted(document.keys()),
        }

    metrics = _discover_operational_driver_metrics(topic, markdown)
    filing_vocabulary = _operational_filing_vocabulary(markdown)
    rows = _extract_operational_driver_rows(
        markdown,
        metrics=metrics,
        year=int(args["year"]),
        quarter=int(args["quarter"]),
        filing_vocabulary=filing_vocabulary,
    )
    warnings: list[str] = []
    if not rows:
        warnings.append("No operational KPI driver rows matched the requested topic in the filing document.")
    warnings.extend(_operational_basis_conflict_warnings(rows))
    return {
        "status": "success",
        "ticker": str(args["ticker"]).upper(),
        "year": args["year"],
        "quarter": args["quarter"],
        "source": source,
        "topic": topic,
        "metrics_requested": [metric["canonical"] for metric in metrics],
        "metric_candidates": [
            {
                key: metric.get(key)
                for key in ("canonical", "match_source", "match_score", "unit")
                if key in metric
            }
            for metric in metrics
        ],
        "row_count": len(rows),
        "rows": rows,
        "filing": {
            key: document.get(key)
            for key in ("filing_type", "sections_returned", "cache_status", "citation_state")
            if key in document
        },
        "warnings": warnings,
    }


def _proxy_get_filing_cover_facts(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source != "auto":
        return {"status": "error", "message": "cover facts currently support source=auto only"}

    params: dict[str, Any] = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "source": source,
        "fact_name": args.get("fact_name", "EntityCommonStockSharesOutstanding"),
    }
    return _call_api("/api/filing/cover-facts", params, timeout=120)


def _proxy_search_filing_text(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    params: dict[str, Any] = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "source": source,
        "query": args["query"],
    }
    return _call_api("/api/filing/text/search", params, timeout=60)


def _proxy_get_filing_evidence(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    payload: dict[str, Any] = {
        "ticker": args["ticker"],
        "year": args["year"],
        "quarter": args["quarter"],
        "query": args["query"],
        "source": source,
        "max_hits": args.get("max_hits", 12),
        "include_full_sections": bool(args.get("include_full_sections", False)),
        "include_planner_trace": bool(args.get("include_planner_trace", False)),
    }
    if args.get("task_intent") is not None:
        payload["task_intent"] = args["task_intent"]
    for key in ("filing_date_from", "filing_date_to"):
        if args.get(key) is not None:
            payload[key] = args[key]
    for key in ("form_types", "related_tickers"):
        if args.get(key):
            payload[key] = args[key]
    return _post_api("/api/filing/evidence", payload, timeout=120)


def _proxy_extract_filing_file(args: dict) -> dict:
    try:
        resolved_path = validate_file_path(args["file_path"])
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    if _deadline_expired(args):
        return {"status": "error", "message": "Request timed out before the filing could be read"}

    try:
        original_text = Path(resolved_path).read_text(encoding="utf-8")
    except Exception as exc:
        return {"status": "error", "message": f"Failed to read filing file: {exc}"}

    section_headers = _list_filing_section_headers(original_text)
    requested_filters = [
        str(value).strip()
        for value in (args.get("sections_filter") or [])
        if str(value).strip()
    ]
    if requested_filters:
        if not section_headers:
            return {
                "status": "error",
                "message": "Section filtering requires a filing file with `## SECTION:` headers.",
            }
        sections_processed = _match_section_headers(section_headers, requested_filters)
        if not sections_processed:
            available = ", ".join(section_headers)
            return {
                "status": "error",
                "message": (
                    f"No filing sections matched filters {requested_filters}. "
                    f"Available sections: {available}"
                ),
            }
    else:
        sections_processed = section_headers

    ingest_result = _post_api(
        "/api/documents/ingest",
        {
            "content": original_text,
            "source_name": Path(resolved_path).name,
        },
        timeout=300,
    )
    if ingest_result.get("status") == "error":
        return ingest_result

    filing_id = str(ingest_result.get("filing_id") or "").strip()
    if not filing_id:
        return {"status": "error", "message": "Document ingest response missing filing_id"}

    schema_name = str(args["schema_name"]).strip()
    if not schema_name:
        return {"status": "error", "message": "Missing required parameter: schema_name"}

    extractions: list[dict[str, Any]] = []
    target_sections = sections_processed if requested_filters else [""]

    for section_name in target_sections:
        if _deadline_expired(args):
            return {"status": "error", "message": "Request timed out during document extraction"}
        extract_result = _post_api(
            "/api/documents/extract",
            {
                "filing_id": filing_id,
                "section": section_name,
                "schemas": [schema_name],
            },
            timeout=300,
        )
        if extract_result.get("status") == "error":
            return extract_result

        rows = (
            extract_result.get("extractions_by_schema", {}).get(schema_name, [])
            if isinstance(extract_result.get("extractions_by_schema"), dict)
            else []
        )
        if isinstance(rows, list):
            extractions.extend(row for row in rows if isinstance(row, dict))

    normalized = _normalize_extraction_rows(extractions)
    grounded_count = sum(1 for item in normalized if item.get("grounded", True))
    return {
        "status": "ok",
        "filing_id": filing_id,
        "schema_used": schema_name,
        "file_path": resolved_path,
        "sections_processed": sections_processed,
        "grounded_count": grounded_count,
        "total_count": len(normalized),
        "extractions": normalized,
    }


def _proxy_get_filing_extractions(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}
    return _post_api(
        "/api/extractions",
        {
            "ticker": args["ticker"],
            "year": args["year"],
            "quarter": args["quarter"],
            "schema": args["schema"],
            "source": source,
            "allow_stale": bool(args.get("allow_stale", False)),
        },
        timeout=600,
    )


def _proxy_search_extractions(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    params: dict[str, Any] = {
        "ticker": args["ticker"],
        "schema": args["schema"],
        "period_from": args["period_from"],
        "period_to": args["period_to"],
        "source": source,
        "allow_stale": str(bool(args.get("allow_stale", False))).lower(),
        "include_candidates": str(bool(args.get("include_candidates", True))).lower(),
        "limit": args.get("limit", 500),
    }
    class_value = args.get("class_")
    if isinstance(class_value, list):
        params["class"] = ",".join(str(item).strip() for item in class_value if str(item).strip())
    elif class_value is not None:
        params["class"] = str(class_value)
    if args.get("form_type") is not None:
        params["form_type"] = args["form_type"]

    attributes = args.get("attributes")
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            param_key = f"attr.{key}"
            if isinstance(value, list):
                params[param_key] = [str(item) for item in value]
            elif value is not None:
                params[param_key] = str(value)

    return _call_api("/api/extractions/search", params, timeout=60)


def _proxy_get_extraction_series(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    params: dict[str, Any] = {
        "ticker": args["ticker"],
        "schema": args["schema"],
        "period_from": args["period_from"],
        "period_to": args["period_to"],
        "source": source,
        "allow_stale": str(bool(args.get("allow_stale", False))).lower(),
        "include_candidates": str(bool(args.get("include_candidates", True))).lower(),
        "include_hits": str(bool(args.get("include_hits", False))).lower(),
        "limit_hits_per_period": args.get("limit_hits_per_period", 50),
    }
    class_value = args.get("class_")
    if isinstance(class_value, list):
        params["class"] = ",".join(str(item).strip() for item in class_value if str(item).strip())
    elif class_value is not None:
        params["class"] = str(class_value)
    if args.get("form_type") is not None:
        params["form_type"] = args["form_type"]

    attributes = args.get("attributes")
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            param_key = f"attr.{key}"
            if isinstance(value, list):
                params[param_key] = [str(item) for item in value]
            elif value is not None:
                params[param_key] = str(value)

    return _call_api("/api/extractions/series", params, timeout=60)


def _proxy_search_filing_tables(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    params: dict[str, Any] = {
        "ticker": args["ticker"],
        "source": source,
        "limit": args.get("limit", 200),
    }
    for key in ("description", "table_type", "period_from", "period_to", "form_type", "section_key"):
        if args.get(key) is not None:
            params[key] = args[key]
    return _call_api("/api/tables/search", params, timeout=60)


def _proxy_compare_filing_tables(args: dict) -> dict:
    source = _normalize_agent_source_arg(args.get("source"))
    if source not in _AGENT_SOURCES:
        return {"status": "error", "message": _AGENT_SOURCE_ERROR}

    tickers = args.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        return {"status": "error", "message": "tickers must be a non-empty list"}

    payload: dict[str, Any] = {
        "tickers": tickers,
        "source": source,
        "include_full_tables": bool(args.get("include_full_tables", True)),
        "limit_per_ticker": args.get("limit_per_ticker", 3),
    }
    for key in ("description", "table_type", "period_from", "period_to", "form_type", "section_key"):
        if args.get(key) is not None:
            payload[key] = args[key]
    return _post_api("/api/tables/compare", payload, timeout=120)


def _proxy_list_extraction_schemas(args: dict) -> dict:
    del args
    return _call_api("/api/documents/schemas", {}, timeout=300)


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

def _json_text(payload: dict) -> str:
    try:
        return json.dumps(payload, indent=2, default=str)
    except Exception as exc:
        fallback = {
            "status": "error",
            "message": "Failed to serialize MCP tool response",
            "details": str(exc),
        }
        return json.dumps(fallback, indent=2)


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

_TOOL_DISPATCH = {
    "get_filings": _proxy_get_filings,
    "get_event_filings": _proxy_get_event_filings,
    "describe_filing": _proxy_describe_filing,
    "get_financials": _proxy_get_financials,
    "get_metric": _proxy_get_metric,
    "get_concept": _proxy_get_concept,
    "cite_concept": _proxy_cite_concept,
    "compare_concept": _proxy_compare_concept,
    "concept_trend": _proxy_concept_trend,
    "get_statement": _proxy_get_statement,
    "get_metric_series": _proxy_get_metric_series,
    "warm_metric_cache": _proxy_warm_metric_cache,
    "warm_metric_cache_status": _proxy_warm_metric_cache_status,
    "list_metrics": _proxy_list_metrics,
    "search_metrics": _proxy_search_metrics,
    "get_filing_sections": _proxy_get_filing_sections,
    "get_filing_document": _proxy_get_filing_document,
    "get_filing_cover_facts": _proxy_get_filing_cover_facts,
    "get_operational_kpi_drivers": _proxy_get_operational_kpi_drivers,
    "search_filing_text": _proxy_search_filing_text,
    "get_filing_evidence": _proxy_get_filing_evidence,
    "get_filing_tables": _proxy_get_filing_tables,
    "get_filing_extractions": _proxy_get_filing_extractions,
    "search_extractions": _proxy_search_extractions,
    "get_extraction_series": _proxy_get_extraction_series,
    "search_filing_tables": _proxy_search_filing_tables,
    "compare_filing_tables": _proxy_compare_filing_tables,
    "extract_filing_file": _proxy_extract_filing_file,
    "list_extraction_schemas": _proxy_list_extraction_schemas,
}

_TOOL_TIMEOUT = {
    "get_filings": 30,
    "get_event_filings": 60,
    "describe_filing": 30,
    "get_financials": 300,
    "get_metric": 300,
    "get_concept": 300,
    "cite_concept": 600,
    "compare_concept": 300,
    "concept_trend": 600,
    "get_statement": 600,
    "get_metric_series": 600,
    "warm_metric_cache": 300,
    "warm_metric_cache_status": 60,
    "list_metrics": 300,
    "search_metrics": 300,
    "get_filing_sections": 300,
    "get_filing_document": 120,
    "get_filing_cover_facts": 120,
    "get_operational_kpi_drivers": 120,
    "search_filing_text": 60,
    "get_filing_evidence": 120,
    "get_filing_tables": 300,
    "get_filing_extractions": 600,
    "search_extractions": 60,
    "get_extraction_series": 60,
    "search_filing_tables": 60,
    "compare_filing_tables": 120,
    "extract_filing_file": 300,
    "list_extraction_schemas": 300,
}


async def _run_tool_guarded(name: str, arguments: dict | None = None) -> dict:
    handler = _TOOL_DISPATCH.get(name)
    if not handler:
        return {"status": "error", "message": f"Unknown tool: {name}"}

    timeout = _TOOL_TIMEOUT.get(name, 60)
    call_args = dict(arguments or {})
    call_args["__deadline_monotonic"] = time.monotonic() + timeout
    try:
        with redirect_stdout(sys.stderr):
            result = await asyncio.wait_for(
                asyncio.to_thread(handler, call_args),
                timeout=timeout,
            )
    except asyncio.TimeoutError:
        result = {"status": "error", "message": f"Tool '{name}' timed out after {timeout}s"}
    except Exception as exc:
        result = {"status": "error", "message": f"Unhandled error in MCP tool '{name}': {exc}"}

    try:
        normalized = json.loads(_json_text(result))
    except Exception as exc:
        return {
            "status": "error",
            "message": "Failed to deserialize MCP tool response",
            "details": str(exc),
        }

    if isinstance(normalized, dict):
        return normalized
    return {"status": "success", "result": normalized}


@mcp.tool()
async def get_filings(
    ticker: str,
    year: int,
    quarter: int,
    source: _FILING_LIST_SOURCES = "auto",
) -> dict:
    """
    Fetch SEC filing metadata for a company. Defaults to 10-Q, 10-K, and 8-K
    earnings release filings. Use source='20f' or source='6k' to list foreign
    issuer filings and discover accessions for targeted table retrieval.

    Do not use source='proxy' here: get_filings does not list DEF 14A proxy
    filings. For proxy discovery, call get_event_filings with
    form_types=['DEF 14A'] and a filing-date range. For proxy content, call
    get_filing_document, get_filing_sections, or get_filing_tables with
    source='proxy' and quarter=4 because DEF 14A is annual.
    """
    return await _run_tool_guarded(
        "get_filings",
        {"ticker": ticker, "year": year, "quarter": quarter, "source": source},
    )


@mcp.tool()
async def get_event_filings(
    ticker: str | None = None,
    cik: str | None = None,
    filing_date_from: str | None = None,
    filing_date_to: str | None = None,
    form_types: list[str] | None = None,
    query: str | None = None,
    related_tickers: list[str] | None = None,
    limit: int = 50,
    sort_order: Literal["asc", "desc"] = "desc",
) -> dict:
    """
    Discover SEC event filings by ticker/CIK, filing-date range, and form type.
    Use this for deal, financing, proxy, filed-communication, and status-update
    materials that are not tied to a fiscal quarter's 10-K/10-Q. Do not pass
    10-K or 10-Q here; use get_filings for periodic filing discovery, or
    get_filing_document/get_filing_sections when periodic filing content is
    needed. Event-form examples include 8-K, 425, proxy forms, S-3/S-3ASR,
    S-4, FWP, 424B*, 8-A12B, CERT, and SC 13D. Returns accession, form, filing
    date, primary document URL, and deterministic event_type labels; it does
    not perform broad corpus search.
    """
    return await _run_tool_guarded(
        "get_event_filings",
        {
            "ticker": ticker,
            "cik": cik,
            "filing_date_from": filing_date_from,
            "filing_date_to": filing_date_to,
            "form_types": form_types,
            "query": query,
            "related_tickers": related_tickers,
            "limit": limit,
            "sort_order": sort_order,
        },
    )


@mcp.tool()
async def get_financials(
    ticker: str,
    year: int,
    quarter: int,
    full_year_mode: bool = False,
    source: Literal["auto", "8k"] = "auto",
    output: Literal["inline", "file"] = "file",
) -> dict:
    """
    Extract all financial facts from SEC filings. Returns structured JSON with income
    statement, balance sheet, and cash flow data. If returned, `scope_warnings`
    flag mixed-scope cash-flow facts and `scope_bridges` provide evidence-backed
    adjustment candidates; for FCF margin/trend work, show reported values and use
    bridge candidates only as normalized/same-scope views.
    """
    return await _run_tool_guarded(
        "get_financials",
        {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "full_year_mode": full_year_mode,
            "source": source,
            "output": output,
        },
    )


@mcp.tool()
async def describe_filing(
    ticker: str,
    year: int,
    quarter: int,
    source: _FILING_SOURCES = "auto",
) -> dict:
    """
    Show cached data availability for one filing across XBRL, sections,
    tables, extraction schemas, and markdown. Read-only; does not populate caches.
    """
    return await _run_tool_guarded(
        "describe_filing",
        {"ticker": ticker, "year": year, "quarter": quarter, "source": source},
    )


@mcp.tool()
async def get_metric(
    ticker: str,
    year: int,
    quarter: int,
    metric_name: str,
    full_year_mode: bool = False,
    source: Literal["auto", "8k"] = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
    role: list[str] | None = None,
) -> dict:
    """
    Get a specific financial metric by common name or XBRL tag and return
    current/prior values with YoY comparison. For guessed or user-facing names,
    call search_metrics or list_metrics first and pass an exact discovered
    metric_name into this tool; do not retry near-miss names blindly. Optional
    role filters accept friendly statement roles such as ["cash_flow"] or
    ["balance_sheet"]. If the response says the cache is cold, call
    warm_metric_cache with warm_hint.body.items, poll warm_metric_cache_status,
    then retry get_metric. Cash-flow metrics may include `scope_warnings` and
    `scope_bridges`; treat bridges as adjustment candidates for same-scope
    analysis, not replacements for reported facts. Filing-native operational KPI
    misses include a remediation hint for get_operational_kpi_drivers when the
    name looks like a non-XBRL operating metric.

    Discovery: call list_metrics or search_metrics for the same ticker, year,
    quarter, source, and date_type before choosing metric_name. Use get_metric
    for one period and get_metric_series when the same metric is needed across
    periods.
    """
    args = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "metric_name": metric_name,
        "full_year_mode": full_year_mode,
        "source": source,
    }
    if date_type is not None:
        args["date_type"] = date_type
    if role:
        args["role"] = role
    return await _run_tool_guarded("get_metric", args)


@mcp.tool()
async def get_concept(
    ticker: str,
    year: int,
    quarter: int,
    concept_name: str,
    full_year_mode: bool = False,
    source: _CONCEPT_SOURCE_LITERAL = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
) -> dict:
    """
    Get a registry-backed financial concept value from cached XBRL financials.
    v1 resolves tag-backed concepts only and returns available=false for
    derivation or unsupported concepts. It never warms caches; call
    get_financials first when cache_status is cold.

    Discovery: choose concept_name from the v1 concept registry
    (data/concept_registry_v1.json) or from statement rows returned by
    get_statement. Pass the canonical concept name exactly.
    """
    args = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "concept_name": concept_name,
        "full_year_mode": full_year_mode,
        "source": source,
    }
    if date_type is not None:
        args["date_type"] = date_type
    return await _run_tool_guarded("get_concept", args)


@mcp.tool()
async def cite_concept(
    ticker: str,
    year: int,
    quarter: int,
    concept_name: str,
    full_year_mode: bool = False,
    source: _CONCEPT_SOURCE_LITERAL = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
    narrative_sources: list[str] | None = None,
    include_extractions: bool = False,
    max_narrative_spans_per_source: int = 5,
    allow_stale_extractions: bool = False,
) -> dict:
    """
    Cite a registry-backed concept by returning the canonical concept value and
    the filing prose around that value from MD&A, notes, risk factors, and
    optional cached extraction spans.

    Use `cite_concept` for metric-anchored narrative joins (concept value first → find the prose around it). For question-shaped narrative retrieval (no anchoring concept), use `get_filing_evidence` instead. The two tools are sharply distinguished: `cite_concept` ALWAYS resolves a canonical concept value and returns the narrative around it; `get_filing_evidence` runs a source-pack planner and surfaces evidence for an arbitrary query.

    Discovery: choose concept_name from the v1 concept registry
    (data/concept_registry_v1.json) or from get_statement output, then pass the
    canonical name exactly.
    """
    args: dict[str, Any] = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "concept_name": concept_name,
        "full_year_mode": full_year_mode,
        "source": source,
        "include_extractions": include_extractions,
        "max_narrative_spans_per_source": max_narrative_spans_per_source,
        "allow_stale_extractions": allow_stale_extractions,
    }
    if date_type is not None:
        args["date_type"] = date_type
    if narrative_sources is not None:
        args["narrative_sources"] = narrative_sources
    return await _run_tool_guarded("cite_concept", args)


@mcp.tool()
async def compare_concept(
    concept_name: str,
    tickers: list[str],
    year: int,
    quarter: int,
    full_year_mode: bool = False,
    source: _CONCEPT_SOURCE_LITERAL = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
    ticker_periods: dict[str, Any] | None = None,
) -> dict:
    """
    Compare one registry-backed concept across a caller-provided list of filers.
    This is cross-filer comparison, not search: caller ticker order is preserved,
    and partial coverage is returned per row with available=false.

    Discovery: choose concept_name from the v1 concept registry
    (data/concept_registry_v1.json) or from a prior get_statement/get_concept
    call, then reuse that exact canonical name for every ticker.
    """
    args = {
        "concept_name": concept_name,
        "tickers": tickers,
        "year": year,
        "quarter": quarter,
        "full_year_mode": full_year_mode,
        "source": source,
    }
    if date_type is not None:
        args["date_type"] = date_type
    if ticker_periods is not None:
        args["ticker_periods"] = ticker_periods
    return await _run_tool_guarded("compare_concept", args)


@mcp.tool()
async def concept_trend(
    ticker: str,
    concept_name: str,
    period_from: ConceptPeriod,
    period_to: ConceptPeriod,
    source: _CONCEPT_SOURCE_LITERAL = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
) -> dict:
    """
    Return a cache-only time series for a registry-backed concept over a period
    range. v1 uses the latest filing's reported series and has no restatement
    awareness: it does not return restated or original_value fields.

    Discovery: choose concept_name from the v1 concept registry
    (data/concept_registry_v1.json), get_statement output, or a successful
    get_concept call before requesting the trend.
    """
    args = {
        "ticker": ticker,
        "concept_name": concept_name,
        "period_from": period_from,
        "period_to": period_to,
        "source": source,
    }
    if date_type is not None:
        args["date_type"] = date_type
    return await _run_tool_guarded("concept_trend", args)


@mcp.tool()
async def get_statement(
    ticker: str,
    statement: _STATEMENT_TYPE_LITERAL,
    year: int | None = None,
    quarter: int | None = None,
    full_year_mode: bool = False,
    period_from: ConceptPeriod | None = None,
    period_to: ConceptPeriod | None = None,
    source: _CONCEPT_SOURCE_LITERAL = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
) -> dict:
    """
    Return a bulk structured statement (income statement, balance sheet, or cash flow statement) from a deterministic v1 statement template and the v1 concept registry. Supports single-period mode with year+quarter and range mode with period_from+period_to; the two modes are mutually exclusive. v1 templates contain tag-kind concepts only, so computed or derivation rows such as margins and ratios belong in separate concept_trend calls.

    Note for model_build / AI-excel-addin populator consumers: this tool is NOT pre-approved as a model_build input. Production model_build flows must continue to use `/api/financials` + `/api/metric/series` until model_build explicitly accepts this tool's contract, provenance shape, error semantics, and `concept_registry_version` pinning. `get_statement` is additive and opt-in for ad-hoc use, FinanceBench harness, and standalone Edgar product.
    """
    args: dict[str, Any] = {
        "ticker": ticker,
        "statement": statement,
        "source": source,
    }
    if year is not None:
        args["year"] = year
    if quarter is not None:
        args["quarter"] = quarter
    if full_year_mode:
        args["full_year_mode"] = full_year_mode
    if period_from is not None:
        args["period_from"] = period_from
    if period_to is not None:
        args["period_to"] = period_to
    if date_type is not None:
        args["date_type"] = date_type
    return await _run_tool_guarded("get_statement", args)


@mcp.tool()
async def get_metric_series(
    ticker: str,
    metric_name: str,
    end_year: int,
    end_quarter: int,
    periods: int = 8,
    full_year_mode: bool = False,
    source: Literal["auto", "8k"] = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
    cached_only: bool = False,
    role: list[str] | None = None,
    axis_key: str | None = None,
) -> dict:
    """
    Fetch one metric across multiple periods in a single call, using cache-first
    execution and per-period status reporting. Optional role filters are applied
    per period before the best match is selected. Pass `axis_key` exactly as
    returned by search_metrics/list_metrics when the user needs a dimensional
    fact such as product revenue. For FCF, FCF margin, cash conversion, or trend
    questions, inspect per-period `scope_warnings` and `scope_bridges`; if
    bridges exist, present raw and same-scope trends.

    `metric_name` must be an exact discovered metric name, not a natural-language
    description. If the metric label came from the user, a filing heading, or your
    own shorthand, call `search_metrics` or `list_metrics` for the ticker/period
    first and pass the returned `metric_name`/`metric_id` exactly. Do not guess
    nearby names such as `Revenues`, `capital expenditures`, `cash flow from
    operations`, or `GrossBookingValue`; use discovered values such as `Revenue`,
    `NetCashProvidedByUsedInOperatingActivities`, or
    `PaymentsToAcquirePropertyPlantAndEquipment` when those are the returned
    matches. Required parameters are `metric_name`, `end_year`, and `end_quarter`;
    do not use `metric`, `years`, or `period`.

    If the result says "Metric series has no values," treat it as a metric-name
    miss and retry after discovery unless the message explicitly says uncached. If
    the message says uncached, call `warm_metric_cache` with `warm_hint.body.items`,
    poll `warm_metric_cache_status`, then retry this tool.

    Discovery: call list_metrics or search_metrics for the same ticker, period,
    source, and date_type to choose metric_name. Use get_metric for a single
    period and get_metric_series for multi-period retrieval.
    """
    args = {
        "ticker": ticker,
        "metric_name": metric_name,
        "end_year": end_year,
        "end_quarter": end_quarter,
        "periods": periods,
        "full_year_mode": full_year_mode,
        "source": source,
        "cached_only": cached_only,
    }
    if date_type is not None:
        args["date_type"] = date_type
    if role:
        args["role"] = role
    if axis_key:
        args["axis_key"] = axis_key
    return await _run_tool_guarded("get_metric_series", args)


@mcp.tool()
async def warm_metric_cache(
    items: list[dict[str, Any]],
) -> dict:
    """
    Enqueue async cache warming for metric periods. Paid/internal API tiers only.
    Each item requires ticker, year, quarter, and optional full_year_mode.
    Returns a job_id for warm_metric_cache_status polling.
    """
    return await _run_tool_guarded("warm_metric_cache", {"items": items})


@mcp.tool()
async def warm_metric_cache_status(
    job_id: str,
) -> dict:
    """
    Poll an async metric cache warming job by job_id.

    Discovery: job_id is returned by warm_metric_cache; do not invent one.
    """
    return await _run_tool_guarded("warm_metric_cache_status", {"job_id": job_id})


@mcp.tool()
async def list_metrics(
    ticker: str,
    year: int,
    quarter: int,
    full_year_mode: bool = False,
    source: Literal["auto", "8k"] = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
    limit: int = 200,
    include_values: bool = True,
) -> dict:
    """
    List available metric tags for a filing period so an agent can choose
    an exact metric_name before calling get_metric. Cash-flow candidates may carry
    `scope_warning` and `scope_bridge_ids`; validate those before using CFO in
    margin or trend formulas. Debt candidates may carry `debt_component_kind`
    values such as `instrument_principal`, `coupon_rate`, `finance_lease`,
    `other_notes`, or `total_debt_rollup`; consumers own any source-basis sum.
    """
    args = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "full_year_mode": full_year_mode,
        "source": source,
        "limit": limit,
        "include_values": include_values,
    }
    if date_type is not None:
        args["date_type"] = date_type
    return await _run_tool_guarded("list_metrics", args)


@mcp.tool()
async def search_metrics(
    ticker: str,
    year: int,
    quarter: int,
    query: str,
    full_year_mode: bool = False,
    source: Literal["auto", "8k"] = "auto",
    date_type: Literal["Q", "YTD", "FY"] | None = None,
    role: list[str] | None = None,
    limit: int = 20,
    include_values: bool = True,
) -> dict:
    """
    Search available filing metrics by natural-language query and return
    ranked candidates. Optional role filters accept friendly statement roles
    such as ["cash_flow"], ["balance_sheet"], or ["income_statement"]. Cash-flow
    matches may carry mixed-scope warnings and bridge IDs; inspect the returned
    top-level `scope_bridges` before using CFO in FCF-style formulas. Debt
    refinance/tranche queries surface structured XBRL components with
    `debt_component_kind`; this tool does not pre-sum refinanceable debt.
    """
    args = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "query": query,
        "full_year_mode": full_year_mode,
        "source": source,
        "limit": limit,
        "include_values": include_values,
    }
    if date_type is not None:
        args["date_type"] = date_type
    if role:
        args["role"] = role
    return await _run_tool_guarded("search_metrics", args)


@mcp.tool()
async def get_filing_sections(
    ticker: str,
    year: int,
    quarter: int,
    sections: list[str] | str | None = None,
    source: Literal["8k", "proxy", "20f", "6k"] | None = None,
    format: Literal["summary", "full"] = "summary",
    max_words: int | None = 3000,
    include_tables: bool = False,
    tables_only: bool = False,
    fallback: bool = False,
    output: Literal["inline", "file"] = "file",
) -> dict:
    """
    Parse qualitative sections from SEC filings and return narrative/tables
    with metadata. Section filters accept canonical keys like "item_1" and
    "part1_item2", or filing headings like "Item 1. Business". Use source='8k'
    for 8-K earnings release sections or source='proxy' for DEF 14A proxy
    statements. Use source='20f' or source='6k' for foreign private issuer
    filings exposed as whole-document sections. get_filings does not list proxy
    filings; use this routed access path directly. Do not use as the first step
    for broad qualitative questions; prefer corpus search
    (research-corpus-mcp.filings_search) when available, or search_filing_text
    once it lands. Pass explicit sections only after evidence discovery has
    identified them. Related tools: get_filing_document for readable markdown,
    get_filing_evidence for question-shaped evidence, and get_filing_cover_facts
    for exact DEI cover-page facts.
    """
    args = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "format": format,
        "max_words": max_words,
        "include_tables": include_tables,
        "tables_only": tables_only,
        "fallback": fallback,
        "output": output,
    }
    if sections:
        args["sections"] = sections
    if source is not None:
        args["source"] = source
    return await _run_tool_guarded("get_filing_sections", args)


@mcp.tool()
async def get_filing_tables(
    ticker: str,
    year: int,
    quarter: int,
    section: str | None = None,
    table_id: str | None = None,
    source: Literal["8k", "proxy", "20f", "6k"] | None = None,
    accession: str | None = None,
) -> dict:
    """
    Fetch structured filing tables. Listing modes return metadata only; supplying
    table_id returns a single table with full row data. Use source='proxy' for
    DEF 14A proxy tables; use source='20f' or source='6k' with accession to
    target a specific foreign issuer filing from get_filings. Related tools:
    get_filing_document reads markdown, get_filing_evidence finds prose
    evidence, and get_filing_cover_facts returns exact cover-page DEI facts.
    """
    args = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
    }
    if section is not None:
        args["section"] = section
    if table_id is not None:
        args["table_id"] = table_id
    if source is not None:
        args["source"] = source
    if accession is not None:
        args["accession"] = accession
    return await _run_tool_guarded("get_filing_tables", args)


@mcp.tool()
async def get_filing_document(
    ticker: str | None = None,
    year: int | None = None,
    quarter: int | None = None,
    source: _FILING_SOURCES = "auto",
    accession: str | None = None,
    cik: str | None = None,
    form_type: str | None = None,
    primary_document: str | None = None,
    sections: list[str] | str | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    max_chars: int = 200_000,
) -> dict:
    """
    Get a SEC filing as readable markdown. Returns sectioned markdown with
    ## SECTION: headers. Supports section filtering and pagination via char
    range or max_chars cap. Pass accession plus cik/form_type/primary_document
    from get_event_filings to read an exact event filing without falling back
    to the latest filing for a fiscal quarter. Period-keyed reads are cache
    backed; exact-accession reads are fetched live.

    sections accepts canonical keys such as item_1a, item_7, item_7a,
    item_8, part1_item1, part1_item2, earnings_release, and
    proxy_statement; Item notation such as "Item 7" is also accepted. Common
    aliases include MD&A, Risk Factors, Financial Statements, Notes, Debt,
    Leases, Non-GAAP reconciliation / non_gaap_reconciliation, Director
    Compensation, Voting Rights, and Proposal 1. For narrow topical phrases
    inside a filing section, call search_filing_text first and then read the
    returned section key or char range with get_filing_document. Related tools:
    get_filing_evidence plans evidence retrieval, get_filing_cover_facts reads
    exact DEI facts, and get_filing_extractions returns cached structured spans.
    """
    args: dict[str, Any] = {
        "source": source,
        "max_chars": max_chars,
    }
    if ticker is not None:
        args["ticker"] = ticker
    if year is not None:
        args["year"] = year
    if quarter is not None:
        args["quarter"] = quarter
    if accession is not None:
        args["accession"] = accession
    if cik is not None:
        args["cik"] = cik
    if form_type is not None:
        args["form_type"] = form_type
    if primary_document is not None:
        args["primary_document"] = primary_document
    if sections is not None:
        args["sections"] = sections
    if char_start is not None:
        args["char_start"] = char_start
    if char_end is not None:
        args["char_end"] = char_end
    return await _run_tool_guarded("get_filing_document", args)


@mcp.tool()
async def get_operational_kpi_drivers(
    ticker: str,
    year: int,
    quarter: int,
    topic: str,
    source: _FILING_SOURCES = "auto",
    sections: list[str] | str | None = None,
    max_chars: int = 200_000,
) -> dict:
    """
    Return structured operational KPI values and driver growth rates from a
    filing's MD&A/earnings discussion. Use this for non-XBRL operating metrics
    such as Gross Bookings, MAPCs, Trips, customer counts, revenue driver
    bridges, take-rate inputs, segment growth rates, constant-currency growth,
    and volume-vs-price decomposition. It discovers candidate labels from the
    filing document and returns citation-ready rows with snippets; it does not
    run the slower generic KPI catalog extractor.
    """
    args: dict[str, Any] = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "topic": topic,
        "source": source,
        "max_chars": max_chars,
    }
    if sections is not None:
        args["sections"] = sections
    return await _run_tool_guarded("get_operational_kpi_drivers", args)


@mcp.tool()
async def get_filing_cover_facts(
    ticker: str,
    year: int,
    quarter: int,
    fact_name: str = "EntityCommonStockSharesOutstanding",
) -> dict:
    """
    Get exact cover-page DEI facts from a 10-K/10-Q, including citation-ready
    source metadata. Use this for outstanding-share questions instead of
    balance-sheet common-stock rows or weighted-average share metrics. Related
    tools: get_filing_document for markdown, get_filing_evidence for qualitative
    evidence, and get_filing_extractions for cached structured spans.
    """
    return await _run_tool_guarded(
        "get_filing_cover_facts",
        {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "source": "auto",
            "fact_name": fact_name,
        },
    )


@mcp.tool()
async def search_filing_text(
    ticker: str,
    year: int,
    quarter: int,
    query: str,
    source: _FILING_SOURCES = "auto",
) -> dict:
    """
    Search cached markdown within one SEC filing keyed by ticker, year, quarter,
    and source. Read-only: cold caches return cache_status='cold' and are not
    warmed. CC5 prefix exception: although search_* usually denotes
    cross-filing search, search_filing_text is intentionally same-filing only
    and requires the per-filing ticker/year/quarter inputs. Use
    search_filing_tables for cross-filing table metadata search.
    """
    return await _run_tool_guarded(
        "search_filing_text",
        {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "source": source,
            "query": query,
        },
    )


@mcp.tool()
async def get_filing_evidence(
    ticker: str,
    year: int,
    quarter: int,
    query: str,
    task_intent: str | None = None,
    source: _FILING_SOURCES = "auto",
    max_hits: int = 12,
    include_full_sections: bool = False,
    include_planner_trace: bool = False,
    filing_date_from: str | None = None,
    filing_date_to: str | None = None,
    form_types: list[str] | None = None,
    related_tickers: list[str] | None = None,
) -> dict:
    """
    Plan and retrieve filing evidence in one call for a qualitative SEC filing
    question. The source planner is internal; inspect its output by setting
    include_planner_trace=true, or by reading the X-Edgar-Planner-Trace
    response header when using the HTTP route directly. For metric-anchored
    joins from a concept value to supporting prose, use cite_concept instead
    once that Phase 6 tool is shipped. Supported task_intent values include
    regulatory_risk, concentration_risk, acquisition_strategy,
    revenue_disaggregation, debt_terms, deal_status, security_offering_terms,
    and guidance_actuals; deal_terms is accepted as a deal_status alias for
    merger/acquisition-status questions. For event backed intents, pass
    filing_date_from/to and form_types when known. Related tools:
    get_filing_document for source markdown, get_filing_cover_facts for exact
    cover facts, and get_filing_extractions for cached structured spans.
    """
    return await _run_tool_guarded(
        "get_filing_evidence",
        {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "query": query,
            "task_intent": task_intent,
            "source": source,
            "max_hits": max_hits,
            "include_full_sections": include_full_sections,
            "include_planner_trace": include_planner_trace,
            "filing_date_from": filing_date_from,
            "filing_date_to": filing_date_to,
            "form_types": form_types or [],
            "related_tickers": related_tickers or [],
        },
    )


@mcp.tool()
async def get_filing_extractions(
    ticker: str,
    year: int,
    quarter: int,
    schema: str,
    source: _FILING_SOURCES = "auto",
    allow_stale: bool = False,
) -> dict:
    """
    Return cached langextract spans for one filing, or fetch the filing and run
    extraction on cache miss. Use this for ticker/year/quarter-keyed filings.
    Related tools: get_filing_document reads markdown, get_filing_evidence
    plans qualitative evidence, and get_filing_cover_facts returns DEI facts.
    """
    return await _run_tool_guarded(
        "get_filing_extractions",
        {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "schema": schema,
            "source": source,
            "allow_stale": allow_stale,
        },
    )


@mcp.tool()
async def search_extractions(
    ticker: str,
    schema: str,
    period_from: ExtractionPeriod,
    period_to: ExtractionPeriod,
    class_: list[str] | str | None = None,
    form_type: _FILING_FORM_TYPES | None = None,
    source: _FILING_SOURCES = "auto",
    attributes: dict[str, str | list[str]] | None = None,
    allow_stale: bool = False,
    include_candidates: bool = True,
    limit: int = 500,
) -> dict:
    """
    Search cached langextract spans across filings using structured filters.
    Periods must use YYYY-Qn or YYYY-FY, not ISO dates. Read-only; cache
    misses are reported and never trigger LLM extraction.
    """
    args = {
        "ticker": ticker,
        "schema": schema,
        "period_from": period_from,
        "period_to": period_to,
        "source": source,
        "allow_stale": allow_stale,
        "include_candidates": include_candidates,
        "limit": limit,
    }
    if class_ is not None:
        args["class_"] = class_
    if form_type is not None:
        args["form_type"] = form_type
    if attributes is not None:
        args["attributes"] = attributes
    return await _run_tool_guarded("search_extractions", args)


@mcp.tool()
async def get_extraction_series(
    ticker: str,
    schema: str,
    period_from: ExtractionPeriod,
    period_to: ExtractionPeriod,
    class_: list[str] | str | None = None,
    form_type: _FILING_FORM_TYPES | None = None,
    source: _FILING_SOURCES = "auto",
    attributes: dict[str, str | list[str]] | None = None,
    allow_stale: bool = False,
    include_candidates: bool = True,
    include_hits: bool = False,
    limit_hits_per_period: int = 50,
) -> dict:
    """
    Periodized counts (and optional hits) for langextract spans across a
    YYYY-Qn/YYYY-FY period range. ISO dates are not accepted. Cache read only.
    For raw cross-filing hits, use search_extractions.
    """
    args = {
        "ticker": ticker,
        "schema": schema,
        "period_from": period_from,
        "period_to": period_to,
        "source": source,
        "allow_stale": allow_stale,
        "include_candidates": include_candidates,
        "include_hits": include_hits,
        "limit_hits_per_period": limit_hits_per_period,
    }
    if class_ is not None:
        args["class_"] = class_
    if form_type is not None:
        args["form_type"] = form_type
    if attributes is not None:
        args["attributes"] = attributes
    return await _run_tool_guarded("get_extraction_series", args)


@mcp.tool()
async def search_filing_tables(
    ticker: str,
    description: str | None = None,
    table_type: str | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
    form_type: _FILING_FORM_TYPES | None = None,
    source: _FILING_SOURCES = "auto",
    section_key: str | None = None,
    limit: int = 200,
) -> dict:
    """
    Cross-filing search over an Edgar table index for a ticker. Matches on
    table description, table_type, and section. Returns metadata only; use
    get_filing_tables with the table_id to fetch full table contents.
    Use search_filing_text for same-filing prose search.
    """
    args: dict[str, Any] = {
        "ticker": ticker,
        "source": source,
        "limit": limit,
    }
    if description is not None:
        args["description"] = description
    if table_type is not None:
        args["table_type"] = table_type
    if period_from is not None:
        args["period_from"] = period_from
    if period_to is not None:
        args["period_to"] = period_to
    if form_type is not None:
        args["form_type"] = form_type
    if section_key is not None:
        args["section_key"] = section_key
    return await _run_tool_guarded("search_filing_tables", args)


@mcp.tool()
async def compare_filing_tables(
    tickers: list[str],
    description: str | None = None,
    table_type: str | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
    form_type: _FILING_FORM_TYPES | None = None,
    source: _FILING_SOURCES = "auto",
    section_key: str | None = None,
    include_full_tables: bool = True,
    limit_per_ticker: int = 3,
) -> dict:
    """
    Compare structured filing tables across caller-provided tickers in one
    response. Preserves ticker order and returns per-ticker availability,
    matching table metadata, and optionally hydrated table rows.
    """
    args: dict[str, Any] = {
        "tickers": tickers,
        "source": source,
        "include_full_tables": include_full_tables,
        "limit_per_ticker": limit_per_ticker,
    }
    if description is not None:
        args["description"] = description
    if table_type is not None:
        args["table_type"] = table_type
    if period_from is not None:
        args["period_from"] = period_from
    if period_to is not None:
        args["period_to"] = period_to
    if form_type is not None:
        args["form_type"] = form_type
    if section_key is not None:
        args["section_key"] = section_key
    return await _run_tool_guarded("compare_filing_tables", args)


@mcp.tool()
async def extract_filing_file(
    file_path: str,
    schema_name: str,
    sections_filter: list[str] | None = None,
) -> dict:
    """
    Validate a local filing markdown path, ingest its content into the document API,
    then run extraction for the requested schema.

    Discovery: choose schema_name from list_extraction_schemas. file_path should
    be an existing local markdown filing path produced by get_filing_document
    output=file or another trusted local export; paths outside the allowed
    output area are rejected.
    """
    args = {
        "file_path": file_path,
        "schema_name": schema_name,
    }
    if sections_filter:
        args["sections_filter"] = sections_filter
    return await _run_tool_guarded("extract_filing_file", args)


@mcp.tool()
async def list_extraction_schemas() -> dict:
    """List the document extraction schemas available through the EDGAR API."""
    return await _run_tool_guarded("list_extraction_schemas", {})


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _version_text() -> str:
    if not __package__:
        return "edgar-mcp 0+local"
    try:
        from . import __version__ as local_version

        return f"edgar-mcp {local_version}"
    except ImportError:
        pass
    try:
        version = package_version("edgar-mcp")
    except PackageNotFoundError:
        version = "0+local"
    return f"edgar-mcp {version}"


def main() -> None:
    sys.stdout = _real_stdout
    if any(arg in {"--version", "-V"} for arg in sys.argv[1:]):
        print(_version_text())
        return

    # Validate API key on startup
    _, api_key = _get_api_config()
    if not api_key:
        print("WARNING: EDGAR_API_KEY not set — remote API tools will fail", file=sys.stderr)

    mcp.run()


def _kill_previous_instance():
    """Kill any previous edgar MCP server instance spawned by the same parent session."""
    import signal
    from pathlib import Path
    server_dir = Path(__file__).resolve().parent
    ppid = os.getppid()
    pid_file = server_dir / f".edgar_mcp_server_{ppid}.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    pid_file.write_text(str(os.getpid()))
    # Clean up stale PID files from dead sessions
    for stale in server_dir.glob(".edgar_mcp_server_*.pid"):
        if stale == pid_file:
            continue
        try:
            session_pid = int(stale.stem.split("_")[-1])
            os.kill(session_pid, 0)  # check if parent session is alive
        except (ValueError, ProcessLookupError):
            stale.unlink(missing_ok=True)
        except PermissionError:
            pass  # process exists but owned by another user


if __name__ == "__main__":
    _kill_previous_instance()
    main()
