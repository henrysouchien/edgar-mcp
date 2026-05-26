# edgar-mcp

SEC EDGAR financial data for AI agents, exposed as an MCP stdio server.

<!-- Generated from docs/tool_manifest.yaml by scripts/render_mcp_readme.py. -->

## Install

```bash
pip install edgar-mcp
```

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `EDGAR_API_KEY` | yes | API key for the hosted EDGAR API. |
| `EDGAR_API_URL` | no | Override the API base URL. Defaults to `https://www.edgarparser.com`. |
| `EDGAR_MCP_OUTPUT_DIR` | no | Directory for `output="file"` tool responses. |

Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "edgar-financials": {
      "type": "stdio",
      "command": "edgar-mcp",
      "env": {
        "EDGAR_API_KEY": "YOUR_KEY_HERE"
      }
    }
  }
}
```

## Tool Inventory

This package exposes 29 public MCP tools. The inventory below is generated from the same manifest that drives the hosted API documentation.

| Tool | Summary | Tier | Cache Behavior |
| --- | --- | --- | --- |
| `get_filings` | Return filing metadata and SEC accession details for one ticker period. | `public` | `cache_only` |
| `get_event_filings` | Discover event filings by ticker, CIK, date range, form type, or query. | `public` | `cold_allowed` |
| `describe_filing` | Show cached availability for financials, sections, tables, markdown, and extractions for one filing. | `public` | `cache_only` |
| `get_filing_document` | Return a readable markdown filing document with section filtering and pagination. | `registered` | `cold_allowed` |
| `get_filing_sections` | Extract selected qualitative filing sections and optional embedded tables. | `public` | `cold_allowed` |
| `get_filing_cover_facts` | Return exact DEI cover-page facts such as shares outstanding with citations. | `registered` | `cold_allowed` |
| `get_filing_evidence` | Plan and retrieve citation-ready filing evidence for qualitative SEC questions. | `paid` | `llm_call` |
| `get_operational_kpi_drivers` | Extract non-XBRL operational KPI values and driver growth rates from filing narrative. | `paid` | `llm_call` |
| `get_filing_extractions` | Return cached langextract spans for one filing or run paid extraction on cache miss. | `paid` | `llm_call` |
| `search_extractions` | Search cached langextract spans across filings with structured filters. | `paid` | `cache_only` |
| `get_extraction_series` | Return periodized counts and optional hits for cached langextract spans. | `paid` | `cache_only` |
| `extract_filing_file` | Ingest a trusted local markdown filing and run a selected extraction schema. | `internal` | `llm_call` |
| `list_extraction_schemas` | List document extraction schemas available to the internal document API. | `internal` | `cache_only` |
| `get_filing_tables` | Return filing table metadata or one hydrated structured table. | `public` | `cold_allowed` |
| `search_filing_tables` | Search cached filing table metadata across a ticker and period range. | `public` | `cache_only` |
| `compare_filing_tables` | Compare matching filing tables across multiple tickers. | `public` | `cache_only` |
| `search_filing_text` | Search cached markdown within one filing and return matching spans. | `registered` | `cache_only` |
| `get_financials` | Return the full structured XBRL financial fact payload for one filing period. | `public` | `cache_only` |
| `get_metric` | Return one discovered metric or XBRL tag with current, prior, and YoY values. | `public` | `cache_only` |
| `get_metric_series` | Return one metric across multiple periods with per-period cache and coverage status. | `public` | `cache_only` |
| `list_metrics` | List exact metric candidates available in one filing period. | `public` | `cache_only` |
| `search_metrics` | Search filing metrics by natural-language query and return ranked exact candidates. | `public` | `cold_allowed` |
| `get_statement` | Return a template-backed income statement, balance sheet, or cash flow statement from cached concepts. | `public` | `cache_only` |
| `get_concept` | Resolve one registry-backed financial concept from cached filing facts. | `public` | `cache_only` |
| `compare_concept` | Compare one registry-backed concept across a caller-provided ticker set. | `public` | `cache_only` |
| `concept_trend` | Return a cache-only concept time series across a requested period range. | `public` | `cache_only` |
| `cite_concept` | Join a registry-backed concept value to filing prose and optional extraction evidence. | `paid` | `cache_only` |
| `warm_metric_cache` | Queue paid-tier background warming for metric periods before follow-up reads. | `paid` | `cold_allowed` |
| `warm_metric_cache_status` | Poll a background cache-warming job returned by warm_metric_cache. | `paid` | `cache_only` |

## Tool Families

### Filing Metadata and Documents

`get_filings`, `get_event_filings`, `describe_filing`, `get_filing_document`, `get_filing_sections`, `get_filing_cover_facts`, `get_filing_evidence`

### Operational KPIs

`get_operational_kpi_drivers`

### LangExtract

`get_filing_extractions`, `search_extractions`, `get_extraction_series`, `extract_filing_file`, `list_extraction_schemas`

### Filing Tables

`get_filing_tables`, `search_filing_tables`, `compare_filing_tables`

### Filing Text

`search_filing_text`

### XBRL and Concepts

`get_financials`, `get_metric`, `get_metric_series`, `list_metrics`, `search_metrics`, `get_statement`, `get_concept`, `compare_concept`, `concept_trend`, `cite_concept`, `warm_metric_cache`, `warm_metric_cache_status`

## Runtime Notes

- This is a thin client for the hosted EDGAR API; it does not parse filings locally.
- Missing or invalid `EDGAR_API_KEY` lets the server start, but tool calls return authentication errors.
- Large payload tools support `output="file"` and write to `EDGAR_MCP_OUTPUT_DIR` or a local fallback directory.
- API rate limits and paid-tool access are enforced by the hosted service.

## Links

- Hosted API: https://www.edgarparser.com
- Documentation: https://docs.edgarparser.com
- MCP setup: https://docs.edgarparser.com/mcp
- Tool reference: https://docs.edgarparser.com/tools
- Changelog RSS: https://docs.edgarparser.com/rss.xml
- In-app developer page: https://www.edgarparser.com/developers
- Open parser package: https://github.com/henrysouchien/edgar-parser
- Source package: https://github.com/henrysouchien/edgar-mcp
