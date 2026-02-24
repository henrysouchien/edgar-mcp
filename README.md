# edgar-mcp

SEC EDGAR financial data for your AI agent.

## What your AI can do

- `get_filings`: fetch filing metadata (10-Q/10-K/8-K earnings release references)
- `get_financials`: fetch full period facts (inline JSON or local file output)
- `get_metric`: fetch a specific metric/tag for a period
- `list_metrics`: list candidate metrics/tags in a filing period
- `search_metrics`: fuzzy-search metrics by natural-language query
- `get_filing_sections`: extract filing narrative sections/tables (10-K/10-Q default, or 8-K earnings release with `source="8k"`)

## Install

```bash
pip install edgar-mcp
```

## Configuration

- `EDGAR_API_KEY` (required): API key for EDGAR Financial API
- `EDGAR_API_URL` (optional): defaults to `https://www.financialmodelupdater.com`
- `EDGAR_MCP_OUTPUT_DIR` (optional): path for `output="file"` responses
  - default: `./exports/file_output` from current working directory
  - fallback: `~/.cache/edgar-mcp/file_output`

## Run

```bash
edgar-mcp
```

Claude Code config snippet:

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

## Requirements

- Python 3.11+
- EDGAR API key

## See also

- **[edgar-parser](https://github.com/henrysouchien/edgar-parser)** — The underlying Python library for parsing SEC EDGAR filings. Use this directly if you want to integrate EDGAR data into your own Python application without MCP.
