## Mandatory Plan-First Workflow

NO COWBOY CODING. This codebase has carefully designed layers and abstractions. Do NOT make quick fixes, first-glance patches, or lazy code changes — they WILL break things, violate architectural patterns, or produce garbage code.

**ALL code changes MUST follow this workflow:**

1. **Plan first** — enter plan mode, research the codebase, understand the architecture. This is where errors get caught. No exceptions for "it looks simple."
2. **Codex review** — send the plan to Codex for review. Codex catches what Claude misses. Iterate until PASS.
3. **Implement via Codex** — send the approved plan to Codex for implementation. Claude does not write the code directly unless explicitly told to.

**The ONLY exception**: trivial fixes (a few lines, typo-level) that the user expressly approves for direct edit. If in doubt, it's not trivial — plan it.

**Why this exists**: Claude's first instinct is to glance at code and "fix" it. This almost always breaks something, misses architectural context, or produces code that doesn't match the codebase's patterns. The plan→review→implement pipeline forces proper investigation before any code is written.

---


NEVER edit files in any `-dist` package directory (app-platform-dist, brokerage-connect-dist, agent-gateway-dist, finance-cli-dist, fmp-mcp-dist, ibkr-mcp-dist, portfolio-risk-engine-dist, taskflow-agent-dist, web-app-platform-dist). These are synced deployment repos — always edit the source repo instead. If you need to change code that lives in a -dist package, identify which source repo owns it and make changes there.
