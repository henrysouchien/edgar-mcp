## Mandatory Plan-First Workflow

NO COWBOY CODING. This codebase has carefully designed layers and abstractions. Do NOT make quick fixes, first-glance patches, or lazy code changes — they WILL break things, violate architectural patterns, or produce garbage code.

**ALL code changes MUST follow this workflow:**

1. **Plan first** — enter plan mode, research the codebase, understand the architecture. This is where errors get caught. No exceptions for "it looks simple."
2. **Codex review** — send the plan to Codex for review. Codex catches what Claude misses. Iterate until PASS.
3. **Implement via Codex** — send the approved plan to Codex for implementation. Claude does not write the code directly unless explicitly told to.

**The ONLY exception**: trivial fixes (a few lines, typo-level) that the user expressly approves for direct edit. If in doubt, it's not trivial — plan it.

**Why this exists**: Claude's first instinct is to glance at code and "fix" it. This almost always breaks something, misses architectural context, or produces code that doesn't match the codebase's patterns. The plan→review→implement pipeline forces proper investigation before any code is written.

### Codex Implementation via MCP

When implementing via `mcp__codex__codex` (the write-path Codex calls in step 3 of the workflow above):

- **Model + reasoning:** DO NOT pass `model` or `config.model_reasoning_effort`. Inherit whatever is in `~/.codex/config.toml` — that's the source of truth and tracks the current model as it's upgraded.
- **Approval policy:** `approval-policy: "never"`.
- **Sandbox:** default to `sandbox: "workspace-write"`. Escalate to `"danger-full-access"` only when the task genuinely requires cross-repo, system-level, or network access — and state the reason in the prompt.
- **Working directory:** always pass `cwd` explicitly (repo root).

Note: the `/codex` skill is read-only (review/challenge/consult). Implementation is not covered by the skill — these MCP conventions are how you stay consistent across sessions.

## Don't defer to dodge friction

In the design/build phase there is no usage signal to "wait for" — pre-launch deferral hides a decision behind a signal that will never arrive, and shifts the catch-tax onto the user.

Three places this shows up:
- **Implementation plans:** if a vetted design doc says X is in scope, keep X in scope. Scope changes belong in the design discussion with the user, not baked into the plan. Planning only part this session? Name the rest as "in scope, not in this plan" — don't demote to "deferred."
- **Bug triage:** the answer is fix it, file it with explicit triage rationale, or decide it's not a bug. "Defer the fix" with no reason is can-kicking.
- **Design pushback:** address the concern. "Wait and see if anyone asks for it" pre-launch is a non-answer — there's no mechanism for demand to surface.

**Forbidden rationales:** "wait for usage pressure," "build if demand emerges," "build if needed," "MVP doesn't need this," "defer to future iteration," "we can add this later."

**Frame: product-saving, not engineer-saving.** Cutting scope to save build effort pattern-matches to a world where human engineering time was the binding constraint. In AI-assisted build it isn't. The question is not "what can we drop to ship cheaper" but "what does the product need to be coherent." If it passes that bar, build it. If it fails, kill it — don't park it.

**Why this exists:** silent deferral fragments coherent designs, lets bugs rot, lets contested decisions linger, and forces the user to catch every punt. This pattern has surfaced repeatedly across sessions — both design docs and implementation plans derived from them have used "build if needed" / "build if usage pressure emerges" rationale on items that don't yet exist. Make the deferral question explicit, or don't make it.

---


NEVER edit files in any `-dist` package directory (app-platform-dist, brokerage-connect-dist, agent-gateway-dist, finance-cli-dist, fmp-mcp-dist, ibkr-mcp-dist, portfolio-risk-engine-dist, taskflow-agent-dist, web-app-platform-dist). These are synced deployment repos — always edit the source repo instead. If you need to change code that lives in a -dist package, identify which source repo owns it and make changes there.

## NEVER CREATE OR SWITCH BRANCHES

**HARD RULE — NO EXCEPTIONS.** Claude must NEVER create or switch branches. Specifically NEVER run:
- `git checkout -b <name>` / `git switch -c <name>` — creates + switches branch
- `git checkout <branch>` / `git switch <branch>` — switches branch
- `git branch <name>` — creates branch

**Why:** Multiple Claude sessions run in parallel, often sharing the same working directory. Any branch creation or checkout moves HEAD for every session rooted in that cwd — silently placing one session's commits on another session's branch, destroying work. This has happened repeatedly.

**What to do instead:** Work on whatever branch the session is currently on. Commit there. If the user wants work on a different branch, THE USER creates it — ask them, do not do it yourself. Never suggest "let me create a branch for this."

