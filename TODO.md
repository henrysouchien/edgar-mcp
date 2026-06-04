# TODO — edgar-mcp

## Housekeeping — iCloud sync duplicates / cruft (filed 2026-06-04)

- **Clean ~2 macOS ` N` sync duplicates** caused by iCloud "Desktop & Documents" syncing `~/Documents`. Safe to delete: regenerable dirs (`node_modules`/`.venv`/`.next`/`cache`/`logs`/`build`) and inert git cruft (`.git/index 2`…`9`, `tmp_obj_*` — git never reads these). Verify any authored ` 2.md`/` 2.py` dupes with `diff` against the original before deleting. Root cause + reusable exclusion plan: `risk_module/docs/planning/ICLOUD_NOSYNC_EXCLUSION_PLAN.md`.
