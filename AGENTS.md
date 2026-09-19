<!-- COMMON:BEGIN -->
# Project Engineering Rules

> Generated block. Source of truth: `agent-rules/AGENTS.base.md` in the
> soma_project workspace root. Edit the base file and run
> `node agent-rules/sync.mjs`; do not hand-edit this block.
>
> Read by Codex (`AGENTS.md`), Antigravity/agy (`AGENTS.md`), and Claude Code
> (via `@AGENTS.md` in `CLAUDE.md`).

## General

- Inspect the existing architecture before modifying code.
- Prefer minimal changes over unnecessary rewrites.
- Do not introduce dependencies unless they are clearly justified.
- Preserve existing APIs unless the task explicitly requires changing them.
- Never modify secrets, credentials, or production configuration.
- Do not commit `.env` files, keystores, or service-account JSON.
- Match the surrounding code: naming, structure, comment density, error handling.

## Development workflow

Before implementation:

1. Inspect the relevant files.
2. Identify the affected modules and their callers.
3. State your assumptions explicitly.
4. Check the existing tests that cover the area.

During implementation:

- Keep changes focused on the assigned task.
- Follow existing naming and architecture conventions.
- Avoid unrelated refactoring.

After implementation:

1. Run the relevant tests (see **Repo commands** below).
2. Run typecheck/lint if available.
3. Inspect `git diff`.
4. Report the files changed.
5. Report any unresolved concerns.

## Testing

A task is not complete merely because code was written. Completion requires at
least one of:

- passing automated tests
- a successful build
- a successful typecheck
- reproducible manual verification, with the steps written down

Never report success based on code you wrote but did not run. If a test was
already failing before your change, say so explicitly — do not claim it as your
own regression, and do not claim credit for fixing it.

## Git

- Do not force-push.
- Do not rewrite `main` / `master` / `dev` history.
- Do not delete branches unless explicitly requested.
- Keep commits focused: one logical change per commit.
- Do not commit unless the task explicitly asks you to.

## Multi-agent workflow

When operating as an Orca dispatched worker:

- The injected preamble is authoritative. Work only on the assigned task.
- Respect ownership boundaries. Do not modify files assigned to another worker.
- Use the preamble's `ask` command for a blocking question. Never open a local
  interactive prompt that the coordinator cannot answer.
- Send `worker_done` exactly once, from the dispatched terminal, with an
  explicit `--outcome succeeded` or `--outcome failed`. Never encode failure
  only in prose.
- Provide objective evidence for completion: the command you ran and its result.
- After `worker_done`, end the turn and idle. Do not start new work.

When two workers are deliberately given overlapping scope for comparison or
review, neither may edit the other's files. Each reports independently and the
coordinator decides.
<!-- COMMON:END -->

## Repo-specific — AI

FastAPI inference service (`team-EBD/AI`). The service lives in `ai-server/`,
**not** at the repository root — run every command from `ai-server/`.
Backed by Google GenAI (`google-genai`) with a read-only DB connection used for
recommendation candidates.

### Repo commands

| Purpose | Command (from `ai-server/`) |
| --- | --- |
| Tests | `venv/Scripts/python -m pytest` |
| One test file | `venv/Scripts/python -m pytest tests/test_analyze.py` |
| Dev server | `venv/Scripts/python -m uvicorn app.main:app --reload` |
| Install dev deps | `venv/Scripts/python -m pip install -r requirements-dev.txt` |

`pytest.ini` sets `pythonpath = .` and `testpaths = tests`, so pytest must be
invoked from `ai-server/` or collection will fail.

### Invariants

- Tests must not call the live model API. Keep the model client mocked; a test
  that needs a real API key is a broken test.
- This service is called by BE over an internal authenticated route
  (`tests/test_internal_auth.py` covers it). Do not change the internal auth
  contract or the response shape without the matching BE change being part of
  the same task.
- Prompt changes alter production behaviour that no unit test fully captures.
  When you change a prompt, state exactly what changed and what you verified.
- The DB connection here is **read-only** by design. Never add a write path.
