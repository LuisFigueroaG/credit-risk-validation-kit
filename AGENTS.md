# AGENTS.md

Project instructions for autonomous agents working in this repository.

## Operating Mode

- Work as a professional software engineer: read the repository first, make scoped changes, verify them, and report concrete outcomes.
- Prefer autonomous execution over stopping for minor decisions. Ask the user only when a decision is genuinely ambiguous, risky, destructive, or requires credentials/secrets.
- Keep changes focused on the requested goal. Do not introduce unrelated refactors or churn.
- Preserve user work. Never revert or overwrite changes you did not make unless the user explicitly asks.

## Git Workflow

- Use `develop` as the active development branch for implementation work.
- Use `master` as the protected release/default branch.
- Move changes from `develop` back to `master` through reviewable commits and pull requests.
- Before starting implementation, check the current branch and worktree state.
- Commit logically grouped changes with clear messages once the work is verified.
- Push the working branch and open or update a PR when the change is ready for review.
- Do not push directly to `master` unless the user explicitly requests it.
- Do not rename `master` to `main` unless the user explicitly requests it.

## Verification

- Run the most relevant tests, linters, formatters, or smoke checks for the changed surface area.
- If verification cannot run because dependencies, credentials, network, or services are unavailable, state that clearly and include the exact command that was attempted.
- Do not treat work as complete until implementation and verification are both addressed.

## Kaggle

- The Kaggle CLI is installed at `/home/luchopy/.local/bin/kaggle`.
- Network access for Kaggle requires escalated permissions in this environment. Run Kaggle CLI commands with `sandbox_permissions: "require_escalated"`.
- Use the absolute CLI path when needed:

```bash
/home/luchopy/.local/bin/kaggle
```

- Do not print Kaggle tokens or credentials. Check only whether credentials are present.
- Prefer storing Kaggle access tokens in `~/.kaggle/access_token` or standard Kaggle credential files instead of embedding secrets in commands, scripts, notebooks, or commits.
- Use only datasets that can be downloaded without accepting additional terms during autonomous runs. If Kaggle requires manual acceptance, report the dataset and let the user handle it.

## Release And CI

- Build distribution artifacts locally when requested, but do not publish to PyPI unless the user explicitly requests it.
- Configure CI where possible and run local checks that are available in the environment. If GitHub Actions, repository settings, or third-party services block full validation, report the limitation clearly.

## Data Handling

- Keep downloaded datasets out of source control unless the user explicitly asks otherwise.
- Prefer `data/raw/` for original downloads and document dataset source, slug, and retrieval command.
- Avoid committing large generated artifacts, archives, model binaries, or credentials.
