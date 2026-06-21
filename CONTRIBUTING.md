# Contributing to ap-verify

Thanks for your interest. ap-verify is a portfolio-grade project with a strict
quality bar, so contributions are held to the same gates the CI enforces.

## Development setup

System dependencies (for the live pipeline — OCR + PDF rasterising):

```bash
brew install tesseract poppler        # macOS; apt: tesseract-ocr poppler-utils
```

Then:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install                    # optional: runs ruff + mypy on commit
```

You do **not** need any API keys to develop or run the test suite — the tests are
deterministic and never call a provider. To run the live pipeline locally with no
keys, use the [Ollama path](README.md#run-locally--no-api-keys).

## The quality gate (must be green before a PR)

```bash
ruff check src tests
ruff format --check src tests
mypy --strict src tests
pytest
```

Conventions:

- **TDD** — write the failing test first, watch it fail, then implement.
- **Clean architecture** — the dependency rule is strict: `domain` imports only the
  standard library and `domain`; `application` defines ports; `infrastructure` /
  `interface` / `eval` depend inward, never the reverse.
- **Domain layer is 100% covered.** New domain logic needs tests that exercise it.
- **Honesty** — this project reports real, imperfect numbers. Don't add metrics,
  badges, or claims that aren't measured and reproducible.

## Pull requests

1. Branch from `main`: `git checkout -b feat/your-change`.
2. Make the change with tests; keep all four gates green.
3. Confirm the eval gate is unaffected: `apverify-eval --count 50 --min-catch-rate 0.99 --max-false-hold 0.0`.
4. Open a PR describing the change and the measurement (if it touches accuracy/eval).

## Reporting issues

Use GitHub Issues. For bugs, include a minimal reproduction and the output of the
failing command. For accuracy/eval findings, include the dataset, split, and counts.
