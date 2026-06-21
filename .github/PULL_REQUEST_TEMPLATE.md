<!-- Thanks for contributing to ap-verify. -->

## Summary

<!-- What does this change and why? -->

## Measurement (if it touches accuracy / eval / the critic)

<!-- Dataset, split, counts, and the before/after numbers. Real numbers only. -->

## Checklist

- [ ] `ruff check src tests` is clean
- [ ] `ruff format --check src tests` is clean
- [ ] `mypy --strict src tests` is clean
- [ ] `pytest` passes (domain layer stays 100% covered)
- [ ] Eval gate unaffected: `apverify-eval --count 50 --min-catch-rate 0.99 --max-false-hold 0.0`
- [ ] New behaviour is covered by tests written test-first
