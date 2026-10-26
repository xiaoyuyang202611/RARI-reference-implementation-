# Code validation (software checks only)

Executed on the local CPU Python/PyTorch environment:

- Python syntax compilation: PASS.
- `pytest -q tests`: 3 tests passed.
- Synthetic dataset manifest validation and complete 4+16 two-step action trajectories: PASS.
- Offline training for two epochs, checkpoint serialization: PASS.
- Validation-only temperature/coverage calibration: PASS.
- Test policy evaluation with 11 methods, source-derived CSV and JSON: PASS.
- Risk/gain calibration calculation, chart generation and single-case inference: PASS.
- `produce_trajectories` with the explicitly demo-only adapter: PASS.
- Manuscript-style four-table CSV export: PASS.

**These tests demonstrate engineering functionality, not RARI's clinical or
scientific efficacy.** No real CT/MRI patient data or RARI authors' model
weights are bundled.
