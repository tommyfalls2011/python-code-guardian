# Python Code Guardian

Python Code Guardian is a deterministic Python syntax checking and repair tool.

It scans Python files, identifies supported syntax problems, proposes conservative repairs, validates candidates, and can apply changes with backup and rollback protection.

## Requirements

- Python 3.10 or newer
- pytest for running the test suite

## Installation

```bash
python -m pip install -e .
```

## Basic Usage

Scan a Python file:

```bash
codeguardian example.py
```

Repair supported syntax errors:

```bash
codeguardian example.py --repair
```

Preview repairs without modifying files:

```bash
codeguardian example.py --repair --dry-run
```

## Repair Policies

`--policy safe` allows only high-confidence repairs. `--policy tested` allows medium- and high-confidence repairs with test validation. `--policy off` disables automatic repair. Low-confidence repairs are never automatically applied.

## Safety

The repair pipeline uses parser diagnostics, token-aware strategies, diagnostic freshness checks, candidate syntax validation, confidence policies, backups, and rollback on validation or test failure.

## Tests

```bash
python -m pytest -q
```

## Development Status

Python Code Guardian is under active development. The current focus is deterministic, validated, and regression-tested Python syntax repair.
