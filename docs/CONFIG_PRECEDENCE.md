# Configuration Precedence

Values are resolved in this order (highest priority first):

1. CLI arguments (e.g., `--learning-rate 0.0005`)
2. YAML config (e.g., `configs/halfmile.yaml: learning_rate: 0.001`)
3. Python dataclass default (e.g., `SeismicConfig.learning_rate = 1e-3`)

## Rules for contributors

When adding a new CLI flag that maps to a `SeismicConfig` field:
- Use `default=None` on the Click option.
- Pass the value through the `overrides` dict pattern.
- Let `load_and_override_config` apply it only if it's not None.

When adding a CLI-only flag (`--output`, `--dry-run`, `--verbose`):
- Concrete defaults are fine.

## Why this matters

If a CLI default is not `None`, it silently overrides the YAML config,
even when the user doesn't pass the flag. This makes it impossible to
change config values from YAML alone and can invalidate experiments.

## History

- 2026-09-14: Fixed `train.py` flags (`--loss`, `--dice-weight`,
  `--focal-gamma`, `--checkpoint-every`, `--early-stopping`) that
  silently overrode config values.