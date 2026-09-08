"""
Pre-compiled regex patterns for error detection in batch training.
"""

import re

# ============================================================
# ERROR PATTERNS - PRE-COMPILED
# ============================================================

_REAL_ERROR_PATTERNS = [
    # Python exceptions
    re.compile(r"^.*RuntimeError:", re.IGNORECASE),
    re.compile(r"^.*ValueError:", re.IGNORECASE),
    re.compile(r"^.*TypeError:", re.IGNORECASE),
    re.compile(r"^.*AttributeError:", re.IGNORECASE),
    re.compile(r"^.*KeyError:", re.IGNORECASE),
    re.compile(r"^.*IndexError:", re.IGNORECASE),
    re.compile(r"^.*ImportError:", re.IGNORECASE),
    re.compile(r"^.*ModuleNotFoundError:", re.IGNORECASE),
    re.compile(r"^.*FileNotFoundError:", re.IGNORECASE),
    re.compile(r"^.*PermissionError:", re.IGNORECASE),
    re.compile(r"^.*ConnectionError:", re.IGNORECASE),
    re.compile(r"^.*TimeoutError:", re.IGNORECASE),
    re.compile(r"^.*MemoryError:", re.IGNORECASE),
    re.compile(r"^.*OutOfMemoryError:", re.IGNORECASE),
    # PyTorch specific
    re.compile(r"^.*MPS out of memory", re.IGNORECASE),
    re.compile(r"^.*CUDA out of memory", re.IGNORECASE),
    re.compile(r"^.*torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    # Train.py specific
    re.compile(r"^.*Error:", re.IGNORECASE),
    re.compile(r"^.*Exception:", re.IGNORECASE),
    re.compile(r"^.*AssertionError", re.IGNORECASE),
    # Stack trace
    re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    # Exit code patterns
    re.compile(r"exited with code [1-9]", re.IGNORECASE),
    re.compile(r"Process exited with code [1-9]", re.IGNORECASE),
    re.compile(r"returned non-zero exit code", re.IGNORECASE),
]

_MEMORY_PATTERNS = [
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"OOM", re.IGNORECASE),
    re.compile(r"MPS out of memory", re.IGNORECASE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"cannot allocate", re.IGNORECASE),
    re.compile(r"memory exhausted", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
    re.compile(r"MemoryError", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"RuntimeError: MPS", re.IGNORECASE),
    re.compile(r"RuntimeError: CUDA", re.IGNORECASE),
    re.compile(r"MPS: out of memory", re.IGNORECASE),
    re.compile(r"Out of memory\. Try reducing", re.IGNORECASE),
]

_FAILED_PATTERNS = [
    re.compile(r"failed with exit code", re.IGNORECASE),
    re.compile(r"command failed", re.IGNORECASE),
    re.compile(r"training failed", re.IGNORECASE),
]

_EXIT_PATTERNS = [
    re.compile(r"exited with code [1-9]", re.IGNORECASE),
    re.compile(r"Process exited with code [1-9]", re.IGNORECASE),
    re.compile(r"returned non-zero exit code", re.IGNORECASE),
]


# ============================================================
# DETECTION FUNCTIONS
# ============================================================


def is_real_error(output: str) -> bool:
    """Check if the output contains a REAL error using pre-compiled patterns."""

    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Skip MLflow info/warning lines
        if "mlflow" in line.lower():
            continue

        # Skip INFO/WARNING/DEBUG log lines (they're not errors)
        if (
            any(
                level in line
                for level in [" INFO ", " WARNING ", " DEBUG ", " CRITICAL "]
            )
            and " ERROR " not in line
        ):
            continue

        # Check real error patterns
        for pattern in _REAL_ERROR_PATTERNS:
            if pattern.search(line):
                return True

    # Check for exit codes
    if "sys.exit(1)" in output or "exit(1)" in output:
        return True

    for pattern in _EXIT_PATTERNS:
        if pattern.search(output):
            return True

    for pattern in _FAILED_PATTERNS:
        if pattern.search(output):
            return True

    return False


def is_memory_error(error_message: str) -> bool:
    """
    Check if an error is a REAL memory error using pre-compiled patterns.
    """
    # First check if it's even a real error
    if not is_real_error(error_message):
        # Direct memory pattern check for short messages
        for pattern in _MEMORY_PATTERNS:
            if pattern.search(error_message):
                return True
        return False

    # Check for memory-specific patterns
    for pattern in _MEMORY_PATTERNS:
        if pattern.search(error_message):
            return True

    return False
