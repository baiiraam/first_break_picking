# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture
def sample_shots():
    """Provide sample shot IDs as numpy array."""
    return np.array([100 + i for i in range(100)])


@pytest.fixture
def sample_shot_ids():
    """Provide sample shot IDs as list."""
    return [100 + i for i in range(100)]


@pytest.fixture
def sample_shot_ids_np():
    """Provide sample shot IDs as numpy array for manifest tests."""
    return np.array([100 + i for i in range(100)])


@pytest.fixture
def sample_picks():
    """Provide sample picks for testing."""
    # Simulates Halfmile picks (in milliseconds)
    return np.array(
        [20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 881.0]
    )


@pytest.fixture
def sample_trace_data():
    """Create sample trace data."""
    n_traces = 10
    n_samples = 751
    data = np.random.randn(n_traces, n_samples).astype(np.float32)
    return data


@pytest.fixture
def temp_chunk_dir(tmp_path):
    """Create temporary chunk directory."""
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    return chunk_dir
