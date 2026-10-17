import numpy as np
import pytest


@pytest.fixture(autouse=True)
def preserve_numpy_random_state():
    """Legacy RNG seeding in one test must not affect later tests."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)
