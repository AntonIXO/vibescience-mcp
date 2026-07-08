import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # make fixture.py importable

from fixture import seed  # noqa: E402


@pytest.fixture()
def vault(tmp_path):
    return tmp_path / "vault"


@pytest.fixture()
def seeded(vault):
    """A Store seeded with the §13 worked example."""
    return seed(vault)
