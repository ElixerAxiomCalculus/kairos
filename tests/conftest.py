import pytest

from kairos.config import load_config
from kairos.engine import EngineContext


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def ctx(cfg):
    return EngineContext(cfg)
