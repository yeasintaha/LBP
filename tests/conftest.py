import os

import pytest

ORL_DIR = os.environ.get("ORL_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ORL"))


@pytest.fixture(scope="session")
def orl_dir():
    if not os.path.isdir(os.path.join(ORL_DIR, "s1")):
        pytest.skip(f"ORL not found at {ORL_DIR} (set ORL_DIR or see README)")
    return ORL_DIR
