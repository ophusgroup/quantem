"""
pytest config file -- e.g. for making data or fixtures that are used by multiple tests
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def image_file(tmp_path_factory: pytest.TempdirFactory):
    img = np.arange(16**2).reshape((16, 16))
    fn = tmp_path_factory.mktemp("data") / "img.npy"
    np.save(fn, img)
    return fn
