"""
File demoing pytest and features we might want to use

Call with .../quantem/tests$pytest
useful flags are controlling verbosity : -q -v -vv etc.

"""

import sys

import numpy as np
import pytest


### basic function calls
def f1(x):
    return x + 1


def test_f1():
    assert f1(4) == 5


def f2():
    raise RuntimeError


def test_f2():
    with pytest.raises(RuntimeError):
        f2()


### tests can be put in classes
### pytest will find all classes prefixed with "Test" and will call all "test_" prefixed methods


class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    ### fails
    # def test_two(self):
    #     x = "hello"
    #     assert hasattr(x, "check")

    def test_three(self):
        x = "hi"
        assert "h" in x


### each test will instantiate a new instance of the class, so this will fail on test_two
class TestClassInstance:
    value = 0

    def test_one(self):
        self.value = 1
        assert self.value == 1

    ### fails
    # def test_two(self):
    #     assert self.value == 1


### one nice feature we will likely want to use is temp files
### tmp_path and tmp_path_factory are pytest fixtures which make a temporary path (as a path.Path
### object and a pytest.TempDirFactory respectively), and which can be used in individual tests
### or for an entire testing session
### we created a temp file called "image_file" in conftest.py and can read it in these tests
### more on this here: https://docs.pytest.org/en/stable/how-to/tmp_path.html


def test_image1(image_file):
    load_img = np.load(image_file)
    test = np.arange(16**2).reshape((16, 16))
    assert np.array_equal(load_img, test)


### fails, note it also shows the tempfile path
# def test_image2(image_file):
#     load_img = np.load(image_file)
#     test = np.ones((16,16))
# assert np.array_equal(load_img, test)


### there are also markers for skipping certain tests or expected failures,
### e.g based on platform or python version


# @pytest.mark.skipif(not sys.platform.startswith("win"), reason="windows only test")
@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="test only for python3.10 or older"
)
def test_function():
    assert 0
