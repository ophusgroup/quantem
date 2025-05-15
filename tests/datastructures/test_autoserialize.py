import numpy as np
import pytest

from quantem.core.io.serialize import AutoSerialize, load


class Child(AutoSerialize):
    def __init__(self):
        self.values = np.arange(4).reshape(2, 2)
        self.flag = True


class Dummy(AutoSerialize):
    def __init__(self):
        self.arr = np.linspace(0, 1, 5)
        self.count = 123
        self.child = Child()
        self._private = "secret"


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_save_load_roundtrip(tmp_path, store):
    """Saving and loading with no skips preserves everything."""
    d = Dummy()
    # choose an appropriate path
    target = tmp_path / ("t.zip" if store == "zip" else "t_dir")
    d.save(str(target), mode="w", store=store)
    loaded = load(str(target))

    # basic type checks
    assert isinstance(loaded, Dummy)
    assert isinstance(loaded.child, Child)

    # numpy array round‐trip
    np.testing.assert_allclose(loaded.arr, d.arr)

    # scalar round‐trip
    assert loaded.count == d.count

    # nested round‐trip
    np.testing.assert_allclose(loaded.child.values, d.child.values)
    assert loaded.child.flag is True

    # private attr
    assert loaded._private == "secret"


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_skip_top_level_scalar(tmp_path, store):
    """Skip a top‐level scalar attribute."""
    d = Dummy()
    target = tmp_path / ("t2.zip" if store == "zip" else "t2_dir")
    d.save(str(target), mode="w", store=store, skip=["count"])
    loaded = load(str(target))

    # arr and child still there
    np.testing.assert_allclose(loaded.arr, d.arr)
    assert isinstance(loaded.child, Child)

    # count should have been skipped
    assert not hasattr(loaded, "count")


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_skip_nested_group(tmp_path, store):
    """Skip a nested AutoSerialize subtree."""
    d = Dummy()
    target = tmp_path / ("t3.zip" if store == "zip" else "t3_dir")
    d.save(str(target), mode="w", store=store, skip=["child"])
    loaded = load(str(target))

    # top‐level arr still there
    np.testing.assert_allclose(loaded.arr, d.arr)

    # child subtree removed
    assert not hasattr(loaded, "child")


@pytest.mark.parametrize("store", ["zip", "dir"])
def test_skip_private_by_name(tmp_path, store):
    """You can also skip private attrs if you list them explicitly."""
    d = Dummy()
    target = tmp_path / ("t4.zip" if store == "zip" else "t4_dir")
    d.save(str(target), mode="w", store=store, skip=["_private"])
    loaded = load(str(target))

    # arr and count still present
    np.testing.assert_allclose(loaded.arr, d.arr)
    assert hasattr(loaded, "count")

    # private should be gone
    assert not hasattr(loaded, "_private")
