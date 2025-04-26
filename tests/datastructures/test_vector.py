import numpy as np
import pytest
from numpy.typing import NDArray

from quantem.core.datastructures.vector import Vector


class TestVector:
    """Test suite for the Vector class."""

    def test_initialization(self):
        """Test Vector initialization with different parameters."""
        # Test with fields
        v1 = Vector.from_shape(shape=(2, 3), fields=["field0", "field1", "field2"])
        assert v1.shape == (2, 3)
        assert v1.num_fields == 3
        assert v1.fields == ["field0", "field1", "field2"]
        assert v1.units == ["none", "none", "none"]
        assert v1.name == "2d ragged array"

        # Test with num_fields
        v2 = Vector.from_shape(shape=(2, 3), num_fields=3)
        assert v2.shape == (2, 3)
        assert v2.num_fields == 3
        assert v2.fields == ["field_0", "field_1", "field_2"]
        assert v2.units == ["none", "none", "none"]

        # Test with custom name and units
        v3 = Vector.from_shape(
            shape=(2, 3),
            fields=["field0", "field1", "field2"],
            name="my_vector",
            units=["unit0", "unit1", "unit2"],
        )
        assert v3.name == "my_vector"
        assert v3.units == ["unit0", "unit1", "unit2"]

        # Test error cases
        with pytest.raises(
            ValueError, match="Must specify either num_fields or fields"
        ):
            Vector.from_shape(shape=(2, 3))

        with pytest.raises(ValueError, match="Specified num_fields"):
            Vector.from_shape(shape=(2, 3), num_fields=3, fields=["field0", "field1"])

        with pytest.raises(ValueError, match="Duplicate field names"):
            Vector.from_shape(shape=(2, 3), fields=["field0", "field0", "field2"])

    def test_data_access(self):
        """Test data access and assignment."""
        v = Vector.from_shape(shape=(2, 3), fields=["field0", "field1", "field2"])

        # Set data at specific indices
        data1 = np.array([[1.0, 2.0, 3.0]])
        v[0, 0] = data1
        np.testing.assert_array_equal(v.get_data(0, 0), data1)  # type: ignore

        # Test get_data method
        assert np.array_equal(v.get_data(0, 0), data1)

        # Test set_data method
        data2 = np.array([[4.0, 5.0, 6.0]])
        v.set_data(data2, 0, 1)
        assert np.array_equal(v.get_data(0, 1), data2)

        # Test error cases
        with pytest.raises(IndexError):
            v[2, 0] = data1  # Out of bounds

        with pytest.raises(ValueError):
            v[0, 0] = np.array([[1.0, 2.0]])  # Wrong number of fields

        with pytest.raises(ValueError):
            v.set_data(np.array([[1.0, 2.0]]), 0, 0)  # Wrong number of fields

    def test_field_operations(self):
        """Test field-level operations."""
        v = Vector.from_shape(shape=(2, 3), fields=["field0", "field1", "field2"])

        # Set initial data
        v[0, 0] = np.array([[1.0, 2.0, 3.0]])
        v[0, 1] = np.array([[4.0, 5.0, 6.0]])
        v[0, 2] = np.array([[7.0, 8.0, 9.0]])

        # Test field access
        field_view = v["field0"]
        assert (
            hasattr(field_view, "vector")
            and hasattr(field_view, "field_name")
            and hasattr(field_view, "field_index")
        )

        # Test field operations
        v["field0"] += 10
        np.testing.assert_array_equal(v.get_data(0, 0)[:, 0], np.array([11.0]))  # type: ignore
        np.testing.assert_array_equal(v.get_data(0, 1)[:, 0], np.array([14.0]))  # type: ignore
        np.testing.assert_array_equal(v.get_data(0, 2)[:, 0], np.array([17.0]))  # type: ignore

        # Test applying a function to a field
        v["field1"] *= 2  # Using multiplication instead of lambda
        np.testing.assert_array_equal(v.get_data(0, 0)[:, 1], np.array([4.0]))  # type: ignore
        np.testing.assert_array_equal(v.get_data(0, 1)[:, 1], np.array([10.0]))  # type: ignore
        np.testing.assert_array_equal(v.get_data(0, 2)[:, 1], np.array([16.0]))  # type: ignore

        # Test field flattening
        flat = v["field2"].flatten()
        np.testing.assert_array_equal(flat, np.array([3.0, 6.0, 9.0]))  # type: ignore

        # Test setting flattened data
        v["field2"].set_flattened(np.array([18.0, 18.0, 18.0]))

        # Test error cases
        with pytest.raises(KeyError):
            v["nonexistent_field"]

        with pytest.raises(ValueError):
            v["field0"].set_flattened(np.array([1.0, 2.0]))  # Wrong length

    def test_slicing(self):
        """Test slicing operations."""
        v = Vector.from_shape(shape=(4, 3), fields=["field0", "field1", "field2"])

        # Set data
        for i in range(4):
            for j in range(3):
                v[i, j] = np.array(
                    [[float(i * 3 + j), float(i * 3 + j + 1), float(i * 3 + j + 2)]]
                )

        # Test slicing
        sliced = v[1:3, 1]
        assert isinstance(sliced, Vector)
        assert sliced.shape == (2, 1)

        # Compare arrays directly
        expected1 = np.array([[4.0, 5.0, 6.0]])
        expected2 = np.array([[7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(sliced.get_data(0, 0), expected1)  # type: ignore
        np.testing.assert_array_equal(sliced.get_data(1, 0), expected2)  # type: ignore

        # Test field access on sliced vector
        field_sliced = sliced["field1"]
        np.testing.assert_array_equal(field_sliced.flatten(), np.array([5.0, 8.0]))  # type: ignore

    def test_field_management(self):
        """Test adding and removing fields."""
        v = Vector.from_shape(shape=(2, 3), fields=["field0", "field1", "field2"])

        # Set initial data
        v[0, 0] = np.array([[1.0, 2.0, 3.0]])

        # Test adding fields
        v.add_fields(["field3", "field4"])
        assert v.num_fields == 5
        assert v.fields == ["field0", "field1", "field2", "field3", "field4"]
        assert v.units == ["none", "none", "none", "none", "none"]

        # Check that new fields are initialized to zeros
        np.testing.assert_array_equal(v.get_data(0, 0)[:, 3:5], np.array([[0.0, 0.0]]))  # type: ignore

        # Test removing fields
        v.remove_fields(["field1", "field3"])
        assert v.num_fields == 3
        assert v.fields == ["field0", "field2", "field4"]
        assert v.units == ["none", "none", "none"]

        # Check that data is preserved for remaining fields
        np.testing.assert_array_equal(v.get_data(0, 0)[:, 0], np.array([1.0]))  # type: ignore
        np.testing.assert_array_equal(v.get_data(0, 0)[:, 1], np.array([3.0]))  # type: ignore
        np.testing.assert_array_equal(v.get_data(0, 0)[:, 2], np.array([0.0]))  # type: ignore

        # Test error cases
        with pytest.raises(ValueError):
            v.add_fields(["field0"])  # Duplicate field

        v.remove_fields(["nonexistent_field"])  # Should just print a warning

    def test_copy(self):
        """Test deep copying."""
        v = Vector.from_shape(shape=(2, 3), fields=["field0", "field1", "field2"])
        v[0, 0] = np.array([[1.0, 2.0, 3.0]])

        # Create a copy
        v_copy = v.copy()

        # Check that it's a deep copy
        assert v_copy is not v
        assert v_copy.shape == v.shape
        assert v_copy.fields == v.fields
        assert v_copy.units == v.units
        np.testing.assert_array_equal(v_copy.get_data(0, 0), v.get_data(0, 0))  # type: ignore

        # Modify the copy and check that the original is unchanged
        v_copy[0, 0] = np.array([[4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(v.get_data(0, 0), np.array([[1.0, 2.0, 3.0]]))  # type: ignore

    def test_flatten(self):
        """Test flattening the entire vector."""
        v = Vector.from_shape(shape=(2, 3), fields=["field0", "field1", "field2"])

        # Set data
        v[0, 0] = np.array([[1.0, 2.0, 3.0]])
        v[0, 1] = np.array([[4.0, 5.0, 6.0]])
        v[0, 2] = np.array([[7.0, 8.0, 9.0]])
        v[1, 0] = np.array([[10.0, 11.0, 12.0]])
        v[1, 1] = np.array([[13.0, 14.0, 15.0]])
        v[1, 2] = np.array([[16.0, 17.0, 18.0]])

        # Flatten the vector
        flattened = v.flatten()

        # Check the flattened array
        expected = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ]
        )
        np.testing.assert_array_equal(flattened, expected)  # type: ignore

    def test_from_data(self):
        """Test creating a Vector from ragged lists or numpy arrays."""
        # Create test data
        data = [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[7.0, 8.0, 9.0]]),
            np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]),
        ]

        # Test with explicit fields
        v1 = Vector.from_ragged_lists(
            data=data,
            fields=["field0", "field1", "field2"],
            name="test_vector",
            units=["unit0", "unit1", "unit2"],
        )

        # Check properties
        assert v1.shape == (3,)
        assert v1.num_fields == 3
        assert v1.fields == ["field0", "field1", "field2"]
        assert v1.units == ["unit0", "unit1", "unit2"]
        assert v1.name == "test_vector"

        # Check data
        np.testing.assert_array_equal(v1.get_data(0), data[0])  # type: ignore
        np.testing.assert_array_equal(v1.get_data(1), data[1])  # type: ignore
        np.testing.assert_array_equal(v1.get_data(2), data[2])  # type: ignore

        # Test with inferred fields
        v2 = Vector.from_ragged_lists(data=data, num_fields=3)

        # Check properties
        assert v2.shape == (3,)
        assert v2.num_fields == 3
        assert v2.fields == ["field_0", "field_1", "field_2"]
        assert v2.units == ["none", "none", "none"]

        # Check data
        np.testing.assert_array_equal(v2.get_data(0), data[0])  # type: ignore
        np.testing.assert_array_equal(v2.get_data(1), data[1])  # type: ignore
        np.testing.assert_array_equal(v2.get_data(2), data[2])  # type: ignore

        # Test error cases
        with pytest.raises(TypeError, match="Data must be a list"):
            Vector.from_ragged_lists(data=np.array([1, 2, 3]))  # type: ignore

        with pytest.raises(ValueError, match="Specified num_fields"):
            Vector.from_ragged_lists(
                data=data,
                fields=["field0", "field1"],  # Wrong number of fields
            )

        with pytest.raises(ValueError, match="Duplicate field names"):
            Vector.from_ragged_lists(
                data=data,
                fields=["field0", "field0", "field2"],  # Duplicate field names
            )
