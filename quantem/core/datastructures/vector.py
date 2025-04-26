from typing import Any, List, Optional, Sequence, Tuple, TypeVar, Union, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import (
    validate_fields,
    validate_num_fields,
    validate_shape,
    validate_vector_data,
    validate_vector_units,
)


class Vector(AutoSerialize):
    """
    A class for holding vector data with ragged array lengths. This class supports any number of fixed dimensions
    (indexed first) followed by a ragged numpy array that can have any number of entries (rows) and columns (fields).
    Inherits from AutoSerialize for serialization support.

    Basic Usage:
    -----------
    # Create a 2D vector with shape=(4, 3) and 3 named fields
    v = Vector.from_shape(shape=(4, 3), fields=['field0', 'field1', 'field2'])

    # Alternative creation with num_fields instead of fields
    v = Vector.from_shape(shape=(4, 3), num_fields=3)  # Fields will be named field_0, field_1, field_2

    # Create with custom name and units
    v = Vector.from_shape(
        shape=(4, 3),
        fields=['field0', 'field1', 'field2'],
        name='my_vector',
        units=['unit0', 'unit1', 'unit2'],
    )

    # Access data at specific indices
    data = v[0, 1]  # Returns numpy array at position (0,1)

    # Set data at specific indices
    v[0, 1] = np.array([[1.0, 2.0, 3.0]])  # Must match num_fields

    # Create a deep copy
    v_copy = v.copy()

    Example usage of from_data:
    -----------------------------------
    data = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8], [9, 10]])
    ]
    v = Vector.from_data(
        data,
        fields=['x', 'y'],
        name='my_ragged_vector',
        units=['m', 'm']
    )

    # Or using lists instead of numpy arrays:
    data = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8], [9, 10]],
    ]
    v = Vector.from_data(
        data,
        fields=['x', 'y'],
        name='my_ragged_vector',
        units=['m', 'm']
    )

    Field Operations:
    ----------------
    # Access a specific field
    field_data = v['field0']  # Returns a FieldView object

    # Perform operations on a field
    v['field0'] += 16  # Add 16 to all field0 values

    # Apply a function to a field
    v['field2'] = lambda x: x * 2  # Double all field2 values

    # Get flattened field data
    field_flat = v['field0'].flatten()  # Returns 1D numpy array

    # Set field data from flattened array
    v['field2'].set_flattened(new_values)  # Must match total length

    Advanced Operations:
    -------------------
    # Complex field calculations
    scale = v['field0'].flatten() / (v['field0'].flatten()**2 + v['field1'].flatten()**2)
    v['field2'].set_flattened(v['field2'].flatten() * scale)

    # Slicing and assignment
    v[2:4, 1] = v[1:3, 1]  # Copy data from one region to another

    # Boolean indexing
    mask = v['field0'].flatten() > 0
    v['field2'].set_flattened(v['field2'].flatten() * mask)

    # Field management
    v.add_fields(('field3', 'field4', 'field5'))  # Add new fields
    v.remove_fields(('field3', 'field4', 'field5'))  # Remove fields

    Direct Data Access:
    ------------------
    # Get data with integer indexing
    data = v.get_data(0, 1)  # Returns numpy array at (0,1)

    # Get data with slice indexing
    data = v.get_data(slice(0, 2), 1)  # Returns list of arrays for rows 0-1 at column 1

    # Set data with integer indexing
    v.set_data(np.array([[1.0, 2.0, 3.0]]), 0, 1)  # Set data at (0,1)

    # Set data with slice indexing
    v.set_data([np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])],
               slice(0, 2), 1)  # Set data for rows 0-1 at column 1

    Notes:
    -----
    - All numpy arrays stored in the vector must have the same number of columns (fields)
    - Field names must be unique
    - Slicing operations return new Vector instances
    - Field operations are performed in-place
    - Units are stored for each field and can be accessed via the units attribute
    - The name attribute can be used to identify the vector in a larger context
    """

    _token = object()

    def __init__(
        self,
        shape: Tuple[int, ...],
        num_fields: Optional[int] = None,
        name: Optional[str] = None,
        fields: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
        _token: object | None = None,
    ) -> None:
        if _token is not self._token:
            raise RuntimeError("Use Vector.from_shape() to instantiate this class.")

        # Initialize attributes
        self._shape = validate_shape(shape)
        self._ndim = len(self._shape)

        if fields is not None:
            self._num_fields = len(fields)
            if num_fields is not None and num_fields != self._num_fields:
                raise ValueError(
                    f"Specified num_fields ({num_fields}) does not match length of fields ({self._num_fields})."
                )
            if len(set(fields)) != len(fields):
                raise ValueError("Duplicate field names are not allowed.")
        elif num_fields is not None:
            self._num_fields = num_fields
        else:
            raise ValueError("Must specify either num_fields or fields.")

        self._name = name or f"{self._ndim}d ragged array"
        self._fields = (
            list(fields)
            if fields is not None
            else [f"field_{i}" for i in range(self._num_fields)]
        )
        self._units = units if units is not None else ["none"] * self._num_fields

        # Initialize empty data structure
        self._data = nested_list(self._shape, fill=None)

    @classmethod
    def from_shape(
        cls,
        shape: Tuple[int, ...],
        num_fields: Optional[int] = None,
        name: Optional[str] = None,
        fields: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
    ) -> "Vector":
        """
        Factory method to create a Vector with the specified shape and fields.

        Parameters
        ----------
        shape : Tuple[int, ...]
            The shape of the vector (dimensions)
        num_fields : Optional[int]
            Number of fields in the vector
        name : Optional[str]
            Name of the vector
        fields : Optional[List[str]]
            List of field names
        units : Optional[List[str]]
            List of units for each field

        Returns
        -------
        Vector
            A new Vector instance
        """
        # Validate inputs
        validated_shape = validate_shape(shape)
        _ndim = len(validated_shape)

        if fields is not None:
            _num_fields = len(fields)
            if num_fields is not None and num_fields != _num_fields:
                raise ValueError(
                    f"Specified num_fields ({num_fields}) does not match length of fields ({_num_fields})."
                )
            if len(set(fields)) != len(fields):
                raise ValueError("Duplicate field names are not allowed.")
        elif num_fields is not None:
            _num_fields = num_fields
        else:
            raise ValueError("Must specify either num_fields or fields.")

        _name = name or f"{_ndim}d ragged array"
        _fields = (
            list(fields)
            if fields is not None
            else [f"field_{i}" for i in range(_num_fields)]
        )
        _units = units if units is not None else ["none"] * _num_fields

        return cls(
            shape=validated_shape,
            num_fields=_num_fields,
            name=_name,
            fields=_fields,
            units=_units,
            _token=cls._token,
        )

    @classmethod
    def from_data(
        cls,
        data: List[Any],
        num_fields: Optional[int] = None,
        name: Optional[str] = None,
        fields: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
    ) -> "Vector":
        """
        Factory method to create a Vector from a list of
        ragged lists or ragged numpy arrays.

        Parameters
        ----------
        data : List[Any]
            A list of ragged lists containing the vector data.
            Each element should be a numpy array with shape (n, num_fields).
        num_fields : Optional[int]
            Number of fields in the vector. If not provided, it will be inferred from the data.
        name : Optional[str]
            Name of the vector
        fields : Optional[List[str]]
            List of field names
        units : Optional[List[str]]
            List of units for each field

        Returns
        -------
        Vector
            A new Vector instance with the provided data

        Raises
        ------
        ValueError
            If the data structure is invalid or inconsistent
        TypeError
            If the data contains invalid types
        """
        # Validate that data is a list
        if not isinstance(data, list):
            raise TypeError("Data must be a list")

        # Validate and determine num_fields
        first_item = data[0]
        if isinstance(first_item, list):
            first_item = np.array(first_item)
        elif not isinstance(first_item, np.ndarray):
            raise TypeError(
                f"Data elements must be numpy arrays or lists, got {type(first_item).__name__}"
            )

        inferred_num_fields = first_item.shape[1]

        # Validate all elements and convert lists to numpy arrays if needed
        for idx, item in enumerate(data):
            if isinstance(item, list):
                data[idx] = np.array(item)
            elif not isinstance(item, np.ndarray):
                raise TypeError(
                    f"Data elements must be numpy arrays or lists, got {type(item).__name__}"
                )

            if data[idx].shape[1] != inferred_num_fields:
                raise ValueError("All arrays must have the same number of fields.")

        num_fields = num_fields or inferred_num_fields

        shape = (len(data),)
        vector = cls.from_shape(
            shape,
            num_fields=num_fields,
            name=name,
            fields=fields,
            units=units,
        )
        vector.data = data

        return vector

    def get_data(self, *indices: Union[int, slice]) -> Union[NDArray, List[NDArray]]:
        """
        Get data at specified indices.

        Parameters:
        -----------
        *indices : int or slice
            Indices to access. Must match the number of dimensions in the vector.

        Returns:
        --------
        numpy.ndarray or list
            The data at the specified indices.

        Raises:
        -------
        IndexError
            If indices are out of bounds.
        ValueError
            If the number of indices does not match the vector dimensions.
        """
        if len(indices) != len(self._shape):
            raise ValueError(f"Expected {len(self._shape)} indices, got {len(indices)}")

        ref: Any = self._data
        for dim, idx in enumerate(indices):
            if isinstance(idx, int) and (idx < 0 or idx >= self._shape[dim]):
                raise IndexError(
                    f"Index {idx} out of bounds for axis {dim} with size {self._shape[dim]}"
                )
            ref = ref[idx]
        return cast(Union[NDArray, List[NDArray]], ref)

    def set_data(self, value: NDArray, *indices: Union[int, slice]) -> None:
        """
        Set data at specified indices.

        Parameters
        ----------
        value : NDArray
            The numpy array to set at the specified indices. Must have shape (_, num_fields).
        *indices : Union[int, slice]
            Indices to set data at. Must match the number of dimensions in the vector.

        Raises
        ------
        IndexError
            If indices are out of bounds.
        ValueError
            If the number of indices does not match the vector dimensions,
            or if the value shape doesn't match the expected shape.
        TypeError
            If the value is not a numpy array.
        """
        if len(indices) != len(self._shape):
            raise ValueError(f"Expected {len(self._shape)} indices, got {len(indices)}")

        ref: Any = self._data
        for dim, idx in enumerate(indices[:-1]):
            if isinstance(idx, int) and (idx < 0 or idx >= self._shape[dim]):
                raise IndexError(
                    f"Index {idx} out of bounds for axis {dim} with size {self._shape[dim]}"
                )
            ref = ref[idx]

        last_idx = indices[-1]
        if isinstance(last_idx, int) and (last_idx < 0 or last_idx >= self._shape[-1]):
            raise IndexError(
                f"Index {last_idx} out of bounds for last axis with size {self._shape[-1]}"
            )

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Value must be a numpy array, got {type(value).__name__}")

        if value.ndim != 2 or value.shape[1] != self._num_fields:
            raise ValueError(
                f"Expected a numpy array with shape (_, {self._num_fields}), got {value.shape}"
            )

        ref[last_idx] = value

    @overload
    def __getitem__(self, idx: str) -> "_FieldView": ...
    @overload
    def __getitem__(
        self, idx: Tuple[Union[int, slice], ...]
    ) -> Union[NDArray, "Vector"]: ...
    @overload
    def __getitem__(self, idx: Union[int, slice]) -> Union[NDArray, "Vector"]: ...

    def __getitem__(
        self, idx: Union[str, Tuple[Union[int, slice], ...], int, slice]
    ) -> Union["_FieldView", NDArray, "Vector"]:
        """
        Get data or a view of the vector at specified indices.

        Parameters
        ----------
        idx : Union[str, Tuple[Union[int, slice], ...], int, slice]
            If str: field name to access
            If tuple: indices for each dimension
            If int/slice: single index or slice

        Returns
        -------
        Union[_FieldView, NDArray, Vector]
            If str: FieldView object for the specified field
            If tuple/int/slice: numpy array or Vector view of the data

        Raises
        ------
        KeyError
            If field name doesn't exist
        IndexError
            If indices are out of bounds
        ValueError
            If the number of indices doesn't match dimensions
        """
        if isinstance(idx, str):  # field-level access
            if idx not in self._fields:
                raise KeyError(f"Field '{idx}' not found.")
            return _FieldView(self, idx)

        if not isinstance(idx, tuple):
            idx = (idx,)

        return_np = True
        for ind in range(min(len(self._shape), len(idx))):
            if not isinstance(idx[ind], int):
                return_np = False
        if len(idx) < len(self._shape):
            return_np = False

        if return_np:
            # Return a view into the numpy array at the user-specified index
            view: Any = self._data
            for i in range(len(idx)):
                view = view[idx[i]]
            return cast(NDArray, view)

        else:
            # Return a view as a new Vector class
            full_idx = list(idx) + [slice(None)] * (len(self._shape) - len(idx))

            def resolve_index(
                dim_idx: Union[int, slice, NDArray[np.bool_], Sequence[int]],
                dim_size: int,
            ) -> List[int]:
                """Convert various index types to a list of integers."""
                if isinstance(dim_idx, slice):
                    return list(range(*dim_idx.indices(dim_size)))  # type: ignore
                elif isinstance(dim_idx, np.ndarray) and dim_idx.dtype == bool:
                    # Convert boolean array to indices
                    return np.flatnonzero(dim_idx).tolist()  # type: ignore
                elif isinstance(dim_idx, (list, np.ndarray)):
                    # Convert sequence to list of integers
                    indices = np.asarray(dim_idx, dtype=np.int64).ravel()
                    return indices.tolist()  # type: ignore
                elif isinstance(dim_idx, int):
                    return [dim_idx]  # type: ignore
                else:
                    # Convert any other sequence to list
                    return [int(i) for i in dim_idx]  # type: ignore

            def recursive_index(
                data: Any,
                dims: Sequence[Union[int, slice, NDArray[np.bool_], Sequence[int]]],
                shape: Tuple[int, ...],
            ) -> Any:
                if len(dims) == 0 or not isinstance(data, list):
                    return data
                i, *rest = dims
                size = shape[0] if shape else len(data)
                idx_list = resolve_index(i, size)
                return [recursive_index(data[j], rest, shape[1:]) for j in idx_list]

            sliced_data = recursive_index(self._data, full_idx, self._shape)

            new_shape: List[int] = []
            for i, s in enumerate(full_idx):
                if i >= len(self._shape):
                    continue
                resolved = resolve_index(s, self._shape[i])
                new_shape.append(len(resolved))

            vector_new = Vector.from_shape(
                shape=tuple(new_shape),
                num_fields=self._num_fields,
                name=self._name + "[view]",
                fields=self._fields,
                units=self._units,
            )
            vector_new._data = sliced_data
            return vector_new

    def __setitem__(
        self,
        idx: Union[str, Tuple[Union[int, slice], ...], int, slice],
        value: Union[NDArray, "_FieldView", "Vector", List[NDArray]],
    ) -> None:
        """
        Set data at specified indices or field.

        Parameters
        ----------
        idx : Union[str, Tuple[Union[int, slice], ...], int, slice]
            If str: field name to set
            If tuple: indices for each dimension
            If int/slice: single index or slice
        value : Union[NDArray, _FieldView, Vector, List[NDArray]]
            The value to set. Must match the expected shape and number of fields.

        Raises
        ------
        KeyError
            If field name doesn't exist
        IndexError
            If indices are out of bounds
        ValueError
            If the number of indices doesn't match dimensions,
            or if the value shape doesn't match expectations
        TypeError
            If the value type is not supported
        """
        if isinstance(idx, str):  # field-level assignment
            if idx not in self._fields:
                raise KeyError(f"Field '{idx}' not found.")
            field_index = self._fields.index(idx)

            if isinstance(value, _FieldView):
                if value.vector is not self:
                    raise ValueError("Cannot assign from another Vector.")
                src_index = value.field_index

                def set_field(arr: Any) -> None:
                    if isinstance(arr, np.ndarray):
                        arr[:, field_index] = arr[:, src_index]
                    elif isinstance(arr, list):
                        for sub in arr:
                            set_field(sub)

                set_field(self._data)
                return

            def set_field(arr: Any) -> None:
                if isinstance(arr, np.ndarray):
                    if callable(value):
                        arr[:, field_index] = value(arr[:, field_index])
                    else:
                        arr[:, field_index] = value
                elif isinstance(arr, list):
                    for sub in arr:
                        set_field(sub)

            set_field(self._data)
            return

        if not isinstance(idx, tuple):
            idx = (idx,)

        def recursive_assign(ref: Any, dims: List[Union[int, slice]], val: Any) -> Any:
            if len(dims) == 0:
                return val
            i, *rest = dims
            if isinstance(i, slice):
                if not isinstance(val, list):
                    raise ValueError(
                        "Slice assignment value must be a list for slice indexing."
                    )
                val_len = len(val)
                ref_len = len(range(*i.indices(len(ref))))
                if val_len != ref_len:
                    raise ValueError(
                        f"Slice assignment mismatch: got {val_len} items, expected {ref_len}."
                    )
                for sub_ref, sub_val in zip(range(*i.indices(len(ref))), val):
                    if (
                        isinstance(sub_val, np.ndarray)
                        and sub_val.shape[1] != self._num_fields
                    ):
                        raise ValueError(
                            f"Expected a numpy array with shape (_, {self._num_fields}), got {sub_val.shape}"
                        )
                    ref[sub_ref] = recursive_assign(ref[sub_ref], rest, sub_val)
                return ref
            else:
                if isinstance(val, np.ndarray) and val.shape[1] != self._num_fields:
                    raise ValueError(
                        f"Expected a numpy array with shape (_, {self._num_fields}), got {val.shape}"
                    )
                ref[i] = recursive_assign(ref[i], rest, val)
                return ref

        if isinstance(value, Vector):
            value = value._data
        recursive_assign(self._data, list(idx), value)

    def add_fields(self, new_fields: Union[str, List[str]]) -> None:
        """
        Add new fields to the vector.

        Parameters
        ----------
        new_fields : Union[str, List[str]]
            Field name(s) to add. Must be unique and not already present.

        Raises
        ------
        ValueError
            If any field name already exists or if there are duplicates
        """
        if isinstance(new_fields, str):
            new_fields = [new_fields]
        else:
            new_fields = list(new_fields)

        if any(name in self._fields for name in new_fields):
            raise ValueError("One or more new field names already exist.")

        if len(set(new_fields)) != len(new_fields):
            raise ValueError("Duplicate field names in input are not allowed.")

        self._fields = list(self._fields) + list(new_fields)
        self._num_fields += len(new_fields)
        self._units = list(self._units) + ["none"] * len(new_fields)

        def expand_array(arr: Any) -> Any:
            if isinstance(arr, np.ndarray):
                if arr.shape[1] != self._num_fields - len(new_fields):
                    raise ValueError(
                        f"Expected arrays with {self._num_fields - len(new_fields)} fields, got {arr.shape[1]}"
                    )
                pad = np.zeros((arr.shape[0], len(new_fields)))
                return np.hstack([arr, pad])
            elif isinstance(arr, list):
                return [expand_array(sub) for sub in arr]
            else:
                return arr

        self._data = expand_array(self._data)

    def remove_fields(self, fields_to_remove: Union[str, List[str]]) -> None:
        """
        Remove fields from the vector.

        Parameters
        ----------
        fields_to_remove : Union[str, List[str]]
            Field name(s) to remove. Must exist in the vector.

        Raises
        ------
        ValueError
            If any field doesn't exist
        """
        if isinstance(fields_to_remove, str):
            fields_to_remove = [fields_to_remove]
        else:
            fields_to_remove = list(fields_to_remove)

        field_to_index = {name: i for i, name in enumerate(self._fields)}
        indices_to_remove = []
        for field in fields_to_remove:
            if field not in field_to_index:
                print(f"Warning: field '{field}' not found.")
            else:
                indices_to_remove.append(field_to_index[field])

        if not indices_to_remove:
            return

        indices_to_remove = sorted(set(indices_to_remove))
        keep_indices = [
            i for i in range(self._num_fields) if i not in indices_to_remove
        ]

        # Update metadata
        self._fields = [self._fields[i] for i in keep_indices]
        self._units = [self._units[i] for i in keep_indices]
        self._num_fields = len(self._fields)

        def prune_array(arr: Any) -> Any:
            if isinstance(arr, np.ndarray):
                if arr.shape[1] < max(indices_to_remove) + 1:
                    raise ValueError(
                        f"Cannot remove field index {max(indices_to_remove)} from array with shape {arr.shape}"
                    )
                return arr[:, keep_indices]
            elif isinstance(arr, list):
                return [prune_array(sub) for sub in arr]
            else:
                return arr

        self._data = prune_array(self._data)

    def copy(self) -> "Vector":
        """
        Create a deep copy of the vector.

        Returns
        -------
        Vector
            A new Vector instance with the same data, shape, fields, and units.
        """
        import copy

        vector_copy = Vector.from_shape(
            shape=self._shape,
            num_fields=self._num_fields,
            name=self._name,
            fields=list(self._fields),
            units=list(self._units),
        )
        vector_copy._data = copy.deepcopy(self._data)
        return vector_copy

    def flatten(self) -> NDArray:
        """
        Flatten the vector into a 2D numpy array.

        Returns
        -------
        NDArray
            A 2D numpy array containing all data, with shape (total_rows, num_fields).
        """

        def collect_arrays(data: Any) -> List[NDArray]:
            if isinstance(data, np.ndarray):
                return [data]
            elif isinstance(data, list):
                arrays = []
                for item in data:
                    arrays.extend(collect_arrays(item))
                return arrays
            else:
                return []

        arrays = collect_arrays(self._data)
        if not arrays:
            return np.empty((0, self._num_fields))
        return np.vstack(arrays)

    def __repr__(self) -> str:
        description = [
            f"quantem.Vector, shape={self._shape}, name={self._name}",
            f"  fields = {self._fields}",
            f"  units: {self._units}",
        ]
        return "\n".join(description)

    def __str__(self) -> str:
        description = [
            f"quantem.Vector, shape={self._shape}, name={self._name}",
            f"  fields = {self._fields}",
            f"  units: {self._units}",
        ]
        return "\n".join(description)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the vector.

        Returns
        -------
        Tuple[int, ...]
            The dimensions of the vector.
        """
        return self._shape

    @shape.setter
    def shape(self, value: Tuple[int, ...]) -> None:
        """
        Set the shape of the vector.

        Parameters
        ----------
        value : Tuple[int, ...]
            The new shape. All dimensions must be positive.

        Raises
        ------
        ValueError
            If any dimension is not positive.
        TypeError
            If value is not a tuple or contains non-integer values.
        """
        self._shape = validate_shape(value)
        self._ndim = len(self._shape)

    @property
    def num_fields(self) -> int:
        """
        Get the number of fields in the vector.

        Returns
        -------
        int
            The number of fields.
        """
        return self._num_fields

    @num_fields.setter
    def num_fields(self, value: int) -> None:
        """
        Set the number of fields in the vector.

        Parameters
        ----------
        value : int
            The new number of fields. Must be positive.

        Raises
        ------
        ValueError
            If value is not positive or doesn't match existing fields.
        TypeError
            If value is not an integer.
        """
        self._num_fields = validate_num_fields(value, self._fields)

    @property
    def name(self) -> str:
        """
        Get the name of the vector.

        Returns
        -------
        str
            The name of the vector
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Set the name of the vector.

        Parameters
        ----------
        value : str
            The new name of the vector
        """
        self._name = value

    @property
    def fields(self) -> List[str]:
        """
        Get the field names of the vector.

        Returns
        -------
        List[str]
            The list of field names.
        """
        return self._fields

    @fields.setter
    def fields(self, value: List[str]) -> None:
        """
        Set the field names of the vector.

        Parameters
        ----------
        value : List[str]
            The new field names. Must match num_fields and be unique.

        Raises
        ------
        ValueError
            If length doesn't match num_fields or if there are duplicates.
        TypeError
            If value is not a list or contains non-string values.
        """
        self._fields = validate_fields(value, self._num_fields)

    @property
    def units(self) -> List[str]:
        """
        Get the units of the vector's fields.

        Returns
        -------
        List[str]
            The list of units, one per field.
        """
        return self._units

    @units.setter
    def units(self, value: List[str]) -> None:
        """
        Set the units of the vector's fields.

        Parameters
        ----------
        value : List[str]
            The new units. Must match num_fields.

        Raises
        ------
        ValueError
            If length doesn't match num_fields.
        TypeError
            If value is not a list or contains non-string values.
        """
        self._units = validate_vector_units(value, self._num_fields)

    @property
    def data(self) -> List[Any]:
        """
        Get the raw data of the vector.

        Returns
        -------
        List[Any]
            The nested list structure containing the vector's data.
        """
        return self._data

    @data.setter
    def data(self, value: List[Any]) -> None:
        """
        Set the raw data of the vector.

        Parameters
        ----------
        value : List[Any]
            The new data structure. Must match the vector's shape and num_fields.

        Raises
        ------
        ValueError
            If the data structure doesn't match shape or num_fields.
        TypeError
            If value is not a list or contains invalid data types.
        """
        self._data = validate_vector_data(value, self._shape, self._num_fields)


# Helper function for nexting lists
def nested_list(shape: Tuple[int, ...], fill: Any = None) -> Any:
    if len(shape) == 0:
        return fill
    return [nested_list(shape[1:], fill) for _ in range(shape[0])]


# Helper class for numerical field operations
class _FieldView:
    def __init__(self, vector: Vector, field_name: str) -> None:
        self.vector = vector
        self.field_name = field_name
        self.field_index = vector._fields.index(field_name)

    def _apply_op(self, op: Any) -> None:
        def apply(arr: Any) -> None:
            if isinstance(arr, np.ndarray):
                arr[:, self.field_index] = op(arr[:, self.field_index])
            elif isinstance(arr, list):
                for sub in arr:
                    apply(sub)

        apply(self.vector._data)

    def __iadd__(self, other: Any) -> "_FieldView":
        self._apply_op(lambda x: x + other)
        return self

    def __isub__(self, other: Any) -> "_FieldView":
        self._apply_op(lambda x: x - other)
        return self

    def __imul__(self, other: Any) -> "_FieldView":
        self._apply_op(lambda x: x * other)
        return self

    def __itruediv__(self, other: Any) -> "_FieldView":
        self._apply_op(lambda x: x / other)
        return self

    def __ipow__(self, other: Any) -> "_FieldView":
        self._apply_op(lambda x: x**other)
        return self

    def flatten(self) -> NDArray:
        def collect(arr: Any) -> List[NDArray]:
            if isinstance(arr, np.ndarray):
                return [arr[:, self.field_index]]
            elif isinstance(arr, list):
                result = []
                for sub in arr:
                    result.extend(collect(sub))
                return result
            else:
                return []

        arrays = collect(self.vector._data)
        if not arrays:
            return np.empty((0,), dtype=float)
        return np.concatenate(arrays, axis=0)

    def set_flattened(self, values: ArrayLike) -> None:
        """
        Set the field values across the entire Vector from a 1D flattened array.
        """

        def fill(arr: Any, values: NDArray, cursor: int) -> int:
            if isinstance(arr, np.ndarray):
                n = arr.shape[0]
                arr[:, self.field_index] = values[cursor : cursor + n]
                return cursor + n
            elif isinstance(arr, list):
                for sub in arr:
                    cursor = fill(sub, values, cursor)
                return cursor
            return cursor

        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("Input to set_flattened must be a 1D array.")

        expected = self.flatten().shape[0]
        if values.shape[0] != expected:
            raise ValueError(f"Expected {expected} values, got {values.shape[0]}")

        fill(self.vector._data, values, cursor=0)

    def __getitem__(
        self, idx: Union[Tuple[Union[int, slice], ...], int, slice]
    ) -> Union[NDArray, "_FieldView"]:
        # Optionally allow v['field0'][0, 1] to get subregion, or v['field0'][...] slice
        sub = self.vector[idx]
        if isinstance(sub, Vector):
            return sub[self.field_name]
        elif isinstance(sub, np.ndarray):
            return sub[:, self.field_index]
        return cast(NDArray, None)

    def __array__(self) -> None:
        raise TypeError(
            "Cannot convert FieldView to array directly. Use `.flatten()` if needed."
        )
