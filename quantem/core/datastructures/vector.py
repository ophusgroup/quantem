import numpy as np

from quantem.core.io.serialize import AutoSerialize


class Vector(AutoSerialize):
    """
    A class for holding vector data with ragged array lengths. This class supports any number of fixed dimensions
    (indexed first) followed by a ragged numpy array that can have any number of entries (rows) and columns (fields).
    Inherits from AutoSerialize for serialization support.

    Basic Usage:
    -----------
    # Create a 2D vector with shape=(4, 3) and 3 named fields
    v = Vector(shape=(4, 3), fields=['field0', 'field1', 'field2'])

    # Alternative creation with num_fields instead of fields
    v = Vector(shape=(4, 3), num_fields=3)  # Fields will be named field_0, field_1, field_2

    # Create with custom name and units
    v = Vector(
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

    def __init__(
        self,
        shape,
        num_fields=None,
        name=None,
        fields=None,
        units=None,
    ):
        self.shape = shape
        if fields is not None:
            self.num_fields = len(fields)
            if num_fields is not None and num_fields != self.num_fields:
                raise ValueError(
                    f"Specified num_fields ({num_fields}) does not match length of fields ({self.num_fields})."
                )
            if len(set(fields)) != len(fields):
                raise ValueError("Duplicate field names are not allowed.")
        elif num_fields is not None:
            self.num_fields = num_fields
        else:
            raise ValueError("Must specify either num_fields or fields.")
        self.ndim = len(shape)

        self.name = name or f"{self.ndim}d ragged array"
        self.fields = (
            list(fields)
            if fields is not None
            else [f"field_{i}" for i in range(num_fields)]
        )
        self.units = units if units is not None else ["none"] * num_fields

        # initialize empty set of lists
        self.data = nested_list(self.shape, fill=None)

    def get_data(self, *indices):
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
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        ref = self.data
        for dim, idx in enumerate(indices):
            if isinstance(idx, int) and (idx < 0 or idx >= self.shape[dim]):
                raise IndexError(
                    f"Index {idx} out of bounds for axis {dim} with size {self.shape[dim]}"
                )
            ref = ref[idx]
        return ref

    def set_data(self, value, *indices):
        """
        Set data at specified indices.

        Parameters:
        -----------
        value : numpy.ndarray
            The numpy array to set.
        *indices : int or slice
            Indices specifying the location to set data.
            Must match the number of dimensions in the vector.

        Raises:
        -------
        IndexError
            If indices are out of bounds.
        ValueError
            If the number of indices doesn't match the vector dimensions,
            or if the value shape doesn't match the expected shape.
        TypeError
            If the provided value is not a numpy array.
        """
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        ref = self.data
        for dim, idx in enumerate(indices[:-1]):
            if isinstance(idx, int) and (idx < 0 or idx >= self.shape[dim]):
                raise IndexError(
                    f"Index {idx} out of bounds for axis {dim} with size {self.shape[dim]}"
                )
            ref = ref[idx]

        last_idx = indices[-1]
        if isinstance(last_idx, int) and (last_idx < 0 or last_idx >= self.shape[-1]):
            raise IndexError(
                f"Index {last_idx} out of bounds for last axis with size {self.shape[-1]}"
            )

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Value must be a numpy array, got {type(value).__name__}")

        if value.ndim != 2 or value.shape[1] != self.num_fields:
            raise ValueError(
                f"Expected a numpy array with shape (_, {self.num_fields}), got {value.shape}"
            )

        ref[last_idx] = value

    def __getitem__(self, idx):
        if isinstance(idx, str):  # field-level access
            if idx not in self.fields:
                raise KeyError(f"Field '{idx}' not found.")
            return _FieldView(self, idx)

        if not isinstance(idx, tuple):
            idx = (idx,)

        return_np = True
        for ind in range(min(len(self.shape), len(idx))):
            if type(idx[ind]) is not int:
                return_np = False
        if len(idx) < len(self.shape):
            return_np = False

        if return_np:
            # Return a view into the numpy array at the user-specified index
            view = self.data
            for i in range(len(idx)):
                view = view[idx[i]]
            return view

        if return_np:
            # Return a view into the numpy array at the user-specified index
            view = self.data
            for i in range(len(idx)):
                view = view[idx[i]]
            return view

        else:
            # Return a view as a new Vector class
            full_idx = list(idx) + [slice(None)] * (len(self.shape) - len(idx))

            def resolve_index(dim_idx, dim_size):
                if isinstance(dim_idx, slice):
                    return list(range(*dim_idx.indices(dim_size)))
                elif isinstance(dim_idx, np.ndarray) and dim_idx.dtype == bool:
                    return np.where(dim_idx)[0].tolist()
                elif isinstance(dim_idx, (list, np.ndarray)):
                    return list(dim_idx)
                else:
                    return [dim_idx]

            def recursive_index(data, dims, shape):
                if len(dims) == 0 or not isinstance(data, list):
                    return data
                i, *rest = dims
                size = shape[0] if shape else len(data)
                idx_list = resolve_index(i, size)
                return [recursive_index(data[j], rest, shape[1:]) for j in idx_list]

            sliced_data = recursive_index(self.data, full_idx, self.shape)

            new_shape = []
            for i, s in enumerate(full_idx):
                if i >= len(self.shape):
                    continue
                resolved = resolve_index(s, self.shape[i])
                new_shape.append(len(resolved))

            vector_new = Vector(
                shape=tuple(new_shape),
                num_fields=self.num_fields,
                name=self.name + "[view]",
                fields=self.fields,
                units=self.units,
            )
            vector_new.data = sliced_data
            return vector_new

    def __setitem__(self, idx, value):
        if isinstance(idx, str):  # field-level assignment
            if idx not in self.fields:
                raise KeyError(f"Field '{idx}' not found.")
            field_index = self.fields.index(idx)

            if isinstance(value, _FieldView):
                if value.vector is not self:
                    raise ValueError("Cannot assign from another Vector.")
                src_index = value.field_index

                def set_field(arr):
                    if isinstance(arr, np.ndarray):
                        arr[:, field_index] = arr[:, src_index]
                    elif isinstance(arr, list):
                        for sub in arr:
                            set_field(sub)

                set_field(self.data)
                return

            def set_field(arr):
                if isinstance(arr, np.ndarray):
                    if callable(value):
                        arr[:, field_index] = value(arr[:, field_index])
                    else:
                        arr[:, field_index] = value
                elif isinstance(arr, list):
                    for sub in arr:
                        set_field(sub)

            set_field(self.data)
            return

        if not isinstance(idx, tuple):
            idx = (idx,)

        def recursive_assign(ref, dims, val):
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
                        and sub_val.shape[1] != self.num_fields
                    ):
                        raise ValueError(
                            f"Expected a numpy array with shape (_, {self.num_fields}), got {sub_val.shape}"
                        )
                    ref[sub_ref] = recursive_assign(ref[sub_ref], rest, sub_val)
                return ref
            # else:
            #     ref[i] = recursive_assign(ref[i], rest, val)
            #     return ref
            else:
                if isinstance(val, np.ndarray) and val.shape[1] != self.num_fields:
                    raise ValueError(
                        f"Expected a numpy array with shape (_, {self.num_fields}), got {val.shape}"
                    )
                ref[i] = recursive_assign(ref[i], rest, val)
                return ref

        if isinstance(value, Vector):
            value = value.data
        recursive_assign(self.data, list(idx), value)

    def add_fields(self, new_fields):
        if isinstance(new_fields, str):
            new_fields = [new_fields]
        else:
            new_fields = list(new_fields)

        if any(name in self.fields for name in new_fields):
            raise ValueError("One or more new field names already exist.")

        if len(set(new_fields)) != len(new_fields):
            raise ValueError("Duplicate field names in input are not allowed.")

        self.fields = list(self.fields) + list(new_fields)
        self.num_fields += len(new_fields)
        self.units = list(self.units) + ["none"] * len(new_fields)

        def expand_array(arr):
            if isinstance(arr, np.ndarray):
                if arr.shape[1] != self.num_fields - len(new_fields):
                    raise ValueError(
                        f"Expected arrays with {self.num_fields - len(new_fields)} fields, got {arr.shape[1]}"
                    )
                pad = np.zeros((arr.shape[0], len(new_fields)))
                return np.hstack([arr, pad])
            elif isinstance(arr, list):
                return [expand_array(sub) for sub in arr]
            else:
                return arr

        self.data = expand_array(self.data)

    def remove_fields(self, fields_to_remove):
        if isinstance(fields_to_remove, str):
            fields_to_remove = [fields_to_remove]
        else:
            fields_to_remove = list(fields_to_remove)

        field_to_index = {name: i for i, name in enumerate(self.fields)}
        indices_to_remove = []
        for field in fields_to_remove:
            if field not in field_to_index:
                print(f"Warning: field '{field}' not found.")
            else:
                indices_to_remove.append(field_to_index[field])

        if not indices_to_remove:
            return

        indices_to_remove = sorted(set(indices_to_remove))
        keep_indices = [i for i in range(self.num_fields) if i not in indices_to_remove]

        # Update metadata
        self.fields = [self.fields[i] for i in keep_indices]
        self.units = [self.units[i] for i in keep_indices]
        self.num_fields = len(self.fields)

        def prune_array(arr):
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

        self.data = prune_array(self.data)

    def copy(self):
        import copy

        vector_copy = Vector(
            shape=self.shape,
            num_fields=self.num_fields,
            name=self.name,
            fields=list(self.fields),
            units=list(self.units),
        )
        vector_copy.data = copy.deepcopy(self.data)
        return vector_copy

    def flatten(self):
        """
        Flatten and stack all non-None numpy arrays along axis 0.
        """

        def collect_arrays(data):
            if isinstance(data, np.ndarray):
                return [data]
            elif isinstance(data, list):
                arrays = []
                for item in data:
                    arrays.extend(collect_arrays(item))
                return arrays
            else:
                return []

        arrays = collect_arrays(self.data)
        if not arrays:
            return np.empty((0, self.num_fields))
        return np.vstack(arrays)

    def __repr__(self):
        description = [
            f"quantem.Vector, shape={self.shape}, name={self.name}",
            f"  fields = {self.fields}",
            f"  units: {self.units}",
        ]
        return "\n".join(description)

    def __str__(self):
        description = [
            f"quantem.Vector, shape={self.shape}, name={self.name}",
            f"  fields = {self.fields}",
            f"  units: {self.units}",
        ]
        return "\n".join(description)


# Helper function for nexting lists
def nested_list(shape, fill=None):
    if len(shape) == 0:
        return fill
    return [nested_list(shape[1:], fill) for _ in range(shape[0])]


# Helper class for numerical field operations
class _FieldView:
    def __init__(self, vector, field_name):
        self.vector = vector
        self.field_name = field_name
        self.field_index = vector.fields.index(field_name)

    def _apply_op(self, op):
        def apply(arr):
            if isinstance(arr, np.ndarray):
                arr[:, self.field_index] = op(arr[:, self.field_index])
            elif isinstance(arr, list):
                for sub in arr:
                    apply(sub)

        apply(self.vector.data)

    def __iadd__(self, other):
        self._apply_op(lambda x: x + other)
        return self

    def __isub__(self, other):
        self._apply_op(lambda x: x - other)
        return self

    def __imul__(self, other):
        self._apply_op(lambda x: x * other)
        return self

    def __itruediv__(self, other):
        self._apply_op(lambda x: x / other)
        return self

    def __ipow__(self, other):
        self._apply_op(lambda x: x**other)
        return self

    def flatten(self):
        def collect(arr):
            if isinstance(arr, np.ndarray):
                return [arr[:, self.field_index]]
            elif isinstance(arr, list):
                result = []
                for sub in arr:
                    result.extend(collect(sub))
                return result
            else:
                return []

        arrays = collect(self.vector.data)
        if not arrays:
            return np.empty((0,), dtype=float)
        return np.concatenate(arrays, axis=0)

    def set_flattened(self, values):
        """
        Set the field values across the entire Vector from a 1D flattened array.
        """

        def fill(arr, values, cursor):
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

        fill(self.vector.data, values, cursor=0)

    def __getitem__(self, idx):
        # Optionally allow v['field0'][0, 1] to get subregion, or v['field0'][...] slice
        sub = self.vector[idx]
        if isinstance(sub, Vector):
            return sub[self.field_name]
        elif isinstance(sub, np.ndarray):
            return sub[:, self.field_index]
        return None

    def __array__(self):
        raise TypeError(
            "Cannot convert FieldView to array directly. Use `.flatten()` if needed."
        )
