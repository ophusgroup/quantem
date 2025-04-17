import numpy as np

from quantem.core.io.serialize import AutoSerialize


class Vector(AutoSerialize):
    """
    Class for holding vector data, with ragged array lengths.
    This class has any number of fixed dimensions (which are indexed first)
    followed by a ragged numpy array, which can have any number of entries (rows)
    and columns (fields).
    """

    def __init__(
        self,
        shape,
        num_fields,
        name=None,
        field_names=None,
        origin=None,
        sampling=None,
        units=None,
    ):
        self.shape = shape
        self.num_fields = num_fields
        self.ndim = len(shape) + 2

        self.name = name or f"{self.ndim}d Vector"
        self.field_names = field_names or [f"field_{i}" for i in range(num_fields)]
        self.origin = np.array(origin) if origin is not None else np.zeros(num_fields)
        self.sampling = (
            np.array(sampling) if sampling is not None else np.ones(num_fields)
        )
        self.units = units if units is not None else ["pixels"] * num_fields

        # initialize empty set of lists
        self.data = nested_list(self.shape, fill=None)

    def set_data(self, *indices_and_value):
        value, *indices = indices_and_value
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        ref = self.data
        for idx in indices[:-1]:
            ref = ref[idx]
        ref[indices[-1]] = value

    # def get_data(self, *indices):
    #     if len(indices) != len(self.shape):
    #         raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

    #     ref = self.data
    #     for idx in indices:
    #         ref = ref[idx]
    #     return ref

    def __getitem__(self, idx):
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
                field_names=self.field_names,
                origin=self.origin,
                sampling=self.sampling,
                units=self.units,
            )
            vector_new.data = sliced_data
            return vector_new

    def __setitem__(self, idx, value):
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
                    ref[sub_ref] = recursive_assign(ref[sub_ref], rest, sub_val)
                return ref
            else:
                ref[i] = recursive_assign(ref[i], rest, val)
                return ref

        if isinstance(value, Vector):
            value = value.data
        recursive_assign(self.data, list(idx), value)

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

    def __str__(self):
        return f"Vector(name={self.name}, shape={self.shape}, num_fields={self.num_fields})"


def nested_list(shape, fill=None):
    if len(shape) == 0:
        return fill
    return [nested_list(shape[1:], fill) for _ in range(shape[0])]
