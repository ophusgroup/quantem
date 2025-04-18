import numpy as np

from quantem.core.io.serialize import AutoSerialize


class Vector(AutoSerialize):
    """
    base class for holding vector data, with ragged array lengths.
    This class has any number of fixed dimensions (which are indexed first)
    followed by a ragged numpy array, which can have any number of entries (rows)
    and columns (fields).
    """

    def __init__(
        self,
        shape,
        num_fields=None,
        name=None,
        field_names=None,
        origin=None,
        sampling=None,
        units=None,
    ):
        self.shape = shape
        if field_names is not None:
            self.num_fields = len(field_names)
            if num_fields is not None and num_fields != self.num_fields:
                raise ValueError(
                    f"Specified num_fields ({num_fields}) does not match length of field_names ({self.num_fields})."
                )
        elif num_fields is not None:
            self.num_fields = num_fields
        else:
            raise ValueError("Must specify either num_fields or field_names.")
        self.ndim = len(shape) + 2

        self.name = name or f"{self.ndim}d ragged array"
        self.field_names = (
            list(field_names)
            if field_names is not None
            else [f"field_{i}" for i in range(num_fields)]
        )
        self.origin = np.array(origin) if origin is not None else np.zeros(num_fields)
        self.sampling = (
            np.array(sampling) if sampling is not None else np.ones(num_fields)
        )
        self.units = units if units is not None else ["pixels"] * num_fields

        # initialize empty set of lists
        self.data = nested_list(self.shape, fill=None)

    def get_data(self, *indices):
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        ref = self.data
        for idx in indices:
            ref = ref[idx]
        return ref

    def set_data(self, *indices_and_value):
        value, *indices = indices_and_value
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        ref = self.data
        for idx in indices[:-1]:
            ref = ref[idx]
        if not isinstance(value, np.ndarray) or value.shape[1] != self.num_fields:
            raise ValueError(
                f"Expected a numpy array with shape (_, {self.num_fields}), got {value.shape if isinstance(value, np.ndarray) else type(value)}"
            )
        ref[indices[-1]] = value

    # get_data and set_data allowing slices:
    # def get_data(self, *indices):
    #     if len(indices) != len(self.shape):
    #         raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

    #     def recursive_get(ref, idx_list):
    #         if not idx_list:
    #             return ref
    #         idx, *rest = idx_list
    #         if isinstance(idx, slice):
    #             return [recursive_get(ref[i], rest) for i in range(*idx.indices(len(ref)))]
    #         else:
    #             return recursive_get(ref[idx], rest)

    #     return recursive_get(self.data, indices)

    # def set_data(self, *indices_and_value):
    #     value, *indices = indices_and_value
    #     if len(indices) != len(self.shape):
    #         raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")

    #     def recursive_set(ref, idx_list, val):
    #         idx, *rest = idx_list
    #         if not rest:
    #             if isinstance(idx, slice):
    #                 expected_length = len(range(*idx.indices(len(ref))))
    #                 if len(val) != expected_length:
    #                     raise ValueError(f"Mismatch: expected {expected_length} items, got {len(val)}")
    #                 for i, v in zip(range(*idx.indices(len(ref))), val):
    #                     if not isinstance(v, np.ndarray) or v.shape[1] != self.num_fields:
    #                         raise ValueError(f"Expected shape (_, {self.num_fields}), got {v.shape}")
    #                     ref[i] = v
    #             else:
    #                 if not isinstance(val, np.ndarray) or val.shape[1] != self.num_fields:
    #                     raise ValueError(f"Expected shape (_, {self.num_fields}), got {val.shape}")
    #                 ref[idx] = val
    #         else:
    #             if isinstance(idx, slice):
    #                 expected_length = len(range(*idx.indices(len(ref))))
    #                 if len(val) != expected_length:
    #                     raise ValueError(f"Mismatch: expected {expected_length} items, got {len(val)}")
    #                 for i, v in zip(range(*idx.indices(len(ref))), val):
    #                     recursive_set(ref[i], rest, v)
    #             else:
    #                 recursive_set(ref[idx], rest, val)

    #     recursive_set(self.data, indices, value)

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

    def add_fields(self, new_field_names):
        if isinstance(new_field_names, str):
            new_field_names = [new_field_names]
        else:
            new_field_names = list(new_field_names)

        if any(name in self.field_names for name in new_field_names):
            raise ValueError("One or more new field names already exist.")

        self.field_names = list(self.field_names) + list(new_field_names)
        self.num_fields += len(new_field_names)
        self.origin = np.append(self.origin, np.zeros(len(new_field_names)))
        self.sampling = np.append(self.sampling, np.ones(len(new_field_names)))
        self.units = list(self.units) + ["pixels"] * len(new_field_names)

        def expand_array(arr):
            if isinstance(arr, np.ndarray):
                if arr.shape[1] != self.num_fields - len(new_field_names):
                    raise ValueError(
                        f"Expected arrays with {self.num_fields - len(new_field_names)} fields, got {arr.shape[1]}"
                    )
                pad = np.zeros((arr.shape[0], len(new_field_names)))
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

        field_to_index = {name: i for i, name in enumerate(self.field_names)}
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
        self.field_names = [self.field_names[i] for i in keep_indices]
        self.origin = self.origin[keep_indices]
        self.sampling = self.sampling[keep_indices]
        self.units = [self.units[i] for i in keep_indices]
        self.num_fields = len(self.field_names)

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
            field_names=list(self.field_names),
            origin=np.copy(self.origin),
            sampling=np.copy(self.sampling),
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
            f"quantem Vector, shape={self.shape}, name={self.name}",
            f"  fields = {self.field_names}",
        ]
        return "\n".join(description)

    def __str__(self):
        description = [
            f"quantem Vector, shape={self.shape}, name={self.name}",
            f"  fields = {self.field_names}",
            f"  units: {self.units}",
            f"  origin: {self.origin}",
            f"  sampling: {self.sampling}",
        ]
        return "\n".join(description)


def nested_list(shape, fill=None):
    if len(shape) == 0:
        return fill
    return [nested_list(shape[1:], fill) for _ in range(shape[0])]
