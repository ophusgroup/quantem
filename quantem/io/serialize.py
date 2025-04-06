# The base class for class serialization

import zarr
import dill
import numpy as np

class AutoSerialize:
    def save(self, path):
        store = zarr.ZipStore(path, mode='w')
        root = zarr.group(store=store)
        self._recursive_save(self, root)
        store.close()

    @classmethod
    def load(cls, path):
        store = zarr.ZipStore(path, mode='r')
        root = zarr.group(store=store)
        obj = cls._recursive_load(root)
        store.close()
        return obj

    def _recursive_save(self, obj, group):
        for attr_name, attr_value in obj.__dict__.items():
            if isinstance(attr_value, np.ndarray):
                group.create_dataset(attr_name, data=attr_value, compressor=zarr.Blosc())
            elif isinstance(attr_value, (int, float, str, bool, type(None))):
                group.attrs[attr_name] = attr_value
            elif isinstance(attr_value, (list, dict, tuple)):
                group.attrs[attr_name] = dill.dumps(attr_value).hex()
            elif hasattr(attr_value, '__dict__'):  # nested objects
                subgroup = group.create_group(attr_name)
                self._recursive_save(attr_value, subgroup)
            else:
                # Methods, functions, other complex objects
                group.attrs[attr_name] = dill.dumps(attr_value).hex()

        # Save the class itself (to reconstruct methods)
        group.attrs['_class_def'] = dill.dumps(obj.__class__).hex()

    @classmethod
    def _recursive_load(cls, group):
        class_def = dill.loads(bytes.fromhex(group.attrs['_class_def']))
        obj = class_def.__new__(class_def)  # Instantiate without calling __init__

        for attr_name in group.attrs:
            if attr_name == '_class_def':
                continue
            attr_value = group.attrs[attr_name]
            try:
                # Attempt dill deserialization (for complex attributes)
                deserialized = dill.loads(bytes.fromhex(attr_value))
                setattr(obj, attr_name, deserialized)
            except (ValueError, dill.UnpicklingError):
                # Simple attributes (int, float, str, bool, None)
                setattr(obj, attr_name, attr_value)

        for subgroup_name, subgroup in group.groups():
            nested_obj = cls._recursive_load(subgroup)
            setattr(obj, subgroup_name, nested_obj)

        for ds_name in group.array_keys():
            setattr(obj, ds_name, group[ds_name][:])

        return obj
