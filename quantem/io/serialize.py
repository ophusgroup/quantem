# Base class for serializing data

import os
import dill
import zarr
import numpy as np
# import numcodecs
import tempfile
import shutil
from zipfile import ZipFile
from zarr.storage import LocalStore

class AutoSerialize:
    def save(self, path):
        if os.path.exists(path):
            os.remove(path)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(tmpdir)
            root = zarr.group(store=store, overwrite=True)
            self._recursive_save(self, root)
            with ZipFile(path, mode='w') as zf:
                for dirpath, _, filenames in os.walk(tmpdir):
                    for filename in filenames:
                        full_path = os.path.join(dirpath, filename)
                        rel_path = os.path.relpath(full_path, tmpdir)
                        zf.write(full_path, arcname=rel_path)

    @classmethod
    def load(cls, path):
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(path, 'r') as zf:
                zf.extractall(tmpdir)
            store = LocalStore(tmpdir)
            root = zarr.group(store=store)
            obj = cls._recursive_load(root)
            return obj

    def _recursive_save(self, obj, group):
        if '_class_def' not in group.attrs:
            group.attrs['_class_def'] = dill.dumps(obj.__class__).hex()

        for attr_name, attr_value in obj.__dict__.items():
            if isinstance(attr_value, np.ndarray):
                if attr_name not in group:
                    group.create_dataset(
                        name=attr_name,
                        data=attr_value,
                        shape=attr_value.shape,
                        dtype=attr_value.dtype,
                        # compressors=[numcodecs.Blosc()]
                    )
            elif isinstance(attr_value, (int, float, str, bool, type(None))):
                group.attrs[attr_name] = attr_value
            elif isinstance(attr_value, AutoSerialize):
                if attr_name in group:
                    subgroup = group[attr_name]
                else:
                    subgroup = group.create_group(attr_name)
                self._recursive_save(attr_value, subgroup)
            else:
                group.attrs[attr_name] = dill.dumps(attr_value).hex()

    @classmethod
    def _recursive_load(cls, group):
        class_def = dill.loads(bytes.fromhex(group.attrs['_class_def']))
        obj = class_def.__new__(class_def)

        for attr_name in group.attrs:
            if attr_name == '_class_def':
                continue
            try:
                deserialized = dill.loads(bytes.fromhex(group.attrs[attr_name]))
                setattr(obj, attr_name, deserialized)
            except Exception:
                setattr(obj, attr_name, group.attrs[attr_name])

        for ds_name in group.array_keys():
            setattr(obj, ds_name, group[ds_name][:])

        for subgroup_name, subgroup in group.groups():
            setattr(obj, subgroup_name, cls._recursive_load(subgroup))

        return obj


# load serialized files without instantiating AutoSerialize
def load(path):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(path, 'r') as zf:
            zf.extractall(tmpdir)
        store = LocalStore(tmpdir)
        root = zarr.group(store=store)
        class_def = dill.loads(bytes.fromhex(root.attrs['_class_def']))
        return class_def._recursive_load(root)