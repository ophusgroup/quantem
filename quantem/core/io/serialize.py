import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import dill
import numpy as np
import zarr
from zarr.storage import LocalStore


# Base class for automatic serialization of classes
class AutoSerialize:
    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
    ) -> None:
        path = str(path)
        if store == "auto":
            store = "zip" if path.endswith(".zip") else "dir"

        if store == "zip" and not path.endswith(".zip"):
            print(f"Warning: appending .zip to path '{path}'")
            path += ".zip"

        if os.path.exists(path):
            if mode == "o":
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            else:
                raise FileExistsError(
                    f"File '{path}' already exists. Use mode='o' to overwrite."
                )

        if store == "zip":
            if not path.endswith(".zip"):
                print(f"Warning: appending .zip to path '{path}'")
                path += ".zip"
            with tempfile.TemporaryDirectory() as tmpdir:
                store_obj = LocalStore(tmpdir)
                root = zarr.group(store=store_obj, overwrite=True)
                self._recursive_save(self, root)
                with ZipFile(path, mode="w") as zf:
                    for dirpath, _, filenames in os.walk(tmpdir):
                        for filename in filenames:
                            full_path = os.path.join(dirpath, filename)
                            rel_path = os.path.relpath(full_path, tmpdir)
                            zf.write(full_path, arcname=rel_path)
        elif store == "dir":
            if os.path.splitext(path)[1]:
                raise ValueError(
                    f"Expected a directory path for store='dir', but got file-like path '{path}'"
                )
            os.makedirs(path, exist_ok=True)
            store_obj = LocalStore(path)
            root = zarr.group(store=store_obj, overwrite=True)
            self._recursive_save(self, root)
        else:
            raise ValueError(f"Unknown store type: {store}")

    def _recursive_save(self, obj: "AutoSerialize", group: zarr.Group) -> None:
        if "_class_def" not in group.attrs:
            group.attrs["_class_def"] = dill.dumps(obj.__class__).hex()

        # Get attributes either from __dict__ or attrs fields
        attrs_fields = getattr(obj.__class__, "__attrs_attrs__", None)
        if attrs_fields is not None:
            items = [(field.name, getattr(obj, field.name)) for field in attrs_fields]
        else:
            items = obj.__dict__.items()

        for attr_name, attr_value in items:
            if isinstance(attr_value, np.ndarray):
                if attr_name not in group:
                    arr = group.create_array(
                        name=attr_name,
                        shape=attr_value.shape,
                        dtype=attr_value.dtype,
                    )
                    arr[:] = attr_value

            elif isinstance(attr_value, (int, float, str, bool, type(None))):
                group.attrs[attr_name] = attr_value
            elif isinstance(attr_value, AutoSerialize):
                if attr_name in group:
                    subgroup = group[attr_name]
                    if not isinstance(subgroup, zarr.Group):
                        subgroup = group.create_group(attr_name)
                else:
                    subgroup = group.create_group(attr_name)
                self._recursive_save(attr_value, subgroup)
            else:
                group.attrs[attr_name] = dill.dumps(attr_value).hex()

    @classmethod
    def _recursive_load(cls, group) -> Any:
        class_def = dill.loads(bytes.fromhex(group.attrs["_class_def"]))
        obj = class_def.__new__(class_def)

        # Initialize attrs classes if needed
        if hasattr(class_def, "__attrs_post_init__"):
            # Set all fields to None initially
            for field in class_def.__attrs_attrs__:
                setattr(obj, field.name, None)

        for attr_name in group.attrs:
            if attr_name == "_class_def":
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

        # Call post_init for attrs classes if it exists
        if hasattr(obj, "__attrs_post_init__"):
            obj.__attrs_post_init__()

        return obj


# Load an autoserialized class
def load(path) -> Any:
    if os.path.isdir(path):
        store = LocalStore(path)
        root = zarr.group(store=store)
        if "_class_def" not in root.attrs:
            raise KeyError(
                "Missing '_class_def' in Zarr root attributes. This directory may not have been saved using AutoSerialize."
            )
        class_def = dill.loads(bytes.fromhex(str(root.attrs["_class_def"])))
        return class_def._recursive_load(root)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)
            store = LocalStore(tmpdir)
            root = zarr.group(store=store)
            class_def = dill.loads(bytes.fromhex(str(root.attrs["_class_def"])))
            return class_def._recursive_load(root)
