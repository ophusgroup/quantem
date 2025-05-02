import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, Sequence
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
        skip: Sequence[str] = (),
    ) -> None:
        """
        Save this object, optionally skipping any attributes whose names appear in `skip`.
        """
        path = str(path)
        skip_set = set(skip)

        # decide zip vs dir
        if store == "auto":
            store = "zip" if path.endswith(".zip") else "dir"

        # ensure .zip extension if needed
        if store == "zip" and not path.endswith(".zip"):
            print(f"Warning: appending .zip to path '{path}'")
            path += ".zip"

        # handle overwrite
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
            with tempfile.TemporaryDirectory() as tmpdir:
                store_obj = LocalStore(tmpdir)
                root = zarr.group(store=store_obj, overwrite=True)
                self._recursive_save(self, root, skip_set)
                with ZipFile(path, mode="w") as zf:
                    for dirpath, _, filenames in os.walk(tmpdir):
                        for filename in filenames:
                            full_path = os.path.join(dirpath, filename)
                            rel_path = os.path.relpath(full_path, tmpdir)
                            zf.write(full_path, arcname=rel_path)

        elif store == "dir":
            if os.path.splitext(path)[1]:
                raise ValueError(
                    f"Expected a directory path for store='dir', but got '{path}'"
                )
            os.makedirs(path, exist_ok=True)
            store_obj = LocalStore(path)
            root = zarr.group(store=store_obj, overwrite=True)
            self._recursive_save(self, root, skip_set)

        else:
            raise ValueError(f"Unknown store type: {store}")

    def _recursive_save(
        self,
        obj: "AutoSerialize",
        group: zarr.Group,
        skip_set: set[str],
    ) -> None:
        # write class definition once
        if "_class_def" not in group.attrs:
            group.attrs["_class_def"] = dill.dumps(obj.__class__).hex()

        # gather items
        attrs_fields = getattr(obj.__class__, "__attrs_attrs__", None)
        if attrs_fields is not None:
            items = [(field.name, getattr(obj, field.name)) for field in attrs_fields]
        else:
            items = obj.__dict__.items()

        for attr_name, attr_value in items:
            # skip if requested
            if attr_name in skip_set:
                continue

            # numpy arrays
            if isinstance(attr_value, np.ndarray):
                if attr_name not in group:
                    arr = group.create_array(
                        name=attr_name,
                        shape=attr_value.shape,
                        dtype=attr_value.dtype,
                    )
                    arr[...] = attr_value

            # primitives
            elif isinstance(attr_value, (int, float, str, bool, type(None))):
                group.attrs[attr_name] = attr_value

            # nested AutoSerialize
            elif isinstance(attr_value, AutoSerialize):
                # skip entire subtree if name in skip_set
                if attr_name in skip_set:
                    continue
                if attr_name in group:
                    subgroup = group[attr_name]
                    if not isinstance(subgroup, zarr.Group):
                        subgroup = group.create_group(attr_name)
                else:
                    subgroup = group.create_group(attr_name)
                self._recursive_save(attr_value, subgroup, skip_set)

            # anything else
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

    def print_tree(self, name: str | None = None) -> None:
        root_label = name or self.__class__.__name__
        print(root_label)

        def _recurse(obj, prefix=""):
            # sort the keys so they print alphabetically
            keys = sorted(obj.__dict__.keys())
            for idx, key in enumerate(keys):
                val = obj.__dict__[key]
                last = idx == len(keys) - 1
                branch = "└── " if last else "├── "
                if isinstance(val, AutoSerialize):
                    print(prefix + branch + key)
                    _recurse(val, prefix + ("    " if last else "│   "))
                else:
                    print(prefix + branch + f"{key}: {type(val).__name__}")

        _recurse(self)


# Load an autoserialized class
def load(path: str, skip: Sequence[str] = ()) -> Any:
    """
    Load an AutoSerialize object from a directory or .zip file,
    optionally skipping any attributes or subgroups named in `skip`.
    """
    skip_set = set(skip)
    skip_set.discard("_class_def")  # never skip the class definition

    def _prune(group: zarr.Group) -> None:
        # Remove any attrs or arrays/groups at this level whose name is in skip_set
        for key in list(skip_set):
            if key in group.attrs:
                del group.attrs[key]
            if key in group:
                del group[key]
        # Recurse into remaining subgroups
        for subname, subgroup in group.groups():
            _prune(subgroup)

    # Open the zarr tree
    if os.path.isdir(path):
        store = LocalStore(path)
        root = zarr.group(store=store)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)
            store = LocalStore(tmpdir)
            root = zarr.group(store=store)

    # Prune out skipped names at every level
    _prune(root)

    # Reconstruct the object
    if "_class_def" not in root.attrs:
        raise KeyError("Missing '_class_def' in Zarr root attributes.")
    cls = dill.loads(bytes.fromhex(root.attrs["_class_def"]))
    return cls._recursive_load(root)
