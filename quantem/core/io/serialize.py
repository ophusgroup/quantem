import base64
import gzip
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, Union
from zipfile import ZipFile

import dill
import numpy as np
import zarr
from zarr.storage import LocalStore

from quantem.core import config

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch


MAX_ATTR_SIZE = 1_000_000  # 1 MB threshold for storing in attrs


# Base class for automatic serialization of classes
class AutoSerialize:
    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: Union[str, type, Sequence[Union[str, type]]] = (),
        compression_level: int | None = 4,
    ) -> None:
        """
        Save the current object to disk using Zarr serialization.

        Parameters
        ----------
        path : str or Path
            Target file path. Use '.zip' extension for zip format, otherwise a directory.
        mode : {'w', 'o'}
            'w' = write only if file doesn't exist, 'o' = overwrite if it does.
        store : {'auto', 'zip', 'dir'}
            Storage format. 'auto' infers from file extension.
        skip : str, type, or list of (str or type)
            Attributes to skip saving by name or type.
        compression_level : int or None
            If set (0–9), applies Zstandard compression with Blosc backend at that level.
            Level 0 disables compression. Raises ValueError if > 9.
        """
        if compression_level is not None:
            if not (0 <= compression_level <= 9):
                raise ValueError(
                    f"compression_level must be between 0 and 9, got {compression_level}"
                )
            compressors = [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": int(compression_level),
                        "shuffle": "bitshuffle",
                    },
                }
            ]
        else:
            compressors = None

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
                raise FileExistsError(f"File '{path}' already exists. Use mode='o' to overwrite.")

        if isinstance(skip, (str, type)):
            skip = [skip]
        skip_names = {s for s in skip if isinstance(s, str)}
        skip_types = tuple(s for s in skip if isinstance(s, type))

        if store == "zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                store_obj = LocalStore(tmpdir)
                root = zarr.group(store=store_obj, overwrite=True)
                self._recursive_save(self, root, skip_names, skip_types, compressors)
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
            self._recursive_save(self, root, skip_names, skip_types, compressors)
        else:
            raise ValueError(f"Unknown store type: {store}")

    def _should_skip(self, value, skip_names, skip_types):
        """Recursively check if value or any nested value should be skipped."""
        if isinstance(value, skip_types):
            return True
        if isinstance(value, dict):
            for k, v in value.items():
                if k in skip_names or self._should_skip(v, skip_names, skip_types):
                    return True
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                if self._should_skip(v, skip_names, skip_types):
                    return True
        return False

    def _recursive_save(
        self,
        obj,
        group: zarr.Group,
        skip_names: set[str] = set(),
        skip_types: tuple[type, ...] = (),
        compressors=None,
    ) -> None:
        if "_class_def" not in group.attrs:
            group.attrs["_class_def"] = dill.dumps(obj.__class__).hex()

        attrs_fields = getattr(obj.__class__, "__attrs_attrs__", None)
        if attrs_fields is not None:
            items = [(field.name, getattr(obj, field.name)) for field in attrs_fields]
        else:
            items = obj.__dict__.items()

        torch_tensor_attrs = []

        for attr_name, attr_value in items:
            if attr_name in skip_names or isinstance(attr_value, skip_types):
                continue

            if isinstance(attr_value, dict):
                filtered_value = {
                    k: v
                    for k, v in attr_value.items()
                    if k not in skip_names and not self._should_skip(v, skip_names, skip_types)
                }
                attr_value = filtered_value

            elif isinstance(attr_value, (list, tuple, set)):
                filtered_value = type(attr_value)(
                    v for v in attr_value if not self._should_skip(v, skip_names, skip_types)
                )
                attr_value = filtered_value

            # converting cupy and torch tensors to numpy arrays for faster saving
            if config.get("has_cupy"):
                if isinstance(attr_value, cp.ndarray):
                    attr_value = cp.asnumpy(attr_value)
            if config.get("has_torch"):
                if isinstance(attr_value, torch.Tensor):
                    attr_value = attr_value.detach().cpu().numpy()
                    torch_tensor_attrs.append(attr_name)
                    # torch tensors are loaded back to torch on cpu

            if isinstance(attr_value, np.ndarray):
                if attr_name not in group:
                    arr = group.create_array(
                        name=attr_name,
                        shape=attr_value.shape,
                        dtype=attr_value.dtype,
                        compressors=compressors,
                    )
                    arr[:] = attr_value
            elif isinstance(attr_value, (int, float, str, bool, type(None))):
                group.attrs[attr_name] = attr_value
            elif isinstance(attr_value, AutoSerialize):
                subgroup = group.require_group(attr_name)
                self._recursive_save(attr_value, subgroup, skip_names, skip_types, compressors)
            else:
                serialized = dill.dumps(attr_value)
                compressed = gzip.compress(serialized)
                if len(compressed) < MAX_ATTR_SIZE:
                    group.attrs[attr_name] = base64.b16encode(compressed).decode("ascii")
                else:
                    ds = group.create_dataset(
                        name=attr_name,
                        shape=(len(compressed),),
                        dtype="uint8",
                        compressors=compressors,
                    )
                    ds[:] = np.frombuffer(compressed, dtype="uint8")

        # Save torch tensor attribute names as metadata
        if torch_tensor_attrs:
            group.attrs["_torch_tensor_attrs"] = torch_tensor_attrs

    @classmethod
    def _recursive_load(
        cls,
        group: zarr.Group,
        skip_names: set[str] = frozenset(),
        skip_types: tuple[type, ...] = (),
    ) -> object:
        """
        Recursively loads an AutoSerialize object and its children.
        """
        # reconstitute the class
        class_def = dill.loads(bytes.fromhex(group.attrs["_class_def"]))
        obj = class_def.__new__(class_def)

        # init attrs-classes if needed
        if hasattr(class_def, "__attrs_post_init__"):
            for f in class_def.__attrs_attrs__:
                setattr(obj, f.name, None)

        # 1) scalar attrs
        for attr_name, raw in group.attrs.items():
            if attr_name == "_class_def" or attr_name in skip_names:
                continue
            val = raw
            if isinstance(val, str):
                try:
                    dec = gzip.decompress(base64.b16decode(val))
                    val = dill.loads(dec)
                except Exception:
                    pass
            setattr(obj, attr_name, val)

        # 2) array datasets
        torch_tensor_attrs = set(group.attrs.get("_torch_tensor_attrs", []))
        for ds_name in group.array_keys():
            if ds_name in skip_names:
                continue
            arr = group[ds_name][:]
            try:
                val = dill.loads(gzip.decompress(arr.tobytes()))
            except Exception:
                val = arr
            # Convert back to torch tensor if needed/possible
            if ds_name in torch_tensor_attrs:
                if config.get("has_torch"):
                    val = torch.from_numpy(val)
            setattr(obj, ds_name, val)

        # 3) sub-groups
        for subgroup_name, subgroup in group.groups():
            # skip by name
            if subgroup_name in skip_names:
                continue
            # peek at its class, skip by type
            sub_cls = dill.loads(bytes.fromhex(subgroup.attrs["_class_def"]))
            if issubclass(sub_cls, skip_types):
                continue
            # otherwise recurse
            setattr(
                obj,
                subgroup_name,
                cls._recursive_load(subgroup, skip_names, skip_types),
            )

        # post-init hook
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
def load(
    path: str | Path,
    skip: Union[str, type, Sequence[Union[str, type]]] = (),
) -> Any:
    """
    Load an AutoSerialize object from disk, optionally skipping attributes
    by name or by type.
    """
    # normalize skip into names vs types
    if isinstance(skip, (str, type)):
        skip = [skip]
    skip_names = {s for s in skip if isinstance(s, str)}
    skip_types = tuple(s for s in skip if isinstance(s, type))

    if os.path.isdir(path):
        store = LocalStore(path)
        root = zarr.group(store=store)
        if "_class_def" not in root.attrs:
            raise KeyError("Missing '_class_def' in Zarr root attrs.")
        class_def = dill.loads(bytes.fromhex(str(root.attrs["_class_def"])))
        return class_def._recursive_load(root, skip_names=skip_names, skip_types=skip_types)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)
            store = LocalStore(tmpdir)
            root = zarr.group(store=store)
            class_def = dill.loads(bytes.fromhex(str(root.attrs["_class_def"])))
            return class_def._recursive_load(root, skip_names=skip_names, skip_types=skip_types)


def print_file(path: str | Path) -> None:
    """Print the saved structure of a serialized object (dir or zip) without loading."""
    if os.path.isdir(path):
        store = LocalStore(path)
    else:
        # Extract zip to temp dir
        tempdir = tempfile.TemporaryDirectory()
        with ZipFile(path, "r") as zf:
            zf.extractall(tempdir.name)
        store = LocalStore(tempdir.name)

    root = zarr.group(store=store)

    def _recurse(group: zarr.Group, prefix: str = "") -> None:
        keys = sorted(set(group.attrs.keys()) | set(group.array_keys()) | set(group.group_keys()))
        for idx, key in enumerate(keys):
            last = idx == len(keys) - 1
            branch = "└── " if last else "├── "
            new_prefix = prefix + ("    " if last else "│   ")

            if key in group.group_keys():
                print(prefix + branch + key)
                _recurse(group[key], new_prefix)
            elif key in group.array_keys():
                arr = group[key]
                print(prefix + branch + f"{key}: ndarray{arr.shape}")
            else:
                val = group.attrs[key]
                if key == "_class_def":
                    print(prefix + branch + f"{key}: class def")
                else:
                    try:
                        print(prefix + branch + f"{key}: {type(val).__name__}")
                    except Exception:
                        print(prefix + branch + f"{key}: <unreadable>")

    print(Path(path).name)
    _recurse(root)
