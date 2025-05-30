import gzip
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, Sequence, Union
from zipfile import ZipFile

import dill
import numpy as np
import torch
import zarr
from zarr.storage import LocalStore


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
                raise FileExistsError(
                    f"File '{path}' already exists. Use mode='o' to overwrite."
                )

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

    def _recursive_save(
        self,
        obj,
        group: zarr.Group,
        skip_names: set[str] = set(),
        skip_types: tuple[type, ...] = (),
        compressors=None,
    ) -> None:
        if "_autoserialize" not in group.attrs:
            group.attrs["_autoserialize"] = {
                "version": 1,
                "class_module": obj.__class__.__module__,
                "class_name": obj.__class__.__qualname__,
            }

        attrs_fields = getattr(obj.__class__, "__attrs_attrs__", None)
        if attrs_fields is not None:
            items = [(field.name, getattr(obj, field.name)) for field in attrs_fields]
        else:
            items = obj.__dict__.items()

        for attr_name, attr_value in items:
            if attr_name in skip_names or isinstance(attr_value, skip_types):
                continue

            if isinstance(attr_value, torch.Tensor):
                tensor_np = (
                    attr_value.detach().cpu().numpy()
                    if attr_value.requires_grad
                    else attr_value.cpu().numpy()
                )
                if attr_name not in group:
                    arr = group.create_dataset(
                        name=attr_name,
                        shape=tensor_np.shape,
                        dtype=tensor_np.dtype,
                        compressors=compressors,
                    )
                    arr[:] = tensor_np
                    group.attrs[f"{attr_name}.torch_save"] = True

            elif isinstance(attr_value, np.ndarray):
                if attr_name not in group:
                    arr = group.create_dataset(
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
                self._recursive_save(
                    attr_value, subgroup, skip_names, skip_types, compressors
                )

            elif isinstance(attr_value, (list, tuple, dict)):
                subgroup = group.require_group(attr_name)
                self._serialize_container(attr_value, subgroup, compressors)

            else:
                serialized = dill.dumps(attr_value)
                compressed = gzip.compress(serialized)
                ds = group.create_dataset(
                    name=attr_name,
                    shape=(len(compressed),),
                    dtype="uint8",
                    compressors=compressors,
                )
                ds[:] = np.frombuffer(compressed, dtype="uint8")

    def _serialize_container(self, value, group: zarr.Group, compressors=None):
        if isinstance(value, (list, tuple)):
            group.attrs["_container_type"] = type(value).__name__
            for i, v in enumerate(value):
                key = str(i)
                if isinstance(v, (list, tuple, dict)):
                    subgroup = group.require_group(key)
                    self._serialize_container(v, subgroup, compressors)
                elif isinstance(v, np.ndarray):
                    group.create_dataset(
                        name=key,
                        shape=v.shape,
                        dtype=v.dtype,
                        data=v,
                        compressors=compressors,
                    )
                else:
                    group.attrs[key] = v

        elif isinstance(value, dict):
            group.attrs["_container_type"] = "dict"
            for k, v in value.items():
                if isinstance(v, (list, tuple, dict)):
                    subgroup = group.require_group(str(k))
                    self._serialize_container(v, subgroup, compressors)
                elif isinstance(v, np.ndarray):
                    group.create_dataset(
                        name=str(k),
                        shape=v.shape,
                        dtype=v.dtype,
                        data=v,
                        compressors=compressors,
                    )
                else:
                    group.attrs[str(k)] = v

    @classmethod
    def _recursive_load(
        cls,
        group: zarr.Group,
        skip_names: set[str] = frozenset(),
        skip_types: tuple[type, ...] = (),
    ) -> object:
        import torch

        # Load class metadata and instantiate object
        meta = group.attrs["_autoserialize"]
        version = meta.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported AutoSerialize version: {version}")
        mod = __import__(meta["class_module"], fromlist=[meta["class_name"]])
        cls = getattr(mod, meta["class_name"])
        obj = cls.__new__(cls)

        # Initialize attrs-based classes if present
        attrs_fields = getattr(cls, "__attrs_attrs__", None)
        if attrs_fields is not None:
            for f in attrs_fields:
                setattr(obj, f.name, None)

        # Restore scalar attributes
        for attr_name, raw in group.attrs.items():
            if attr_name == "_class_def" or attr_name in skip_names:
                continue
            if attr_name.endswith(".torch_save"):
                continue  # skip tensor metadata flags
            setattr(obj, attr_name, raw)

        # Restore datasets (arrays or dill objects)
        for ds_name in group.array_keys():
            if ds_name in skip_names:
                continue
            arr = group[ds_name][:]
            try:
                val = dill.loads(gzip.decompress(arr.tobytes()))
            except Exception:
                val = arr
                if group.attrs.get(f"{ds_name}.torch_save", False):
                    val = torch.from_numpy(val)
            setattr(obj, ds_name, val)

        # Restore subgroups
        for subgroup_name in group.group_keys():
            if subgroup_name in skip_names:
                continue
            subgroup = group[subgroup_name]
            if "_autoserialize" in subgroup.attrs:
                meta = subgroup.attrs["_autoserialize"]
                sub_cls = getattr(
                    __import__(meta["class_module"], fromlist=[meta["class_name"]]),
                    meta["class_name"],
                )
                if issubclass(sub_cls, skip_types):
                    continue
                setattr(
                    obj,
                    subgroup_name,
                    sub_cls._recursive_load(subgroup, skip_names, skip_types),
                )
            elif "_container_type" in subgroup.attrs:
                setattr(obj, subgroup_name, cls._deserialize_container(subgroup))

        if hasattr(obj, "__attrs_post_init__"):
            obj.__attrs_post_init__()

        return obj

    @classmethod
    def _deserialize_container(cls, group: zarr.Group) -> Any:
        import torch

        ctype = group.attrs["_container_type"]
        if ctype in ("list", "tuple"):
            int_keys = []
            for k in (
                list(group.attrs.keys())
                + list(group.array_keys())
                + list(group.group_keys())
            ):
                try:
                    int_keys.append(int(k))
                except ValueError:
                    continue
            length = max(int_keys) + 1 if int_keys else 0

            items = []
            for i in range(length):
                key = str(i)
                if key in group.attrs:
                    items.append(group.attrs[key])
                elif key in group.array_keys():
                    arr = group[key][:]
                    val = arr
                    if group.attrs.get(f"{key}.torch_save", False):
                        val = torch.from_numpy(val)
                    items.append(val)
                elif key in group.group_keys():
                    items.append(cls._deserialize_container(group[key]))
                else:
                    raise KeyError(f"Missing expected key '{key}' in container")
            return items if ctype == "list" else tuple(items)

        elif ctype == "dict":
            result = {}
            for key in sorted(group.attrs.keys()):
                if key == "_container_type" or key.endswith(".torch_save"):
                    continue
                result[key] = group.attrs[key]
            for key in group.array_keys():
                arr = group[key][:]
                val = arr
                if group.attrs.get(f"{key}.torch_save", False):
                    val = torch.from_numpy(val)
                result[key] = val
            for key in group.group_keys():
                result[key] = cls._deserialize_container(group[key])
            return result

        else:
            raise ValueError(f"Unknown container type: {ctype}")

    def print_tree(self, name: str | None = None, depth: int | None = None) -> None:
        root_label = name or self.__class__.__name__
        print(root_label)

        def _recurse(val, prefix: str, current_depth: int):
            if isinstance(val, AutoSerialize):
                keys = sorted(val.__dict__.keys())
                for idx, key in enumerate(keys):
                    subval = val.__dict__[key]
                    last = idx == len(keys) - 1
                    branch = "└── " if last else "├── "
                    new_prefix = prefix + ("    " if last else "│   ")
                    print(
                        prefix
                        + branch
                        + key
                        + (
                            ""
                            if isinstance(subval, AutoSerialize)
                            else f": {type(subval).__name__}"
                        )
                    )
                    if depth is None or current_depth < depth - 1:
                        _recurse(subval, new_prefix, current_depth + 1)
            elif isinstance(val, (list, tuple)):
                for idx, item in enumerate(val):
                    last = idx == len(val) - 1
                    branch = "└── " if last else "├── "
                    new_prefix = prefix + ("    " if last else "│   ")
                    print(prefix + branch + f"[{idx}]: {type(item).__name__}")
                    if depth is None or current_depth < depth - 1:
                        _recurse(item, new_prefix, current_depth + 1)
            elif isinstance(val, dict):
                keys = sorted(val.keys())
                for idx, key in enumerate(keys):
                    item = val[key]
                    last = idx == len(keys) - 1
                    branch = "└── " if last else "├── "
                    new_prefix = prefix + ("    " if last else "│   ")
                    print(prefix + branch + f"{repr(key)}: {type(item).__name__}")
                    if depth is None or current_depth < depth - 1:
                        _recurse(item, new_prefix, current_depth + 1)

        _recurse(self, prefix="", current_depth=0)


# Load an autoserialized class
def load(
    path: str | Path,
    skip: Union[str, type, Sequence[Union[str, type]]] = (),
) -> Any:
    if isinstance(skip, (str, type)):
        skip = [skip]
    skip_names = {s for s in skip if isinstance(s, str)}
    skip_types = tuple(s for s in skip if isinstance(s, type))

    if os.path.isdir(path):
        store = LocalStore(path)
    else:
        tempdir = tempfile.TemporaryDirectory()
        with ZipFile(path, "r") as zf:
            zf.extractall(tempdir.name)
        store = LocalStore(tempdir.name)

    root = zarr.group(store=store)
    if "_autoserialize" not in root.attrs:
        raise KeyError("Missing '_autoserialize' metadata in Zarr root attrs.")
    meta = root.attrs["_autoserialize"]
    version = meta.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported AutoSerialize version: {version}")
    mod = __import__(meta["class_module"], fromlist=[meta["class_name"]])
    cls = getattr(mod, meta["class_name"])
    return cls._recursive_load(root, skip_names=skip_names, skip_types=skip_types)


def print_file(
    path: str | Path, depth: int | None = None, show_values: bool = True
) -> None:
    """Print the saved structure of a serialized object (dir or zip) up to a given depth."""
    if os.path.isdir(path):
        store = LocalStore(path)
    else:
        tempdir = tempfile.TemporaryDirectory()
        with ZipFile(path, "r") as zf:
            zf.extractall(tempdir.name)
        store = LocalStore(tempdir.name)

    root = zarr.group(store=store)

    def _recurse(obj: Any, prefix: str = "", current_depth: int = 0) -> None:
        if isinstance(obj, zarr.Group):
            keys = sorted(
                set(obj.attrs.keys()) | set(obj.array_keys()) | set(obj.group_keys())
            )
            for idx, key in enumerate(keys):
                last = idx == len(keys) - 1
                branch = "└── " if last else "├── "
                new_prefix = prefix + ("    " if last else "│   ")

                if key in obj.group_keys():
                    print(prefix + branch + key)
                    if depth is None or current_depth < depth - 1:
                        _recurse(obj[key], new_prefix, current_depth + 1)
                elif key in obj.array_keys():
                    arr = obj[key]
                    print(prefix + branch + f"{key}: ndarray shape={arr.shape}")
                else:
                    val = obj.attrs[key]
                    type_str = type(val).__name__
                    display_val = (
                        f" = {repr(val)}"
                        if show_values
                        and isinstance(val, (int, float, str, bool, type(None)))
                        else ""
                    )
                    print(prefix + branch + f"{key}: {type_str}{display_val}")

                    if isinstance(val, (list, tuple)):
                        for i, item in enumerate(val):
                            last_item = i == len(val) - 1
                            sub_branch = "└── " if last_item else "├── "
                            sub_prefix = new_prefix + ("    " if last_item else "│   ")
                            val_str = (
                                f" = {repr(item)}"
                                if show_values
                                and isinstance(
                                    item, (int, float, str, bool, type(None))
                                )
                                else f" shape={item.shape}"
                                if show_values and isinstance(item, np.ndarray)
                                else ""
                            )
                            print(
                                new_prefix
                                + sub_branch
                                + f"[{i}]: {type(item).__name__}{val_str}"
                            )
                            if depth is None or current_depth < depth - 1:
                                _recurse(item, sub_prefix, current_depth + 2)

                    elif isinstance(val, dict):
                        for i, (k, v) in enumerate(sorted(val.items())):
                            last_item = i == len(val) - 1
                            sub_branch = "└── " if last_item else "├── "
                            sub_prefix = new_prefix + ("    " if last_item else "│   ")
                            val_str = (
                                f" = {repr(v)}"
                                if show_values
                                and isinstance(v, (int, float, str, bool, type(None)))
                                else f" shape={v.shape}"
                                if show_values and isinstance(v, np.ndarray)
                                else ""
                            )
                            print(
                                new_prefix
                                + sub_branch
                                + f"{repr(k)}: {type(v).__name__}{val_str}"
                            )
                            if depth is None or current_depth < depth - 1:
                                _recurse(v, sub_prefix, current_depth + 2)

    print(Path(path).name)
    _recurse(root)
