import gzip
import io
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

            elif isinstance(attr_value, torch.optim.Optimizer):
                # Save optimizer state_dict as bytes
                opt_group = group.require_group(attr_name)
                opt_group.attrs["_torch_optimizer"] = True
                opt_group.attrs["class_name"] = attr_value.__class__.__name__

                # Save state_dict as byte array
                buffer = io.BytesIO()
                torch.save(attr_value.state_dict(), buffer)
                buffer.seek(0)
                byte_arr = np.frombuffer(buffer.read(), dtype="uint8")
                opt_group.create_dataset(
                    "state_dict", data=byte_arr, shape=byte_arr.shape, dtype="uint8"
                )

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
                # Fallback: dill + gzip
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
                elif isinstance(v, torch.Tensor):
                    tensor_np = (
                        v.detach().cpu().numpy() if v.requires_grad else v.cpu().numpy()
                    )
                    group.create_dataset(
                        name=key,
                        shape=tensor_np.shape,
                        dtype=tensor_np.dtype,
                        data=tensor_np,
                        compressors=compressors,
                    )
                    group.attrs[f"{key}.torch_save"] = True
                else:
                    group.attrs[key] = v

        elif isinstance(value, dict):
            group.attrs["_container_type"] = "dict"
            for k, v in value.items():
                k = str(k)
                if isinstance(v, (list, tuple, dict)):
                    subgroup = group.require_group(k)
                    self._serialize_container(v, subgroup, compressors)
                elif isinstance(v, np.ndarray):
                    group.create_dataset(
                        name=k,
                        shape=v.shape,
                        dtype=v.dtype,
                        data=v,
                        compressors=compressors,
                    )
                elif isinstance(v, torch.Tensor):
                    tensor_np = (
                        v.detach().cpu().numpy() if v.requires_grad else v.cpu().numpy()
                    )
                    group.create_dataset(
                        name=k,
                        shape=tensor_np.shape,
                        dtype=tensor_np.dtype,
                        data=tensor_np,
                        compressors=compressors,
                    )
                    group.attrs[f"{k}.torch_save"] = True
                else:
                    group.attrs[k] = v

    @classmethod
    def _recursive_load(
        cls,
        group: zarr.Group,
        skip_names: set[str] = frozenset(),
        skip_types: tuple[type, ...] = (),
    ) -> object:
        import io

        import torch

        meta = group.attrs["_autoserialize"]
        version = meta.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported AutoSerialize version: {version}")
        mod = __import__(meta["class_module"], fromlist=[meta["class_name"]])
        cls = getattr(mod, meta["class_name"])
        obj = cls.__new__(cls)

        attrs_fields = getattr(cls, "__attrs_attrs__", None)
        if attrs_fields is not None:
            for f in attrs_fields:
                setattr(obj, f.name, None)

        for attr_name, raw in group.attrs.items():
            if (
                attr_name == "_class_def"
                or attr_name.endswith(".torch_save")
                or attr_name in skip_names
            ):
                continue
            setattr(obj, attr_name, raw)

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

        for subgroup_name in group.group_keys():
            if subgroup_name in skip_names:
                continue
            subgroup = group[subgroup_name]

            if subgroup.attrs.get("_torch_optimizer"):
                # Restore optimizer from saved state_dict
                class_name = subgroup.attrs["class_name"]
                state_bytes = bytes(subgroup["state_dict"][:])
                buffer = io.BytesIO(state_bytes)
                state_dict = torch.load(buffer, map_location="cpu")

                # Attempt to recreate optimizer with dummy param
                opt_class = getattr(torch.optim, class_name)
                dummy_param = torch.zeros(1, requires_grad=True)
                opt = opt_class([dummy_param])
                opt.load_state_dict(state_dict)
                setattr(obj, subgroup_name, opt)

            elif "_autoserialize" in subgroup.attrs:
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

    def print_tree(
        self,
        name: str | None = None,
        depth: int | None = None,
        show_values: bool = True,
        show_autoserialize_types: bool = False,
        show_class_origin: bool = False,
    ) -> None:
        mod_cls = (
            f"{self.__class__.__module__}.{self.__class__.__name__}"
            if show_class_origin
            else self.__class__.__name__
        )
        label = name or self.__class__.__name__
        print(f"{label}: class {mod_cls}")

        def _recurse(val, prefix: str, current_depth: int, is_last: bool = True):
            def make_branch(idx, total):
                last = idx == total - 1
                return ("└── ", "    ") if last else ("├── ", "│   ")

            if isinstance(val, AutoSerialize):
                keys = [
                    k
                    for k in sorted(val.__dict__.keys())
                    if show_autoserialize_types
                    or k not in {"_container_type", "_autoserialize", "_class_def"}
                ]
                for idx, key in enumerate(keys):
                    subval = val.__dict__[key]
                    branch, new_indent = make_branch(idx, len(keys))
                    suffix = ""
                    if show_autoserialize_types and hasattr(subval, "_container_type"):
                        suffix = f" (_container_type = '{getattr(subval, '_container_type', '')}')"
                    if isinstance(subval, AutoSerialize):
                        print(
                            prefix
                            + branch
                            + f"{key}: class {subval.__class__.__name__}{suffix}"
                            if show_class_origin
                            else prefix + branch + f"{key}{suffix}"
                        )
                    elif isinstance(subval, torch.Tensor):
                        print(
                            prefix
                            + branch
                            + f"{key}: torch.Tensor shape={tuple(subval.shape)}"
                        )
                    elif isinstance(subval, np.ndarray):
                        print(
                            prefix
                            + branch
                            + f"{key}: ndarray shape={tuple(subval.shape)}"
                        )
                    elif isinstance(subval, (list, tuple)) and show_autoserialize_types:
                        print(
                            prefix + branch + f"{key}: {type(subval).__name__}{suffix}"
                        )
                    else:
                        val_str = (
                            f" = {repr(subval)}"
                            if show_values
                            and isinstance(subval, (int, float, str, bool, type(None)))
                            else ""
                        )
                        print(
                            prefix + branch + f"{key}: {type(subval).__name__}{val_str}"
                        )
                    if depth is None or current_depth < depth - 1:
                        _recurse(
                            subval,
                            prefix + new_indent,
                            current_depth + 1,
                            idx == len(keys) - 1,
                        )

            elif isinstance(val, (list, tuple)):
                for idx, item in enumerate(val):
                    branch, new_indent = make_branch(idx, len(val))
                    if isinstance(item, torch.Tensor):
                        print(
                            prefix
                            + branch
                            + f"[{idx}]: torch.Tensor shape={tuple(item.shape)}"
                        )
                    elif isinstance(item, np.ndarray):
                        print(
                            prefix
                            + branch
                            + f"[{idx}]: ndarray shape={tuple(item.shape)}"
                        )
                    else:
                        val_str = (
                            f" = {repr(item)}"
                            if show_values
                            and isinstance(item, (int, float, str, bool, type(None)))
                            else ""
                        )
                        print(
                            prefix + branch + f"[{idx}]: {type(item).__name__}{val_str}"
                        )
                    if depth is None or current_depth < depth - 1:
                        _recurse(
                            item,
                            prefix + new_indent,
                            current_depth + 1,
                            idx == len(val) - 1,
                        )

            elif isinstance(val, dict):
                keys = sorted(val.keys())
                for idx, key in enumerate(keys):
                    item = val[key]
                    branch, new_indent = make_branch(idx, len(keys))
                    if isinstance(item, torch.Tensor):
                        print(
                            prefix
                            + branch
                            + f"{repr(key)}: torch.Tensor shape={tuple(item.shape)}"
                        )
                    elif isinstance(item, np.ndarray):
                        print(
                            prefix
                            + branch
                            + f"{repr(key)}: ndarray shape={tuple(item.shape)}"
                        )
                    else:
                        val_str = (
                            f" = {repr(item)}"
                            if show_values
                            and isinstance(item, (int, float, str, bool, type(None)))
                            else ""
                        )
                        print(
                            prefix
                            + branch
                            + f"{repr(key)}: {type(item).__name__}{val_str}"
                        )
                    if depth is None or current_depth < depth - 1:
                        _recurse(
                            item,
                            prefix + new_indent,
                            current_depth + 1,
                            idx == len(keys) - 1,
                        )

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
    path: str | Path,
    depth: int | None = None,
    show_values: bool = True,
    show_autoserialize_types: bool = False,
    show_class_origin: bool = False,
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

    def _recurse(
        obj: Any, prefix: str = "", current_depth: int = 0, is_last: bool = True
    ) -> None:
        if isinstance(obj, zarr.Group):
            keys = sorted(
                set(obj.attrs.keys()) | set(obj.array_keys()) | set(obj.group_keys())
            )
            if current_depth == 0:
                class_info = obj.attrs.get("_autoserialize")
                label = Path(path).name
                if class_info and "class_name" in class_info:
                    mod_cls = (
                        f"{class_info['class_module']}.{class_info['class_name']}"
                        if show_class_origin
                        else class_info["class_name"]
                    )
                    label += f": class {mod_cls}"
                print(label)

            printable_keys = []
            for key in keys:
                if not show_autoserialize_types and key in {
                    "_container_type",
                    "_autoserialize",
                    "_class_def",
                }:
                    continue
                if not show_autoserialize_types and key.endswith(".torch_save"):
                    continue
                printable_keys.append(key)

            for idx, key in enumerate(printable_keys):
                last = idx == len(printable_keys) - 1
                branch = "└── " if last else "├── "
                new_prefix = prefix + ("    " if last else "│   ")

                if key in obj.group_keys():
                    child_group = obj[key]
                    group_type = child_group.attrs.get("_container_type")
                    suffix = (
                        f" (_container_type = '{group_type}')"
                        if group_type and show_autoserialize_types
                        else ""
                    )
                    print(prefix + branch + f"{key}{suffix}")
                    if depth is None or current_depth < depth - 1:
                        _recurse(child_group, new_prefix, current_depth + 1, last)

                elif key in obj.array_keys():
                    arr = obj[key]
                    is_torch = obj.attrs.get(f"{key}.torch_save", False)
                    tensor_type = "torch.Tensor" if is_torch else "ndarray"
                    print(prefix + branch + f"{key}: {tensor_type} shape={arr.shape}")

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

    _recurse(root)
