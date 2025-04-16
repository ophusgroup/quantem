from __future__ import annotations

import ast
import os
import threading
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Union

import yaml

no_default = "__no_default__"

PATH = Path(os.getenv("QUANTUM_CONFIG", "~/.config/quantem")).expanduser().resolve()
paths = [PATH]
config: dict = {}
deprecations: dict[str, str | None] = {}

config_lock = threading.Lock()

defaults: list[Mapping] = []

# TODO - add a write option


class set:
    """Temporarily set configuration values within a context manager

    Parameters
    ----------
    arg : mapping or None, optional
        A mapping of configuration key-value pairs to set.
    **kwargs :
        Additional key-value pairs to set. If ``arg`` is provided, values set
        in ``arg`` will be applied before those in ``kwargs``.
        Double-underscores (``__``) in keyword arguments will be replaced with
        ``.``, allowing nested values to be easily set.
    """

    # _record: list[tuple[Literal["insert", "replace"], tuple[str, ...], Any]]

    def __init__(
        self,
        arg: Union[Mapping, None] = None,
        config: dict = config,
        **kwargs,
    ):
        self.config: dict = config
        self._record = []

        if arg is not None:
            for key, value in arg.items():
                key = check_deprecations(key)
                self._assign(key.split("."), value, config)
        if kwargs:
            for key, value in kwargs.items():
                key = key.replace("__", ".")
                key = check_deprecations(key)
                self._assign(key.split("."), value, config)

    def __enter__(self):
        return self.config

    # def __exit__(self, type, value, traceback):
    #     for op, path, value in reversed(self._record):
    #         d = self.config
    #         if op == "replace":
    #             for key in path[:-1]:
    #                 d = d.setdefault(key, {})
    #             d[path[-1]] = value
    #         else:  # insert
    #             for key in path[:-1]:
    #                 try:
    #                     d = d[key]
    #                 except KeyError:
    #                     break
    #             else:
    #                 d.pop(path[-1], None)

    def _assign(
        self,
        keys: Sequence[str],
        value: Any,
        d: dict,
        path: tuple[str, ...] = (),
        record: bool = True,
    ) -> None:
        """Assign value into a nested configuration dictionary

        Parameters
        ----------
        keys : Sequence[str]
            The nested path of keys to assign the value.
        value : object
        d : dict
            The part of the nested dictionary into which we want to assign the
            value
        path : tuple[str], optional
            The path history up to this point.
        record : bool, optional
            Whether this operation needs to be recorded to allow for rollback.
        """
        key = canonical_name(keys[0], d)

        path = path + (key,)

        if len(keys) == 1:
            if record:
                if key in d:
                    self._record.append(("replace", path, d[key]))
                else:
                    self._record.append(("insert", path, None))
            d[key] = value
        else:
            if key not in d:
                if record:
                    self._record.append(("insert", path, None))
                d[key] = {}
                # No need to record subsequent operations after an insert
                record = False
            self._assign(keys[1:], value, d[key], path, record=record)


def refresh(
    config: dict = config, defaults: list[Mapping] = defaults, **kwargs
) -> None:
    """
    Update configuration by re-reading yaml files and env variables

    This mutates the global quantem.config.config, or the config parameter if
    passed in.

    This goes through the following stages:

    1.  Clearing out all old configuration
    2.  Updating from the stored defaults from downstream libraries
        (see update_defaults)
    3.  Updating from yaml files and environment variables

    Note that some functionality only checks configuration once at startup and
    may not change behavior, even if configuration changes.  It is recommended
    to restart your python process if convenient to ensure that new
    configuration changes take place.

    See Also
    --------
    quantem.config.collect: for parameters
    quantem.config.update_defaults
    """
    config.clear()

    for d in defaults:
        update(config, d, priority="new")

    update(config, collect(**kwargs))


def get(
    key: str,
    default: Any = no_default,
    config: dict = config,
    override_with: Any = None,
) -> Any:
    """
    Get elements from global config

    If ``override_with`` is not None this value will be passed straight back.
    Useful for getting kwarg defaults from abtek config.

    Use '.' for nested access
    """
    if override_with is not None:
        return override_with
    keys = key.split(".")
    result = config
    for k in keys:
        k = canonical_name(k, result)
        try:
            result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


def update_defaults(
    new: Mapping, config: dict = config, defaults: list[Mapping] = defaults
) -> None:
    """Add a new set of defaults to the configuration

    It does two things:

    1.  Add the defaults to a global collection to be used by refresh later
    2.  Updates the global config with the new configuration
        prioritizing older values over newer ones
    """
    current_defaults = merge(*defaults)
    defaults.append(new)
    # defaults.append(new)
    update(config, new, priority="new-defaults", defaults=current_defaults)
    # TODO rewrite this so it works like we want


def _initialize() -> None:
    fn = os.path.join(os.path.dirname(__file__), "quantem.yaml")

    with open(fn) as f:
        _defaults = yaml.safe_load(f)

    update_defaults(_defaults)


def canonical_name(k: str, config: dict) -> str:
    """Return the canonical name for a key.

    Handles user choice of '-' or '_' conventions by standardizing on whichever
    version was set first. If a key already exists in either hyphen or
    underscore form, the existing version is the canonical name. If neither
    version exists the original key is used as is.
    """
    try:
        if k in config:
            return k
    except TypeError:
        # config is not a mapping, return the same name as provided
        return k

    altk = k.replace("_", "-") if "_" in k else k.replace("-", "_")

    if altk in config:
        return altk

    return k


def update(
    old: dict,
    new: Mapping,
    priority: Literal["old", "new", "new-defaults"] = "new",
    defaults: Mapping | None = None,
) -> dict:
    """Update a nested dictionary with values from another

    This is like dict.update except that it smoothly merges nested values

    This operates in-place and modifies old

    Parameters
    ----------
    priority: string {'old', 'new', 'new-defaults'}
        If new (default) then the new dictionary has preference.
        Otherwise the old dictionary does.
        If 'new-defaults', a mapping should be given of the current defaults.
        Only if a value in ``old`` matches the current default, it will be
        updated with ``new``.

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b)  # doctest: +SKIP
    {'x': 2, 'y': {'a': 2, 'b': 3}}

    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b, priority='old')  # doctest: +SKIP
    {'x': 1, 'y': {'a': 2, 'b': 3}}

    >>> d = {'x': 0, 'y': {'a': 2}}
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'a': 3, 'b': 3}}
    >>> update(a, b, priority='new-defaults', defaults=d)  # doctest: +SKIP
    {'x': 1, 'y': {'a': 3, 'b': 3}}

    """
    for k, v in new.items():
        k = canonical_name(k, old)

        if isinstance(v, Mapping):
            if k not in old or old[k] is None or not isinstance(old[k], dict):
                old[k] = {}
            update(
                old[k],
                v,
                priority=priority,
                defaults=defaults.get(k) if defaults else None,
            )
        else:
            if (
                priority == "new"
                or k not in old
                or (
                    priority == "new-defaults"
                    and defaults
                    and k in defaults
                    and defaults[k] == old[k]
                )
            ):
                old[k] = v

    return old


def collect(
    paths: Sequence[os.PathLike] = paths, env: Mapping[str, str] | None = None
) -> dict:
    """
    Collect configuration from paths and environment variables

    Parameters
    ----------
    paths : list[str]
        A list of paths to search for yaml config files

    env : Mapping[str, str]
        The system environment variables

    Returns
    -------
    config: dict

    """
    if env is None:
        env = os.environ

    configs = [*collect_yaml(paths=paths), collect_env(env=env)]
    return merge(*configs)


# @overload
# def collect_yaml(
#     paths: Sequence[str], *, return_paths: Literal[False] = False
# ) -> Iterator[dict]: ...


# @overload
# def collect_yaml(
#     paths: Sequence[str], *, return_paths: Literal[True]
# ) -> Iterator[tuple[Path, dict]]: ...


def collect_yaml(
    paths: Sequence[os.PathLike], *, return_paths: bool = False
) -> Iterator[dict | tuple[Path, dict]]:
    """Collect configuration from yaml files

    This searches through a list of paths, expands to find all yaml or json
    files, and then parses each file.
    """
    # Find all paths
    file_paths = []
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    file_paths.extend(
                        sorted(
                            os.path.join(path, p)
                            for p in os.listdir(path)
                            if os.path.splitext(p)[1].lower()
                            in (".json", ".yaml", ".yml")
                        )
                    )
                except OSError:
                    # Ignore permission errors
                    pass
            else:
                file_paths.append(path)

    # Parse yaml files
    for path in file_paths:
        config = _load_config_file(path)
        if config is not None:
            if return_paths:
                yield Path(path), config
            else:
                yield config


def collect_env(env: Mapping[str, str] | None = None) -> dict:
    """Collect config from environment variables

    This grabs environment variables of the form "QUANTEM_FOO__BAR_BAZ=123" and
    turns these into config variables of the form ``{"foo": {"bar-baz": 123}}``
    It transforms the key and value in the following way:

    -  Lower-cases the key text
    -  Treats ``__`` (double-underscore) as nested access
    -  Calls ``ast.literal_eval`` on the value
    """

    if env is None:
        env = os.environ

    d = {}

    for name, value in env.items():
        if name.startswith("QUANTEM_"):
            varname = name[5:].lower().replace("__", ".")
            d[varname] = interpret_value(value)

    result: dict = {}
    set(d, config=result)
    return result


def interpret_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass

    # Avoid confusion of YAML vs. Python syntax
    hardcoded_map = {"none": None, "null": None, "false": False, "true": True}
    return hardcoded_map.get(value.lower(), value)


def merge(*dicts: Mapping) -> dict:
    """Update a sequence of nested dictionaries

    This prefers the values in the latter dictionaries to those in the former

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'y': {'b': 3}}
    >>> merge(a, b)  # doctest: +SKIP
    {'x': 1, 'y': {'a': 2, 'b': 3}}
    """
    result: dict = {}
    for d in dicts:
        update(result, d)
    return result


def _load_config_file(path: str) -> dict | None:
    """A helper for loading a config file from a path, and erroring
    appropriately if the file is malformed."""
    try:
        with open(path) as f:
            config = yaml.safe_load(f.read())
    except OSError:
        # Ignore permission errors
        return None
    except Exception as exc:
        raise ValueError(
            f"A dask config file at {path!r} is malformed, original error message:\n\n{exc}"
        ) from None
    if config is not None and not isinstance(config, dict):
        raise ValueError(
            f"A dask config file at {path!r} is malformed - config files must have "
            f"a dict as the top level object, got a {type(config).__name__} instead"
        )
    return config


def check_deprecations(key: str, deprecations: dict = deprecations) -> str:
    """Check if the provided value has been renamed or removed

    Parameters
    ----------
    key : str
        The configuration key to check
    deprecations : Dict[str, str]
        The mapping of aliases

    Examples
    --------
    >>> deprecations = {"old_key": "new_key", "invalid": None}
    >>> check_deprecations("old_key", deprecations=deprecations)  # doctest: +SKIP
    UserWarning: Configuration key "old_key" has been deprecated. Please use "new_key"
    instead.

    >>> check_deprecations("invalid", deprecations=deprecations)
    Traceback (most recent call last):
        ...
    ValueError: Configuration value "invalid" has been removed

    >>> check_deprecations("another_key", deprecations=deprecations)
    'another_key'

    Returns
    -------
    new: str
        The proper key, whether the original (if no deprecation) or the aliased
        value
    """
    if key in deprecations:
        new = deprecations[key]
        if new:
            warnings.warn(
                'Configuration key "{}" has been deprecated. Please use "{}" instead'.format(
                    key, new
                )
            )
            return new
        else:
            raise ValueError(f'Configuration value "{key}" has been removed')
    else:
        return key


refresh()
_initialize()
