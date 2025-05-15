# quantem

This is the home repository for the quantitative electron microscopy (quantem) data analysis toolkit.

## Installation Instructions

The package is available on the Python Package Index (PyPi), as [quantem](https://pypi.org/project/quantem/).

You can install it using `pip install quantem`.

## Developer Instructions

We use [uv](https://docs.astral.sh/uv/) to manage the package.

Getting started:

- [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- `git clone` the repo and `cd` into the directory
- run `uv sync` to install all the dependencies in an editable environment

Dependency management:

- use `uv add package_name` to add dependencies
- use `uv remove package_name` to remove dependencies
- use `uv add dev_package_name --dev` to add a dev dependency, i.e. that devs need (e.g. pytest) but you don't want shipped to users
- use `uv pip install testing_package_name` to install a package you think you might need, but don't want to add to dependencies just yet

Running python/scripts in environment:

- use `uv run python`, `uv run jupyterlab` etc. to automatically activate the environment and run your command
- alternatively use `source .venv/bin/activate` to explicitly activate environment and use `python`, `jupyterlab` etc. as usual
  - note that if you're using an IDE like VS Code, it probably activates the environment automatically

## License

quantem is free and open source software, distributed under the [MIT License](LICENSE).
