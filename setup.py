from setuptools import setup, find_packages

setup(
    name="quantem",
    # version=version_ns["__version__"],
    packages=find_packages(),
    description="quantitative electron microscopy analysis toolkit",
    # long_description="TODO",
    # long_description_content_type="text/markdown",
    url="https://github.com/ophusgroup/quantem/",
    author="Colin Ophus",
    author_email="cophus@gmail.com",
    license="MIT",
    keywords="STEM, TEM, EM, data analysis",
    python_requires=">=3.11",
    install_requires=[
        "colorspacious >= 1.1.2",
        "dill >= 0.3.3",
        "gdown >= 5.1.0",
        "jupyterlab",
        "matplotlib >= 3.2.2",
        "numpy >= 2.0",
        "scipy >= 1.5.2",
        "tqdm >= 4.46.1",
        "zarr"
    ],
)