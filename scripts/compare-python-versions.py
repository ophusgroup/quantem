import sys

import tomli
from packaging.version import Version

ref_toml = sys.argv[1]
new_toml = sys.argv[2]

with open(ref_toml, "rb") as f:
    ref_version = Version(tomli.load(f)["project"]["version"])

with open(new_toml, "rb") as f:
    new_version = Version(tomli.load(f)["project"]["version"])

if new_version > ref_version:
    print(1)
elif new_version == ref_version:
    print(0)
else:
    print(-1)
