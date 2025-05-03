import sys

from packaging.version import Version

ref_version = Version(sys.argv[1])
new_version = Version(sys.argv[2])

if new_version > ref_version:
    print(1)
elif new_version == ref_version:
    print(0)
else:
    print(-1)
