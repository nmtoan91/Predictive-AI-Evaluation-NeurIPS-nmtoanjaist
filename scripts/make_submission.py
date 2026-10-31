"""Zip submission/ into an archive for the competition (model.py at the root).

    python scripts/make_submission.py                  # -> output/TagCover.zip and its SHA-256

The zip is the same byte for byte for the same files (fixed dates, sorted names).
"""

import argparse
import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_zip(source, zip_path):
    members = []
    for base, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if not name.endswith(".pyc"):
                path = os.path.join(base, name)
                members.append((os.path.relpath(path, source).replace(os.sep, "/"), path))
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in sorted(members):
            info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(path, "rb") as handle:
                archive.writestr(info, handle.read())
    return [arcname for arcname, _ in sorted(members)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=os.path.join(ROOT, "submission"))
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "TagCover.zip"))
    args = parser.parse_args()
    names = make_zip(args.source, args.out)
    with open(args.out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print(f"{os.path.relpath(args.out, ROOT)}: {', '.join(names)}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    sys.exit(main())
