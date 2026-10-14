#!/usr/bin/env python3
"""Read env/versions.toml from shell scripts (system python3 >= 3.11, stdlib only).

    tools/versions.py get images.vllm.digest      -> prints the value
    tools/versions.py keys models                 -> prints the table's keys, one per line
    tools/versions.py image vllm                  -> prints <ref>@<digest>
"""
import sys
import tomllib
from pathlib import Path

V = tomllib.loads((Path(__file__).resolve().parent.parent / "env" / "versions.toml").read_text())


def get(path):
    x = V
    for k in path.split("."):
        x = x[k]
    return x


def main(argv):
    cmd, arg = argv[1], argv[2]
    if cmd == "get":
        x = get(arg)
        print(" ".join(x) if isinstance(x, list) else x)
    elif cmd == "keys":
        print("\n".join(get(arg)))
    elif cmd == "image":
        i = get(f"images.{arg}")
        print(f"{i['ref']}@{i['digest']}")
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
