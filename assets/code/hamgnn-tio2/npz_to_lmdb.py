#!/usr/bin/env python3
"""Convert HamGNN graph_data.npz to HamGNN LMDB format."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np


def parse_size(size_text: str) -> int:
    units = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }
    text = size_text.strip().lower()
    number = ""
    suffix = ""
    for char in text:
        if char.isdigit() or char == ".":
            number += char
        else:
            suffix += char
    if not number:
        raise ValueError(f"Invalid size: {size_text}")
    suffix = suffix or "b"
    if suffix not in units:
        raise ValueError(f"Invalid size suffix in {size_text}")
    return int(float(number) * units[suffix])


def graph_items(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as npz:
        if "graph" not in npz:
            raise KeyError(f"{npz_path} does not contain a 'graph' array")
        graphs = npz["graph"].item()

    if isinstance(graphs, dict):
        def key_order(key):
            try:
                return int(key)
            except Exception:
                return key

        for _, graph in sorted(graphs.items(), key=lambda item: key_order(item[0])):
            yield graph
    else:
        for graph in graphs:
            yield graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map-size", default="256G")
    parser.add_argument("--commit-interval", default=16, type=int)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMDB path: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    map_size = parse_size(args.map_size)
    env = lmdb.open(
        str(args.output),
        map_size=map_size,
        subdir=True,
        meminit=False,
        writemap=False,
        map_async=True,
    )

    count = 0
    txn = env.begin(write=True)
    try:
        for count, graph in enumerate(graph_items(args.input), start=1):
            payload = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(f"graph_{count - 1}".encode(), payload)
            if count % args.commit_interval == 0:
                txn.commit()
                print(f"committed {count} graphs", flush=True)
                txn = env.begin(write=True)

        txn.put(b"num_graphs", str(count).encode())
        txn.commit()
        env.sync()
    except Exception:
        txn.abort()
        raise
    finally:
        env.close()

    print(f"converted_graphs={count}")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"output_size_bytes={sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())}")
    print(f"pid={os.getpid()}")


if __name__ == "__main__":
    main()
