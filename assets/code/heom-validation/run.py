"""Run a frozen validation protocol using only the bundled local package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('harmonic', 'sac'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'source-provenance.json').read_text(encoding='utf-8'))
    for relative, entry in manifest['files'].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise SystemExit(f'Source checksum mismatch: {relative}')
    # Set these before loading NumPy/SciPy; callers can explicitly override them.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output.parent / '.matplotlib-cache'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    sys.argv = ['benchmark', '--output', str(output)]
    module = 'heom_harmonic' if args.case == 'harmonic' else 'heom_sac'
    runpy.run_module('toymodel.benchmarks.' + module, run_name='__main__')
    metrics_file = 'metrics.json' if args.case == 'harmonic' else 'summary.json'
    metrics = json.loads((output / metrics_file).read_text(encoding='utf-8'))
    if not metrics['numerical_gates_passed']:
        raise SystemExit('Numerical acceptance gate failed; inspect the retained diagnostics.')


if __name__ == '__main__':
    main()
