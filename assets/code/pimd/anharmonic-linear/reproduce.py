"""Stage the source-only PIMD VIII benchmark in an explicit output directory.

Example: python reproduce.py --smoke --output quartic-smoke
Full production is opt-in through --stage; smoke is an implementation check.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_files():
    return sorted([PACKAGE / 'README.md', PACKAGE / 'requirements.txt',
                   PACKAGE / 'reproduce.py', *list((PACKAGE / 'inputs').glob('*.json')),
                   *list((PACKAGE / 'work').glob('*.py')),
                   *list((PACKAGE / 'work').glob('*.cpp'))])

def stage(output):
    if output == PACKAGE or PACKAGE in output.parents:
        raise ValueError('Choose an output directory outside the downloadable source package.')
    files = {p.relative_to(PACKAGE).as_posix(): digest(p) for p in source_files()}
    marker = output / 'publication_source.json'
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8'))['files'] != files:
            raise ValueError('Nonempty output has a different source identity; choose a fresh directory.')
    else:
        output.mkdir(parents=True, exist_ok=True)
        for rel in files:
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PACKAGE / rel, target)
        marker.write_text(json.dumps({'files': files}, indent=2)+'\n', encoding='utf-8', newline='\n')
    for name in ['records', 'results']:
        (output / name).mkdir(exist_ok=True)
    return files

def build(output, compiler):
    compiler = shutil.which(compiler)
    if not compiler:
        raise RuntimeError('A GCC-compatible C++17 compiler with OpenMP is required; put g++ on PATH or pass --cxx.')
    if os.name == 'nt':
        os.environ['MINGW_BIN'] = str(Path(compiler).resolve().parent)
        name = 'native_engine.dll'
        options = ['-shared', '-static-libgcc', '-static-libstdc++']
    elif sys.platform == 'darwin':
        name = 'libnative_engine.dylib'
        options = ['-dynamiclib', '-fPIC']
    else:
        name = 'libnative_engine.so'
        options = ['-shared', '-fPIC']
    target = output / 'work' / name
    source = output / 'work/native_engine.cpp'
    receipt = output / 'records/build.json'
    identity = {'source_sha256': digest(source), 'compiler': compiler}
    if target.exists() and receipt.exists():
        previous = json.loads(receipt.read_text(encoding='utf-8'))
        if all(previous.get(k) == v for k, v in identity.items()) and previous.get('library_sha256') == digest(target):
            return
        raise ValueError('Existing native build has a different identity; use a fresh output directory.')
    command = [compiler, '-O3', '-std=c++17', '-fopenmp', *options, str(source), '-o', str(target)]
    subprocess.run(command, cwd=output, check=True)
    if os.name == 'nt':
        os.environ['MINGW_BIN'] = str(Path(compiler).resolve().parent)
    identity.update(command=command, library_sha256=digest(target))
    receipt.write_text(json.dumps(identity, indent=2)+'\n', encoding='utf-8', newline='\n')

def execute(output, script, *arguments):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(output / 'records/matplotlib-cache'))
    subprocess.run([sys.executable, str(output / 'work' / script), *arguments],
                   cwd=output, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--stage', choices=['build', 'exact', 'main', 'refine', 'extension', 'analyze', 'validate', 'all'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    args = parser.parse_args()
    output = args.output.resolve()
    files = stage(output)
    build(output, args.cxx)
    if args.smoke:
        execute(output, 'exact_1d.py', '--alphas', '.5', '--bases', '80', '120', '180')
        execute(output, 'native_smoke_1d.py')
        execute(output, 'run_benchmark.py', '--phase', 'smoke')
        execute(output, 'public_smoke_checks.py')
        print(json.dumps({'status': 'pass', 'scope': 'small implementation checks; not production convergence', 'output': str(output)}))
        return
    stages = ['exact', 'main', 'refine', 'extension', 'analyze', 'validate'] if args.stage == 'all' else [args.stage]
    for name in stages:
        if name == 'exact':
            execute(output, 'exact_1d.py')
        elif name in ['main', 'refine', 'extension']:
            execute(output, 'run_extension.py' if name == 'extension' else 'run_benchmark.py', '--phase', name)
        elif name == 'analyze':
            execute(output, 'analyze_benchmark.py')
        elif name == 'validate':
            execute(output, 'final_validation.py')
    print(json.dumps({'status': 'completed', 'stage': args.stage, 'source_files': len(files), 'output': str(output)}))

if __name__ == '__main__':
    main()
