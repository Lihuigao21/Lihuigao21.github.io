"""Load a locally built native kernel without machine-specific paths."""
from pathlib import Path
import ctypes
import os
import shutil
import sys

LIBRARY_NAME = ('native_engine.dll' if os.name == 'nt' else
                'libnative_engine.dylib' if sys.platform == 'darwin' else
                'libnative_engine.so')
_DLL_DIRECTORY_HANDLES = []

def load_native(root):
    if os.name == 'nt':
        candidates = []
        if os.environ.get('MINGW_BIN'):
            candidates.append(Path(os.environ['MINGW_BIN']))
        compiler = shutil.which(os.environ.get('CXX', 'g++'))
        if compiler:
            candidates.append(Path(compiler).resolve().parent)
        for folder in dict.fromkeys(candidates):
            if folder.is_dir():
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(folder)))
    library = Path(root) / 'work' / LIBRARY_NAME
    if not library.is_file():
        raise FileNotFoundError('Build the native kernel with reproduce.py before running this stage.')
    return ctypes.CDLL(str(library))
