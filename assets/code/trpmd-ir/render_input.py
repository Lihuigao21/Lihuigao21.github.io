#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


template, output, socket = sys.argv[1:4]
text = Path(template).read_text(encoding="utf-8")
text = text.replace("__SOCKET__", socket)
Path(output).write_text(text, encoding="utf-8")
