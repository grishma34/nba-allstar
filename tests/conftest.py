"""Make the repo root importable so tests can `import src.*` regardless of
how pytest is invoked."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
