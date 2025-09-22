import os
import sys

# Ensure project root is on sys.path so 'import gcmi' works when running pytest
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
