import sys
from pathlib import Path

# Ensure project root is importable when running pytest from the tests folder
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
