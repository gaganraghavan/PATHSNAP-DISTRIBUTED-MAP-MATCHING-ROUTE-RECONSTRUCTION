import sys
from pathlib import Path

# Let tests import the synthetic-data helper and the top-level ``api`` package.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
