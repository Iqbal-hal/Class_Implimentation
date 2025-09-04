import sys
import traceback
import importlib
from pathlib import Path

repo_root = Path(__file__).resolve().parent
# ensure repo and package dir on sys.path
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'TradingWorkbench'))

print("Diagnostic: sys.path (first 8 entries):")
for i, p in enumerate(sys.path[:8]):
    print(f"  {i}: {p}")

try:
    m = importlib.import_module('TradingWorkbench.support_files')
    print("IMPORT OK ->", getattr(m, '__file__', '<no file>'))
except Exception as e:
    print("IMPORT FAILED:")
    traceback.print_exc()