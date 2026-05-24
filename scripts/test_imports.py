print("start")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

print("before data")
from adaptive_retrieval.data import load_documents
print("after data")

print("before llm_budget")

import traceback
try:
    import adaptive_retrieval.llm_budget
    print("after llm_budget")
except Exception as e:
    print("FAILED IMPORT")
    traceback.print_exc()
print("after llm_budget")