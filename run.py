"""
DataTrust AI - Launch Script
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Starting DataTrust AI Platform")
    print("  Dashboard: http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(ROOT_DIR))
