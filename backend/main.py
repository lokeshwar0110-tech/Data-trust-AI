import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from datatrust import __version__ as ENGINE_VERSION
from backend.routes.evaluate import router as evaluate_router

app = FastAPI(
    title="DataTrust AI v3 API",
    description="Context-Aware Decision Support & Data Validation Platform for Machine Learning and Analytics",
    version=ENGINE_VERSION
)

# Enable CORS for development & API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(evaluate_router)

# Mount static directory for SaaS dashboard
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=FileResponse)
def serve_dashboard():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "DataTrust AI API is running. Navigate to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DataTrust AI v3",
        "engine_version": ENGINE_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(ROOT_DIR))
