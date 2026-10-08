"""Deploy-mode entrypoint for Render (single process).

Route order (Starlette matches Mounts in order):
  1. /api/*  -> full FastAPI app (frontend api.ts BASE = '/api')
  2. /*      -> bare documented API paths (same app, keeps API contract)
  3. /*      -> static dist files, SPA fallback to index.html (innermost,
                only reached when the bare API app raised 404)

Run:  uvicorn deploy_server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount

from dltime.api import create_app

DIST = Path(__file__).resolve().parent / "frontend" / "dist"

if not DIST.is_dir():
    raise RuntimeError(f"frontend/dist not found at {DIST}; run 'npm run build' first.")

api_app = create_app()

# --- innermost: static dist + SPA fallback --------------------------------
spa = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)


@spa.exception_handler(StarletteHTTPException)
async def _spa_404(request, exc):
    if exc.status_code == 404:
        return FileResponse(DIST / "index.html")
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@spa.get("/{full_path:path}", include_in_schema=False)
async def _spa_files(full_path: str):
    candidate = DIST / full_path
    if full_path and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(DIST / "index.html")


# --- outer shell ----------------------------------------------------------
app = FastAPI(title="Data Loader Time (deploy)", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.router.routes.append(Mount("/api", app=api_app, name="api"))
app.router.routes.append(Mount("/", app=api_app, name="bare-api"))
app.router.routes.append(Mount("/", app=spa, name="spa-front"))
