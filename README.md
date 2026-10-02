# Thumbnail Attention Studio: v1 (archived)

Snapshot of the v1 code. The v2 rewrite lives in a separate repo; this one exists so v1 can be cloned and continued.

## Layout
- `backend/` FastAPI app: `api/`, `pipeline/`, `analyzers/`, `fusion/`, `scoring/`, `evaluation/`, `config/*.yaml`, `models/*.onnx` (about 140 MB, committed), `tests/`, `scripts/`
- `frontend/` React sources (App, components, eye `lab/`) and the webgazer / MediaPipe assets in `public/`
- `docs/`, `ml/` notes and notebooks

## Run
```
cd backend
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                # tests
uvicorn api.main:app --reload --port 8000
```
Storage is local SQLite unless `DATABASE_URL` (Postgres) is set in `backend/.env`.

## Frontend caveat
The archive holds the frontend sources only. `index.html`, `main.tsx`, `vite.config.ts` and the `src/` folder layout
(vitest expects `src/**/*.test.ts`) were not saved with v1, so you must recreate that Vite scaffold around these files.
`frontend/package.json` was reconstructed from the imports in the code, not recovered from the original.
