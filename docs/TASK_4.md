# Task 4 — Local FastAPI interface (final scope)

Task 4 originally included optional Vercel deployment. After completing the web interface and encountering Python function-bundle size limits during a cloud build, the project was intentionally finalized as a **local-first application**. Cloud deployment is **not a requirement, not part of the delivered system, and not claimed as a success**. This file replaces the obsolete cloud-installation instructions.

## Delivered components

- `app.py`: FastAPI entrypoint for the existing local browser application.
- `src/geovalue/web.py`: HTTP API for reports, model inference and supported gradient queries.
- `public/index.html`, `public/styles.css`, `public/app.js`, `public/favicon.svg`: responsive user interface.
- `public/data/*.json`: precomputed surface/point/report inputs for the browser.
- `scripts/prepare_deployment.py`: legacy filename retained for compatibility; copies trusted local artifacts and generates sampled visualization data. It does **not** deploy anything.

## Local run

```bash
source "$HOME/.venvs/geovalue/bin/activate"
uv pip install --link-mode copy -e ".[ml,geo,web,dev]"
python -m pytest -q
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. If the browser lacks reports/maps or if the model isn't found, confirm the local files first, then run `python scripts/prepare_deployment.py` after your completed training and geography analysis.

The accepted release remains partly network-dependent: the browser loads Plotly from a CDN, and some Folium map backgrounds rely on external tiles. These are documented limits; the application requires no cloud account or remote Python API.

## UI behavior

Five sections: Overview, Prediction, Geography, Calculus, Method. User inputs receive validation and supported-location guidance. The returned price is a historical census-area median estimate; the spatial gradient is a numerical property of a smoothed fixed-feature scenario.

## Backups and publication

`python scripts/package_full.py` creates a local complete ZIP including any original CSV, model, JSON, numerical arrays and HTML exports found in the current checkout. **Do not commit this personal backup**. The public GitHub repository already contains the historical deployment copies of the trained model and selected JSON assets; they are retained to avoid breaking compatibility, but no future cloud deployment is expected.

Details: [USER_GUIDE.md](USER_GUIDE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [TESTING.md](TESTING.md).
