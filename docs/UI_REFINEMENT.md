# Task 4 interface refinement

This update changes the browser UX, not the training pipeline or its measured results.

- Light, typography-led interface; top-level navigation and responsive layouts rather than a dense sidebar.
- Prediction inputs grouped into geography and census-area characteristics; loading state, recalculation guidance, validation errors, and out-of-California warnings.
- Geographical points chart: observed, predicted, or absolute-error layers, with click-to-copy location into the prediction form. The chart needs no map-tile API.
- Gradient query: location presets computed from valid surface grid cells, bearing slider, nearest-supported-point helper, error guidance for unsupported coordinates, and clear output interpretation.
- Local 3D Plotly surface with a 2D fallback where WebGL is unavailable.
- Semantic labels, keyboard-friendly navigation, focus outlines, visible missing-data states and an SVG favicon.

## Run and verify

```sh
source "$HOME/.venvs/geovalue/bin/activate"
python -m pytest -q
node --check public/app.js   # optional, if Node.js is installed
python -m uvicorn app:app --reload
```

Open the web UI and check these five paths: **Overview** → **Prediction** (submit and edit fields) → **Geography** (switch data layers, click a point) → **Calculus** (choose preset, query derivative, try unsupported coordinates) → **Method**. Browser chart rendering uses Plotly; a working network connection is needed to load the Plotly script from its CDN.

## Preservation of local files

The UI update intentionally does not contain `data/raw/housing.csv`, your private trained `.joblib`, `public/data/points.json`, `public/data/surface.json`, or generated report files because those files are in your local project, not the public repository. Merge the update ZIP in place. Use `python scripts/package_full.py` to create your own full backup with those assets. The full **source** ZIP has the available sample CSV and JSON reports but is not a substitute for the user's full-data backup.
