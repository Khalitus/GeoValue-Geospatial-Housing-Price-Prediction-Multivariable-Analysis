# Building the original-model GeoValue poster

This poster release uses the **original Mac-trained XGBoost artifact** verified in [`poster/evidence/original_model_verification.json`](evidence/original_model_verification.json), not the independently refitted model from earlier poster revisions.

## Evidence policy

- **Observed 2025 sales:** actual recorded transactions, not model predictions.
- **Original-model 2025 predictions:** 8,691 held-out properties, with R² 0.84119967, MAE USD 58,190.78, and RMSE USD 101,555.74.
- **Spatial-support calibration:** the 2024 component uses a separately fitted **pre-2024 calibration model**, as required by the historical validation design. The 2025 audit uses the original final trained XGBoost model.
- **Gradient figure:** numerical derivatives of a Gaussian-smoothed, fixed-feature geographical scenario from the original model; **not measured price appreciation, an observed gradient, or a causal effect**.
- **No independent-refit statistics or figures:** any older poster image/report labeled "independent XGBoost reproduction" or "R² 0.8378" is superseded for exhibition purposes.

## Rebuild the editable poster

From the repository root on your Mac:

```bash
source "$HOME/.venvs/geovalue/bin/activate"
git pull --ff-only origin main
uv pip install --link-mode copy python-pptx pillow
```

Place the supplied original editable A0 design at:

```text
poster/source/GeoValue_TechXhibit8_Base.pptx
```

The original A0 design file is distributed separately because it contains the licensed university logos. **Do not commit the source design containing superseded independent-refit panels.**

Then build:

```bash
python poster/scripts/build_original_poster.py
```

It validates the exact published original-model metrics, then replaces the four old research panels with the original-model figures from `poster/figures/` and rewrites every result caption that referred to an independent model. No model is trained.

If LibreOffice is installed:

```bash
python poster/scripts/build_original_poster.py --export-pdf
```

Otherwise open the generated PowerPoint in PowerPoint or Keynote and export to PDF, using full A0 landscape size.

## Outputs

```text
poster/final/
  GeoValue_TechXhibit8_Original_Model_Final.pptx
  GeoValue_TechXhibit8_Original_Model_Final.pdf      # when exported
  ORIGINAL_MODEL_RELEASE.json
  original_model_hero.png
  original_support_chart.png
  original_actual_predicted.png
  original_price_bracket.png
```

The release manifest records the original-model hash, all source figure SHA-256 digests, the evaluated test metrics, and the final PowerPoint hash.

**Before committing:** inspect the exported PDF at full A0 size, scan its GitHub QR code, and confirm the labels use the correct 2024/2025 cohorts. Commit `poster/final/` only after the PDF is visually checked.

```bash
python -m pytest -q poster/tests

git add poster/scripts/build_original_poster.py \
  poster/ORIGINAL_POSTER_BUILD.md poster/tests/
git commit -m "docs(poster): build from verified original XGBoost evidence"
git push origin main
```

After you have built and checked the poster:

```bash
git add poster/final/
git commit -m "docs(poster): publish original-model TechXhibit poster"
git push origin main
```

The original model, source datasets, and numerical working files remain on the Mac SSD; their SHA-256 hashes are recorded in the committed evidence.

## Numerical-report caveat

The 2024 spatial-support calibration is **not** produced by the original final 2025 model; it is fitted on pre-2024 data. This is intentional and is disclosed in the poster and manuscript. The original final-model 2025 performance and spatial audit are independently verified by the stored source hashes and saved-model evaluation.
