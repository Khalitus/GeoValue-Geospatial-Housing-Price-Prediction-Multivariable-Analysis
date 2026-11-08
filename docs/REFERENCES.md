# References

Sources are grouped here to avoid duplicating bibliographies across the README and technical documents. URLs were checked against official project, paper or standards pages as available; access date: **8 November 2026**. These references support the tools and techniques used, not numerical model-accuracy claims (which come from the project's own recorded JSON files).

1. **Nugent, C.** *California Housing Prices*. Kaggle. Historical California census data (1990). https://www.kaggle.com/datasets/camnugent/california-housing-prices
2. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html — implementation concepts: pipelines, preprocessing, regression and evaluation. Documentation: https://scikit-learn.org/stable/
3. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*, 785–794. https://doi.org/10.1145/2939672.2939785 — gradient-boosted trees; Python documentation: https://xgboost.readthedocs.io/en/stable/python/python_api.html
4. **EPSG Geodetic Parameter Dataset.** EPSG:3310 — NAD83 / California Albers. https://epsg.io/3310 — projected California coordinate reference system and metre-based east/north axes.
5. **SciPy Developers.** `scipy.ndimage.gaussian_filter`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html — spatial smoothing. **NumPy Developers.** `numpy.gradient`. https://numpy.org/doc/stable/reference/generated/numpy.gradient.html — finite-difference gradients.
6. **NumPy Developers.** NumPy documentation. https://numpy.org/doc/stable/ — array operations and numerical quadrature calculations.
7. **McKinney, W.** (2010). Data Structures for Statistical Computing in Python. *Proceedings of SciPy 2010*, 56–61. https://conference.scipy.org/proceedings/scipy2010/mckinney.html — Pandas origins; https://pandas.pydata.org/docs/
8. **FastAPI Contributors.** FastAPI documentation. https://fastapi.tiangolo.com/ — local ASGI service and request validation.
9. **Plotly.** Plotly JavaScript graphing library documentation. https://plotly.com/javascript/ — responsive browser plotting and 3D display.
10. **pytest development team.** pytest documentation. https://docs.pytest.org/en/stable/ — automated unit and integration testing.
11. **Folium Contributors.** Folium documentation. https://python-visualization.github.io/folium/latest/ — geographic HTML mapping.
12. **Salim Habib University.** *Multivariable Calculus Course Outline, Fall 2026*. Instructor-provided course document. Internal course source (not a public URL); the applied topics include multivariable functions, derivatives, 3D surfaces, double integrals and vector fields.

## Internal evidence (not external references)

- `public/data/training-report.json` — actual data split, software versions, validation model comparisons, ablation, held-out test metrics.
- `public/data/geospatial-report.json` — actual original-data grid, masking, integration and gradient summary.
- GitHub Actions successful run: https://github.com/Khalitus/GeoValue-Geospatial-Housing-Price-Prediction-Multivariable-Analysis/actions/runs/42475421269
- `docs/TESTING.md` — separate audit methodology, synthetic stress tests and screenshot provenance.
