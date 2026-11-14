# Bundled frontend dependency

The official local browser loads a copy of **plotly.js v3.3.1**, bundled under `public/official/plotly.min.js`. Plotly, Inc. owns the library copyright; it is distributed under the MIT license. The required license text is preserved in [`public/official/plotly-LICENSE.txt`](../../public/official/plotly-LICENSE.txt).

The bundled library is an open-source dependency, not an algorithm implemented by GeoValue. Bundling it removes the browser's dependence on an external JavaScript CDN, supporting local operation and repeatable visualization.
