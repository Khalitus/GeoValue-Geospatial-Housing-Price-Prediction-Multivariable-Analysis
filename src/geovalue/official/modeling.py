"""Property-level regression pipeline with no target-derived predictors."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from geovalue.official.data import (CATEGORICAL_INPUTS, ENGINEERED, INPUT_COLUMNS,
                                    NUMERIC_INPUTS, PROHIBITED_INPUTS)

MODELS = ('linear','random_forest','xgboost')
FORMAT = 'geovalue-official-sale-v1'
GEO_FEATURES = frozenset({'latitude','longitude','municipality','zip_code'})


class PropertyFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create transaction-time housing ratios, without fitting any global statistics."""
    def fit(self, X: pd.DataFrame, y=None):
        self._validate(X)
        return self

    @staticmethod
    def _validate(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Property inputs must be a pandas DataFrame')
        missing = set(INPUT_COLUMNS)-set(X.columns)
        if missing:
            raise ValueError(f'Property fields missing: {sorted(missing)}')

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        out = X.loc[:, list(INPUT_COLUMNS)].copy()
        numeric = set(NUMERIC_INPUTS)
        for field in numeric:
            out[field] = pd.to_numeric(out[field], errors='coerce')
        out['property_age_at_sale'] = (out['sale_year'] - out['year_built']).clip(lower=0)
        out['bathroom_equivalents'] = out['full_baths'] + 0.5*out['half_baths']
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = out['living_area_sqft'] / out['lot_area_sqft'].replace(0, np.nan)
        out['floor_area_ratio'] = ratio.replace([np.inf, -np.inf], np.nan)
        return out


def build_model(name: str, *, seed: int = 42, geography: bool = True,
                forest_trees: int = 120, xgboost_trees: int = 220) -> Pipeline:
    if name not in MODELS:
        raise ValueError(f'Unknown official model: {name}')
    if forest_trees < 1 or xgboost_trees < 1:
        raise ValueError('Tree counts must be positive')
    if set(INPUT_COLUMNS) & PROHIBITED_INPUTS:
        raise RuntimeError('Forbidden sale/assessment value has entered the predictor schema')
    numbers = [c for c in NUMERIC_INPUTS + ENGINEERED if geography or c not in GEO_FEATURES]
    categories = [c for c in CATEGORICAL_INPUTS if geography or c not in GEO_FEATURES]
    numeric = Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                        ('scale',StandardScaler())])
    categorical = Pipeline([('fill',SimpleImputer(strategy='most_frequent')),
                            ('encode',OneHotEncoder(handle_unknown='ignore', min_frequency=25,
                                                    sparse_output=True))])
    preprocessor = ColumnTransformer([('numbers', numeric, numbers), ('categories',categorical,categories)],
                                     sparse_threshold=0.30)
    if name=='linear':
        predictor = LinearRegression()
    elif name=='random_forest':
        predictor = RandomForestRegressor(n_estimators=forest_trees, max_depth=18,
                                          min_samples_leaf=2, n_jobs=-1, random_state=seed)
    else:
        predictor = XGBRegressor(objective='reg:squarederror', n_estimators=xgboost_trees,
                                 max_depth=6, learning_rate=0.06, subsample=0.85,
                                 colsample_bytree=0.9, tree_method='hist',
                                 n_jobs=4, random_state=seed)
    return Pipeline([('features',PropertyFeatureEngineer()),
                     ('preprocess',preprocessor), ('regressor', predictor)])


def metrics(actual, predicted) -> dict[str,float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or len(actual)==0:
        raise ValueError('Metrics need nonempty equal-length vectors')
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError('Metrics must be finite')
    return {'mae':float(mean_absolute_error(actual,predicted)),
            'rmse':float(np.sqrt(mean_squared_error(actual,predicted))),
            'r2':float(r2_score(actual,predicted))}


def save_model(pipeline: Pipeline, name: str, path: Path) -> Path:
    if name not in MODELS:
        raise ValueError('Invalid model name')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'format':FORMAT,'name':name,'input_columns':list(INPUT_COLUMNS),
                 'pipeline':pipeline},path,compress=3)
    return path


def load_model(path: Path) -> dict:
    """Load only locally trusted Joblib artifacts (pickle can execute arbitrary code)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Official model missing: {path}')
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get('format')!=FORMAT or artifact.get('name') not in MODELS or artifact.get('input_columns')!=list(INPUT_COLUMNS) or not isinstance(artifact.get('pipeline'), Pipeline):
        raise ValueError('Not a compatible official GeoValue model artifact')
    return artifact


def validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame,pd.DataFrame) or frame.empty:
        raise ValueError('No property records provided')
    missing=set(INPUT_COLUMNS)-set(frame.columns)
    if missing:
        raise ValueError('Missing property inputs: '+', '.join(sorted(missing)))
    result=frame.loc[:,list(INPUT_COLUMNS)].copy()
    for name in NUMERIC_INPUTS:
        result[name]=pd.to_numeric(result[name],errors='coerce')
    mandatory=('latitude','longitude','living_area_sqft','lot_area_sqft','year_built',
               'bedrooms','sale_year','sale_month')
    if result[list(mandatory)].isna().any().any():
        raise ValueError('Missing required numeric property inputs')
    if not result['latitude'].between(39.5,41).all() or not result['longitude'].between(-81,-79).all():
        raise ValueError('Property location outside approximate Allegheny County bounds')
    if not result['sale_year'].between(2013,2025).all() or not result['sale_month'].between(1,12).all():
        raise ValueError('Sale year/month must lie in 2013–2025')
    if not result['year_built'].between(1700,2025).all() or (result['year_built']>result['sale_year']).any():
        raise ValueError('Construction year is invalid for requested sale year')
    if not result['living_area_sqft'].between(250,30000).all() or (result['lot_area_sqft']<=0).any():
        raise ValueError('Invalid living/lot area')
    if not result['bedrooms'].between(0,20).all():
        raise ValueError('Invalid number of bedrooms')
    for col in NUMERIC_INPUTS:
        if np.isinf(result[col].to_numpy(dtype=float)).any():
            raise ValueError(f'Infinite numeric input: {col}')
    for col in CATEGORICAL_INPUTS:
        result[col]=result[col].astype('string').fillna('Unknown').str.strip().replace('', 'Unknown').astype(str)
    return result


def predict(model_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    model=load_model(model_path)['pipeline']
    typed=validate_inputs(frame)
    predicted=np.asarray(model.predict(typed),dtype=float)
    if predicted.shape!=(len(typed),) or not np.isfinite(predicted).all():
        raise ValueError('Model returned invalid property prices')
    output=frame.copy()
    output['predicted_sale_price_usd']=predicted
    return output
