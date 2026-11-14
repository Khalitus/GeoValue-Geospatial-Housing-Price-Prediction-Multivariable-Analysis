"""Leakage-conscious parcel/sale/centroid joins, validation and build audit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geovalue.official.sources import check_source_files

TARGET = 'sale_price'
NUMERIC_INPUTS = (
    'latitude', 'longitude', 'bedrooms', 'full_baths', 'half_baths', 'total_rooms',
    'living_area_sqft', 'lot_area_sqft', 'year_built', 'stories', 'fireplaces',
    'garage_spaces', 'sale_year', 'sale_month',
)
CATEGORICAL_INPUTS = ('construction_grade', 'condition_code', 'heating_cooling',
                      'municipality', 'zip_code')
INPUT_COLUMNS = NUMERIC_INPUTS + CATEGORICAL_INPUTS
ENGINEERED = ('property_age_at_sale', 'bathroom_equivalents', 'floor_area_ratio')
METADATA_COLUMNS = ('parcel_id', 'sale_date')

ASSESSMENT_FIELDS = {
    'YEARBLT': 'year_built', 'BEDROOMS': 'bedrooms', 'FULLBATHS': 'full_baths',
    'HALFBATHS': 'half_baths', 'TOTALROOMS': 'total_rooms',
    'FINISHEDLIVINGAREA': 'living_area_sqft', 'LOTAREA': 'lot_area_sqft',
    'STORIES': 'stories', 'FIREPLACES': 'fireplaces', 'BSMTGARAGE': 'garage_spaces',
    'GRADE': 'construction_grade', 'CONDITION': 'condition_code',
    'HEATINGCOOLINGDESC': 'heating_cooling', 'MUNIDESC': 'municipality',
    'PROPERTYZIP': 'zip_code',
}
REQUIRED_ASSESSMENT = frozenset({'PARID', 'CLASS', 'YEARBLT', 'BEDROOMS',
                                 'FINISHEDLIVINGAREA', 'LOTAREA'})
REQUIRED_SALES = frozenset({'PARID', 'SALEDATE', 'PRICE', 'SALEDESC'})
REQUIRED_CENTROIDS = frozenset({'PIN', 'LAT', 'LONG'})
# Current assessed/fair market values and current/previous sale fields MUST NEVER
# enter INPUT_COLUMNS; using them would mix target/post-sale information into X.
PROHIBITED_INPUTS = frozenset({'PRICE','SALEPRICE','SALEDESC','SALECODE','FAIRMARKETTOTAL',
                               'COUNTYTOTAL','LOCALTOTAL','PREVSALEPRICE','RECORDDATE','PARID'})


@dataclass(frozen=True)
class BuildConfig:
    start_year: int = 2013
    end_year: int = 2025
    minimum_price: float = 10000.0
    maximum_price: float = 5000000.0

    def __post_init__(self):
        if not 2013 <= self.start_year <= self.end_year <= 2025:
            raise ValueError('Allowed sale years are from 2013 to 2025 inclusive')
        if self.minimum_price <= 0 or self.maximum_price <= self.minimum_price:
            raise ValueError('Invalid price limits')


def _normalized_identifiers(series: pd.Series) -> pd.Series:
    # Preserve the county's 16-character alphanumeric PIN, including leading zeroes.
    return series.astype('string').str.strip().str.upper().str.replace(r'\s+', '', regex=True)


def _csv_columns(path: Path) -> set[str]:
    return {str(col).strip().upper() for col in pd.read_csv(path, nrows=0).columns}


def _read_selected(path: Path, selected: set[str], *, key_col: str, chunksize: int | None = None):
    present = _csv_columns(path)
    required = {'PARID'} if key_col == 'PARID' else {'PIN'}
    if not required.issubset(present):
        raise ValueError(f'{path.name} has no parcel key column {key_col}')
    cols = selected & present
    return pd.read_csv(path, usecols=lambda col: col.strip().upper() in cols,
                       dtype={key_col: 'string', 'PROPERTYZIP': 'string'}, low_memory=False,
                       chunksize=chunksize, encoding='utf-8-sig')


def _parse_dates(series: pd.Series) -> pd.Series:
    # WPRDC uses dd/mm/yyyy for slash dates and ISO for other resources.
    return pd.to_datetime(series, format='mixed', dayfirst=True, errors='coerce')


def _load_sales(path: Path, config: BuildConfig) -> tuple[pd.DataFrame, dict]:
    available = _csv_columns(path)
    if not REQUIRED_SALES.issubset(available):
        raise ValueError(f'Sales CSV lacks required fields: {sorted(REQUIRED_SALES - available)}')
    chunks = _read_selected(path, REQUIRED_SALES | {'SALECODE'}, key_col='PARID', chunksize=100_000)
    clean = []
    audit = {'raw_sale_rows': 0, 'rejected_invalid_or_uncertain_sale': 0,
             'rejected_date_or_price': 0, 'retained_sales_before_join': 0}
    for chunk in chunks:
        chunk.columns = chunk.columns.str.strip().str.upper()
        audit['raw_sale_rows'] += len(chunk)
        # Strict reading of the county's own description. No guessing from codes.
        valid = chunk['SALEDESC'].astype('string').str.strip().str.upper().eq('VALID SALE').fillna(False)
        audit['rejected_invalid_or_uncertain_sale'] += int((~valid).sum())
        chunk = chunk.loc[valid].copy()
        dates = _parse_dates(chunk['SALEDATE'])
        prices = pd.to_numeric(chunk['PRICE'].astype('string').str.replace(r'[$,]', '', regex=True), errors='coerce')
        good = dates.dt.year.between(config.start_year, config.end_year) & prices.between(config.minimum_price, config.maximum_price)
        audit['rejected_date_or_price'] += int((~good).sum())
        if good.any():
            clean.append(pd.DataFrame({'parcel_id': _normalized_identifiers(chunk.loc[good, 'PARID']).to_numpy(),
                                       'sale_date': dates[good].to_numpy(), 'sale_price': prices[good].to_numpy()}))
    if not clean:
        raise ValueError('No market-valid transactions in the requested period; check SALEDESC and date formats')
    sales = pd.concat(clean, ignore_index=True)
    valid_id = sales['parcel_id'].str.fullmatch(r'[A-Z0-9]{16}', na=False)
    audit['rejected_invalid_parcel_id'] = int((~valid_id).sum())
    sales = sales.loc[valid_id].drop_duplicates(['parcel_id','sale_date','sale_price']).reset_index(drop=True)
    audit['retained_sales_before_join'] = len(sales)
    return sales, audit


def _load_assessments(path: Path) -> tuple[pd.DataFrame, dict]:
    available = _csv_columns(path)
    if not REQUIRED_ASSESSMENT.issubset(available):
        raise ValueError(f'Assessment CSV lacks fields: {sorted(REQUIRED_ASSESSMENT - available)}')
    selected = REQUIRED_ASSESSMENT | set(ASSESSMENT_FIELDS) | {'ASOFDATE'}
    frame = _read_selected(path, selected, key_col='PARID')
    frame.columns = frame.columns.str.strip().str.upper()
    frame['PARID'] = _normalized_identifiers(frame['PARID'])
    frame = frame.loc[frame['CLASS'].astype('string').str.strip().str.upper().eq('R')].copy()
    frame = frame.loc[frame['PARID'].str.fullmatch(r'[A-Z0-9]{16}', na=False)]
    original = len(frame)
    frame = frame.drop_duplicates(subset=['PARID'], keep='first')
    frame = frame.rename(columns={'PARID': 'parcel_id', **ASSESSMENT_FIELDS})
    for target in ASSESSMENT_FIELDS.values():
        if target not in frame:
            frame[target] = np.nan
    return frame[['parcel_id', *ASSESSMENT_FIELDS.values()]], {'residential_assessments': original,
                                                                'duplicated_assessment_keys': original - len(frame)}


def _load_centroids(path: Path) -> tuple[pd.DataFrame, dict]:
    available = _csv_columns(path)
    if not REQUIRED_CENTROIDS.issubset(available):
        raise ValueError(f'Centroid CSV lacks fields: {sorted(REQUIRED_CENTROIDS - available)}')
    frame = _read_selected(path, REQUIRED_CENTROIDS, key_col='PIN')
    frame.columns = frame.columns.str.strip().str.upper()
    frame['PIN'] = _normalized_identifiers(frame['PIN'])
    frame['LAT'] = pd.to_numeric(frame['LAT'], errors='coerce')
    frame['LONG'] = pd.to_numeric(frame['LONG'], errors='coerce')
    valid = frame['LAT'].between(39.5, 41.0) & frame['LONG'].between(-81.0, -79.0)
    valid &= frame['PIN'].str.fullmatch(r'[A-Z0-9]{16}', na=False)
    frame = frame.loc[valid].copy()
    before = len(frame)
    frame = frame.drop_duplicates('PIN', keep='first').rename(columns={'PIN':'parcel_id','LAT':'latitude','LONG':'longitude'})
    return frame[['parcel_id','latitude','longitude']], {'valid_centroid_rows': before,
                                                        'duplicate_centroid_keys': before-len(frame)}


def build_official_data(assessments: Path, sales: Path, centroids: Path,
                        config: BuildConfig | None = None) -> tuple[pd.DataFrame, dict]:
    config = config or BuildConfig()
    transactions, sales_audit = _load_sales(Path(sales), config)
    structure, assess_audit = _load_assessments(Path(assessments))
    points, points_audit = _load_centroids(Path(centroids))
    merged = transactions.merge(structure, on='parcel_id', how='inner', validate='many_to_one')
    post_assess = len(merged)
    merged = merged.merge(points, on='parcel_id', how='inner', validate='many_to_one')
    post_geo = len(merged)
    for column in ASSESSMENT_FIELDS.values():
        if column not in CATEGORICAL_INPUTS:
            merged[column] = pd.to_numeric(merged[column], errors='coerce')
    good = merged['living_area_sqft'].between(250, 30000)
    good &= merged['lot_area_sqft'].gt(0)
    good &= merged['bedrooms'].between(0, 20)
    good &= merged['year_built'].between(1700, 2025)
    good &= merged['year_built'].le(merged['sale_date'].dt.year)
    good &= merged['full_baths'].isna() | merged['full_baths'].between(0, 20)
    good &= merged['half_baths'].isna() | merged['half_baths'].between(0, 20)
    cleaned = merged.loc[good].copy()
    cleaned['sale_year'] = cleaned['sale_date'].dt.year.astype('int16')
    cleaned['sale_month'] = cleaned['sale_date'].dt.month.astype('int8')
    for category in CATEGORICAL_INPUTS:
        cleaned[category] = cleaned[category].astype('string').fillna('Unknown').str.strip().replace('', 'Unknown').astype(str)
    # Never add assessed valuation, prior sales, or county administrative sale fields to inputs.
    result = cleaned.loc[:, [*METADATA_COLUMNS, *INPUT_COLUMNS, TARGET]].sort_values(['sale_date', 'parcel_id'])
    result = result.reset_index(drop=True)
    years = {str(year): int(count) for year, count in result['sale_year'].value_counts().sort_index().items()}
    audit = {'source': 'WPRDC / Allegheny County official assessment, recorded-sales and parcel-centroid records',
             'period': [config.start_year, config.end_year], 'sale_price_filter_usd': [config.minimum_price, config.maximum_price],
             **sales_audit, **assess_audit, **points_audit,
             'matched_sales_with_residential_assessment': post_assess,
             'matched_sales_with_centroid': post_geo,
             'rejected_physical_or_temporal_inconsistency': post_geo-len(result),
             'final_transaction_rows': len(result), 'unique_parcels': int(result['parcel_id'].nunique()),
             'sales_per_year': years, 'source_snapshot_warning': (
                 'Current assessment attributes are not guaranteed to be those at historic sale dates. '
                 'This is a retrospective reconstruction; chronological holdout does not eliminate this temporal feature leakage risk.'),
             'selection_warning': ('Only explicit SALEDESC=VALID SALE is included; sales lacking a clear description are excluded. '
                                   'County validation decisions may change over time.')}
    if result.empty:
        raise ValueError('No residential sales remained after official joins and quality screening; inspect build report and raw headers')
    return result, audit


def run_build(raw_dir: Path, output: Path, report: Path,
              config: BuildConfig | None = None) -> dict:
    paths = check_source_files(raw_dir)
    data, audit = build_official_data(paths['assessments'], paths['sales'], paths['centroids'], config)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    audit['output_csv'] = str(output)
    source_manifest = Path(raw_dir)/'source-manifest.json'
    if source_manifest.is_file():
        audit['source_manifest'] = json.loads(source_manifest.read_text(encoding='utf-8'))
    report.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return audit


def load_processed_data(path: Path, *, require_target: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Official processed dataset missing: {path}')
    df = pd.read_csv(path, dtype={'parcel_id':'string','zip_code':'string','condition_code':'string'}, low_memory=False)
    required = set(INPUT_COLUMNS) | ({TARGET, 'sale_date', 'parcel_id'} if require_target else set())
    absent = required-set(df.columns)
    if absent:
        raise ValueError('Missing processed columns: '+', '.join(sorted(absent)))
    if require_target:
        df['sale_date'] = _parse_dates(df['sale_date'])
        df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
        if df['sale_date'].isna().any() or not np.isfinite(df['sale_price'].to_numpy(dtype=float)).all():
            raise ValueError('Invalid sale dates or target values in processed file')
    return df
