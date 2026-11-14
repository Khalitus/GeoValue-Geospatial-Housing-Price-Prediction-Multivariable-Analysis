import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geovalue.official.data import (BuildConfig, CATEGORICAL_INPUTS, INPUT_COLUMNS,
                                    PROHIBITED_INPUTS, build_official_data, load_processed_data, run_build)


def test_realistic_three_source_join_and_validity(county_raw):
    data,audit=build_official_data(*(county_raw / (k+'.csv') for k in ('assessments','sales','centroids')))
    assert audit['raw_sale_rows']==112
    assert audit['rejected_invalid_or_uncertain_sale']==1
    assert audit['final_transaction_rows']==111
    assert len(data)==111
    assert data.parcel_id.str.len().eq(16).all()
    assert data.parcel_id.str.startswith('000').all()
    assert data.sale_year.max()==2025
    assert data.sale_year.min()==2023
    assert data.sale_price.min()>=10000
    assert not (set(INPUT_COLUMNS)&PROHIBITED_INPUTS)


def test_builder_does_not_accidentally_include_target_as_feature(official_data):
    assert 'sale_price' not in INPUT_COLUMNS
    assert 'parcel_id' not in INPUT_COLUMNS
    assert 'sale_code' not in INPUT_COLUMNS
    assert all(name in official_data for name in INPUT_COLUMNS)


def test_missing_sale_description_is_fail_closed(county_raw):
    sales=county_raw/'sales.csv'
    frame=pd.read_csv(sales,dtype={'PARID':str})
    frame.loc[0,'SALEDESC']=None
    frame.to_csv(sales,index=False)
    _,audit=build_official_data(*(county_raw / (k+'.csv') for k in ('assessments','sales','centroids')))
    assert audit['rejected_invalid_or_uncertain_sale']==2


def test_bad_sale_date_and_price_filtered(county_raw):
    path=county_raw/'sales.csv'
    records=pd.read_csv(path,dtype={'PARID':str})
    records.loc[3,'SALEDATE']='not a date'
    records.loc[4,'PRICE']='5'
    records.to_csv(path,index=False)
    _,audit=build_official_data(*(county_raw / (k+'.csv') for k in ('assessments','sales','centroids')))
    assert audit['rejected_date_or_price']==2


def test_missing_required_assessment_field_fails_loudly(county_raw):
    path=county_raw/'assessments.csv'
    frame=pd.read_csv(path,dtype={'PARID':str}).drop(columns='YEARBLT')
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Assessment CSV lacks'):
        build_official_data(*(county_raw / (k+'.csv') for k in ('assessments','sales','centroids')))


def test_temporal_construction_date_filter(county_raw):
    path=county_raw/'assessments.csv'
    frame=pd.read_csv(path,dtype={'PARID':str})
    frame.loc[0,'YEARBLT']=2026
    frame.to_csv(path,index=False)
    _,audit=build_official_data(*(county_raw / (k+'.csv') for k in ('assessments','sales','centroids')))
    assert audit['rejected_physical_or_temporal_inconsistency']>=2


def test_build_report_contains_exact_counts_and_snapshot_caveat(tmp_path,county_raw):
    output=tmp_path/'sales.csv';report=tmp_path/'audit.json'
    audit=run_build(county_raw,output,report)
    assert output.is_file() and report.is_file()
    assert audit['final_transaction_rows']==111
    assert 'temporal feature leakage' in audit['source_snapshot_warning']
    loaded=load_processed_data(output)
    assert len(loaded)==111
    assert len(set(loaded.columns))==len(loaded.columns)


def test_build_config_rejects_future_and_zero_prices():
    with pytest.raises(ValueError):BuildConfig(end_year=2026)
    with pytest.raises(ValueError):BuildConfig(minimum_price=0)


def test_join_preserves_categorical_and_coordinate_values(official_data):
    assert official_data.latitude.between(39.5,41).all()
    assert official_data.longitude.between(-81,-79).all()
    for name in CATEGORICAL_INPUTS:
        assert official_data[name].notna().all()


def test_chunked_sale_processing_across_100k_boundary(tmp_path,county_raw):
    # More than the loader's 100,000-row chunk size; bad rows are discarded.
    path=county_raw/'sales.csv'
    valid=pd.read_csv(path,dtype={'PARID':str})
    invalid=pd.DataFrame([{'PARID':'0000000000000099','SALEDATE':'15/06/2023',
                           'PRICE':500000,'SALEDESC':'MULTI-PARCEL SALE','SALECODE':'H'}]*100050)
    pd.concat([valid,invalid],ignore_index=True).to_csv(path,index=False)
    _,audit=build_official_data(*(county_raw / (k+'.csv') for k in ('assessments','sales','centroids')))
    assert audit['raw_sale_rows']==100162
    assert audit['rejected_invalid_or_uncertain_sale']==100051
    assert audit['final_transaction_rows']==111
