"""Small, deterministic county-shaped fixtures; never misrepresented as official data."""
import numpy as np
import pandas as pd
import pytest

from geovalue.official.data import INPUT_COLUMNS


@pytest.fixture
def county_raw(tmp_path):
    raw=tmp_path/'county';raw.mkdir()
    rows=[];sales=[];centroids=[]
    for i in range(110):
        key=f'{i:016d}'
        year=2023 if i<75 else 2024 if i<92 else 2025
        city=f'Municipality_{i%8}'
        rows.append({'PARID':key,'CLASS':'R','YEARBLT':1970+i%40,
                     'BEDROOMS':1+i%5,'FULLBATHS':1+i%3,'HALFBATHS':i%2,
                     'TOTALROOMS':4+i%8,'FINISHEDLIVINGAREA':850+(i%21)*95,
                     'LOTAREA':4500+(i%35)*110,'STORIES':1+(i%3)*0.5,
                     'GRADE':'C' if i%3 else 'B','CONDITION':'3',
                     'HEATINGCOOLINGDESC':'Forced Air','MUNIDESC':city,
                     'PROPERTYZIP':str(15200+i%20),'FIREPLACES':i%2,
                     'BSMTGARAGE':i%3})
        sales.append({'PARID':key,'SALEDATE':f'15/06/{year}','PRICE':160000+i*2000,
                      'SALECODE':'0','SALEDESC':'VALID SALE'})
        centroids.append({'PIN':key,'LAT':40.34+(i%9)*0.013,
                          'LONG':-80.08+(i%13)*0.014})
    # Valid parcel with recorded sales in two periods: keep out of earlier groups.
    sales.append({'PARID':f'{0:016d}','SALEDATE':'15/07/2025','PRICE':285000,'SALECODE':'0','SALEDESC':'VALID SALE'})
    # Multi-parcel sale must be removed despite plausible price.
    sales.append({'PARID':f'{1:016d}','SALEDATE':'15/07/2024','PRICE':380000,'SALECODE':'H','SALEDESC':'MULTI-PARCEL SALE'})
    pd.DataFrame(rows).to_csv(raw/'assessments.csv',index=False)
    pd.DataFrame(sales).to_csv(raw/'sales.csv',index=False)
    pd.DataFrame(centroids).to_csv(raw/'centroids.csv',index=False)
    return raw


@pytest.fixture
def official_data(county_raw):
    from geovalue.official.data import build_official_data
    return build_official_data(county_raw/'assessments.csv',county_raw/'sales.csv',county_raw/'centroids.csv')[0]
