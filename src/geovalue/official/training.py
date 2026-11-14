"""Chronological 2024/2025 holdout with parcel-disjoint models and geographical checks."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from geovalue.official.data import INPUT_COLUMNS, load_processed_data
from geovalue.official.modeling import MODELS, build_model, metrics, save_model


@dataclass(frozen=True)
class Config:
    seed: int = 42
    forest_trees: int = 120
    xgboost_trees: int = 220
    max_training_rows: int | None = None

    def __post_init__(self):
        if self.forest_trees < 1 or self.xgboost_trees < 1:
            raise ValueError('Trees must be positive')
        if self.max_training_rows is not None and self.max_training_rows < 100:
            raise ValueError('max_training_rows must be at least 100 or unlimited')


def _features(df):
    return df.loc[:,list(INPUT_COLUMNS)], df['sale_price']


def split_sales(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    """No parcel appears across train, validation, test (newer partition wins)."""
    needed={'parcel_id','sale_year','sale_price',*INPUT_COLUMNS}
    if missing:=needed-set(data.columns):
        raise ValueError(f'Missing training data columns: {sorted(missing)}')
    train=data.loc[data['sale_year']<=2023].copy()
    val=data.loc[data['sale_year']==2024].copy()
    test=data.loc[data['sale_year']==2025].copy()
    if len(train)<20 or len(val)<5 or len(test)<5:
        raise ValueError('Chronological split requires >=20 sales through 2023 and >=5 each in 2024 and 2025')
    # Prevent any parcel's transaction from leaking across distinct time partitions.
    test_ids=set(test['parcel_id'])
    val=val.loc[~val['parcel_id'].isin(test_ids)].copy()
    val_ids=set(val['parcel_id'])
    train=train.loc[~train['parcel_id'].isin(val_ids | test_ids)].copy()
    if len(train)<20 or len(val)<5:
        raise ValueError('Too few parcel-independent examples remain after excluding repeated sales')
    details={'train_count':len(train),'validation_count':len(val),'test_count':len(test),
             'train_years':'2013–2023','validation_year':2024,'test_year':2025,
             'parcel_overlap_train_validation':len(set(train.parcel_id)&set(val.parcel_id)),
             'parcel_overlap_train_test':len(set(train.parcel_id)&set(test.parcel_id)),
             'parcel_overlap_validation_test':len(set(val.parcel_id)&set(test.parcel_id)),
             'excluded_for_parcel_overlap':len(data.loc[data.sale_year<=2025])-len(train)-len(val)-len(test)}
    return train,val,test,details


def _spatial_experiment(train: pd.DataFrame, name: str, cfg: Config) -> dict:
    """Hold out municipalities inside pre-2024 data; this is a *separate* test."""
    counts=train['municipality'].astype(str).value_counts()
    if len(counts)<3:
        return {'available':False,'reason':'Fewer than 3 municipalities in training data'}
    # Deterministically hold out a small number of municipalities, not random rows.
    groups=sorted(counts.index)
    selected=[group for group in groups if group!='Unknown']
    if len(selected)<3:
        return {'available':False,'reason':'Insufficient known municipality groups'}
    rng=np.random.default_rng(cfg.seed)
    shuffled=list(rng.permutation(selected))
    chosen=shuffled[:max(1,int(round(len(selected)*0.15)))]
    hold=train.loc[train['municipality'].isin(chosen)]
    fit=train.loc[~train['municipality'].isin(chosen)]
    if len(hold)<5 or len(fit)<20:
        return {'available':False,'reason':'Municipality-separated partitions are too small'}
    model=build_model(name,seed=cfg.seed,forest_trees=cfg.forest_trees,xgboost_trees=cfg.xgboost_trees)
    x,y=_features(fit)
    model.fit(x,y)
    hx,hy=_features(hold)
    return {'available':True,'holdout_municipalities':[str(v) for v in chosen],
            'train_rows':len(fit),'holdout_rows':len(hold),
            'metrics':metrics(hy,model.predict(hx)),
            'interpretation':'Distinct municipalities within pre-2024 sales; not a prospective year-2025 test.'}


def train_evaluate(data: pd.DataFrame, cfg: Config | None = None):
    cfg=cfg or Config()
    train,val,test,split=split_sales(data)
    if cfg.max_training_rows is not None and len(train)>cfg.max_training_rows:
        train=train.sample(n=cfg.max_training_rows,random_state=cfg.seed).sort_values('sale_year')
        split['training_rows_after_optional_sampling']=len(train)
    x,y=_features(train)
    xv,yv=_features(val)
    comparisons={}
    for name in MODELS:
        model=build_model(name,seed=cfg.seed,forest_trees=cfg.forest_trees,xgboost_trees=cfg.xgboost_trees)
        model.fit(x,y)
        comparisons[name]={'train':metrics(y,model.predict(x)),
                           'validation':metrics(yv,model.predict(xv))}
    chosen=min(MODELS,key=lambda n: comparisons[n]['validation']['rmse'])
    without=build_model(chosen,seed=cfg.seed,forest_trees=cfg.forest_trees,
                        xgboost_trees=cfg.xgboost_trees,geography=False)
    without.fit(x,y)
    without_metric=metrics(yv,without.predict(xv))
    spatial=_spatial_experiment(train,chosen,cfg)
    # No parcels in the 2025 partition were used during any fitting, even in 2024.
    development=pd.concat([train,val],ignore_index=True)
    xd,yd=_features(development)
    final=build_model(chosen,seed=cfg.seed,forest_trees=cfg.forest_trees,xgboost_trees=cfg.xgboost_trees)
    final.fit(xd,yd)
    xt,yt=_features(test)
    held_out=metrics(yt,final.predict(xt))
    report={'data_source':'Allegheny County / WPRDC official transaction, assessment and centroid records',
            'target':'sale_price','target_units':'recorded sale USD (year dependent, unadjusted for inflation)',
            'model_selected':chosen,'selection_rule':'lowest 2024 validation RMSE',
            'partition':split, 'model_metrics':comparisons,
            'geographical_ablation':{'excluded_features':['latitude','longitude','municipality','zip_code'],
                                      'validation_with_geo':comparisons[chosen]['validation'],
                                      'validation_without_geo':without_metric,
                                      'rmse_reduction_usd':without_metric['rmse']-comparisons[chosen]['validation']['rmse']},
            'municipality_holdout':spatial,'held_out_2025_test':held_out,
            'final_development_rows':len(development),
            'configuration':{'seed':cfg.seed,'forest_trees':cfg.forest_trees,'xgboost_trees':cfg.xgboost_trees,
                             'max_training_rows':cfg.max_training_rows},
            'software':{'python':platform.python_version(),**{k:version(k) for k in ['pandas','numpy','scikit-learn','xgboost','joblib']}},
            'major_limitations':[
                'Most recent assessment snapshot can contain post-sale modifications and assessment-time characteristics; temporal feature leakage remains a risk.',
                'Recorded sales are only a subset of the property stock; only explicit county VALID SALE descriptions are included.',
                'No parcel overlaps across chronological partitions, but municipality generalization is a separate exploratory experiment.',
                'Sale prices from different years are nominal, not inflation-adjusted.',
                'No features derived from past/current assessed value, current sale price, or previous recorded sale are used.',
                'Historic county sales may be corrected/reclassified; results depend on official source extraction date.'
            ]}
    return final,chosen,report


def run_training(data_path: Path, model_path: Path, report_path: Path, cfg: Config | None = None) -> dict:
    data=load_processed_data(data_path)
    final,chosen,report=train_evaluate(data,cfg)
    model_file=save_model(final,chosen,model_path)
    report_path=Path(report_path)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {'selected_model':chosen,'test_2025':report['held_out_2025_test'],
            'train_rows':report['partition']['train_count'],
            'validation_rows':report['partition']['validation_count'],
            'test_rows':report['partition']['test_count'],
            'model_path':str(model_file),'report_path':str(report_path)}
