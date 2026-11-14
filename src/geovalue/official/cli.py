"""Separate official-data CLI, preserving the original GeoValue California commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geovalue.official.data import BuildConfig, INPUT_COLUMNS
from geovalue.official.sources import SOURCE_PAGES, RESOURCE_IDS

RAW=Path('data/official/raw')
PROCESSED=Path('data/official/processed/residential_sales_2013_2025.csv')
MODEL=Path('artifacts/official/residential-sale-model.joblib')
TRAIN_REPORT=Path('reports/official/training-report.json')
BUILD_REPORT=Path('reports/official/build-report.json')


def parser() -> argparse.ArgumentParser:
    parent=argparse.ArgumentParser(prog='geovalue-official',description='Official Allegheny County residential sale analysis (Tasks 6–7)')
    sub=parent.add_subparsers(dest='command',required=True)
    sub.add_parser('sources', help='Show the official source pages and verified CKAN resource IDs')
    dl=sub.add_parser('download', help='Download three official WPRDC CSVs (large files)')
    dl.add_argument('--raw-dir',type=Path,default=RAW)
    dl.add_argument('--overwrite',action='store_true')
    build=sub.add_parser('build',help='Join valid recorded sales, assessment features and coordinates')
    build.add_argument('--raw-dir',type=Path,default=RAW)
    build.add_argument('--output',type=Path,default=PROCESSED)
    build.add_argument('--report',type=Path,default=BUILD_REPORT)
    build.add_argument('--start-year',type=int,default=2013)
    build.add_argument('--end-year',type=int,default=2025)
    build.add_argument('--minimum-price',type=float,default=10000)
    build.add_argument('--maximum-price',type=float,default=5000000)
    prof=sub.add_parser('profile',help='Inspect processed sales count, years and missing fields')
    prof.add_argument('--data',type=Path,default=PROCESSED)
    tr=sub.add_parser('train',help='Train three models and chronologically evaluate against 2024/2025')
    tr.add_argument('--data',type=Path,default=PROCESSED)
    tr.add_argument('--model-output',type=Path,default=MODEL)
    tr.add_argument('--report-output',type=Path,default=TRAIN_REPORT)
    tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--forest-trees',type=int,default=120)
    tr.add_argument('--xgboost-trees',type=int,default=220)
    tr.add_argument('--max-training-rows',type=int,help='Optional train partition cap for memory-limited Macs; default uses all')
    pred=sub.add_parser('predict',help='Batch predictions on CSV with property features')
    pred.add_argument('--model',type=Path,default=MODEL)
    pred.add_argument('--input',type=Path,required=True)
    pred.add_argument('--output',type=Path,default=Path('reports/official/predictions.csv'))
    info=sub.add_parser('model-info', help='Inspect trusted locally saved official model metadata')
    info.add_argument('--model',type=Path,default=MODEL)
    analysis=sub.add_parser('analyze',help='Generate Pennsylvania calculus, reliability calibration and local dashboard assets')
    analysis.add_argument('--data',type=Path,default=PROCESSED)
    analysis.add_argument('--model',type=Path,default=MODEL)
    analysis.add_argument('--training-report',type=Path,default=TRAIN_REPORT)
    analysis.add_argument('--report-dir',type=Path,default=Path('reports/official'))
    analysis.add_argument('--public-dir',type=Path,default=Path('public/official/data'))
    analysis.add_argument('--artifact-dir',type=Path,default=Path('artifacts/official'))
    analysis.add_argument('--grid-size',type=int,default=86)
    analysis.add_argument('--sigma',type=float,default=1.5)
    analysis.add_argument('--support-km',type=float,default=4.5)
    analysis.add_argument('--map-points',type=int,default=1600)
    audit=sub.add_parser('audit',help='Inspect contradictory recorded sale prices for identical parcel/date')
    audit.add_argument('--data',type=Path,default=PROCESSED)
    audit.add_argument('--output',type=Path,default=Path('reports/official/transaction-audit.json'))
    diagnose=sub.add_parser('diagnose',help='Investigate extreme 2024 Linear Regression validation predictions')
    diagnose.add_argument('--data',type=Path,default=PROCESSED)
    diagnose.add_argument('--output',type=Path,default=Path('reports/official/linear-diagnostics.json'))
    diagnose.add_argument('--top-n',type=int,default=5)
    return parent


def main(argv: list[str] | None=None) -> int:
    args=parser().parse_args(argv)
    try:
        if args.command=='sources':
            result={key:{'url':SOURCE_PAGES[key],'resource_id':rid} for key,rid in RESOURCE_IDS.items()}
        elif args.command=='download':
            from geovalue.official.sources import download_sources
            result=download_sources(args.raw_dir,overwrite=args.overwrite)
        elif args.command=='build':
            from geovalue.official.data import run_build
            result=run_build(args.raw_dir,args.output,args.report,BuildConfig(
                start_year=args.start_year,end_year=args.end_year,
                minimum_price=args.minimum_price,maximum_price=args.maximum_price))
        elif args.command=='profile':
            from geovalue.official.data import load_processed_data
            df=load_processed_data(args.data)
            result={'rows':len(df),'unique_parcels':int(df.parcel_id.nunique()),
                    'sales_by_year':{str(k):int(v) for k,v in df.sale_year.value_counts().sort_index().items()},
                    'missing_features':{col:int(df[col].isna().sum()) for col in INPUT_COLUMNS},
                    'median_sale_price_usd':float(df.sale_price.median()),
                    'price_range_usd':[float(df.sale_price.min()),float(df.sale_price.max())]}
        elif args.command=='train':
            from geovalue.official.training import Config,run_training
            result=run_training(args.data,args.model_output,args.report_output,
                                Config(seed=args.seed,forest_trees=args.forest_trees,
                                       xgboost_trees=args.xgboost_trees,
                                       max_training_rows=args.max_training_rows))
        elif args.command=='predict':
            import pandas as pd
            from geovalue.official.modeling import predict
            values=pd.read_csv(args.input)
            result_frame=predict(args.model,values)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            result_frame.to_csv(args.output,index=False)
            result={'output_csv':str(args.output),'predicted_records':len(result_frame)}
        elif args.command=='analyze':
            from geovalue.official.analysis import build_analysis
            result=build_analysis(args.data,args.model,args.training_report,
                                  report_dir=args.report_dir, public_dir=args.public_dir,
                                  artifact_dir=args.artifact_dir,grid_size=args.grid_size,
                                  sigma=args.sigma,support_km=args.support_km,
                                  max_points=args.map_points)
        elif args.command=='diagnose':
            from geovalue.official.analysis import diagnose_linear_baseline
            result=diagnose_linear_baseline(args.data,args.output,top_n=args.top_n)
        elif args.command=='audit':
            from geovalue.official.analysis import audit_same_day_sales
            from geovalue.official.data import load_processed_data
            result=audit_same_day_sales(load_processed_data(args.data))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        elif args.command=='model-info':
            from geovalue.official.modeling import load_model
            result={key:value for key,value in load_model(args.model).items() if key!='pipeline'}
        else:
            raise ValueError('Unsupported command')
        print(json.dumps(result,indent=2,default=str,allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
