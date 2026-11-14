import json

from fastapi.testclient import TestClient
from geovalue.official import cli,web
from geovalue.web import app


def test_sources_have_official_ckan_ids(capsys):
    assert cli.main(['sources'])==0
    data=json.loads(capsys.readouterr().out)
    assert all('data.wprdc.org' in obj['url'] for obj in data.values())


def test_official_cli_build_and_profile(tmp_path,county_raw,capsys):
    out=tmp_path/'processed.csv'
    assert cli.main(['build','--raw-dir',str(county_raw),'--output',str(out),
                     '--report',str(tmp_path/'audit.json')])==0
    assert cli.main(['profile','--data',str(out)])==0
    assert 'unique_parcels' in capsys.readouterr().out


def test_missing_model_reports_503_and_not_a_fake_prediction(tmp_path,monkeypatch,official_data):
    monkeypatch.setattr(web,'official_model_path',lambda:tmp_path/'missing.joblib')
    client=TestClient(app)
    assert client.get('/api/official/health').json()['status']=='model_missing'
    request=official_data.iloc[0].loc[list(web.PropertyInput.model_fields)].to_dict()
    response=client.post('/api/official/predict',json=request)
    assert response.status_code==503


def test_official_api_predicts_from_saved_pipeline(tmp_path,monkeypatch,official_data):
    from geovalue.official.modeling import build_model,save_model
    from geovalue.official.data import INPUT_COLUMNS
    train=official_data.loc[:,list(INPUT_COLUMNS)]
    model=build_model('xgboost',xgboost_trees=3).fit(train,official_data.sale_price)
    path=save_model(model,'xgboost',tmp_path/'official.joblib')
    monkeypatch.setattr(web,'official_model_path',lambda:path)
    payload=official_data.iloc[0].loc[list(web.PropertyInput.model_fields)].to_dict()
    response=TestClient(app).post('/api/official/predict',json=payload)
    assert response.status_code==200,response.text
    assert response.json()['predicted_sale_price_usd']>0
    assert response.json()['price_basis'].startswith('recorded historical')


def test_cli_training_prediction_and_model_info(tmp_path,official_data,capsys):
    data=tmp_path/'processed.csv';official_data.to_csv(data,index=False)
    model=tmp_path/'model.joblib';report=tmp_path/'report.json'
    result=cli.main(['train','--data',str(data),'--model-output',str(model),
                     '--report-output',str(report),'--forest-trees','3','--xgboost-trees','3'])
    assert result==0
    assert model.is_file() and report.is_file()
    assert cli.main(['model-info','--model',str(model)])==0
    example=tmp_path/'records.csv'
    from geovalue.official.data import INPUT_COLUMNS
    official_data.loc[:,list(INPUT_COLUMNS)].head(3).to_csv(example,index=False)
    output=tmp_path/'prediction.csv'
    assert cli.main(['predict','--model',str(model),'--input',str(example),
                     '--output',str(output)])==0
    import pandas as pd
    assert len(pd.read_csv(output))==3


def test_cli_rejects_missing_processed_dataset(capsys):
    assert cli.main(['profile','--data','/not/a/real/path.csv'])==2
    assert 'missing' in capsys.readouterr().err.lower()
