"""Source provenance tests; network calls are fully mocked."""
import hashlib
import io
import json

import pytest

from geovalue.official import sources


class FakeOpener:
    def __init__(self, *, bad_domain=False, html=False, wrong_id=False):
        self.requested=[]
        self.bad_domain=bad_domain
        self.html=html
        self.wrong_id=wrong_id

    def __call__(self,request,timeout=0):
        url=request.full_url
        self.requested.append(url)
        for key,rid in sources.RESOURCE_IDS.items():
            if f'resource_show?id={rid}' in url:
                meta={'success':True,'result':{'id':('wrong' if self.wrong_id else rid),
                       'format':'CSV','url': ('https://bad.invalid/sales.csv' if self.bad_domain else
                               f'https://data.wprdc.org/mock/{key}.csv'),
                       'last_modified':'2025-01-01'}}
                return io.BytesIO(json.dumps(meta).encode())
            if url.endswith(f'/{key}.csv'):
                if self.html:
                    return io.BytesIO(b'<html><body>Access denied</body></html>')
                header=b'PIN,LAT,LONG\n' if key=='centroids' else b'PARID,X\n'
                return io.BytesIO(header+b'0000000000000001,2\n')
        raise RuntimeError('Unexpected URL '+url)


def test_resource_metadata_has_correct_provenance():
    transport=FakeOpener()
    meta=sources.get_metadata('sales',opener=transport)
    assert meta['resource_id']==sources.RESOURCE_IDS['sales']
    assert meta['download_url'].startswith('https://data.wprdc.org/')
    assert len(transport.requested)==1


def test_download_creates_manifest_and_preserves_existing_data(tmp_path):
    transport=FakeOpener()
    manifest=sources.download_sources(tmp_path,opener=transport)
    assert set(manifest['sources'])==set(sources.RESOURCE_IDS)
    for key,name in sources.SOURCE_NAMES.items():
        path=tmp_path/name
        assert path.is_file()
        assert manifest['sources'][key]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    # A changed local file is preserved unless --overwrite is requested.
    target=tmp_path/'sales.csv'
    target.write_bytes(b'PARID,CUSTOM\n0000000000000009,1\n')
    second=sources.download_sources(tmp_path,opener=transport)
    assert b'CUSTOM' in target.read_bytes()
    assert second['sources']['sales']['status']=='existing-file-preserved'
    assert (tmp_path/'source-manifest.json').is_file()


def test_cannot_download_non_official_metadata_url():
    with pytest.raises(ValueError,match='official'):
        sources.get_metadata('sales',opener=FakeOpener(bad_domain=True))


def test_cannot_download_html_and_no_partial_file_remains(tmp_path):
    with pytest.raises(ValueError,match='CSV header'):
        sources.download_sources(tmp_path,opener=FakeOpener(html=True))
    assert not list(tmp_path.glob('*.part'))


def test_incorrect_resource_id_is_rejected():
    with pytest.raises(ValueError,match='Unexpected resource'):
        sources.get_metadata('sales',opener=FakeOpener(wrong_id=True))


def test_missing_raw_sources_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError,match='Missing official sources'):
        sources.check_source_files(tmp_path)
