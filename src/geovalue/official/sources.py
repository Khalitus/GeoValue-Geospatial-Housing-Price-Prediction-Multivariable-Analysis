"""Verified WPRDC resource identifiers and safe, provenance-recording CSV downloads."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PORTAL = 'https://data.wprdc.org'
RESOURCE_IDS = {
    'assessments': '9a1c60bd-f9f7-4aba-aeb7-af8c3aaa44e5',
    'sales': '5bbe6c55-bce6-4edb-9d04-68edeb6bf7b1',
    'centroids': '3fab7152-3f11-4788-8372-4c33f86ea813',
}
SOURCE_PAGES = {
    'assessments': f'{PORTAL}/dataset/property-assessments/resource/{RESOURCE_IDS["assessments"]}',
    'sales': f'{PORTAL}/dataset/real-estate-sales/resource/{RESOURCE_IDS["sales"]}',
    'centroids': f'{PORTAL}/dataset/parcel-centroids-in-allegheny-county-with-geographic-identifiers/resource/{RESOURCE_IDS["centroids"]}',
}
SOURCE_NAMES = {key: key + '.csv' for key in RESOURCE_IDS}


def get_metadata(key: str, *, opener=urlopen) -> dict:
    """Use CKAN's resource_show metadata instead of guessing a download URL."""
    if key not in RESOURCE_IDS:
        raise ValueError(f'Unknown WPRDC resource: {key}')
    url = f'{PORTAL}/api/3/action/resource_show?id={RESOURCE_IDS[key]}'
    request = Request(url, headers={'User-Agent': 'GeoValue-University-Research/1.0'})
    with opener(request, timeout=45) as response:
        result = json.load(response)
    if result.get('success') is not True or not isinstance(result.get('result'), dict):
        raise ValueError(f'Invalid CKAN metadata for {key}')
    resource = result['result']
    if str(resource.get('id', '')).lower() != RESOURCE_IDS[key]:
        raise ValueError(f'Unexpected resource identifier for {key}')
    source_url = str(resource.get('url') or '').strip()
    if not source_url.startswith(('https://data.wprdc.org/', 'https://www.data.wprdc.org/')):
        raise ValueError('Refusing download URL outside official WPRDC host')
    return {'name': key, 'resource_id': resource['id'], 'source_page': SOURCE_PAGES[key],
            'download_url': source_url, 'last_modified': resource.get('last_modified'),
            'format': resource.get('format'), 'description': resource.get('description')}


def download_sources(directory: Path, *, overwrite: bool = False, opener=urlopen) -> dict:
    """Download three official CSVs, with atomic replacement and SHA-256 manifest.

    Files already present are preserved unless explicit overwrite=True. All downloads
    are performed via the resource URL returned by CKAN, not an assumed external URL.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'sources': {}}
    for key, name in SOURCE_NAMES.items():
        target = directory / name
        resource = get_metadata(key, opener=opener)
        if target.is_file() and not overwrite:
            resource['status'] = 'existing-file-preserved'
        else:
            temporary = target.with_suffix('.csv.part')
            sha = hashlib.sha256()
            try:
                request = Request(resource['download_url'], headers={'User-Agent': 'GeoValue-University-Research/1.0'})
                with opener(request, timeout=120) as response, temporary.open('wb') as output:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        sha.update(chunk)
                        output.write(chunk)
                # A portal access/terms error can return HTML; never call that a CSV.
                with temporary.open('rb') as check:
                    header = check.read(4096).decode('utf-8-sig', errors='replace').upper()
                expected = 'PIN' if key == 'centroids' else 'PARID'
                if expected not in header.split('\n', 1)[0]:
                    raise ValueError(f'{key} download did not contain the expected CSV header. Review WPRDC access terms.')
                os.replace(temporary, target)
                resource['status'] = 'downloaded'
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        resource['size_bytes'] = target.stat().st_size
        # SHA also identifies manually placed, previously downloaded resources.
        digest = hashlib.sha256()
        with target.open('rb') as file:
            for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
                digest.update(block)
        resource['sha256'] = digest.hexdigest()
        manifest['sources'][key] = resource
    (directory / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def check_source_files(directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    missing = [file for file in SOURCE_NAMES.values() if not (directory / file).is_file()]
    if missing:
        raise FileNotFoundError('Missing official sources: ' + ', '.join(missing) + '. See docs/TASK_6.md.')
    return {key: directory / filename for key, filename in SOURCE_NAMES.items()}
