from pathlib import Path
import requests
import gzip
import shutil


def download_gz(url: str, path: Path):
    tmp = Path('tmp')
    if not tmp.exists():
        tmp.mkdir()
    tmp = tmp / 'tmp.gz'

    response = requests.get(url, stream=True)
    with tmp.open('wb') as dst:
        for data in response.iter_content():
            dst.write(data)
    with gzip.open(tmp, 'rb') as src:
        with path.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    tmp.unlink()
