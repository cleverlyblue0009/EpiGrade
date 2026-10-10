"""Download the processed beta files for every series in the corpus from NCBI GEO.

Processed series-matrix betas only - never raw IDATs. Files go to external_dir()
(config/paths.yaml), smallest first; a file already present and intact is skipped, a partial
download is resumed. One failure is logged and the rest continue.

GSE116300 is the one exception to "series matrix": its series matrix carries no beta table, so
its depositors' processed beta-values supplementary file is used instead.
"""

from __future__ import annotations

import gzip
import sys

import requests

from epigrade import paths
from epigrade.corpus.loader import SERIES_FILES, matrix_url

SUPPLEMENTARY_URLS = {
    "GSE116300_beta_values.txt.gz":
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116300/suppl/"
        "GSE116300_18.05.30.for.GEO.processed.beta.values.txt.gz",
}
CHUNK = 1 << 20


def _intact(path) -> bool:
    """A gzip that reads to its end without error."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(CHUNK * 8):
                pass
        return True
    except (OSError, EOFError):
        return False


def remote_size(url: str) -> int:
    response = requests.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    return int(response.headers.get("Content-Length", 0))


def download(url: str, dest) -> None:
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code == 416:  # the partial file is already complete
            pass
        else:
            response.raise_for_status()
            mode = "ab" if response.status_code == 206 else "wb"
            with open(part, mode) as f:
                for chunk in response.iter_content(CHUNK):
                    f.write(chunk)
    if not _intact(part):
        raise OSError(f"{part.name} is not a complete gzip file after download")
    part.replace(dest)


def main() -> None:
    targets = []
    for series_id, names in SERIES_FILES.items():
        for name in names:
            url = SUPPLEMENTARY_URLS.get(name, matrix_url(series_id, name))
            targets.append((series_id, name, url))

    pending = []
    for series_id, name, url in targets:
        dest = paths.external_dir() / name
        cache = paths.interim_dir() / f"{series_id}_betas.parquet"
        if dest.exists() or cache.exists():
            print(f"[{series_id}] {name}: already present")
            continue
        try:
            pending.append((remote_size(url), series_id, name, url))
        except requests.RequestException as exc:
            print(f"[{series_id}] {name}: FAILED to reach GEO ({exc})")
            pending.append((sys.maxsize, series_id, name, url))

    failures = []
    for size, series_id, name, url in sorted(pending):
        label = f"{size / 1e6:.0f} MB" if size < sys.maxsize else "size unknown"
        print(f"[{series_id}] {name}: downloading ({label})...", flush=True)
        try:
            download(url, paths.external_dir() / name)
        except (requests.RequestException, OSError) as exc:
            print(f"[{series_id}] {name}: FAILED ({exc})")
            failures.append(name)
    if failures:
        raise RuntimeError(
            f"{len(failures)} file(s) failed to download: {', '.join(failures)} - rerun to "
            "resume"
        )


if __name__ == "__main__":
    main()
