"""Download every input the pipeline reads that is not harvested metadata.

1. Processed beta files for the 11 disorder series, from NCBI GEO. Series-matrix betas only -
   never raw IDATs. (GSE116300's series matrix carries no beta table, so its depositors'
   processed beta-values supplementary file is used instead.)
2. Array manifests for 450K, EPIC v1 and EPIC v2, from the GEO platform records.
3. Two files that GEO does not hold: the published Sotos signature (Choufani et al. 2015,
   Supplementary Data 3 and 7, open access via Europe PMC) and the Chen et al. 2013
   cross-reactive probe list (public mirror). These are the only non-GEO downloads.

Files go to external_dir() (config/paths.yaml), smallest matrix first. A file already present
is skipped and a partial download is resumed. One failure is logged and the rest continue.
"""

from __future__ import annotations

import gzip
import io
import sys
import zipfile

import requests

from epigrade import paths
from epigrade.corpus.loader import SERIES_FILES, matrix_url

GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
SUPPLEMENTARY_URLS = {
    "GSE116300_beta_values.txt.gz":
        f"{GEO}/series/GSE116nnn/GSE116300/suppl/"
        "GSE116300_18.05.30.for.GEO.processed.beta.values.txt.gz",
}
MANIFESTS = {
    "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz":
        f"{GEO}/platforms/GPL13nnn/GPL13534/suppl/"
        "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz",
    "GPL21145_MethylationEPIC_15073387_v-1-0.csv.gz":
        f"{GEO}/platforms/GPL21nnn/GPL21145/suppl/GPL21145_MethylationEPIC_15073387_v-1-0.csv.gz",
}
# EPIC v2 has no manifest file on GEO, only the platform's full table (~2.5 GB of text). It is
# streamed and only the probe-name column is kept.
EPIC_V2_TABLE = ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
                 "?acc=GPL33022&targ=self&form=text&view=data")
EPIC_V2_NAMES = "GPL33022_probe_names.txt"
CHOUFANI_ZIP = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4703864/supplementaryFiles"
CHOUFANI_FILES = {  # member of the Europe PMC zip -> name the pipeline reads
    "ncomms10207-s4.xlsx": "SupplementaryData3_signature_probes.xlsx",
    "ncomms10207-s8.xlsx": "SupplementaryData7_missense_variants.xlsx",
}
CHEN_URL = ("https://github.com/Jfortin1/funnorm_repro/raw/master/bad_probes/"
            "48639-non-specific-probes-Illumina450k.xlsx")
CHEN_NAME = "chen2013_nonspecific_probes.xlsx"
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


def download(url: str, dest, check_gzip: bool = True) -> None:
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code != 416:  # 416: the partial file is already complete
            response.raise_for_status()
            mode = "ab" if response.status_code == 206 else "wb"
            with open(part, mode) as f:
                for chunk in response.iter_content(CHUNK):
                    f.write(chunk)
    if check_gzip and not _intact(part):
        raise OSError(f"{part.name} is not a complete gzip file after download")
    part.replace(dest)


def download_choufani(dest_dir) -> None:
    response = requests.get(CHOUFANI_ZIP, timeout=1800)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member, name in CHOUFANI_FILES.items():
            match = next(m for m in archive.namelist() if m.endswith(member))
            (dest_dir / name).write_bytes(archive.read(match))


def download_epic_v2_names(dest) -> None:
    part = dest.with_name(dest.name + ".part")
    n = 0
    with requests.get(EPIC_V2_TABLE, stream=True, timeout=300) as response, \
            open(part, "w", encoding="utf-8") as out:
        response.raise_for_status()
        for raw in response.iter_lines():
            line = raw.decode("utf-8", errors="replace")
            if not line or line[0] in "!^#":
                continue
            fields = line.split("\t")
            if len(fields) > 2:
                out.write(fields[2] + "\n")
                n += 1
    if n < 900_000:  # the platform has ~930,000 probes; fewer means a truncated stream
        raise OSError(f"EPIC v2 table ended after {n} rows - truncated, rerun to retry")
    part.replace(dest)


def other_inputs() -> list[tuple[str, object]]:
    """(label, zero-argument download function) for every non-matrix input still missing."""
    ext = paths.external_dir()
    jobs = []
    for name, url in MANIFESTS.items():
        if not (ext / name).exists():
            jobs.append((name, lambda url=url, name=name: download(url, ext / name)))
    if not (ext / EPIC_V2_NAMES).exists():
        jobs.append((EPIC_V2_NAMES, lambda: download_epic_v2_names(ext / EPIC_V2_NAMES)))
    choufani_dir = ext / "choufani_2015"
    if not all((choufani_dir / n).exists() for n in CHOUFANI_FILES.values()):
        choufani_dir.mkdir(parents=True, exist_ok=True)
        jobs.append(("Choufani 2015 supplementary data", lambda: download_choufani(choufani_dir)))
    if not (ext / CHEN_NAME).exists():
        jobs.append((CHEN_NAME, lambda: download(CHEN_URL, ext / CHEN_NAME, check_gzip=False)))
    return jobs


def main() -> None:
    pending = []
    for series_id, names in SERIES_FILES.items():
        for name in names:
            url = SUPPLEMENTARY_URLS.get(name, matrix_url(series_id, name))
            dest = paths.external_dir() / name
            cache = paths.interim_dir() / f"{series_id}_betas.parquet"
            if dest.exists() or cache.exists():
                print(f"[{series_id}] {name}: already present")
                continue
            try:
                pending.append((remote_size(url), series_id, name, url))
            except requests.RequestException as exc:
                print(f"[{series_id}] {name}: could not reach GEO for its size ({exc})")
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

    for label, job in other_inputs():
        print(f"[inputs] {label}: downloading...", flush=True)
        try:
            job()
        except (requests.RequestException, OSError, zipfile.BadZipFile, StopIteration) as exc:
            print(f"[inputs] {label}: FAILED ({exc})")
            failures.append(label)

    if failures:
        raise RuntimeError(
            f"{len(failures)} file(s) failed to download: {', '.join(failures)} - rerun to "
            "resume"
        )
    print("All inputs present.")


if __name__ == "__main__":
    main()
