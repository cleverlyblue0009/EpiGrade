"""Parse a GEO series matrix file (processed beta values) into a probe x sample DataFrame.

We use the series matrix, not GEOparse's per-sample table fetch and not raw IDATs - this is the
one place in the pipeline that needs real beta values, and it comes as a single already-
processed file per the project's network constraint (GEO only, no IDAT-level reprocessing).
"""

from __future__ import annotations

import gzip

import pandas as pd


def parse_series_matrix(gz_path) -> pd.DataFrame:
    """Return a DataFrame indexed by probe ID (rows) with one column per GSM accession."""
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    start = end = None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            start = i + 1
        elif line.startswith("!series_matrix_table_end"):
            end = i
            break
    if start is None:
        raise ValueError(f"'!series_matrix_table_begin' not found in {gz_path}")
    if end is None:
        end = len(lines)

    from io import StringIO

    table_text = "".join(lines[start:end])
    df = pd.read_csv(StringIO(table_text), sep="\t", index_col=0, quotechar='"')
    df.index.name = "probe_id"
    df.columns = [c.strip('"') for c in df.columns]
    return df


def parse_beta_values_by_slide_position(gz_path, meta: pd.DataFrame) -> pd.DataFrame:
    """Parse a GEO *_processed.beta.values.txt.gz supplementary file (a plain probe x sample
    TSV with NO !series_matrix_table_begin/end markers - some depositors, e.g. GSE116300, ship
    beta values this way instead of embedding them in the series matrix itself).

    Columns in this file format are "<slide_id>_<array_position>" (e.g. "8655685138_R05C02"),
    not GSM accessions - renamed here using the slide_id/array_position characteristics fields
    already harvested into `meta` (indexed by gsm_accession), which is a complete 1:1 mapping
    verified against every sample in the series (see docs/METHODS.md).
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    df.index.name = "probe_id"

    if "slide_id" not in meta.columns or "array_position" not in meta.columns:
        raise ValueError(
            "meta is missing slide_id/array_position columns needed to map this file's "
            "columns to GSM accessions"
        )
    key_to_gsm = {
        f"{row.slide_id}_{row.array_position}": gsm for gsm, row in meta.iterrows()
        if pd.notna(row.slide_id) and pd.notna(row.array_position)
    }
    missing = [c for c in df.columns if c not in key_to_gsm]
    if missing:
        raise ValueError(
            f"{len(missing)} column(s) in {gz_path} have no matching GSM via "
            f"slide_id/array_position (first few: {missing[:5]}) - mapping is incomplete, "
            "refusing to silently drop or misassign samples"
        )
    df.columns = [key_to_gsm[c] for c in df.columns]
    return df
