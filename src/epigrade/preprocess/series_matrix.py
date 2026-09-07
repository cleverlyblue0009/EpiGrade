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
