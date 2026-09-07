"""Probe filtering against Chen et al. 2013 cross-reactive/SNP probe lists.

Provenance: the cross-reactive probe list (29,233 probes) is Chen et al.'s own original file,
sourced from the Weksberg lab (the same lab that produced the Choufani et al. 2015 Sotos paper
this project reproduces): originally hosted at sickkids.ca, retrieved here via the
Jfortin1/funnorm_repro GitHub mirror (see docs/METHODS.md for the exact URL and retrieval date).

HONEST LIMITATION: Chen et al. 2013 also defines a SNP-containing probe exclusion (probes with a
SNP at the CpG or single-base-extension site). That list could not be retrieved as a direct,
citable machine-readable file from a public mirror in this project - commonly-used substitutes
(e.g. minfi's getSnpInfo() against dbSNP137) are a DIFFERENT, independently-derived probe set,
not Chen et al.'s own list, so it is not applied here as if it were. The probe count this module
reports is therefore cross-reactive-filtered only, and is compared honestly against the paper's
424,586 target rather than forced to match it - see the printed report and docs/LIMITATIONS.md.
"""

from __future__ import annotations

import openpyxl
import pandas as pd

from epigrade import paths

CHEN_CROSS_REACTIVE_PATH = "chen2013_nonspecific_probes.xlsx"
PAPER_TARGET_RETAINED = 424_586


def load_cross_reactive_probes() -> set[str]:
    """The 29,233 cross-reactive cg/ch probes from Chen et al. 2013 (both sheets)."""
    path = paths.external_dir() / CHEN_CROSS_REACTIVE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See docs/METHODS.md for the retrieval command "
            "(Jfortin1/funnorm_repro mirror of the original Weksberg-lab file)."
        )
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    probes = set()
    for sheet_name in ("nonspecific cg probes", "nonspecific ch probes"):
        ws = wb[sheet_name]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0]:
                probes.add(row[0])
    wb.close()
    return probes


def filter_probes(beta_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove Chen et al. cross-reactive probes. Returns (filtered_df, report_dict).

    report_dict is written verbatim to results/tables/probe_filtering.tsv - never hand-edit
    the retained count to match the paper; report what was actually computed.
    """
    cross_reactive = load_cross_reactive_probes()
    total_before = len(beta_df)
    mask = ~beta_df.index.isin(cross_reactive)
    filtered = beta_df[mask]
    report = {
        "total_probes_before_filtering": total_before,
        "cross_reactive_probes_in_chen_list": len(cross_reactive),
        "cross_reactive_probes_actually_present_in_data": total_before - mask.sum(),
        "probes_retained": len(filtered),
        "paper_target_retained_424586": PAPER_TARGET_RETAINED,
        "difference_from_paper_target": len(filtered) - PAPER_TARGET_RETAINED,
        "note": (
            "Cross-reactive filter only (exact Chen et al. list). The paper's SNP-containing "
            "probe exclusion could not be sourced as a direct machine-readable file from a "
            "public mirror in this session - see docs/LIMITATIONS.md. This is reported "
            "honestly rather than forced to match 424,586."
        ),
    }
    return filtered, report
