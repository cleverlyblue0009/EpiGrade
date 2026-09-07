"""Task 4: one-command reproduction, from a clean clone to every figure and table.

    python scripts/reproduce.py            # full pipeline, skips stages already done
    python scripts/reproduce.py --force    # rerun every stage regardless of markers
    python scripts/reproduce.py --quick    # cached-data stages only, for demo use - no
                                            # network access, just regenerates reports/resource
                                            # from whatever is already downloaded/harvested

Each stage writes a marker file to D:/epigrade_data/interim/.reproduce_markers/ on success and
is skipped on a later run unless --force is passed (or the marker is missing). A failed stage is
reported plainly and does NOT stop the rest of the pipeline - a partial, honestly-reported run
beats an all-or-nothing one, consistent with the rest of this project.

Network/download stages (need GEO access): harvest, sotos_demo, srs_matrix, kabuki_charge.
Cached-data-only stages (--quick runs only these): triage_report, calibration,
confounding_gate, summary_figure, provenance, resource_export.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from epigrade import paths

MARKER_DIR = paths.interim_dir() / ".reproduce_markers"

# (stage_name, module_path, needs_network)
STAGES = [
    ("harvest", "epigrade.acquire.harvest", True),
    ("triage_report", "phase1_report", False),
    ("sotos_demo", "demo_sotos", True),
    ("srs_matrix", "phase4_6_srs_and_matrix", True),
    ("kabuki_charge", "phase4_kabuki_charge", True),
    ("calibration", "phase5_calibration", False),
    ("confounding_gate", "phase6_confounding_gate", False),
    ("summary_figure", "make_summary_figure", False),
    ("provenance", "generate_provenance", False),
    ("resource_export", "epigrade.report.resource", False),
]


def _import_stage(module_path: str):
    """Scripts in scripts/ aren't a package; add it to sys.path once so plain module names
    (not dotted epigrade.* paths) import correctly regardless of the caller's cwd."""
    scripts_dir = str(paths.repo_root() / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module(module_path)


def _marker_path(stage_name: str):
    return MARKER_DIR / f"{stage_name}.done"


def run_stage(stage_name: str, module_path: str, force: bool) -> str:
    marker = _marker_path(stage_name)
    if marker.exists() and not force:
        return "skipped (already done - pass --force to rerun)"

    t0 = time.time()
    try:
        module = _import_stage(module_path)
        module.main()
    except Exception as exc:  # noqa: BLE001 - one stage failing must not kill the pipeline
        return f"FAILED after {time.time() - t0:.0f}s: {exc}"

    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"completed in {time.time() - t0:.0f}s\n", encoding="utf-8")
    return f"done in {time.time() - t0:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rerun every stage")
    parser.add_argument(
        "--quick", action="store_true",
        help="cached-data stages only (no network) - regenerates reports/resource from "
             "whatever is already downloaded/harvested",
    )
    args = parser.parse_args()

    print("EpiGrade reproduction pipeline")
    print(f"  mode: {'quick (cached-data only)' if args.quick else 'full'}"
          f"{', force rerun' if args.force else ''}")
    print(f"  marker directory: {MARKER_DIR}\n")

    results = {}
    for stage_name, module_path, needs_network in STAGES:
        if args.quick and needs_network:
            results[stage_name] = "skipped (--quick: needs network)"
            print(f"[{stage_name}] skipped (--quick: needs network)")
            continue
        print(f"[{stage_name}] running...")
        outcome = run_stage(stage_name, module_path, args.force)
        results[stage_name] = outcome
        print(f"[{stage_name}] {outcome}")

    print("\n--- Summary ---")
    for stage_name, outcome in results.items():
        print(f"  {stage_name:20s} {outcome}")

    n_failed = sum(1 for o in results.values() if o.startswith("FAILED"))
    if n_failed:
        print(f"\n{n_failed} stage(s) failed - see the FAILED lines above. This is reported "
              "plainly, not hidden; rerun scripts/reproduce.py once the underlying issue "
              "(usually network access to NCBI GEO) is resolved. Stages that already succeeded "
              "will be skipped, not redone.")
    else:
        print("\nAll stages completed (or were already done). "
              f"Resource: {paths.resource_dir() / 'epigrade_v1.json'}")
        print("Run `streamlit run src/epigrade/app/main.py` to view it.")


if __name__ == "__main__":
    main()
