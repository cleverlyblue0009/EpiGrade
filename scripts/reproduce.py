"""One-command reproduction, from a clean clone to every figure and table.

    python scripts/reproduce.py            # full pipeline, skips stages already done
    python scripts/reproduce.py --force    # rerun every stage regardless of markers
    python scripts/reproduce.py --quick    # no network: rebuild everything that can be rebuilt
                                           # from what is already on disk

Each stage writes a marker file under interim_dir()/.reproduce_markers/ on success and is
skipped on a later run unless --force is passed. A failed stage is reported plainly and does
NOT stop the rest of the pipeline.

Every stage declares what it needs:
  network  downloads from NCBI GEO (skipped by --quick)
  data     reads the harvested metadata or downloaded matrices (--quick runs it only if they
           are already on disk, and says so when they are not)
  tables   reads only what is checked into results/ (always runs; works on a clean clone)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from epigrade import paths

MARKER_DIR = paths.interim_dir() / ".reproduce_markers"
NETWORK, DATA, TABLES = "network", "data", "tables"

# (stage_name, module_path, needs)
STAGES = [
    ("harvest", "epigrade.acquire.harvest", NETWORK),
    ("triage_report", "phase1_report", DATA),
    ("download_matrices", "download_matrices", NETWORK),
    ("sotos_demo", "demo_sotos", DATA),
    ("build_corpus", "build_corpus", DATA),
    ("ceilings", "compute_ceilings", TABLES),
    ("figures", "make_figures", TABLES),
    ("test_signatures", "make_test_signatures", DATA),
    ("byo_resource", "build_byo_resource", DATA),
    ("byo_fixtures", "run_byo_fixtures", TABLES),
    # after build_corpus (scores) and byo_resource (cohort membership), before the resource
    ("calibration_grid", "build_calibration_grid", TABLES),
    ("attribution", "compute_attribution", DATA),
    ("confounding_gate", "phase6_confounding_gate", DATA),
    ("provenance", "generate_provenance", DATA),
    ("resource_export", "epigrade.report.resource", TABLES),
]


def data_on_disk() -> bool:
    """Harvested metadata plus every cached beta matrix: what the `data` stages read."""
    from epigrade.corpus import loader, registry

    if not (paths.interim_dir() / "samples.parquet").exists():
        return False
    return all(loader.is_downloaded(s) for s in registry.BETA_SERIES)


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
        help="no network: rebuild what can be rebuilt from what is already on disk",
    )
    args = parser.parse_args()

    have_data = data_on_disk()
    print("EpiGrade reproduction pipeline")
    print(f"  mode: {'quick (no network)' if args.quick else 'full'}"
          f"{', force rerun' if args.force else ''}")
    print(f"  data root: {paths.data_root()} "
          f"({'corpus present' if have_data else 'corpus not downloaded'})")
    print(f"  marker directory: {MARKER_DIR}\n")

    results = {}
    for stage_name, module_path, needs in STAGES:
        if args.quick and needs == NETWORK:
            outcome = "skipped (--quick: needs network)"
        elif args.quick and needs == DATA and not have_data:
            outcome = ("skipped (--quick: needs the downloaded corpus, which is not on this "
                       "machine - its checked-in outputs are used as they are)")
        else:
            print(f"[{stage_name}] running...", flush=True)
            outcome = run_stage(stage_name, module_path, args.force)
        results[stage_name] = outcome
        print(f"[{stage_name}] {outcome}", flush=True)

    print("\n--- Summary ---")
    for stage_name, outcome in results.items():
        print(f"  {stage_name:20s} {outcome}")

    n_failed = sum(1 for o in results.values() if o.startswith("FAILED"))
    if n_failed:
        print(f"\n{n_failed} stage(s) failed - see the FAILED lines above. Rerun "
              "scripts/reproduce.py once the cause (usually network access to NCBI GEO) is "
              "resolved; stages that already succeeded are skipped, not redone.")
        sys.exit(1)
    print("\nAll stages completed, were already done, or were skipped as stated. "
          f"Resource: {paths.resource_dir() / 'epigrade_v1.json'}")
    print("Run `streamlit run src/epigrade/app/main.py` to view it.")


if __name__ == "__main__":
    main()
