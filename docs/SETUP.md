# Setup

## Why the venv and data live on D:

This machine's C: drive had very little free space when this project started (later checks
showed ~28 GB free, but D: has more headroom at ~87 GB), so by agreement with the project owner
all bulk artifacts - the virtualenv and every downloaded/cached data file - live on D:, while the
git repo itself stays at its existing location on C:. Nothing in `src/` hardcodes either
location; see `config/paths.yaml`.

| What | Where |
|---|---|
| Git repo (this checkout) | `C:\Users\UPASANA\EpiGrade` |
| Virtualenv | `D:\epigrade_venv` |
| GEO downloads / caches / interim parquet | `D:\epigrade_data` (see `config/paths.yaml`) |

## Activate the venv

PowerShell:

```powershell
D:\epigrade_venv\Scripts\Activate.ps1
```

Bash (Git Bash):

```bash
source /d/epigrade_venv/Scripts/activate
```

## Environment variables

- `GEMINI_API_KEY` - required for the phase 1 LLM label-escalation step (ambiguous GEO sample
  metadata). Set it as a user environment variable (`setx GEMINI_API_KEY "..."` then restart the
  terminal) rather than committing it anywhere. Without it, escalated samples are left
  unresolved and logged as such rather than guessed.

## Install / reinstall dependencies

```powershell
D:\epigrade_venv\Scripts\pip.exe install -r requirements.txt
D:\epigrade_venv\Scripts\pip.exe install -e .
```
