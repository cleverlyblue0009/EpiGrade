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

## Optional: plain-language summaries (Groq)

The app shows a short plain-language summary under each result. **It works without any API
key**: the summary is then written by a fixed template from the computed numbers.

With a key, a language model writes the summary instead, and every output is checked against
the computed numbers before it is shown (see `src/epigrade/explain/`). To enable it:

1. Create a free account at https://console.groq.com and create an API key.
2. Set it as an environment variable, never in a file inside the repository:
   - PowerShell, for the current session: `$env:GROQ_API_KEY = "<your key>"`
   - Windows, permanently: `setx GROQ_API_KEY "<your key>"`, then open a new terminal
   - macOS/Linux: `export GROQ_API_KEY=<your key>`
3. Restart the app. A "Regenerate" button appears under each summary.

`.env.example` lists the variable; `.env` is gitignored. The model id is in
`config/explain.yaml`. Accepted model outputs are cached in `results/cache/explanations/`, so
once `scripts/eval_explanations.py` has been run the app shows them with no network call.
