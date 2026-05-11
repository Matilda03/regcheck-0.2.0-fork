# Setup Notes — regcheck-0.2.0-beta

Documenting deviations from the README and what was done instead to get the API running.

---

## Issues Found in the README

### 1. Venv name mismatch
**README says:**
```bash
python -m venv venv
source venv/bin/activate
```
**Reality:** The repo already contained a `.venv/` directory (with a dot prefix), created with Python 3.13 (Homebrew). The README makes no mention of this existing `.venv`.

**What was done:** Created a fresh `venv/` (no dot) using the system Python 3.11, as the README instructs. Used `venv/bin/pip` and `venv/bin/python` directly rather than activating the shell environment (Claude Code runs commands without shell activation).

---

### 2. PyMuPDF 1.24.5 fails to build on Python 3.13 with a path containing spaces
**Root cause:** The existing `.venv/` used Python 3.13. PyMuPDF 1.24.5 has no pre-built wheel for Python 3.13 and falls back to building from source. The build script invokes a shell command that contains the venv's absolute path — which includes spaces (`/Users/matifogato/Documents/HTI Master/...`) — unquoted, causing `sh` to fail with exit code 127 (`/bin/sh: /Users/matifogato/Documents/HTI: No such file or directory`).

**What was done:** Created a new `venv/` with Python 3.11 (system Python at `/Library/Frameworks/Python.framework/Versions/3.11/`). PyMuPDF 1.24.5 has a pre-built wheel for `cp311-macosx_11_0_arm64`, so no source build was needed and installation succeeded.

**Note:** The `.python-version` file specifies `3.12`, but no Python 3.12 is installed on this machine. Python 3.11 (system) and 3.13 (Homebrew) are available. The README says "Python 3.11+" so 3.11 is compliant.

---

### 3. README omits `openai` SDK version jump
The README mentions `openai` as a provider but does not call out that `requirements.txt` pins `openai==2.6.0` — a major version bump from the widely-known 1.x series. This version installed without issue (the package exists on PyPI), but it is worth noting for anyone who expects the 1.x API interface.

---

### 4. Redis not installed locally
**README says:** Redis is required for the web flow.

**Reality:** Redis is not installed on this machine (`redis-server not found`). The web app will fail to connect to Redis on startup of task-based routes. The CLI works without Redis.

**What to do:** Install Redis via Homebrew:
```bash
brew install redis
brew services start redis
```
Or set `REDIS_URL` in `.env` to a remote Redis instance.

---

### 5. `.env` file not present
The README says `cp .env.example .env` (optional). The `.env` was not present.

**What was done:** Copied `.env.example` to `.env`. API keys are still blank — fill in at least one of `OPENAI_API_KEY`, `GROQ_API_KEY`, or `DEEPSEEK_API_KEY` before running comparisons.

---

### 6. NLTK data downloads to `~/nltk_data`, not `nltk_data/`
**README says:** `nltk_data/` — "Not committed; downloaded locally via NLTK."

**Reality:** NLTK downloads to the user's home directory (`/Users/matifogato/nltk_data/`) by default, not to a project-local `nltk_data/` folder. The README implies a local folder but doesn't clarify. This is fine functionally.

---

## What Was Actually Run

```bash
# Step 1 — Create venv with Python 3.11 (not the pre-existing .venv with Python 3.13)
python3 -m venv venv

# Step 2 — Install dependencies (using venv directly, no shell activation needed)
venv/bin/pip install -r requirements.txt

# Step 3 — Download NLTK tokenizer data
venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Step 4 — Create .env
cp .env.example .env
# → Fill in API keys before running

# Step 5 — Verify app loads
venv/bin/python -c "from backend.main import create_app; app = create_app(); print('OK')"
# Output: App created successfully
```

## To Start the Web App

```bash
# Activate venv first (or use venv/bin/uvicorn directly)
source venv/bin/activate

# Option A (as README states, factory pattern):
uvicorn backend.main:create_app --factory --reload

# Option B (app.py shortcut):
uvicorn app:app --reload
```

**Prerequisite:** Redis must be running (`brew services start redis`) for task-based web routes to work.

## To Run the CLI (no Redis needed)

```bash
source venv/bin/activate
python -m backend.cli general \
  --preregistration /path/prereg.pdf \
  --paper /path/paper.pdf \
  --dimensions-csv test_materials/dimensions_example.csv \
  --client openai \
  --output result.json
```

## Script to run clinical trial comparisons in batches
The script is at batch_clinical.py. Here's what it does and how to use it:
                                                                                                                                                                                   
  ### Activate the venv first                                                                                                                                                          
  `source venv/bin/activate`                                                                                                                                                    
                                                                                                                                                                                     
  ### Basic usage                                                                                                                                                                      
```
python batch_clinical.py \
    --papers-dir ./papers \                                                                                                                                                          
    --output-dir ./results   
```                                                                                                                                                      

  ### With all options
```
  python batch_clinical.py \
    --papers-dir ./papers \                                                                                                                                                          
    --output-dir ./results \
    --client openai \                                                                                                                                                                
    --parser-choice grobid \                                                                                                                                                       
    --output-format csv \
    --reasoning-effort medium \
    --dimensions-csv test_materials/dimensions_example.csv \                                                                                                                         
    --append-previous-output \
    --overwrite                                                                                                                                                                      
```                                                                                                                                                                               
  ### What it does, step by step                                                                                                                                                         
                                                                                                                                                                                   
  For each .pdf in --papers-dir:

  1. Extracts the NCT ID — scans every page of the PDF with PyMuPDF looking for the pattern NCT\d{8} (case-insensitive, same regex used in backend/services/trials.py). Takes the    
  first match.
  2. Skips with a warning if no NCT ID is found, or if the output file already exists (unless --overwrite is passed).                                                                
  3. Calls the RegCheck backend directly (same process, no subprocess) via clinical_trial_comparison, which fetches the trial from ClinicalTrials.gov and runs the LLM comparison.   
  4. Saves the result to --output-dir with the name {NCT_ID}_{paper_stem}.csv (or .json), e.g. NCT01234567_smith_2023.csv.                                                           
  5. Prints a summary at the end listing successes, skips, and failures.                                                                                                             
                                                                                                                                                                                     
  Key design decisions                                                                                                                                                               
                                                                                                                                                                                     
  - Direct import, not subprocess — calls clinical_trial_comparison via asyncio.run() directly rather than spawning python -m backend.cli as a subprocess. Faster and avoids         
  re-paying startup cost per paper; each call gets a fresh event loop.
  - First NCT ID wins — scans page-by-page and stops at the first match, which is usually in the abstract or methods section.                                                        
  - `--overwrite` guard — skips already-completed outputs by default, so you can safely re-run a partially completed batch without re-doing finished work.   