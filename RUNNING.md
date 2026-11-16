# Running TakeMeter

Everything about how the starter works.

This pair is different from the last two: **you train a model.** It runs on
your own machine, in a notebook, from inside this repo. No accounts, no keys,
nothing hosted.

---

## Capstone: hosted on Render (Path A)

TakeMeter runs as a web service on Render's free tier. The same model also
runs locally from the command line with `classify.py` (see **Running it
locally** below).

| | |
|---|---|
| **URL** | `https://<FILL IN>.onrender.com` (set after the first deploy) |
| **Platform** | Render, free web service (512 MB memory, shared CPU, sleeps after ~15 min idle) |
| **Config** | [`render.yaml`](render.yaml) (Render Blueprint) |
| **Build command** | `pip install -r requirements-server.txt && python fetch_model.py` |
| **Start command** | `uvicorn server:app --host 0.0.0.0 --port $PORT` |
| **Health check** | `GET /health` |

### What's on the URL

| Endpoint | What it does |
|---|---|
| `GET /` | A page with a text box. Paste a comment and get its label |
| `POST /classify` | `{"text": "..."}` or `{"posts": ["...", ...]}` (max 50). Returns the label, confidence, token count, `truncated` flag, or the rejection reason for each post |
| `GET /health` | `200` with memory use if the model is loaded. `503` with the reason if it isn't |
| `GET /logs?n=50` | The last *n* structured log records (max 500), one JSON object per line. Needs `?token=...` if `LOGS_TOKEN` is set |
| `GET /docs` | Auto-generated API docs |

```bash
curl -X POST https://<FILL IN>.onrender.com/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "Tatum shot 4/19 in the last 3 games, he is in a real slump"}'
```

⚠️ **The first request after ~15 idle minutes takes about a minute.** The free
instance sleeps, and waking it means starting Python and loading the model.
Warm requests are fast. Any timing in this repo says which kind it measured.

### Why the hosted version runs ONNX, not PyTorch

Serving the PyTorch model peaked at **613 MB** (measured on 2026-11-15:
torch + transformers + the model + 6 inferences). Render's free tier has 512
MB, so it would be killed on the first request. The server runs the same
trained weights exported to ONNX and served by `onnxruntime`, which peaked at
**427 MB** while loading and settled at **~362 MB** with the web server
(measured the same way, on Windows).

It's the same model. `export_onnx.py` checks the two on every post in
`test_split.csv`, and the labels match on 35/35, with the largest probability
difference 3×10⁻⁷. `classify.py` loads either kind of folder and gave
identical labels, confidences and token counts on 37 posts, including two
truncated ones.

### How it got there (deploying from scratch)

The model weights (`model.onnx`, 268 MB) are over GitHub's 100 MB file limit,
so they ship as a **GitHub Release asset** and the Render build downloads
them. Everything else is in git.

**1. Build the model and the ONNX export, locally (once).**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install onnx==1.23.2 onnxruntime==1.32.0        # only needed for the export
```

Run notebook sections 1–5 and the **"Save the model for `classify.py`"** cell
(writes `takemeter_model/`, about 10 min on CPU). Then:

```bash
python export_onnx.py
# wrote takemeter_onnx/ (269 MB)
# labels match on 35/35 posts in test_split.csv
```

This writes `takemeter_onnx/`. The small files in it (tokenizer, config,
`model.onnx.sha256`) are committed. `model.onnx` itself is gitignored.

**2. Upload `model.onnx` as a release asset.** On GitHub: **Releases → Draft a
new release**. Tag `model-v1`, title `TakeMeter ONNX model`. Drag
`takemeter_onnx/model.onnx` into the assets box and publish. The download
URL is:

```
https://github.com/<user>/<repo>/releases/download/model-v1/model.onnx
```

The repo must be **public**, or the build can't download it. If you rebuild
the model, re-run `export_onnx.py`, commit the new `model.onnx.sha256` and
upload the new file. `fetch_model.py` refuses a file whose checksum doesn't
match.

**3. Create the service on Render.** Sign in at render.com with GitHub. Go to
**New → Blueprint**, pick this repo, and Render reads `render.yaml`. Paste
the URL from step 2 into `MODEL_URL` when asked. `LOGS_TOKEN` is optional.
Then click **Apply**. The build installs `requirements-server.txt` (no torch)
and runs `fetch_model.py`, which downloads and checksums the model. Then
Render starts uvicorn and polls `/health`.

**4. Check it.** Open `https://<service>.onrender.com/health`. It should say
`"status": "ok"` and `"backend": "onnx"`. Every push to `main` redeploys.

### What the server needs

| | |
|---|---|
| Python | 3.13 (`PYTHON_VERSION` in `render.yaml`; tested locally on 3.13.16) |
| Packages | [`requirements-server.txt`](requirements-server.txt), exact pins: `onnxruntime==1.32.0 tokenizers==0.22.2 numpy==2.5.4 fastapi==0.163.1 uvicorn==0.54.0`. No torch, transformers or pandas |
| Model files | `takemeter_onnx/`: `model.onnx` 268 MB (release asset) plus tokenizer and config (~1 MB, committed) |
| Memory | ~362 MB steady, 427 MB peak while loading (measured locally; `/health` and every `request` log record report the real numbers on Render) |
| Env vars | `MODEL_URL` (required at build), `LOGS_TOKEN` (optional), `TAKEMETER_MODEL` / `TAKEMETER_LOG` (optional overrides) |

### The log

Every event is one JSON object per line. It goes to stdout, so it shows in
Render's **Logs** tab, and to `logs/server_log.jsonl`, which `GET /logs`
serves.

| `event` | When | Fields |
|---|---|---|
| `startup` | The model loaded | `model_load_ms`, `backend`, `runtime`, `labels`, `rss_mb`, `peak_rss_mb`, `python` |
| `fatal` | The model didn't load (the server stays up and answers `503`) | `error`, `rss_mb`, `peak_rss_mb` |
| `post` | Each post in a request | `ts`, `run_id` (= request id), `input` (first 2,000 chars), `input_chars`, `status` (`ok`/`rejected`/`error`), `label`, `confidence`, `tokens`, `truncated`, `reason`, `latency_ms` |
| `request` | Each `/classify` call, plus any failed `/health` | `path`, `status_code`, `latency_ms`, `n_posts`, `error`, `rss_mb`, `peak_rss_mb` |

Bad requests leave a record too: not JSON (`400`), wrong shape (`400`), over
50 posts or over 1 MB (`413`), and no model (`503`).

⚠️ **The free instance's disk is wiped when it sleeps.** Copy the records you
need from `/logs` (or the Logs tab) into `README.md` and commit them **in
the same session**. The copy in the repo is the one that lasts.

### Measuring cost

```bash
python measure_cost.py https://<service>.onrender.com --runs 10
```

The script sends one warm-up request (that's the cold-start time if the
instance was asleep), then 10 timed requests with one post each. It reports
the server's own `latency_ms` and the round trip seen from your machine. It
uses only the standard library.

### When the hosted version goes wrong

| What you see | What it means |
|---|---|
| Build fails at `fetch_model.py` with `HTTP 404` | `MODEL_URL` is wrong, or the repo/release is private |
| Build fails with `checksum mismatch` | The uploaded `model.onnx` isn't the one `model.onnx.sha256` describes. Re-export and re-upload, or commit the matching checksum |
| Deploy fails, "Out of memory (used over 512Mi)" in Events | The model didn't fit. Check `peak_rss_mb` in the last `startup` record |
| `/health` returns `503` | The model didn't load. The `error` field says why |
| First request takes about a minute | The instance was asleep. Expected on the free tier |
| `/logs` is empty | The instance restarted and wiped its disk. Use Render's Logs tab |

---

## Running it locally

Two local run paths, no Render needed.

**The web server, exactly as Render runs it:**

```bash
pip install -r requirements-server.txt
python export_onnx.py              # or MODEL_URL=... python fetch_model.py
uvicorn server:app --port 8000     # then open http://127.0.0.1:8000
```

**The command line (`classify.py`):**

```bash
python classify.py --csv test_split.csv
python classify.py "Jokic is the best player alive and it's not close" "WHAT A BLOCK"
```

Each post prints one line: its label and confidence, `REJECTED` with a reason,
or `ERROR`. Every post is also appended to `logs/classify_log.jsonl` (same
fields as the server's `post` records, plus a `run_end` summary).

| Option | What it does |
|---|---|
| `--csv FILE` | Classify the `text` column of a CSV |
| `--model DIR` | `takemeter_model` (PyTorch, default) or `takemeter_onnx` (ONNX) |
| `--log FILE` | Where the log goes (default `logs/classify_log.jsonl`, appended to) |
| `--no-log` | Don't write a log |

**Exit codes:** `0` means the run finished, even if some posts were rejected.
`2` means it couldn't start: no model, an unreadable CSV, no `text` column or
no input.

Local environment: Python 3.13.16, CPU only. Exact versions it was run with:
`torch==2.14.1 transformers==4.57.6 tokenizers==0.22.2 safetensors==0.8.0
pandas==2.3.3 numpy==2.5.4 scikit-learn==1.9.1 datasets==3.6.0
accelerate==1.15.0`. Peak memory classifying the 35-post `test_split.csv`
with PyTorch was 623 MB.

### When `classify.py` goes wrong

| What you see | What it means |
|---|---|
| `error: model folder not found` | Build it: notebook sections 1–5 plus the save cell |
| `error: model folder ... is incomplete` / `could not be loaded` | The folder is half-copied or corrupt. Delete it and rebuild |
| `error: CSV has no \`text\` column` | The message lists the columns it did find |
| `error: CSV is not UTF-8 text` | Re-save the file as UTF-8 |
| `REJECTED empty` / `no_words` / `link_only` / `too_long` / `not_text` / `missing` | That post can't be classified. The rest carries on |
| `REJECTED malformed_row` | That CSV line had the wrong number of fields. It's listed at the end |
| `[truncated: N tokens]` | Labelled, but the model only read the first 128 tokens |
| `warning: can't write log` | The log path is read-only or is a folder. The labels still print |

---

## Before your first class

One setup, one virtual environment, everything in this repo. The
[environment setup page](../pages/ide_setup) has the per-operating-system
commands.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test.py
```

**`python test.py` passing is what you need before your first session.** It
checks your Python version against the pins, your disk, your memory, and
whether the packages import.

**Your disk needs about 9 GB free, up from 2 in the last two pairs.** The baseline model is roughly
1.6 GB and PyTorch is not small.

**Your machine needs about 6 GB of RAM.** Two things here are memory-hungry:
`baseline.py` peaks around 1.8 GB loading the zero-shot model, and the training
run peaks around 2.1 GB. Neither is comfortable beside an editor and a browser
on a 4 GB machine. `test.py` warns you.

You'll see one `[WARN] Baseline model — not downloaded yet`. That's expected in
unit 5. **Clear it before the unit 6 session** by running `python baseline.py`
once — downloading 1.6 GB during class is not how you want to spend a breakout.

> **No API key and no accounts for this pair.** Nothing here calls a hosted
> model. The baseline and the training both run on your own machine.

### What you'll be training on

`test.py` tells you, and so does section 1 of the notebook. There are three
possibilities and all of them work:

| Device | What it means | Speed |
|---|---|---|
| `cuda` | An NVIDIA GPU | Under a minute per seed |
| `mps` | Apple Silicon — M1 and later | Fast; much quicker than the CPU path |
| `cpu` | Everything else | A few minutes per seed. Three seeds is a coffee, not an afternoon |

⚠️ **Write down which one you get.** It is part of your result, not a detail
about your laptop. Two people running the same notebook on the same data with
the same seed can get different numbers, because different hardware does the
arithmetic in a slightly different order. Unit 6 asks you to report it beside
your three seeds, and `results.json` records it for you.

---

## The notebook

`takemeter.ipynb` runs **on your own machine**, from inside this repo.

1. Open it in VS Code (or `jupyter notebook`, if you prefer).
2. **Pick the kernel: the `.venv` inside this project.** In VS Code that's the
   selector in the top right. Getting this wrong is the most common way the
   first cell fails — it will say a package isn't installed when it is,
   because it's looking at a different Python.
3. Fill in the two lines at the top of section 1's second cell — your name and
   email. They only label your results.

> ⚠️ **Open it from your repo, not from a copy.** The notebook writes next to
> your code and expects `labels.csv` to be there. If you open a copy from your
> Downloads folder, it will tell you so rather than failing strangely.

### It reads and writes this folder

There is nothing to upload and nothing to download. `labels.csv` is already
here; `results.json`, `test_split.csv` and `results_three_seeds_*.json` land
here when the notebook writes them.

**Commit them yourself**, the way you commit anything else. Nothing is pushed
for you.

- Section 5 writes `results.json` and `test_split.csv` — **unit 5**
- Section 6 writes `results_three_seeds_<label>.json` — **unit 6**.
  Set `RUN_LABEL = "after"` in that cell before the Milestone 5 re-run, or the
  improvement overwrites the numbers you're comparing against.

> **Add rows to `labels.csv` mid-session?** Re-run section 2, which is what
> reads the file. The notebook holds the data in memory once it has read it.

> **Your edits to the notebook itself are yours to commit too.** `LABELS`, the
> settings and the seed live in the file — if you change them, that's a change
> to your repo like any other.

### What's in it

| Section | When | What it does |
|---|---|---|
| 1. Setup | unit 5 | Reports your device, and confirms it can see your files |
| 2. Your labels | unit 5 | `LABELS`, the settings, and **the seed** |
| 3. Split | unit 5 | 70/15/15, stratified |
| 4. Train | unit 5 | Minutes on a laptop CPU, under one on a GPU |
| 5. Results | unit 5 | Writes `results.json` and `test_split.csv` |
| 6. Three seeds | **unit 6** | Retrains three times in one cell |
| 7. Confusion matrix | **unit 6** | Prints a markdown table to paste |

Sections 6 and 7 are next unit. Nothing in unit 5 needs them.

---

## Your data

`labels.csv` in your repo. Three columns:

| Column | What's in it |
|---|---|
| `text` | The whole post. Don't trim it to fit |
| `label` | One of your labels, spelled **exactly** as in `LABELS` |
| `note` | `cold` for the 20 you labelled unaided, and a note on any you pre-labelled |

`labels_template.csv` shows the shape. `data/practice_labels.csv` is the set
used in class — 60 posts from a fictional running forum, already labelled, so
you can run the whole notebook before you have data of your own.

**One file, unsplit.** Don't split it yourself — the notebook does the 70/15/15
split, and splitting twice is how examples leak between training and test.

### The check before training

Section 2's last cell looks for the things that break a run or quietly ruin it:

| It says | What it means |
|---|---|
| `[FIX ME] Labels in your CSV that aren't in LABELS` | A spelling or capital mismatch. `Hot_take` and `hot_take` are different |
| `[FIX ME] 'x' is 78% of your data` | Over the brief's 70% cap. The model will learn to guess it |
| `[look at] N duplicate posts` | They'll land in different splits and inflate your score. That's leakage |
| `[look at] Only 12 examples of 'x'` | Its test split will be tiny, so its F1 will jump between seeds |
| `[look at] 60 rows` | Under the submission floor. Training still runs — fine for the practice data |

---

## The baseline — unit 6

```bash
python baseline.py
python baseline.py --trained results.json
```

This runs a general model over your posts using **only your label
definitions**, with no training. It's the fair opponent your fine-tuned model
has to beat.

**It scores `test_split.csv`** — the held-out posts the notebook saved in unit 5,
the same ones your trained model was measured on. Two numbers from two different
sets of posts aren't a comparison, which is why it reads that file rather than
picking its own sample. If the file isn't there it says so, falls back to a
random sample of `labels.csv`, and tells you to disclose that in your README.

Runs **on your machine**. No account, no key. A few seconds per post, so 30
posts takes a couple of minutes — that slowness is itself part of the finding.

Put your definitions in `label_definitions.txt`, one per line as
`label: definition`, copied from your README. **The definitions are the prompt** —
they're the entire input the model gets.

Without the file it falls back to bare label names, which is a much weaker
prompt. Worth doing on purpose once: run it both ways and the gap between the
two accuracies is what your definitions were worth, measured instead of assumed.
If you report the bare-names run as your baseline, say so in your README.

The second command adds a Trained column and prints the **Baseline vs.
Trained** table ready to paste.

---

## The agreement check — unit 6

Staff labelled 30 posts under the taxonomy in `data/staff_taxonomy.md`. You
label the same 30 under **those** definitions — not the ones you designed in
unit 5 — and then compare.

```bash
python agreement.py
```

Four files. The posts and the rules ship with the starter; only the answers
come from your TF, and only after yours are done. Posts and labels join on `id`:

| File | Where it comes from |
|---|---|
| `data/staff_taxonomy.md` | **Ships in the starter.** The three definitions and the decision rules staff used. Read it first — you label under *these*, not your own |
| `data/staff_posts.csv` | **Ships in the starter.** The 30 posts, no labels |
| `data/staff_labels.csv` | **Your TF gives you this**, once your own labels are done |
| `my_staff_labels.csv` | You make it: the same posts, labelled by you, with `text` and `label` columns |

**You label the staff set under the staff taxonomy, not the one you designed in
unit 5.** Your own taxonomy was yours to invent; this one is somebody else's to
apply, and whether you can apply a written rule the way its authors did is the
only evidence you have about whether your own 200 labels were consistent.

Put both staff files in `data/` under those names and the bare command finds
them. Anywhere else, point at them:

```bash
python agreement.py --staff-posts posts.csv --staff-labels labels.csv
```

Posts are matched on their text, so **paste it unchanged**. Retype or trim a
post and it won't match — the script tells you which ones didn't.

It works out the rate and lists every disagreement. **It stops there on
purpose.** The rate is arithmetic; the adjudication is the graded part, and your
README has a block to fill in for each one.

> This is the only instrument you have that can tell a labelling problem from a
> model problem. Milestone 4 asks you to use it as exactly that — a label pair
> you disagreed with staff about, and that your matrix also confuses, is a
> finding about your labels rather than about the model.

---

## Every command

| Command | What it does |
|---|---|
| `python test.py` | Checks your environment |
| `python baseline.py` | The zero-shot baseline on your held-out posts — **unit 6** |
| `python agreement.py` | Your labels vs. the staff set — **unit 6** |
| `python baseline.py --help` | All the options |
| Notebook: sections 1–5 | Train once, write `results.json` — **unit 5** |
| Notebook: sections 6–7 | Three seeds and the matrix — **unit 6** |
| `python classify.py --csv FILE` / `"post"` | Label posts with the saved model, and log the run. **Capstone** |
| `python export_onnx.py` | Export the saved model to ONNX for hosting, and check it matches. **Capstone** |
| `uvicorn server:app --port 8000` | The web service, run locally the way Render runs it. **Capstone** |
| `python measure_cost.py URL` | Ten timed requests against a running server. **Capstone** |

---

## Which piece goes with which milestone

### Unit 5 — the build

| Milestone | What you're doing | Where |
|---|---|---|
| 1 | Pick a community and read it | Nothing to run. Read for 20 minutes |
| 2 | Design your labels | README, **Label Taxonomy** |
| 3 | Write your criteria | `criteria.md` — all five are yours this time |
| 4 | Collect and label 200 | `labels.csv` |
| 5 | Run the training | Notebook, sections 1–5 |
| 6 | Write it up | README |

### Unit 6 — the test

| Milestone | What you're doing | Where |
|---|---|---|
| 1 | Run the baseline | `baseline.py` |
| 2 | Retrain three times, read the matrix | Notebook, sections 6–7 |
| 3 | Check yourself against the staff labels | `agreement.py` |
| 4 | Call each criterion, diagnose the misses | README |
| 5 | Fix one thing and re-run | Notebook, section 6 with `RUN_LABEL = "after"` |
| 6 | Say what's still broken | README |

---

## Where everything lives

| File | What it does |
|---|---|
| `takemeter.ipynb` | The training notebook. Runs here, on your machine |
| `baseline.py` | The zero-shot baseline. Runs on your machine |
| `agreement.py` | Your labels against the staff set. **Unit 6** |
| `criteria.md` | Your five criteria. **You write all five** |
| `labels.csv` | Your labelled data. **You create this** |
| `labels_template.csv` | The shape it needs |
| `label_definitions.txt` | Your definitions, for the baseline |
| `data/practice_labels.csv` | The 60-post practice set used in class |
| `data/staff_taxonomy.md` | The staff set's definitions and decision rules. Ships here |
| `data/staff_posts.csv` | The staff set's 30 posts. Ships here |
| `data/staff_labels.csv` | The staff set's labels. **Your TF gives you this** |
| `README.md` | Your submission |
| `results.json` | Written and committed by the notebook |
| `test_split.csv` | Your held-out posts, committed by the notebook. `baseline.py` reads it |
| `results_three_seeds_before.json` | Written and committed in unit 6, section 6 |
| `results_three_seeds_after.json` | The same cell re-run with `RUN_LABEL = "after"`, in Milestone 5 |
| `baseline_results.json` | Written by `baseline.py`. **Commit it** |
| `agreement_results.json` | Written by `agreement.py`. **Commit it** |
| `my_staff_labels.csv` | Your labels on the staff set. **You create this** |
| `classify.py` | Labels posts with the saved model, outside the notebook. **Capstone** |
| `takemeter_model/` | The saved model. Built by the notebook's save cell, **not committed** |
| `logs/classify_log.jsonl` | Structured log of every `classify.py` run. **Commit it**, it's the run record |
| `server.py` | The hosted web service (Render). **Capstone** |
| `render.yaml`, `requirements-server.txt` | Render config and the server's pinned packages. **Capstone** |
| `fetch_model.py` | Downloads and checksums `model.onnx` during the Render build. **Capstone** |
| `export_onnx.py`, `takemeter_onnx/` | The ONNX export the server runs. `model.onnx` is a release asset, not in git |
| `measure_cost.py` | Measures cost per run against a live server. **Capstone** |

---

## When something goes wrong

| What you see | What it means |
|---|---|
| Section 1 says `cpu` | Correct on most laptops, and fine. A three-seed run is minutes. Nothing to fix |
| `ModuleNotFoundError` in the notebook | Wrong kernel. Pick the `.venv` inside this project, top right in VS Code |
| `FileNotFoundError: labels.csv` | Either you haven't created it yet, or the notebook is open from a copy rather than from your repo. Section 1's second cell says which |
| The machine crawls during training | Close the browser and anything heavy. Training peaks around 2.1 GB |
| `KeyError` during training | A label in your CSV isn't in `LABELS`. Check capitals and spaces |
| Training takes an hour | You're on CPU. See the first row |
| Accuracy swings 15 points between seeds | Your dataset is too small or too lopsided for a stable measure. **That's a diagnosis, not a mistake** — write it down |
| A label's F1 is 0.00 | It probably has almost nothing in the test split. Check the split table |
| `baseline.py` says no `test_split.csv` | Unit 5's results section never finished. Re-run it, or note in your README that your comparison isn't like-for-like |
| `baseline.py` is downloading forever | First run only, ~1.6 GB. Do it before class |
| The baseline is very slow | Expected. A few seconds per post is what no fine-tuning costs at runtime |

The three usual causes when the notebook fails, in order: the kernel is the
system Python rather than this project's `.venv`, the notebook was opened from
a copy outside the repo, or a label in your CSV doesn't match `LABELS` exactly.

If none of those fix it, commit your CSV and criteria and write down the error.
**You have a labelled dataset and a filed standard, which is most of the
unit's grade.**

---

## A note on committing

At least four new commits in unit 5, four more in unit 6. Your commit history is
what shows your criteria existed before your results did.

The `results*.json` files are deliberately **not** in `.gitignore`. They're
evidence the run happened, and so are `test_split.csv` and `labels.csv`. The
notebook writes them; you commit them, along with `criteria.md`,
`baseline_results.json` and your README.

**One machine, one repo.** Everything happens here, so there's nothing to pull
before you push.

**Do not delete and recreate this repository.** You submit the same URL both
units.
