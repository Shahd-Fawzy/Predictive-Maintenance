# Industrial Predictive Maintenance & Failure Prevention

Graduation project — Advanced Data Science & AI track (Project 5).

A decision-support system that estimates machine failure risk from sensor
readings, compares classical ML against deep learning, and ships as a
Streamlit dashboard. Built on the **AI4I 2020 Predictive Maintenance**
dataset (synthetic industrial process data, 10,000 records).

> **This is a demonstration pipeline, not a certified industrial safety
> system.** The dataset is synthetic — good results here validate the
> approach, not real-world reliability. See [Limitations](#limitations).

## Quickstart

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

Download `ai4i2020.csv` from the [UCI AI4I 2020 Predictive Maintenance
dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaint)
and place it in the project root (same folder as the notebook).

### 3. Run the notebook

Open `Project5_Predictive_Maintenance_FullPipeline.ipynb` and run all cells,
top to bottom. This will, in order:

1. Clean the data and engineer features → `ai4i2020_features.csv`
2. Write the shared `src/features.py` and `src/modeling.py` modules
3. Train and compare 4 classical ML models (Part 3)
4. Train an MLP and an Autoencoder (Parts 4–5) — **requires `torch`**
5. Produce the final ML-vs-DL comparison (Part 6)
6. Write `app/app.py` and `requirements.txt` (Part 7)

At the end, `models/` contains the trained artifacts the app needs:
`best_model.pkl`, `best_model_meta.json`, `feature_reference_stats.json`,
plus the MLP and autoencoder weights.

### 4. Run the app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501`. Three tabs: **Single Machine Check**, **Batch
Scoring**, and **Monitoring** (see [App features](#app-features) below).

### 5. Run the tests

```bash
pytest tests/ -v
```

Tests that need trained artifacts (`test_model_artifacts.py`,
`test_app_smoke.py`) skip cleanly if you haven't run the notebook yet,
rather than failing.

---

## Running with Docker

The Streamlit app can be containerized once the notebook has produced
`models/`, `src/`, and `app/` (Docker packages the **deployment** side —
it does not train the model).

```bash
# 1. Run the notebook at least through Part 7 first (see Quickstart above)

# 2. Build the image
docker build -t predictive-maintenance-app .

# 3. Run it
docker run -p 8501:8501 predictive-maintenance-app

# 4. Open http://localhost:8501
```

To stop it: `docker ps` (find the container ID), then `docker stop <id>`.

**Using docker-compose instead** (also persists the prediction log to your
host machine so it survives a rebuild):

```bash
docker compose up --build
```

**Rebuilding after retraining:** if you retrain the model (new
`models/best_model.pkl`), rebuild the image — `docker build` again, or
`docker compose up --build`.

---

## SQL

`sql/predictive_maintenance.sql` (SQL Server) builds the full data layer:

- **Staging → dimension → fact** schema (`Stg_AI4I` → `MachineType` →
  `MachineReadings`), plus a `PredictionHistory` table for logging live
  predictions.
- An ETL step that joins staging data against the `MachineType` dimension
  and computes the same three engineered features as the Python pipeline
  (temperature difference, mechanical power, wear×torque interaction) —
  using the identical physics formula, so the SQL and Python feature sets
  agree.
- **12 analytical queries**, covering JOINs, CTEs, and window functions
  (`RANK`, `ROW_NUMBER`, running `SUM() OVER`).

**Before running it:** the `BULK INSERT` path
(`'E:\teck\final\ai4i2020.csv'`) is specific to the machine it was
written on. `BULK INSERT` reads from the **SQL Server instance's**
filesystem, not the client's — update that path (or convert it to a
`SQLCMD` variable) to point at wherever `ai4i2020.csv` lives relative to
your SQL Server installation before running the script.

`app/prediction_history.csv` (produced by the Streamlit app) uses the
exact same column names as the `PredictionHistory` table, so it can be
bulk-loaded into it directly with no reshaping.

---

## App features

- **Single Machine Check** — enter one machine's sensor readings, get a
  failure probability, a Low/Medium/High risk tier, and a recommended
  action. Also flags any input outside the range the model was trained on.
- **Batch Scoring** — upload a CSV of machines, score all of them, download
  the results as CSV.
- **Monitoring** — every prediction (from either tab) is logged to
  `app/prediction_history.csv`; this tab charts alert rate over time and
  shows the full log.

The app always uses the model's own **business-tuned decision threshold**
(optimized for recall subject to a minimum precision — see the notebook,
Part 3.9), not the default 0.5, since missing a real failure is far more
costly than one extra inspection.

---

## Results (Machine Learning track)

On a held-out 20% test set (2,000 records, ~3.4% failure rate):

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting (tuned, default threshold) | 0.98 | 0.78 | 0.87 | 0.894 | 0.968 |
| Gradient Boosting (tuned + business threshold) | 0.31 | 0.94 | 0.46 | 0.894 | 0.968 |
| Random Forest (tuned) | 0.94 | 0.75 | 0.84 | 0.865 | 0.980 |
| Decision Tree | 0.39 | 0.82 | 0.53 | 0.773 | 0.857 |
| Logistic Regression (balanced) | 0.18 | 0.88 | 0.30 | 0.434 | 0.939 |

**Deployed model:** Gradient Boosting (tuned), at threshold **0.0092** —
chosen to catch 94% of real failures, accepting more false alarms in
exchange (a business call: a missed failure costs far more than an
unnecessary inspection).

Deep learning results (MLP, Autoencoder) depend on a `torch` environment
this repo doesn't assume you have at README-writing time — run Parts 4–6
of the notebook to populate them; the final comparison table in Part 6
is generated automatically from whatever ran.

---

## Project structure (suggested, matches the brief)

```
project-name/
├── data/                    # raw/processed metadata only - not checked in
├── notebooks/                (this repo keeps the one main notebook at root)
├── src/
│   ├── features.py           # feature engineering (shared)
│   └── modeling.py           # metrics + threshold selection (shared)
├── sql/
│   └── predictive_maintenance.sql
├── app/
│   ├── app.py                 # Streamlit dashboard
│   └── prediction_history.csv # generated at runtime
├── tests/
├── models/                    # generated by the notebook - not checked in
├── evaluation/                # generated by the notebook - plots, comparison CSVs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

`models/`, `evaluation/`, `data/processed/`, and `ai4i2020_features.csv`
are generated artifacts — add them to `.gitignore` rather than committing
them; anyone cloning the repo regenerates them by running the notebook.

---

## Limitations

- **AI4I 2020 is a synthetic dataset.** Strong metrics here demonstrate a
  sound ML/DL pipeline — they are not evidence of real industrial
  reliability, and this system is not a certified safety tool.
- The five failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are
  components of the `Machine failure` target, not independent causes — they
  are deliberately excluded from every model's input features (see
  `src/features.get_feature_columns`). Reintroducing them anywhere in the
  pipeline would silently leak the label.
- The deployed decision threshold trades precision for recall
  deliberately; the "false alarm" rate at that threshold is real and would
  need to be weighed against actual inspection costs before any real
  deployment.

## Future work

- Failure-mode classification (which of TWF/HDF/PWF/OSF/RNF, not just
  failure/no-failure).
- Drift detection over time as sensor behavior or product mix shifts.
- A cost-based alerting layer that turns the probability into a maintenance
  schedule, not just a flag.
