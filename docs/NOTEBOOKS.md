# Notebooks Guide

Every notebook in `notebooks/` has a single, documented responsibility. They are meant to be run **in numerical order** the first time so each one's outputs become the next one's inputs. Once artefacts exist under `data/` and `models/`, the backend service can run on its own and the notebooks are only re-run when you change labeling logic, features, or the training corpus.

All notebooks read from / write to the `data/` and `models/` folders. Those folders are **gitignored** on purpose — artefacts are derived, not source.

## Prerequisites

```bash
cd backend
uv sync --extra dev       # installs ipykernel + ipython for notebooks
uv run python -m ipykernel install --user --name decision-assistant
```

Open any notebook with the `decision-assistant` kernel. The raw dataset (`data/raw/twcs.csv`) must be downloaded from Kaggle before `02_load_inspect_full.ipynb`:

[Customer Support on Twitter — thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

## Execution order

| Order | Notebook | Reads | Writes | Purpose |
|:-:|---|---|---|---|
| 1 | `02_load_inspect_full.ipynb` | `data/raw/twcs.csv` | summary logs | Memory-aware load of the full raw tweet dataset. Validates schema, dtypes, row counts, and prints the structural stats referenced by later EDA. |
| 2 | `03_process_conversations_full.ipynb` | `data/raw/twcs.csv` | `data/processed/twcs_enhanced.csv` | Threads individual tweets back into conversations, adds derived columns (inbound vs outbound, thread id, response flags). This is the input for all subsequent work. |
| 3 | `08_eda_and_cleaning.ipynb` | `data/processed/twcs_enhanced.csv` | `data/cleaned/twcs_cleaned_fixed.csv`, `data/cleaned/questions_for_ml_fixed.csv`, `data/cleaned/conversations_for_rag.csv` | RAG-safe cleaning. Produces two curated views: a **tabular** one for the ML baseline and a **conversational** one for Chroma ingestion. |
| 4 | `09_data_quality_checkpoint.ipynb` | `data/cleaned/*.csv` | quality report | Invariants check — row counts, nulls, duplicated ids, length distributions. Fails fast if cleaning regressed. Run after any change to `08_*`. |
| 5 | `10_priority_labeling.ipynb` | `data/cleaned/questions_for_ml_fixed.csv` | `data/labeled/labeled_questions_fixed.csv` | Applies the weak-supervision labeling function (keyword + punctuation + length rules) to assign `urgent` / `normal` priority. Explicitly documents the rule so reviewers can see what the classifier is really learning. |
| 6 | `10_eda_validation_cleaned.ipynb` | `data/cleaned/*.csv`, `data/labeled/*.csv` | charts in-notebook | Final EDA pass on the cleaned + labeled corpus. Sanity-checks class balance, message length per class, common keywords by label. Used to justify labeling rules before training. |
| 7 | `11_ml_features_train.ipynb` | `data/labeled/labeled_questions_fixed.csv` | `data/splits/X_{train,val,test}.csv`, `y_*.csv` | Engineers tabular features (length, urgency-keyword counts, punctuation signals, sentiment via TextBlob, etc.) and produces the train / val / test split used by every model. |
| 8 | `12_ml_training_pipeline.ipynb` | `data/splits/*.csv` | `models/priority_classifier.pkl`, `models/feature_columns.json` | Trains and compares several classifiers (Logistic Regression, Random Forest, XGBoost), picks the best, and exports the artefacts the backend loads at startup. |
| — | `archive/fix_cleaned_datasets.ipynb` | — | — | **One-off repair script** for an earlier dataset issue. Kept for history only; it is **not** part of the normal pipeline and should not be re-run. |

## After the notebooks

Once `models/priority_classifier.pkl` and `models/feature_columns.json` exist, and `data/cleaned/conversations_for_rag.csv` is present, build the vector index that the API reads:

```bash
cd backend
uv run python scripts/ingest_conversations.py
uv run python scripts/validate_vector_store.py   # sanity check
```

Then start the backend (`uv run uvicorn app.main:app --reload`) and the UI proxies through to it.

## Honesty checklist (per the project brief)

- Labels come from a **documented rule**. Treat accuracy on that rule as a floor, not a brag — this is weak supervision.
- No numbers in the UI come from the notebook at runtime. If you want to surface hold-out accuracy in the UI, paste it into `frontend/.env` as `VITE_ML_HOLDOUT_ACCURACY` / `VITE_LLM_HOLDOUT_ACCURACY` — the UI displays these only when set.
- Latency and cost on screen are measured on the **live** call, not copied from the notebook.
