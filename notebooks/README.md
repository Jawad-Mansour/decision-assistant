# Notebooks

Run in numerical order. The full per-notebook guide — inputs, outputs, and when to re-run — lives in **[`../docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md)**.

Quick index:

| Order | Notebook | Purpose |
|:-:|---|---|
| 1 | `02_load_inspect_full.ipynb` | Load + validate raw Twitter support dataset |
| 2 | `03_process_conversations_full.ipynb` | Thread tweets into conversations, add derived columns |
| 3 | `08_eda_and_cleaning.ipynb` | RAG-safe cleaning → ML view + RAG view |
| 4 | `09_data_quality_checkpoint.ipynb` | Invariants / quality gate after cleaning |
| 5 | `10_priority_labeling.ipynb` | Weak-supervision `urgent` / `normal` labels |
| 6 | `10_eda_validation_cleaned.ipynb` | EDA on cleaned + labeled corpus |
| 7 | `11_ml_features_train.ipynb` | Feature engineering + train / val / test split |
| 8 | `12_ml_training_pipeline.ipynb` | Model comparison + export `priority_classifier.pkl` |

`archive/fix_cleaned_datasets.ipynb` is a historical one-off repair — **not** part of the pipeline.
