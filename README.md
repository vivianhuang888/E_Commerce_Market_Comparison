# E_Commerce_Market_Comparison
E_Commerce_Market_Comparison_between_Amazon_UK_and_Amazon_Brazil
Contributors: Wei-An Huang, Hui Gao, Donghyeon Na, Shuomeng Guan, Courtney Vincent

## MIMIC-IV-ED XGBoost training helper
The repository now includes `mimic_ed_xgboost.py`, a command-line helper for training
an XGBoost model on the cleaned triage Parquet file (for example,
`MIMIC-IV-ED_triage_cleaned.parquet`). The pipeline uses ordinal encoding for
categorical features to respect the "no one-hot" constraint and validates that the
provided target column is a single-label field.

### Basic usage
```bash
python mimic_ed_xgboost.py \
  --data-path /path/to/MIMIC-IV-ED_triage_cleaned.parquet \
  --target your_target_column
```

Key behavior:
- Automatically separates numeric and categorical predictors.
- Imputes missing numeric values with the median and categorical values with the most frequent entry.
- Uses ordinal encoding for categorical features (no one-hot, no multi-label targets).
- Computes accuracy and macro F1 scores after a stratified train/test split.

You can adjust core XGBoost hyperparameters through CLI flags such as `--n-estimators`,
`--learning-rate`, and `--max-depth`. Use `--help` for the full list of options.
