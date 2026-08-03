# Dataset Generator

Utilities for appending additional binary datasets to the project CSV files in
`datasets/`.

The generator is idempotent for a generated dataset/model pair: rerunning it
removes matching rows first, then appends freshly generated rows.

## CSV replacement keys

- `metadata.csv`: `appId`
- `values.csv`: `appId`
- `none.csv`: `appId`, `modelName`
- `logistic_regression.csv`: `appId`, `model`, `variant`
- `decision_tree.csv`: `appId`, `model`, `depth`

## Default datasets

The default run tries to add six popular binary classification datasets:

- `breast_cancer`
- `banknote_authentication`
- `spambase`
- `pima_diabetes`
- `ionosphere`
- `blood_transfusion`

`breast_cancer` is bundled with scikit-learn. The others are fetched from
OpenML by scikit-learn.

## Usage

Preview without writing. Start with one dataset so progress is easy to see:

```powershell
python dataset_generator/generate_datasets.py --dry-run --datasets breast_cancer
```

Generate and append/replace rows for one dataset:

```powershell
python dataset_generator/generate_datasets.py --datasets breast_cancer
```

Run a subset progressively:

```powershell
python dataset_generator/generate_datasets.py --datasets breast_cancer spambase
```

If an OpenML dataset is slow, interrupt safely and rerun a smaller subset. Rows
are replaced by key on each run, so rerunning the same dataset/model does not
duplicate rows.
