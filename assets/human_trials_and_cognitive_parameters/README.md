# Human-response and cognitive-model fitting data

This directory is the database-staging boundary for CoAX and CoXAM. Source
CSVs and binary model artifacts remain immutable in their current locations;
the builder creates a normalized SQLite snapshot without duplicating source
files in Git.

## Build a local snapshot

For production-quality participant pseudonyms, set a secret that is not stored
in the repository:

```bash
export HUMAN_FITTING_HMAC_KEY='use-a-secret-from-your-secret-manager'
python src/data_loaders/human_model_fitting_store.py
```

The default output is `consolidated.sqlite3` in this directory and is ignored
by Git. A machine-readable audit can be written with:

```bash
python src/data_loaders/human_model_fitting_store.py \
  --audit-json /tmp/human-fitting-audit.json
```

## Identity rules

- An instance ID is never globally unique. Its identity is
  `(dataset_version_id, source_instance_id)`.
- CoAX and CoXAM datasets with the same `appId` are different versions. Their
  feature sets and row order differ.
- A human response is an event, not merely a participant/trial pair. CoAX can
  contain both without-XAI and with-XAI inference events for one source trial.
- Raw participant IDs are not copied into the database. The builder stores an
  HMAC-SHA-256 pseudonym when `HUMAN_FITTING_HMAC_KEY` is set. The unsalted
  local fallback is explicitly flagged in `data_quality_issues`.
- `.pth`, `.zip`, and `.jmp` files should stay in filesystem/object storage.
  `model_artifacts` stores their URI, checksum, size, and linkage metadata.

## Canonical contents

- `dataset_versions`, `dataset_features`, `instances`: versioned task data.
- `dataset_provenance`: checksummed raw/preprocessed dataset and generation-code
  candidates. Links are explicitly marked unverified where no row map exists.
- `ai_models`, `ai_predictions`, `model_artifacts`: AI and cognitive-policy
  artifacts with checksums.
- `local_explanations`: CoAX per-instance importance/attribution vectors.
- `surrogate_explanations`: CoXAM logistic-regression and decision-tree
  surrogate representations.
- `studies`, `participants`, `trials`: pseudonymized behavioral events.
- `cognitive_fit_runs`, `cognitive_parameters`: fitted participant parameters.
- `data_quality_issues`: preserved, queryable ingestion problems.
- `fit_ready_human_responses` and `fitted_cognitive_parameters`: convenient
  analysis views.

The schema is defined in
`src/data_loaders/human_model_fitting_schema.sql`.

## Audit findings for the current source snapshot

### CoAX

- 4,661 instances across `adult`, `forest_cover`, `mushrooms`, and
  `wine_quality`.
- 50,364 human event rows from 391 participant IDs.
- Every human row joins to an instance and AI prediction. Every importance or
  attribution event joins to its generated explanation, and stored AI
  predictions agree across the joined files.
- 42,524 inference rows contain responses; 7,840 feedback rows intentionally
  have no response.
- 1,133 fitted strategy rows cover 330 participants and three datasets
  (`adult`, `forest_cover`, `wine_quality`). No fitted rows are present for
  `mushrooms`; 61 human-study participants have no fitted parameter row.
- Parameter nulls are strategy-dependent: `sensitivity` is absent in 511 rows
  and `scaling_factor` is absent in 622 rows. Store parameters as child rows,
  not as one wide fixed vector.
- The second copy of the five dataset/explanation CSVs under
  `coax/UI/xai_methods` is byte-identical and should not be ingested.

### CoXAM

- 3,873 instances across 12 datasets. Metadata also declares
  `loan_approval`, but no instance rows are present for it.
- 26,000 human rows cover the `mushrooms` and `wine_quality` studies.
- 25,992 rows join to an instance and AI prediction with no prediction
  disagreement. Eight counterfactual rows have no instance/XAI identifiers.
- 80 rows have no participant ID. In total, 12,960 forward rows and 12,902
  counterfactual rows have the identifiers and observed response fields needed
  for fitting.
- All otherwise valid human conditions have the required LR/DT surrogate
  representation.
- The fitting notebooks reference `datasets/combined.csv`; it is absent.
  Intended participant-fit outputs such as
  `participant_parameters_fit_lr_v0.1.csv`,
  `participant_parameters_fit_dt_v0.1.csv`, and `outputs/rl_fit_trials*.csv`
  are also absent. Therefore CoXAM cognitive parameters are not currently
  available as durable artifacts.

### Shared repository assets

- Raw or prepared dataset artifacts exist under `assets/original_datasets` for
  every dataset used in the two human studies (`adult`, `forest_cover`,
  `mushrooms`, and `wine_quality`). However, there is no durable map from a
  study `instanceId` to the raw-row ID and train/dev/test split, so provenance
  is directory-name matched but not fully reproducible at row level.
- Several additional CoXAM datasets are reproducibly named in
  `dataset_generator/generate_datasets.py` (scikit-learn or statsmodels
  sources), while others only have prepared CSV rows. The importer records
  both local artifacts and generation code in `dataset_provenance`.
- Candidate AI weight files exist under `assets/model_weights`, and CoXAM has
  saved PPO policy ZIPs under its output tree. Filename matching alone does
  not prove that a weight file produced a particular prediction table, so the
  importer marks those links `filename_match_unverified`.
- The existing `assets/ai_dataset` and `assets/explanations` copies are older
  subsets and do not hash-match the source snapshots used by the human studies.
  They should not replace the versioned source files during fitting.

## Recommended server deployment

Use PostgreSQL for metadata, instances, responses, and fitted parameters.
Translate SQLite JSON text columns to `jsonb`, identity integers to `bigint`
or UUIDs, and retain the same composite uniqueness constraints. Put large
binary artifacts in S3-compatible object storage and keep only immutable URIs
plus SHA-256 checksums in PostgreSQL.

Recommended access controls:

1. Keep the raw-to-pseudonym mapping outside the research database.
2. Restrict participant-level tables to the fitting service role.
3. Expose de-identified views to analysts.
4. Treat source files as immutable ingestion batches; create a new
   `dataset_version` or study snapshot when data changes.
