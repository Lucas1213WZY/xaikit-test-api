PRAGMA foreign_keys = ON;

CREATE TABLE source_files (
    source_file_id INTEGER PRIMARY KEY,
    framework TEXT NOT NULL CHECK (framework IN ('coax', 'coxam', 'shared')),
    role TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    row_count INTEGER,
    ingested_at TEXT NOT NULL,
    UNIQUE (relative_path, sha256)
);

CREATE TABLE dataset_versions (
    dataset_version_id INTEGER PRIMARY KEY,
    framework TEXT NOT NULL CHECK (framework IN ('coax', 'coxam')),
    data_id TEXT NOT NULL,
    version_key TEXT NOT NULL UNIQUE,
    feature_schema_sha256 TEXT NOT NULL,
    snapshot_sha256 TEXT NOT NULL,
    metadata_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    metadata_json TEXT NOT NULL
);

CREATE TABLE dataset_features (
    dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(dataset_version_id),
    feature_index INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    feature_kind TEXT NOT NULL,
    min_value REAL,
    max_value REAL,
    options_json TEXT,
    PRIMARY KEY (dataset_version_id, feature_index)
);

CREATE TABLE instances (
    dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(dataset_version_id),
    source_instance_id TEXT NOT NULL,
    target_value TEXT,
    feature_values_json TEXT NOT NULL,
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    source_row_number INTEGER NOT NULL,
    PRIMARY KEY (dataset_version_id, source_instance_id),
    UNIQUE (source_file_id, source_row_number)
);

CREATE TABLE dataset_provenance (
    dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(dataset_version_id),
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    provenance_role TEXT NOT NULL,
    link_status TEXT NOT NULL,
    PRIMARY KEY (dataset_version_id, source_file_id, provenance_role)
);

CREATE TABLE ai_models (
    ai_model_id INTEGER PRIMARY KEY,
    dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(dataset_version_id),
    model_name TEXT NOT NULL,
    model_role TEXT NOT NULL DEFAULT 'black_box',
    UNIQUE (dataset_version_id, model_name, model_role)
);

CREATE TABLE ai_predictions (
    ai_model_id INTEGER NOT NULL REFERENCES ai_models(ai_model_id),
    dataset_version_id INTEGER NOT NULL,
    source_instance_id TEXT NOT NULL,
    prediction TEXT,
    normalization_max REAL,
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    source_row_number INTEGER NOT NULL,
    PRIMARY KEY (ai_model_id, source_instance_id),
    FOREIGN KEY (dataset_version_id, source_instance_id)
        REFERENCES instances(dataset_version_id, source_instance_id),
    UNIQUE (source_file_id, source_row_number)
);

CREATE TABLE local_explanations (
    local_explanation_id INTEGER PRIMARY KEY,
    ai_model_id INTEGER NOT NULL REFERENCES ai_models(ai_model_id),
    dataset_version_id INTEGER NOT NULL,
    source_instance_id TEXT NOT NULL,
    explanation_type TEXT NOT NULL,
    explanation_method TEXT NOT NULL,
    prediction TEXT,
    normalization_max REAL,
    payload_json TEXT NOT NULL,
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    source_row_number INTEGER NOT NULL,
    FOREIGN KEY (dataset_version_id, source_instance_id)
        REFERENCES instances(dataset_version_id, source_instance_id),
    UNIQUE (
        ai_model_id,
        source_instance_id,
        explanation_type,
        explanation_method
    ),
    UNIQUE (source_file_id, source_row_number)
);

CREATE TABLE surrogate_explanations (
    surrogate_explanation_id INTEGER PRIMARY KEY,
    ai_model_id INTEGER NOT NULL REFERENCES ai_models(ai_model_id),
    surrogate_type TEXT NOT NULL,
    variant TEXT NOT NULL,
    fidelity REAL,
    payload_json TEXT NOT NULL,
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    source_row_number INTEGER NOT NULL,
    UNIQUE (ai_model_id, surrogate_type, variant),
    UNIQUE (source_file_id, source_row_number)
);

CREATE TABLE studies (
    study_id INTEGER PRIMARY KEY,
    framework TEXT NOT NULL CHECK (framework IN ('coax', 'coxam')),
    study_key TEXT NOT NULL UNIQUE,
    study_name TEXT NOT NULL,
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
);

CREATE TABLE participants (
    participant_id INTEGER PRIMARY KEY,
    study_id INTEGER NOT NULL REFERENCES studies(study_id),
    participant_key TEXT NOT NULL,
    pseudonym_method TEXT NOT NULL,
    UNIQUE (study_id, participant_key)
);

CREATE TABLE trials (
    trial_id INTEGER PRIMARY KEY,
    study_id INTEGER NOT NULL REFERENCES studies(study_id),
    participant_id INTEGER REFERENCES participants(participant_id),
    dataset_version_id INTEGER REFERENCES dataset_versions(dataset_version_id),
    ai_model_id INTEGER REFERENCES ai_models(ai_model_id),
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    source_row_number INTEGER NOT NULL,
    source_trial_index TEXT,
    source_instance_id TEXT,
    session TEXT,
    phase TEXT,
    step TEXT,
    trial_type TEXT,
    tested_with_xai INTEGER CHECK (tested_with_xai IN (0, 1) OR tested_with_xai IS NULL),
    xai_type TEXT,
    explanation_method TEXT,
    condition_name TEXT,
    complexity TEXT,
    ai_prediction TEXT,
    human_response TEXT,
    is_correct INTEGER CHECK (is_correct IN (0, 1) OR is_correct IS NULL),
    response_time REAL,
    counterfactual_response_json TEXT,
    fit_eligible INTEGER NOT NULL CHECK (fit_eligible IN (0, 1)),
    quality_status TEXT NOT NULL,
    UNIQUE (source_file_id, source_row_number),
    FOREIGN KEY (dataset_version_id, source_instance_id)
        REFERENCES instances(dataset_version_id, source_instance_id)
);

CREATE TABLE cognitive_fit_runs (
    fit_run_id INTEGER PRIMARY KEY,
    framework TEXT NOT NULL CHECK (framework IN ('coax', 'coxam')),
    study_id INTEGER REFERENCES studies(study_id),
    participant_id INTEGER REFERENCES participants(participant_id),
    dataset_version_id INTEGER REFERENCES dataset_versions(dataset_version_id),
    ai_model_id INTEGER REFERENCES ai_models(ai_model_id),
    source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
    source_row_number INTEGER NOT NULL,
    session TEXT,
    xai_type TEXT,
    explanation_method TEXT,
    tested_with_xai INTEGER CHECK (tested_with_xai IN (0, 1) OR tested_with_xai IS NULL),
    strategy TEXT,
    fit_algorithm TEXT,
    assignment_probability REAL,
    objective_name TEXT,
    objective_value REAL,
    metrics_json TEXT NOT NULL,
    UNIQUE (source_file_id, source_row_number)
);

CREATE TABLE cognitive_parameters (
    fit_run_id INTEGER NOT NULL REFERENCES cognitive_fit_runs(fit_run_id),
    parameter_name TEXT NOT NULL,
    numeric_value REAL,
    text_value TEXT,
    PRIMARY KEY (fit_run_id, parameter_name)
);

CREATE TABLE model_artifacts (
    model_artifact_id INTEGER PRIMARY KEY,
    framework TEXT NOT NULL CHECK (framework IN ('coax', 'coxam', 'shared')),
    artifact_kind TEXT NOT NULL,
    ai_model_id INTEGER REFERENCES ai_models(ai_model_id),
    relative_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    link_status TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE data_quality_issues (
    issue_id INTEGER PRIMARY KEY,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    issue_code TEXT NOT NULL,
    framework TEXT,
    source_file_id INTEGER REFERENCES source_files(source_file_id),
    source_row_number INTEGER,
    entity_type TEXT,
    entity_key TEXT,
    details_json TEXT NOT NULL
);

CREATE INDEX idx_instances_lookup
    ON instances(dataset_version_id, source_instance_id);
CREATE INDEX idx_dataset_provenance
    ON dataset_provenance(dataset_version_id, provenance_role);
CREATE INDEX idx_predictions_lookup
    ON ai_predictions(ai_model_id, source_instance_id);
CREATE INDEX idx_local_explanations_lookup
    ON local_explanations(ai_model_id, explanation_type, explanation_method, source_instance_id);
CREATE INDEX idx_trials_participant_sequence
    ON trials(participant_id, session, phase, source_trial_index);
CREATE INDEX idx_trials_instance
    ON trials(dataset_version_id, source_instance_id);
CREATE INDEX idx_trials_fit_eligible
    ON trials(study_id, fit_eligible, phase, step);
CREATE INDEX idx_fit_runs_participant
    ON cognitive_fit_runs(participant_id, dataset_version_id, strategy);
CREATE INDEX idx_quality_issue_code
    ON data_quality_issues(severity, issue_code);

CREATE VIEW fit_ready_human_responses AS
SELECT
    t.trial_id,
    s.framework,
    s.study_key,
    p.participant_key,
    dv.version_key AS dataset_version_key,
    t.source_instance_id,
    am.model_name,
    t.source_trial_index,
    t.session,
    t.phase,
    t.step,
    t.trial_type,
    t.tested_with_xai,
    t.xai_type,
    t.explanation_method,
    t.condition_name,
    t.complexity,
    t.ai_prediction,
    t.human_response,
    t.is_correct,
    t.response_time,
    t.counterfactual_response_json
FROM trials AS t
JOIN studies AS s ON s.study_id = t.study_id
JOIN participants AS p ON p.participant_id = t.participant_id
JOIN dataset_versions AS dv ON dv.dataset_version_id = t.dataset_version_id
JOIN ai_models AS am ON am.ai_model_id = t.ai_model_id
WHERE t.fit_eligible = 1;

CREATE VIEW fitted_cognitive_parameters AS
SELECT
    fr.fit_run_id,
    fr.framework,
    p.participant_key,
    dv.version_key AS dataset_version_key,
    am.model_name,
    fr.session,
    fr.xai_type,
    fr.explanation_method,
    fr.tested_with_xai,
    fr.strategy,
    fr.fit_algorithm,
    fr.assignment_probability,
    fr.objective_name,
    fr.objective_value,
    cp.parameter_name,
    cp.numeric_value,
    cp.text_value
FROM cognitive_fit_runs AS fr
LEFT JOIN participants AS p ON p.participant_id = fr.participant_id
LEFT JOIN dataset_versions AS dv ON dv.dataset_version_id = fr.dataset_version_id
LEFT JOIN ai_models AS am ON am.ai_model_id = fr.ai_model_id
JOIN cognitive_parameters AS cp ON cp.fit_run_id = fr.fit_run_id;
