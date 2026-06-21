# Colab Tutorial Full Workflow Table

This table abstracts the end-to-end flow in
`tutorials/colab_tutorial_full_workflow.ipynb`.

| Phase | Workflow Step | Purpose | Main Inputs | Main Outputs |
| --- | --- | --- | --- | --- |
| 1 | Configure experimental design | Define the experiment factors and validation rules before any data or trial generation runs. | `iv_config`, `CVs`, `DVs`; IV type (`within` or `between`); within-subject randomization mode (`block` or `trial`) | Validated IV, CV, and DV dictionaries |
| 2 | Select and split dataset | Load the selected dataset, normalize features, preserve raw row ids, and create a stable train/test split. | `selected_dataset_ids`; source CSV such as `winequality-red.csv`; `test_size`; `random_state` | `dataset_id`, `df`, `X_train`, `X_test`, `y_train`, `y_test`, `train_instance_ids`, `test_instance_ids`, `feature_names` |
| 3 | Train AI model | Train the model that will later be explained by XAI adapters. | Dataset split from phase 2; `ModelManager`; model type such as `mlp` | `trained_model`, `trained_engine`, `train_data_X`, `test_data_X`, test accuracy |
| 4 | Generate explanation CSVs | Run each configured XAI method and export method-specific plus combined explanation files. | `trained_engine`, train/test data, labels, `feature_names`, `test_instance_ids`, XAI IV levels such as `shap` and `lime` | `generated_explanation/{method}_mlp_{dataset_id}.csv`; `generated_explanation/de_mlp_{dataset_id}.csv` |
| 5 | Build experimental trial design | Split IVs into between-subject, block-level within-subject, and trial-randomized roles; generate counterbalanced participant assignments and trial rows. | `iv_config`, `CVs`, combined explanation CSV, participants per between condition, trials per participant | `assignments`, `trials`, `experiment_output/trials.csv`, `experiment_output/trials.json`, `experiment_output/design_summary.json` |
| 6 | Convert UI JSON to trial info | Rebuild the same trial-generation pipeline from a UI-exported JSON configuration. | `src/experiment_design/example_json_from_ui/template.json`; dataset, sampling, and output config from JSON | Trial CSV, trial JSON, and design summary using UI-provided settings |
| 7 | Prepare cognitive model inputs | For each trial, combine trial metadata, raw instance attributes, AI prediction, and the matching explanation values. | Trial row, raw dataset `df`, explanation pool CSV, label column | Per-trial cognitive input dictionary with `trial_info`, `instance_attributes`, `instance_explanation`, and `ai_prediction` |
| 8 | Run cognitive simulation | Execute a cognitive model over one trial, one participant, one condition, or the whole experiment. | `trials`, `DVs`, cognitive parameters, raw dataset, explanation pool, execution mode | Executed trial result rows with DV outputs, cognitive parameters, explanation columns, and correctness vs AI |
| 9 | Export simulated results | Persist simulated experiment results for downstream analysis or dashboard use. | `executed` or full experiment simulation output | `experiment_output/simulated_results.csv`; `experiment_output/simulated_results.json` |

## Dependency Flow

| Upstream Artifact | Consumed By | Why It Matters |
| --- | --- | --- |
| `iv_config`, `CVs`, `DVs` | Trial design, XAI method selection, cognitive simulation | These define what conditions exist and what outcomes the workflow measures. |
| Dataset split outputs | Model training and explanation generation | The model trains on `X_train`; explanations are generated for `X_test` using stable `test_instance_ids`. |
| Trained model engine | XAI adapters | Explanations require the trained model's prediction interface. |
| Combined explanation CSV | Trial generation and cognitive simulation | Trial generation samples instance ids from the explanation pool; simulation retrieves explanation values and AI predictions. |
| `trials` | Cognitive executor and result export | Trial rows are the central schedule for participants, conditions, instances, and XAI exposure. |
| Simulated results | Analysis or UI/dashboard layers | Final table contains trial metadata plus simulated DV measurements. |
