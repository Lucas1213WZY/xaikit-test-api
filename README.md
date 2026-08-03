# xaikit-test
XAI Interpretation Simulator Toolkit

## 📖 Documentation

**<https://lucas1213wzy.github.io/xaikit-test-api/>** — user guide and full API
reference, rebuilt automatically on every push to `main`.

To build it locally, see [docs/README.md](docs/README.md):

```bash
python docs/build_docs.py && python -m http.server -d site 8000
```

## 🛖Project  Architecture

```markdown
xaikit-test/
├── assets/ 
│   └── data/     
│       ├── ai_dataset/
│       ├── explanations/
│       └──  human_trials_and_cognitive_paramters/  
│ 
├── src/                          ← IMPLEMENTATION (internal)
│   ├── experiment_design
│   ├── cognitive_models/     (cognitive models - reasoning strategies & strategy selector)
│   │   ├── forward_rs/              (CoAX/CoXAM forward strategies)
│   │   ├── counterfactual_rs/       (CoXAM counterfactual strategies)
│   │   ├── memory/               (cognitive memory backends)
│   │   └── cr_agent/             (CoXAM CR agent orchestration)
│   ├── models/                   (AI model implementations)
│   ├── data_loaders/             (data processing, XAI dataset CSV parsing)
│   │   ├── ai_dataloader/        (loading AI datasets for training and explanation generation)      
│   │   └── xai_adapter/.         (XAI methods: attribution, rules/weights, adapters to convert into standard data format))
│   └── virtual_experiment_executor/ (API-driven virtual experiment simulation) [experiments for hypothesis generation]
│ 
├── tests/                        ← Integration tests and import/path setup
│   ├── conftest.py
│   └── integration/
│       ├── test_unified_memory.py
│       ├── test_xai_adapter_api.py
│       ├── test_models_registry.py
│       ├── tmp_integration_check_sim.py
│       └── verify_refactored_models.py
│ 
├── tutorials/                    ← Tutorial notebooks and rendered quickstarts
│   ├── xai_adapter_quickstart.ipynb
│   ├── virtual_experiment_executor_quickstart.ipynb
│   └── simulated_results/
│
├── UI
│   ├── instance_visualization
│   ├── results_dashboard
│   └── experimental_design
│
└── README.md                      
```

## Tutorial workflow summaries

- [Colab full workflow table](tutorials/colab_tutorial_full_workflow_table.md) abstracts the end-to-end sequence from `tutorials/colab_tutorial_full_workflow.ipynb`.
