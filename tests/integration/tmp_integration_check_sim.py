from src.data_loaders import UnifiedDataLoader
from src.xai_adapter import create_coxam_xai_method
from src.user_simulation import SessionGenerator, SessionConfig, StrategyConfig
from src.cognitive_models import initialize_strategies, StrategyRegistry


def main() -> None:
    initialize_strategies()

    coax_loader = UnifiedDataLoader.from_assets(
        source="coax",
        assets_root="assets",
        app_id="wine_quality",
    )

    coxam_loader = UnifiedDataLoader.from_assets(
        source="coxam",
        assets_root="assets",
        app_id="wine_quality",
    )
    explainer = create_coxam_xai_method(
        coxam_loader,
        method_type="decision_tree",
        app_id="wine_quality",
        model_name="mlp",
        depth=3,
    )

    config = SessionConfig(
        dataset_name="wine_quality",
        n_participants=2,
        n_trials_per_participant=5,
        ai_dataset_loader=coax_loader,
        distribution_file="src/user_simulation/three datasets strategies_distributions.json",
        strategy_configs=[
            StrategyConfig(
                strategy_name="sensitive_features",
                percentage=100.0,
                xai_type="importance",
                tested_with_xai=True,
                explainer=explainer,
            )
        ],
        random_seed=42,
    )

    gen = SessionGenerator()
    gen.simulator.setup_dependencies(strategy_registry=StrategyRegistry)

    results = gen.generate(config)
    df = gen.results_to_dataframe(results)

    print("ok_results", len(results))
    print("ok_participants", int(df["Participant ID"].nunique()))
    print(
        "ok_trials_per_participant",
        int(len(df) / max(1, int(df["Participant ID"].nunique()))),
    )
    print(
        "has_response_cols",
        sorted(
            [
                c
                for c in [
                    "AI Prediction",
                    "Explainer Prediction",
                    "Response",
                    "Response==AI",
                    "Response==Explainer",
                ]
                if c in df.columns
            ]
        ),
    )
    print(
        "inference_methods",
        sorted(df["inference_method"].dropna().unique().tolist())
        if "inference_method" in df.columns
        else [],
    )
    print(
        "sample_row",
        df[
            [
                "Participant ID",
                "Trial Index",
                "Instance Id",
                "AI Prediction",
                "Explainer Prediction",
                "Response",
            ]
        ]
        .head(1)
        .to_dict(orient="records")[0],
    )


if __name__ == "__main__":
    main()
