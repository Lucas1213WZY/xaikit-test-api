"""
Comprehensive examples demonstrating the unified data loader system.

This module shows how to use the UnifiedDataLoader with CoAX, CoXAM, 
and custom data sources, including filtering, normalization, and explanation methods.
"""

import sys
import pandas as pd
import numpy as np

# Example 1: Loading CoAX synthetic data
def example_coax_loading():
    """Example: Load and use CoAX data."""
    from src.data_loaders import UnifiedDataLoader, MinMaxNormalizer
    
    print("=" * 60)
    print("Example 1: Loading CoAX Synthetic Data")
    print("=" * 60)
    
    # Create loader from the standardized assets directory
    loader = UnifiedDataLoader.from_assets(
        source="coax",
        assets_root="assets",
        coax_explanation_type="importance",
    )
    
    # Get summary
    summary = loader.get_summary()
    print(f"\nLoaded summary: {summary}")
    print(f"Available apps: {loader.list_apps()}")
    
    # Get instances
    try:
        features, predictions = loader.get_instances([0, 1, 2], normalize=True)
        print(f"\nFeatures shape: {len(features)} instances, {len(features[0])} features each")
        print(f"First instance predictions: {predictions[0]}")
    except Exception as e:
        print(f"(Note: Data files may not be available: {e})")


# Example 2: Loading CoXAM experiment data
def example_coxam_loading():
    """Example: Load and use CoXAM data."""
    from src.data_loaders import UnifiedDataLoader
    
    print("\n" + "=" * 60)
    print("Example 2: Loading CoXAM Experiment Data")
    print("=" * 60)
    
    # Create loader from the standardized assets directory
    loader = UnifiedDataLoader.from_assets(
        source="coxam",
        assets_root="assets",
    )
    
    # Get summary
    summary = loader.get_summary()
    print(f"\nLoaded summary: {summary}")
    
    # Get participant-specific data (CoXAM only)
    try:
        participants = loader.get_participant_ids()
        if participants:
            first_participant = participants[0]
            trials = loader.get_participant_trials(first_participant)
            print(f"\nParticipant {first_participant} has {len(trials)} trials")
    except Exception as e:
        print(f"(Note: Participant data may not be available: {e})")


# Example 3: Composable filtering
def example_filtering():
    """Example: Use composable filters."""
    from src.data_loaders import UnifiedDataLoader, FilterBuilder
    
    print("\n" + "=" * 60)
    print("Example 3: Composable Filtering")
    print("=" * 60)
    
    # Create a filter builder
    filter_builder = FilterBuilder()
    
    # Chain multiple filter conditions
    filtered = (
        filter_builder
        .by_app("wine_quality")
        .by_condition("LR")
        .by_xai_type("importance")
    )
    
    print(f"Filter chain created with {len(filtered.conditions)} conditions")
    print("Filters applied:")
    print("  - by_app('wine_quality')")
    print("  - by_condition('LR')")
    print("  - by_xai_type('importance')")


def example_xai_methods():
    """Example: Create and use explanation methods."""
    from src.xai_adapter import get_adapter_registry
    
    print("\n" + "=" * 60)
    print("Example 4: XAI Method Management")
    print("=" * 60)
    
    # Get global registry
    registry = get_adapter_registry()
    
    print(f"\nRegistry initialized")
    print(f"Available explanation method types: {registry.list_available()}")
    
    # Check if explanation method is registered
    if registry.is_registered('decision_tree'):
        print("\n✓ Decision Tree method is registered")
    
    if registry.is_registered('logistic_regression'):
        print("✓ Logistic Regression method is registered")


# Example 5: Full workflow with UnifiedDataLoader
def example_full_workflow():
    """Example: Complete workflow combining all features."""
    from src.data_loaders import UnifiedDataLoader
    
    print("\n" + "=" * 60)
    print("Example 5: Full Workflow")
    print("=" * 60)
    
    # Step 1: Create loader
    print("\n[1] Creating UnifiedDataLoader for CoAX...")
    try:
        loader = UnifiedDataLoader.from_assets(
            source="coax",
            assets_root="assets",
        )
        print(f"    Loaded: {loader}")
        
        # Step 2: Inspect data
        print("\n[2] Inspecting loaded data...")
        summary = loader.get_summary()
        print(f"    Apps: {summary.get('app_ids')}")
        print(f"    Instances: {summary.get('n_instances')}")
        print(f"    Features: {summary.get('n_features')}")
        
        # Step 3: Get instances
        print("\n[3] Retrieving instances...")
        try:
            features, predictions = loader.get_instances([0, 1], normalize=True)
            print(f"    Retrieved {len(features)} instances with features shape {len(features[0])}")
        except Exception as e:
            print(f"    (Instance retrieval requires actual CSV files)")
        
        # Step 4: XAI methods
        print("\n[4] Using XAI methods...")
        from src.xai_adapter import get_adapter_registry
        xai_methods = get_adapter_registry().list_available()
        print(f"    Available XAI methods: {xai_methods}")
        
        print("\n✓ Full workflow completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error in workflow: {e}")
        print("  (Note: Some features require actual data files)")


# Example 6: API Reference
def print_api_reference():
    """Print quick API reference."""
    print("\n" + "=" * 60)
    print("API REFERENCE")
    print("=" * 60)
    
    api_reference = """
    
    CREATING LOADERS:
    ─────────────────
    
    # From CoAX data
    loader = UnifiedDataLoader.from_coax(
        feature_file="assets/ai_datasets/coax/values.csv",
        metadata_file="assets/ai_datasets/coax/metadata.csv",
        prediction_file="assets/ai_datasets/coax/none.csv"
    )

    # From CoXAM data
    loader = UnifiedDataLoader.from_coxam(
        feature_file="assets/ai_datasets/coxam/values.csv",
        metadata_file="assets/ai_datasets/coxam/metadata.csv",
        prediction_file="assets/ai_datasets/coxam/none.csv"
    )
    
    # From custom source
    loader = UnifiedDataLoader.from_custom(my_data_source)
    
    
    ACCESSING DATA:
    ───────────────
    
    # Get features and predictions
    features, predictions = loader.get_instances([0, 1, 2], normalize=True)
    
    # Get just features
    features = loader.get_features([0, 1, 2])
    
    # Get just predictions
    predictions = loader.get_predictions([0, 1, 2])
    
    # Get explanations
    explanations = loader.get_explanations([0, 1, 2])
    
    
    FILTERING:
    ──────────
    
    # Chain multiple filters
    filter_builder = loader.filter()
    filter_builder.by_app("wine_quality")
    filter_builder.by_condition("LR")
    filter_builder.by_xai_type("importance")
    loader.apply_filter(filter_builder)
    
    
    EXPLAINERS:
    ───────────
    
    # Get registry
    from src.xai_adapter import get_adapter_registry
    registry = get_adapter_registry()
    
    # List available
    xai_methods = registry.list_available()
    
    # Create explanation method
    dt_exp = registry.create(
        'decision_tree',
        explanation_df=dt_df,
        metadata_df=metadata_df,
        app_id="wine_quality",
        model_name="mlp"
    )
    
    # Apply to instance
    result = dt_exp.apply(instance_features)
    
    
    INSPECTION:
    ───────────
    
    # Get summary
    summary = loader.get_summary()
    
    # List apps
    apps = loader.list_apps()
    
    # Get metadata
    metadata = loader.get_metadata()
    
    # Participant data (CoXAM only)
    trials = loader.get_participant_trials(participant_id=1, phase="forward")
    
    """
    
    print(api_reference)


if __name__ == "__main__":
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " Unified Data Loader - Comprehensive Examples ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run examples
    example_coax_loading()
    example_coxam_loading()
    example_filtering()
    example_xai_methods()
    example_full_workflow()
    print_api_reference()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
