"""
Tutorial: Using the Unified Data Loader for CoAX and CoXAM

This tutorial demonstrates the core patterns for loading and accessing 
explanations from both CoAX (pre-computed) and CoXAM (on-demand) systems.
"""

# ============================================================================
# PATTERN 1: CoAX - Get Pre-Stored Explanations from CSV
# ============================================================================
# CoAX stores pre-computed explanations in CSV files. Simply load and access.

def tutorial_coax_explanations():
    """
    Load CoAX data and retrieve pre-stored explanations from CSV.
    
    Use case: You have pre-computed importance/attribution values in CSV format
    and want to access them alongside feature values and metadata.
    """
    from src.data_loaders import UnifiedDataLoader
    
    # Step 1: Create loader from CoAX data
    loader = UnifiedDataLoader.from_assets(
        source="coax",
        assets_root="assets",
        coax_explanation_type="importance",
    )
    
    # Step 2: Get explanations for instances [0, 1, 2]
    explanations = loader.get_explanations([0, 1, 2])  # From CSV
    
    print(f"✓ Loaded explanations for 3 instances")
    print(f"  Shape: {explanations.shape}")
    print(f"  Columns: {list(explanations.columns)}")
    
    return loader, explanations


# ============================================================================
# PATTERN 2: CoXAM - Compute Explanations On-Demand using Registry
# ============================================================================
# CoXAM uses interpreter classes that analyze models and generate explanations.

def tutorial_coxam_explanations():
    """
    Load CoXAM data and use the explanation method registry to compute explanations.
    
    Use case: You have trained models (Decision Trees, Logistic Regression)
    and want to generate interpretable explanations by analyzing model behavior.
    """
    from src.data_loaders import UnifiedDataLoader
    import json
    
    # Step 1: Create loader from CoXAM data
    loader = UnifiedDataLoader.from_assets(
        source="coxam",
        assets_root="assets",
    )
    
    from src.xai_adapter import create_coxam_xai_method, get_adapter_registry
    available = get_adapter_registry().list_available()
    print(f"✓ Available explanation methods: {available}")

    dt_method = create_coxam_xai_method(
        loader,
        method_type="rules",
        app_id="wine_quality",
        model_name="mlp",
        depth=3,
    )
    lr_method = create_coxam_xai_method(
        loader,
        method_type="weights",
        app_id="wine_quality",
        model_name="mlp",
        variant="sparse",
    )

    features = loader.get_features([0, 1, 2], normalize=False)
    print(f"✓ Decision Tree predictions: {dt_method.apply_batch(features)}")
    print(f"✓ Logistic Regression predictions: {lr_method.apply_batch(features)}")
    
    return loader, dt_method, lr_method


# ============================================================================
# PATTERN 3: Comparing CoAX vs CoXAM - Same Instances, Different Access
# ============================================================================
# Demonstrate the difference in how explanations are retrieved.

def tutorial_comparison():
    """
    Side-by-side comparison of CoAX (CSV-based) vs CoXAM (model-based).
    """
    from src.data_loaders import UnifiedDataLoader
    
    print("\n" + "="*70)
    print("COMPARISON: CoAX vs CoXAM Explanation Access")
    print("="*70)
    
    # -------- CoAX: Pre-computed from CSV --------
    print("\n1. CoAX Pattern (Pre-computed Explanations)")
    print("-" * 70)
    
    loader_coax = UnifiedDataLoader.from_assets(
        source="coax",
        assets_root="assets",
        coax_explanation_type="importance",
    )
    
    # Direct access to CSV-stored explanations
    explanations_coax = loader_coax.get_explanations([0, 1, 2])
    print(f"  • Access method: loader.get_explanations()")
    print(f"  • Source: CSV files (pre-computed)")
    print(f"  • Speed: ⚡ Very fast (just loading from disk)")
    print(f"  • Result shape: {explanations_coax.shape}")
    
    # -------- CoXAM: Model-based Interpretation --------
    print("\n2. CoXAM Pattern (On-Demand Model Interpretation)")
    print("-" * 70)
    
    loader_coxam = UnifiedDataLoader.from_assets(
        source="coxam",
        assets_root="assets",
    )
    
    # Model-based explanations via registry
    from src.xai_adapter import get_adapter_registry
    registry = get_adapter_registry()
    print(f"  • Access method: registry.create() then .apply()")
    print(f"  • Source: Model analysis algorithms")
    print(f"  • Speed: ⏱️  Depends on algorithm complexity")
    print(f"  • Available explanation methods: {registry.list_available()}")
    
    # -------- Summary --------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    ┌─ CoAX ──────────────────────────────────────────────────────────┐
    │ • Pre-computed explanation values stored in CSV files           │
    │ • Fast, deterministic, no model required                        │
    │ • Use: loader.get_explanations([instance_ids])                  │
    │ • Example: Load precomputed LIME or SHAP values                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─ CoXAM ─────────────────────────────────────────────────────────┐
    │ • Model interpretation algorithms (DT, LR, etc.)                │
    │ • Compute explanations on-demand, flexible & extensible         │
    │ • Use: create_coxam_xai_method(...).apply_batch(instances)       │
    │ • Example: Generate rules from Decision Tree model              │
    └─────────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# PATTERN 4: Mixed Access - Combining Both Approaches
# ============================================================================
# Advanced: Use CoAX data but also generate CoXAM-style explanations.

def tutorial_mixed_access():
    """
    Load CoAX data but also register and use custom explanation methods.
    
    Use case: You have pre-computed explanations but want to augment
    them with on-demand model interpretations.
    """
    from src.data_loaders import UnifiedDataLoader
    from src.xai_adapter import XAIAdapter
    
    # Load CoAX data
    loader = UnifiedDataLoader.from_assets(
        source="coax",
        assets_root="assets",
        coax_explanation_type="importance",
    )
    
    print("\n1. Original CoAX explanations:")
    explanations_csv = loader.get_explanations([0, 1])
    print(f"   Loaded from CSV: {explanations_csv.shape}")
    
    # Get the registry and register a custom explanation method
    print("\n2. Register a custom explanation method:")
    from src.xai_adapter import get_adapter_registry
    registry = get_adapter_registry()
    
    class CustomMethod(XAIAdapter):
        """Example custom explanation method."""
        def explain(self, instances):
            return {"custom_score": len(instances)}

        def apply(self, instance):
            return {"custom_score": sum(instance) / len(instance)}
        
        def apply_batch(self, instances):
            return [self.apply(inst) for inst in instances]
        
        def get_info(self):
            return {"name": "custom_xai_method", "version": "0.1"}
    
    registry.register('custom', CustomMethod)
    print(f"   Available explanation methods now: {registry.list_available()}")
    
    # Use the custom explanation method
    print("\n3. Use the custom explanation method alongside CSV explanations:")
    custom_exp = registry.create('custom')
    instances = loader.get_features([0, 1])
    custom_results = custom_exp.apply_batch(instances)
    print(f"   Custom explanations: {custom_results}")


# ============================================================================
# PATTERN 5: Filtering and Advanced Features
# ============================================================================
# Use the filter builder to subset data, then access explanations.

def tutorial_filtering():
    """
    Demonstrate composable filtering for both CoAX and CoXAM.
    """
    from src.data_loaders import UnifiedDataLoader
    
    # Load data
    loader = UnifiedDataLoader.from_assets(
        source="coxam",
        assets_root="assets",
    )
    
    print("\nFiltering examples:")
    print("-" * 70)
    
    # Example 1: Filter by app
    print("1. Get data for 'wine' app only:")
    filtered = loader.filter().by_app("wine")
    print(f"   Filtered indices: {filtered.get_indices()[:5]}...")
    
    # Example 2: Chain filters
    print("\n2. Filter by app AND condition (model name):")
    filtered = loader.filter().by_app("wine").by_condition("LR")
    print(f"   Filtered indices: {filtered.get_indices()[:5]}...")
    
    # Example 3: Get participant-specific data (CoXAM only)
    print("\n3. Get trials for specific participant:")
    participant_ids = loader.get_participant_ids()
    if participant_ids:
        trials = loader.get_participant_trials(participant_ids[0])
        print(f"   Trials for {participant_ids[0]}: {len(trials)} entries")
    
    # Example 4: Apply filter and get explanations
    print("\n4. Filter + get explanations:")
    filtered = loader.filter().by_app("wine")
    indices = filtered.get_indices()[:3]
    if indices:
        explanations = loader.get_explanations(indices)
        print(f"   Explanations for {len(indices)} filtered instances: {explanations.shape}")


# ============================================================================
# MAIN: Run All Tutorials
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIFIED DATA LOADER TUTORIAL")
    print("="*70)
    
    try:
        print("\n[Tutorial 1] CoAX - Pre-stored Explanations from CSV")
        print("-" * 70)
        tutorial_coax_explanations()
    except Exception as e:
        print(f"  (Skipped: {type(e).__name__})")
    
    try:
        print("\n[Tutorial 2] CoXAM - On-Demand Explainer Registry")
        print("-" * 70)
        tutorial_coxam_explanations()
    except Exception as e:
        print(f"  (Skipped: {type(e).__name__})")
    
    try:
        print("\n[Tutorial 3] Comparison: CoAX vs CoXAM")
        tutorial_comparison()
    except Exception as e:
        print(f"  (Skipped: {type(e).__name__})")
    
    try:
        print("\n[Tutorial 4] Mixed Access - Combine Both Approaches")
        print("-" * 70)
        tutorial_mixed_access()
    except Exception as e:
        print(f"  (Skipped: {type(e).__name__})")
    
    try:
        print("\n[Tutorial 5] Filtering and Advanced Features")
        print("-" * 70)
        tutorial_filtering()
    except Exception as e:
        print(f"  (Skipped: {type(e).__name__})")
    
    print("\n" + "="*70)
    print("✓ Tutorials completed!")
    print("="*70)
