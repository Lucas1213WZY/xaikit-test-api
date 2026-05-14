#!/usr/bin/env python3
"""Test unified models layer"""

import sys
sys.path.insert(0, '/Users/wangzhuoyulucas/Documents/GitHub/xaik-tool-cognitive-agent')

from src.models import ModelManager, ModelRegistry

print("\n" + "="*70)
print("UNIFIED MODELS LAYER - REGISTRY TEST")
print("="*70)

registry = ModelRegistry()
print(f"\n✓ Registry initialized")
print(f"✓ Model sources: {registry.list_sources()}")
print(f"✓ Total available models: {len(registry.list_available_models())}")

# Show models by source
for source in registry.list_sources():
    models_in_source = [m for m in registry.list_available_models() if f'_{source}' in m]
    print(f"\n  {source.upper()}: {len(models_in_source)} models")
    for model_key in sorted(models_in_source)[:5]:
        info = registry.available_models[model_key]
        print(f"    - {info['dataset']} ({info['model_type']})")

# Test specific lookups
print("\n" + "-"*70)
print("MODEL LOOKUP TESTS:")
print("-"*70)

datasets_to_test = ['wine_quality', 'forest_cover', 'mushrooms', 'heart_disease']

for dataset in datasets_to_test:
    for source in registry.list_sources():
        info_mlp = registry.get_model_info(dataset, 'mlp', source)
        info_xgb = registry.get_model_info(dataset, 'xgboost', source)
        
        mlp_status = "✓" if info_mlp else "✗"
        xgb_status = "✓" if info_xgb else "✗"
        print(f"  {dataset:20} ({source:5}): MLP {mlp_status} | XGBoost {xgb_status}")

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)
