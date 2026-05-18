"""
XAIK Models Layer - Complete Reference
========================================

A unified, simple interface for managing ML models (MLP & XGBoost).
"""

# QUICK START
# ===========

# 1. Load a pre-trained model
from src.models import ModelManager
import numpy as np

manager = ModelManager()
model = manager.load_model('wine_quality', 'mlp', source='coxam')

# 2. Make predictions
X_test = np.array([[7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8]])
predictions = manager.predict(X_test)
print(predictions)  # [[prob_0, prob_1]]

# 3. Train a new model
model = manager.create_model('dataset', 'mlp', input_dim=20, num_classes=3)
manager.train(X_train, y_train, epochs=100)

# 4. List available models
print(manager.list_available_pretrained()['available_models'])


# DIRECTORY STRUCTURE
# ===================

# /src/models/
# ├── __init__.py              # Exports
# ├── models.py                # Core classes (400+ LOC consolidated)
# ├── registry.py              # Model discovery
# ├── examples.py              # Integration examples
# ├── README.md                # This file
# │
# ├── coxam/                   # CoXAM models (11 datasets)
# │   ├── mlp/   (11 weights)
# │   └── xgboost/ (11 weights)
# │
# └── coax/                    # CoAX models (42 MLP + 8 XGBoost)
#     ├── mlp/   (42 weight variations)
#     └── xgboost/ (8 weights)


# CORE CLASSES
# ============

# ModelManager - Main API
# ---------
# manager = ModelManager()
# 
# # Loading
# model = manager.load_model(dataset, model_type, source='coxam')
# 
# # Training
# manager.train(X, y, X_dev, y_dev, epochs=100)
# 
# # Prediction
# predictions = manager.predict(X)
# 
# # Evaluation
# accuracy = manager.evaluate(X_test, y_test)
# 
# # Model management
# manager.save_model('path.pth')
# manager.set_active_model('model_key')
# manager.list_loaded_models()


# UnifiedModel (Abstract)
# --------
# Implemented by: MLPUnifiedModel, XGBoostUnifiedModel
# 
# Methods (all models):
#   - predict(X) → np.ndarray
#   - train(X, y, X_dev, y_dev) → Dict
#   - evaluate(X, y) → float
#   - load(weight_path) → None
#   - save(weight_path) → None
#   - get_info() → Dict


# ModelRegistry
# --------
# registry = ModelRegistry()
# 
# Methods:
#   - get_model_info(dataset, model_type, source) → Dict
#   - list_available_models() → List[str]
#   - list_datasets() → List[str]
#   - list_model_types() → List[str]
#   - list_sources() → List[str]
#   - filter_by_dataset(dataset) → Dict
#   - filter_by_model_type(model_type) → Dict
#   - to_dict() → Dict


# EXAMPLES
# ========

# Example 1: Load and Predict
# ---------
from src.models import ModelManager
import numpy as np

manager = ModelManager()
model = manager.load_model('wine_quality', 'mlp', 'coxam')
X = np.random.randn(5, 11)
predictions = manager.predict(X)
print(f"Predictions shape: {predictions.shape}")  # (5, 2)


# Example 2: Compare Models
# ---------
mlp_coxam = manager.load_model('wine_quality', 'mlp', 'coxam', auto_activate=False)
mlp_coax = manager.load_model('wine_quality', 'mlp', 'coax', auto_activate=False)

pred1 = mlp_coxam.predict(X)
pred2 = mlp_coax.predict(X)

print(f"CoXAM: {pred1}")
print(f"CoAX:  {pred2}")


# Example 3: Train New Model
# ---------
manager = ModelManager()
model = manager.create_model(
    dataset='my_data',
    model_type='mlp',
    input_dim=20,
    num_classes=3,
    hidden_dimension=64,
    dropout_rate=0.2
)

X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 3, 1000)
X_dev = np.random.randn(200, 20)
y_dev = np.random.randint(0, 3, 200)

manager.train(X_train, y_train, X_dev=X_dev, y_dev=y_dev, epochs=50)
manager.save_model('my_model.pth')


# Example 4: FastAPI Integration
# ---------
# from fastapi import FastAPI
# from src.models import ModelManager
# import numpy as np
#
# app = FastAPI()
# manager = ModelManager()
#
# @app.get("/models")
# def list_models():
#     return manager.list_available_pretrained()
#
# @app.post("/predict")
# def predict(dataset: str, model_type: str, data: list):
#     model = manager.load_model(dataset, model_type)
#     X = np.array(data).reshape(1, -1)
#     pred = manager.predict(X)
#     return {"predictions": pred.tolist()}


# Example 5: Batch Predictions
# ---------
X_batch = np.random.randn(1000, 11)
predictions = manager.predict(X_batch)  # (1000, 2)

predicted_classes = np.argmax(predictions, axis=1)
confidence = np.max(predictions, axis=1)

print(f"Classes shape: {predicted_classes.shape}")
print(f"Confidence shape: {confidence.shape}")


# AVAILABLE DATASETS
# ==================
# wine_quality, forest_cover, mushrooms, heart_disease,
# king_county_housing, prima_diabetes, breast_cancer,
# cardiotocography, adult, german_credit, loan_approval, mpg


# MODEL SOURCES
# =============
# 'coxam' - 11 datasets, both MLP & XGBoost
# 'coax'  - 11 datasets MLP (42 weight variations) + XGBoost


# API REFERENCE
# =============

# ModelManager.load_model(dataset, model_type, source='coxam', auto_activate=True)
#     Load pre-trained model.
#     Returns: UnifiedModel

# ModelManager.create_model(dataset, model_type, input_dim, num_classes, **kwargs)
#     Create new untrained model.
#     Returns: UnifiedModel

# ModelManager.predict(X, model=None)
#     Predict on X using active or specified model.
#     Returns: np.ndarray shape (n_samples, n_classes)

# ModelManager.train(X, y, model=None, X_dev=None, y_dev=None, **kwargs)
#     Train model. See individual model docs for kwargs.
#     Returns: Dict with training metadata

# ModelManager.evaluate(X, y, model=None)
#     Evaluate accuracy.
#     Returns: float [0, 1]

# ModelManager.save_model(weight_path, model=None)
#     Save model weights.
#     Returns: None

# ModelManager.get_active_model()
#     Get active model.
#     Returns: UnifiedModel or None

# ModelManager.set_active_model(model_key)
#     Set active model by key.
#     Returns: UnifiedModel

# ModelManager.list_loaded_models()
#     List cached models.
#     Returns: Dict[str, Dict]

# ModelManager.list_available_pretrained()
#     List all discoverable models from registry.
#     Returns: Dict


# HYPERPARAMETERS
# ===============

# MLP:
#   input_dim: int
#   num_classes: int
#   hidden_dimension: int (default: 50)
#   dropout_rate: float (default: 0)
#   device_id: int (default: -1, CPU)
#   epochs: int (for training, default: 300)
#   batch_size: int (for training, default: 1000)

# XGBoost:
#   learning_rate: float (default: 0.05)
#   num_boost_round: int (default: 50)


# COMMON ISSUES
# =============

# Q: Model not found error
# A: Check manager.list_available_pretrained()
#    Verify dataset name (case-sensitive)

# Q: Input dimension mismatch
# A: X.shape[1] must equal model's input_dim
#    Check dataset's feature count

# Q: Out of memory
# A: Reduce batch size, use smaller batches

# Q: Model behaves differently than training script
# A: Verify preprocessing matches:
#    - One-hot encoding consistent
#    - Feature scaling consistent
#    - Data types consistent (float32 for PyTorch)


# PERFORMANCE TIPS
# ================

# Load models on startup:
#   manager = ModelManager()
#   model = manager.load_model(...)  # ~1-2s first time, <100ms cached

# Reuse manager across requests:
#   Global app.manager = ModelManager()

# Use batch predictions:
#   predictions = manager.predict(X_batch)  # Better than loop

# Expected speeds:
#   MLP: 10-50ms per batch
#   XGBoost: 5-20ms per batch


# FILE STRUCTURE (REFACTORED)
# ============================

# Before (7 files):
#   - model_registry.py (115 LOC)
#   - model_interface.py (170 LOC)
#   - unified_model_loader.py (280 LOC)
#   - api_examples.py (250 LOC)
#   - MODELS_API_README.md (800 LOC)
#   - IMPLEMENTATION_SUMMARY.md (400 LOC)
#   - Total: Complex, scattered logic

# After (3 files + examples):
#   - models.py (450 LOC - all logic consolidated)
#   - registry.py (125 LOC - model discovery)
#   - examples.py (examples)
#   - README.md (this file)
#   - Total: Simple, organized, same functionality
#
# Status: ✓ Simplified without losing functionality


# INTEGRATION CHECKLIST
# =====================

# [ ] Import ModelManager: from src.models import ModelManager
# [ ] Initialize in startup: manager = ModelManager()
# [ ] Create API endpoint for /models (list_available_pretrained)
# [ ] Create endpoint for /predict (load_model + predict)
# [ ] Handle errors: ValueError if model not found
# [ ] Cache manager in app state
# [ ] Test with all model sources (coax, coxam)
# [ ] Benchmark prediction latency
# [ ] Document API responses


__doc__ = """XAIK Models Layer - Complete Reference"""
