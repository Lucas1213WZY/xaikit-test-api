"""
API Integration Examples
========================
Shows how to integrate the unified models layer with your API.
"""

# ============================================================================
# EXAMPLE 1: FastAPI Integration
# ============================================================================

"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from src.ai_models import ModelManager

app = FastAPI()
manager = ModelManager()

class PredictionRequest(BaseModel):
    dataset: str
    model_type: str = "mlp"
    data: list  # Array of features

class PredictionResponse(BaseModel):
    dataset: str
    model_type: str
    predictions: list
    confidence: float


@app.get("/models/available")
def list_models():
    \"\"\"Get all available pre-trained models.\"\"\"
    return manager.list_available_pretrained()


@app.post("/predict")
def predict(request: PredictionRequest):
    \"\"\"Make predictions using a model.\"\"\"
    try:
        # Load model if not already loaded
        model = manager.load_model(
            dataset=request.dataset,
            model_type=request.model_type
        )
        
        # Convert list to numpy array
        X = np.array([request.data])
        
        # Get predictions
        predictions = model.predict(X)[0]
        
        return PredictionResponse(
            dataset=request.dataset,
            model_type=request.model_type,
            predictions=predictions.tolist(),
            confidence=float(np.max(predictions))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch-predict")
def batch_predict(request: PredictionRequest):
    \"\"\"Make batch predictions.\"\"\"
    try:
        model = manager.load_model(
            dataset=request.dataset,
            model_type=request.model_type
        )
        
        # Convert to numpy, assuming it's 2D (multiple samples)
        X = np.array(request.data)
        predictions = model.predict(X)
        
        return {
            'dataset': request.dataset,
            'model_type': request.model_type,
            'batch_size': len(X),
            'predictions': predictions.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
def train(request: PredictionRequest):
    \"\"\"Train a new model.\"\"\"
    try:
        # Create new model (not pre-trained)
        model = manager.create_model(
            dataset=request.dataset,
            model_type=request.model_type,
            input_dim=len(request.data[0]),  # First sample determines input_dim
            num_classes=2,  # Adjust based on your use case
        )
        
        X = np.array(request.data)
        # Assuming labels are last column or passed separately
        y = np.zeros(len(X))  # Placeholder
        
        training_info = manager.train(X, y, epochs=100)
        
        return {
            'dataset': request.dataset,
            'model_type': request.model_type,
            'training_info': training_info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
"""


# ============================================================================
# EXAMPLE 2: Basic CLI Usage
# ============================================================================

def example_cli_usage():
    \"\"\"Show basic command-line usage patterns.\"\"\"
    
    from src.ai_models import ModelManager
    import numpy as np
    
    # Initialize manager
    manager = ModelManager()
    
    # List available models
    print("Available models:")
    available = manager.list_available_pretrained()
    print(available['available_models'])
    
    # Load a pre-trained model
    print("\nLoading wine_quality MLP model...")
    model = manager.load_model(
        dataset='wine_quality',
        model_type='mlp',
        source='coxam'
    )
    
    # Make predictions
    X_test = np.random.randn(5, 11)  # 5 samples, 11 features
    predictions = manager.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")
    
    # Get model info
    print(f"\nModel info: {model.get_info()}")
    
    # List all loaded models
    print(f"\nLoaded models: {manager.list_loaded_models()}")


# ============================================================================
# EXAMPLE 3: Training Workflow
# ============================================================================

def example_training_workflow():
    \"\"\"Show how to train new models.\"\"\"
    
    from src.ai_models import ModelManager
    import numpy as np
    
    manager = ModelManager()
    
    # Create new model
    print("Creating new MLP model...")
    model = manager.create_model(
        dataset='custom_data',
        model_type='mlp',
        input_dim=20,
        num_classes=3,
        hidden_dimension=64,
        dropout_rate=0.2,
    )
    
    # Generate dummy training data
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 3, 1000)
    X_dev = np.random.randn(200, 20)
    y_dev = np.random.randint(0, 3, 200)
    
    # Train model
    print("Training model...")
    training_info = manager.train(
        X_train, y_train,
        X_dev=X_dev, y_dev=y_dev,
        epochs=50,
        batch_size=32,
    )
    
    print(f"Training completed: {training_info}")
    
    # Evaluate
    X_test = np.random.randn(100, 20)
    y_test = np.random.randint(0, 3, 100)
    accuracy = manager.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save model
    manager.save_model('my_trained_model.pth')
    print("Model saved!")


# ============================================================================
# EXAMPLE 4: Multi-Model Management
# ============================================================================

def example_multi_model_management():
    \"\"\"Show how to manage multiple models simultaneously.\"\"\"
    
    from src.ai_models import ModelManager
    import numpy as np
    
    manager = ModelManager()
    
    # Load different models
    print("Loading multiple models...")
    mlp_model = manager.load_model('wine_quality', 'mlp', auto_activate=False)
    xgb_model = manager.load_model('wine_quality', 'xgboost', auto_activate=False)
    
    # Test data
    X_test = np.random.randn(3, 11)
    
    # Compare predictions
    print("\nComparing predictions:")
    mlp_pred = mlp_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
    print(f"MLP predictions:\n{mlp_pred}")
    print(f"XGBoost predictions:\n{xgb_pred}")
    
    # Switch between models
    manager.set_active_model('wine_quality_mlp_coxam')
    print(f"\nActive model: {manager.get_active_model().get_info()}")


# ============================================================================
# EXAMPLE 5: Integration with CoAX/CoXAM Training Pipeline
# ============================================================================

def example_training_pipeline_integration():
    \"\"\"Show how to use the models layer with your existing training scripts.\"\"\"
    
    from src.ai_models import ModelManager
    import numpy as np
    
    # Your existing data loading/preprocessing
    # (from main_new_v0.1.py or main_with_training_on_all_features.py)
    
    manager = ModelManager()
    
    # Create model for your dataset
    model = manager.create_model(
        dataset='wine_quality',
        model_type='mlp',
        input_dim=11,
        num_classes=2,
    )
    
    # Your data comes from the training pipeline
    X_train = np.random.randn(1000, 11)
    y_train = np.random.randint(0, 2, 1000)
    X_dev = np.random.randn(200, 11)
    y_dev = np.random.randint(0, 2, 200)
    X_test = np.random.randn(100, 11)
    y_test = np.random.randint(0, 2, 100)
    
    # Train using unified interface
    print("Training model via unified layer...")
    manager.train(X_train, y_train, X_dev=X_dev, y_dev=y_dev, epochs=100)
    
    # Evaluate
    test_acc = manager.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Make predictions for explanation generation
    predictions = manager.predict(X_test)
    
    # Now you can pass to your XAI explainers:
    # explainer = xai.get_explainer('shap', model.engine, train_data, ...)
    # explanations = explainer.explain(X_test, ...)
    
    print("Ready for XAI explanation generation!")


if __name__ == "__main__":
    print("=" * 70)
    print("XAIK Models Layer - API Integration Examples")
    print("=" * 70)
    
    print("\n[1] CLI Usage:")
    example_cli_usage()
    
    # Uncomment to run other examples:
    # print("\n[2] Training Workflow:")
    # example_training_workflow()
    
    # print("\n[3] Multi-Model Management:")
    # example_multi_model_management()
    
    # print("\n[4] Training Pipeline Integration:")
    # example_training_pipeline_integration()
