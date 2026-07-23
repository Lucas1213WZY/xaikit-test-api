"""
Model Registry
==============
Central registry of available datasets, models, and their locations.
"""

from typing import Dict, List, Optional
import os

from .paths import MODEL_WEIGHTS_ROOT


class ModelRegistry:
    """
    Registry for tracking available models across different sources.
    Supports multiple model sources: CoXAM, CoAX, and custom paths.
    """

    # Pre-trained weights live under
    #   <repo>/assets/model_weights/<model_type>/<cognitive_agent>/
    MODEL_AGENTS = ['coxam', 'coax']
    
    # Supported datasets
    DATASETS = [
        'wine_quality',
        'forest_cover',
        'mushrooms',
        'heart_disease',
        'king_county_housing',
        'prima_diabetes',
        'breast_cancer',
        'cardiotocography',
        'adult',
        'german_credit',
        'loan_approval',
        'mpg',
    ]
    
    # Supported model types
    MODEL_TYPES = ['mlp', 'xgboost']
    
    def __init__(self):
        """Initialize registry with available models."""
        self._discover_models()
    
    def _discover_models(self):
        """Discover available pre-trained models in the model sources."""
        self.available_models = {}
        
        for model_type in self.MODEL_TYPES:
            for source_name in self.MODEL_AGENTS:
                model_dir = os.path.join(str(MODEL_WEIGHTS_ROOT), model_type, source_name)

                if not os.path.exists(model_dir):
                    continue
                
                # List available model weights
                for dataset in self.DATASETS:
                    weight_patterns = [
                        f'{dataset}_model_weights.pth',
                        f'{dataset}_model_weights.json',
                    ]
                    
                    for pattern in weight_patterns:
                        weight_path = os.path.join(model_dir, pattern)
                        if os.path.exists(weight_path):
                            key = f'{dataset}_{model_type}_{source_name}'
                            self.available_models[key] = {
                                'dataset': dataset,
                                'model_type': model_type,
                                'source': source_name,
                                'weight_path': weight_path,
                                'model_dir': model_dir,
                            }
    
    def get_model_info(self, dataset: str, model_type: str, source: str = 'coxam') -> Optional[Dict]:
        """
        Get information about a specific model.
        
        Args:
            dataset: Dataset name (e.g., 'wine_quality')
            model_type: Model type ('mlp' or 'xgboost')
            source: Model source ('coxam' or 'coax'), defaults to 'coxam'
        
        Returns:
            Dict with model info or None if not found
        """
        key = f'{dataset}_{model_type}_{source}'
        return self.available_models.get(key)
    
    def list_available_models(self) -> List[str]:
        """List all available model identifiers."""
        return sorted(list(self.available_models.keys()))
    
    def list_datasets(self) -> List[str]:
        """List all supported datasets."""
        return sorted(self.DATASETS)
    
    def list_model_types(self) -> List[str]:
        """List all supported model types."""
        return self.MODEL_TYPES
    
    def list_sources(self) -> List[str]:
        """List all model sources."""
        return list(self.MODEL_AGENTS)
    
    def filter_by_dataset(self, dataset: str) -> Dict:
        """Get all model variants for a dataset."""
        return {k: v for k, v in self.available_models.items() if v['dataset'] == dataset}
    
    def filter_by_model_type(self, model_type: str) -> Dict:
        """Get all models of a specific type."""
        return {k: v for k, v in self.available_models.items() if v['model_type'] == model_type}
    
    def to_dict(self) -> Dict:
        """Return registry as dictionary for API responses."""
        return {
            'available_models': self.list_available_models(),
            'datasets': self.list_datasets(),
            'model_types': self.list_model_types(),
            'sources': self.list_sources(),
            'model_details': self.available_models,
        }
