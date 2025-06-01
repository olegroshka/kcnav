# agent/complexity_models.py
import os
from typing import List, Dict, Any, Optional
import torch

from .complexity_model_interface import ComplexityModelInterface
from .complexity_model_transformer import ComplexityTransformer
from .complexity_model_diffusion import ComplexityDiffusionModel

class PlaceholderComplexityModel(ComplexityModelInterface):
    """A placeholder complexity model that does minimal work, used when no specific model is needed."""
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, **kwargs): # Added **kwargs
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initialized PlaceholderComplexityModel. Path: {model_path}, Device: {self.device}")

    def load_model(self, path: str, device: Optional[str] = None):
        print(f"PlaceholderComplexityModel: 'Loading' model from {path} onto {device or self.device}. No actual loading.")
        self.model_path = path
        if device:
            self.device = device

    def score_candidates(self, trajectory: Any, candidates_text: List[str]) -> List[float]:
        # print(f"PlaceholderComplexityModel: Scoring {len(candidates_text)} candidates. Returning uniform scores.")
        if not candidates_text:
            return []
        # For baseline (num_candidates_rerank=1), this won't be heavily used for selection.
        # If it were, providing uniform scores or a simple heuristic is fine.
        return [1.0 / len(candidates_text)] * len(candidates_text) if candidates_text else []


    def evaluate_trajectory_appropriateness(self, trajectory: Any) -> Dict[str, Any]:
        # print("PlaceholderComplexityModel: Evaluating trajectory. Returning dummy evaluation.")
        return {
            'overall_score': 0.5, # Dummy score, not critical for Phase 1 baseline
            'problematic_steps': [],
            'guidance': 'This is dummy guidance from PlaceholderComplexityModel.'
        }

def get_complexity_model(model_type: str, model_path: Optional[str] = None, device: Optional[str] = None, **kwargs) -> ComplexityModelInterface:
    """
    Factory function to get an instance of a complexity model.
    **kwargs can pass model-specific parameters from the config.
    """
    resolved_device = device if device else ('cuda' if torch.cuda.is_available() and kwargs.get('use_gpu', True) else 'cpu')
    print(f"Getting complexity model: type='{model_type}', path='{model_path}', device='{resolved_device}'")

    if model_type.lower() == "transformer":
        model = ComplexityTransformer(
            feature_dim=kwargs.get('transformer_feature_dim', 4),
            hidden_size=kwargs.get('transformer_hidden_size', 128),
            num_layers=kwargs.get('transformer_num_layers', 2),
            nhead=kwargs.get('transformer_nhead', 4)
        ).to(resolved_device) # Move to device after init
        if model_path and os.path.exists(model_path):
            print(f"Attempting to load Transformer model from {model_path}")
            model.load_model(model_path, device=resolved_device) # load_model is a placeholder itself
        else:
            print(f"Transformer model path '{model_path}' not found or not provided. Using fresh/untrained Transformer skeleton.")
        return model

#    diffusion_seq_len: 60  # Max sequence length for VAE input, should match TrajectoryFeatureDataset max_seq_len
                            # and transformer_max_seq_len if features are shared.
#     diffusion_feature_dim: 6  # Input feature dim for VAE encoder (STEP_TYPE_DIM + 3)
#     diffusion_hidden_size: 128  # GRU hidden size in VAE encoder
#     diffusion_latent_dim: 32  # Dimensionality of VAE latent space z0 (smaller than Transformer hidden)
#     diffusion_denoiser_hidden_size: 256  # Width of MLP layers in DDPM denoiser
#     diffusion_timesteps: 1000  # T in DDPM (number of noising steps)

    elif model_type.lower() == "diffusion":
        model = ComplexityDiffusionModel(
            seq_len=kwargs.get('diffusion_seq_len', 50),
            feature_dim=kwargs.get('diffusion_feature_dim', 2),
            denoiser_hidden_size=kwargs.get('diffusion_denoiser_hidden_size', 128)
        ).to(resolved_device)
        if model_path and os.path.exists(model_path):
            model.load_model(model_path, device=resolved_device)
        else:
             print(f"Diffusion model path '{model_path}' not found or not provided. Using fresh/untrained Diffusion skeleton.")
        return model

    elif model_type.lower() == "placeholder":
        # Pass kwargs to Placeholder in case it uses any, though current version doesn't
        model = PlaceholderComplexityModel(model_path=model_path, device=resolved_device, **kwargs)
        if model_path: # Call load_model for consistency
            model.load_model(model_path, device=resolved_device)
        return model
    else:
        raise ValueError(f"Unknown complexity model type: {model_type}")
