# agent/complexity_model_transformer.py
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

from .complexity_model_interface import ComplexityModelInterface


class ComplexityTransformer(nn.Module, ComplexityModelInterface):
    def __init__(self,
                 feature_dim: int,  # This MUST match the features from TrajectoryFeatureDataset
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 nhead: int = 4,
                 dropout_rate: float = 0.1,
                 max_seq_len_for_pos_encoding: int = 50):  # Max steps in a trajectory for pos encoding
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.model_path = None  # To store path if load_model is called

        self.input_embedding = nn.Linear(feature_dim, hidden_size)

        # Learned positional encoding
        self.positional_encoding = nn.Embedding(max_seq_len_for_pos_encoding, hidden_size)
        # Or fixed sinusoidal:
        # self.positional_encoding = PositionalEncoding(hidden_size, dropout_rate, max_len=max_seq_len_for_pos_encoding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,  # Common practice
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu'  # GELU is common in transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head for predicting next-step complexities (local NCD, global complexity)
        self.complexity_regressor = nn.Linear(hidden_size, 2)

        # Output head for predicting overall trajectory success probability
        self.success_classifier = nn.Linear(hidden_size, 1)  # Outputs a logit

        print(
            f"Initialized ComplexityTransformer: features_in={feature_dim}, hidden={hidden_size}, layers={num_layers}, heads={nhead}")

    def forward(self,
                sequences: torch.Tensor,  # Shape: [batch_size, seq_len, feature_dim]
                padding_mask: Optional[torch.Tensor] = None  # Shape: [batch_size, seq_len], True for padded
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, _ = sequences.shape

        if self.input_embedding.in_features != self.feature_dim:
            raise ValueError(f"Model feature_dim ({self.input_embedding.in_features}) "
                             f"does not match input data feature_dim ({self.feature_dim}). "
                             "Ensure config 'transformer_feature_dim' is correct.")

        # 1. Embed input features
        x_emb = self.input_embedding(sequences)  # [batch_size, seq_len, hidden_size]

        # 2. Add positional encoding
        # Create position IDs (0, 1, 2, ..., seq_len-1) for each sequence in the batch
        position_ids = torch.arange(0, seq_len, device=sequences.device).unsqueeze(0).repeat(batch_size, 1)
        pos_enc = self.positional_encoding(position_ids)  # [batch_size, seq_len, hidden_size]
        x_emb = x_emb + pos_enc

        # 3. Pass through Transformer Encoder
        # src_key_padding_mask: if an element is True, it is masked.
        transformer_output = self.transformer_encoder(x_emb, src_key_padding_mask=padding_mask)
        # Output shape: [batch_size, seq_len, hidden_size]

        # 4. Get representation for prediction
        # We use the hidden state of the *last actual token* in each sequence.
        # If using padding_mask, we need to find the actual length of each sequence.
        if padding_mask is not None:
            # `~padding_mask` gives True for actual tokens. Sum along seq_len dim.
            actual_lengths = (~padding_mask).sum(dim=1)
            # Indices of the last actual token (0-indexed)
            last_token_indices = actual_lengths - 1
            # Handle cases where a sequence might be entirely padding (actual_lengths could be 0)
            last_token_indices = torch.clamp(last_token_indices, min=0)

            # Gather the hidden states for these last tokens
            # Need to use advanced indexing: output[batch_indices, last_token_indices_for_each_batch]
            batch_indices = torch.arange(batch_size, device=sequences.device)
            last_hidden_state = transformer_output[batch_indices, last_token_indices]
            # Shape: [batch_size, hidden_size]
        else:
            # If no padding mask, assume all sequences have length seq_len
            last_hidden_state = transformer_output[:, -1, :]  # Take the last hidden state
            # Shape: [batch_size, hidden_size]

        # 5. Output heads
        # Predict next step's complexity based on the representation of the current sequence
        predicted_next_complexities = self.complexity_regressor(last_hidden_state)  # [batch_size, 2]

        # Predict overall success probability of the trajectory ending successfully
        success_logit = self.success_classifier(last_hidden_state)  # [batch_size, 1]

        return predicted_next_complexities, success_logit

    def load_model(self, path: str, device: Optional[str] = None):
        print(f"ComplexityTransformer: Loading model state_dict from {path}")
        self.model_path = path
        try:
            map_location = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
            self.load_state_dict(torch.load(path, map_location=map_location))
            self.eval()  # Set to evaluation mode after loading
            print(f"  ComplexityTransformer model loaded successfully onto {map_location}.")
        except FileNotFoundError:
            print(f"  ERROR: Model file not found at {path}. Transformer remains untrained or uses initial weights.")
        except Exception as e:
            print(f"  ERROR: Failed to load model from {path}: {e}. Check model compatibility.")

    def score_candidates(self, trajectory: Any, candidates_text: List[str]) -> List[float]:
        # This method will be fully implemented in Phase 4 (Agent Integration)
        # It requires creating hypothetical future states and running them through the trained model.
        print(
            f"ComplexityTransformer.score_candidates (Not fully implemented for Phase 3): Scoring {len(candidates_text)}.")
        if not candidates_text: return []
        # For now, return dummy scores or scores based on a simple heuristic.
        # A real implementation would predict success_prob for each candidate.
        return [1.0 / (i + 1) for i in range(len(candidates_text))]

    def evaluate_trajectory_appropriateness(self, trajectory: Any) -> Dict[str, Any]:
        # This method will be fully implemented in Phase 4/5 (Agent Integration/Reflection)
        print("ComplexityTransformer.evaluate_trajectory_appropriateness (Not fully implemented for Phase 3).")
        # A real implementation would process the trajectory, get predictions, and format guidance.
        return {
            'overall_score': 0.5,  # Dummy score
            'problematic_steps': [],
            'guidance': 'Dummy guidance from ComplexityTransformer (not fully implemented for evaluation yet).'
        }

# Example of a fixed PositionalEncoding module if not using nn.Embedding
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0) # .transpose(0, 1) if not batch_first
#         self.register_buffer('pe', pe)
#     def forward(self, x): # x shape: [batch_size, seq_len, embedding_dim]
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
