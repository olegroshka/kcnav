# agent/complexity_model_transformer.py
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import math  # For log1p

# Assuming these are correctly imported via robust __init__.py or sys.path in consuming scripts
# These imports are primarily for _create_feature_vector_for_step and type hints.
# If this file is part of the 'agent' package, and 'data' is a sibling,
# 'from data.trajectory_processor import ...' should work if project root is in PYTHONPATH.
try:
    from data.trajectory_processor import encode_step_type, STEP_TYPE_DIM
except ImportError:
    # Fallback for contexts where direct import might fail (e.g. isolated testing of this file)
    # This is less ideal than ensuring PYTHONPATH is set correctly by the calling script (e.g., main.py)
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))  # agent/
    project_root = os.path.dirname(current_dir)  # project_root/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from data.trajectory_processor import encode_step_type, STEP_TYPE_DIM

        print("Loaded trajectory_processor components in ComplexityTransformer via sys.path modification.")
    except ImportError as e_inner:
        print(f"FATAL: Could not import from data.trajectory_processor in ComplexityTransformer: {e_inner}")
        raise

from .complexity_model_interface import ComplexityModelInterface
# It's good practice to import specific classes for type hinting if they are well-defined.
from agent.trajectory import Trajectory
from agent.complexity_calculator import NcdCalculator


class ComplexityTransformer(nn.Module, ComplexityModelInterface):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 nhead: int = 4,
                 dropout_rate: float = 0.1,
                 max_seq_len_for_pos_encoding: int = 50):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.model_path = None
        self.max_seq_len = max_seq_len_for_pos_encoding

        self.input_embedding = nn.Linear(feature_dim, hidden_size)
        self.positional_encoding = nn.Embedding(max_seq_len_for_pos_encoding, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
            dropout=dropout_rate, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.complexity_regressor = nn.Linear(hidden_size, 2)
        self.success_classifier = nn.Linear(hidden_size, 1)

        print(
            f"Initialized ComplexityTransformer: features_in={feature_dim}, hidden={hidden_size}, layers={num_layers}, heads={nhead}")

    def _create_feature_vector_for_step(self,
                                        step_content_text: str,
                                        step_type: str,
                                        step_index: int,
                                        total_steps_in_hypothetical_traj: int,
                                        previous_step_content_for_ncd: Optional[str],
                                        full_hypothetical_history_text_for_ncd: str,
                                        ncd_calculator: NcdCalculator
                                        ) -> List[float]:
        step_features = []
        step_features.extend(encode_step_type(step_type))

        local_ncd = ncd_calculator.calculate_local_ncd(previous_step_content_for_ncd, step_content_text)
        step_features.append(float(local_ncd))

        # Calculate raw global complexity for this hypothetical step
        raw_global_complexity_for_this_step = ncd_calculator.calculate_global_complexity(
            full_hypothetical_history_text_for_ncd)
        # Apply the same scaling as used during training (log1p)
        scaled_global_complexity = math.log1p(float(raw_global_complexity_for_this_step))
        step_features.append(scaled_global_complexity)

        normalized_time_step = (step_index - 1) / max(1,
                                                      total_steps_in_hypothetical_traj - 1) if total_steps_in_hypothetical_traj > 1 else 0.0
        step_features.append(normalized_time_step)

        if len(step_features) != self.feature_dim:
            print(
                f"Warning (_create_feature_vector_for_step): Feature vector len {len(step_features)} != model_feature_dim {self.feature_dim}.")
            if len(step_features) < self.feature_dim:
                step_features.extend([0.0] * (self.feature_dim - len(step_features)))
            else:
                step_features = step_features[:self.feature_dim]
        return step_features

    def _prepare_sequence_for_scoring(self,
                                      current_trajectory: Trajectory,
                                      candidate_step_text: str,
                                      ncd_calculator: NcdCalculator
                                      ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        existing_step_features = []
        last_content_for_ncd = current_trajectory.problem_statement
        current_history_text_for_ncd_so_far = f"Problem: {current_trajectory.problem_statement}"

        for step_data in current_trajectory.steps:
            step_idx = step_data['step_index']
            # Total steps if candidate is added: len(current_trajectory.steps) + 1
            total_steps_if_candidate_added = len(current_trajectory.steps) + 1

            # We need raw ncd_global from metadata to scale it, or re-calculate it
            # Assuming metadata['ncd_global'] stores raw compressed length as per react_agent logging
            raw_global_ncd_at_step = step_data['metadata'].get('ncd_global', 0.0)
            scaled_global_complexity_at_step = math.log1p(float(raw_global_ncd_at_step))

            single_step_feature_vec = [
                *encode_step_type(step_data['step_type']),
                float(step_data['metadata'].get('ncd_local', 0.0)),
                scaled_global_complexity_at_step,  # Use scaled global NCD from past steps
                (step_idx - 1) / max(1,
                                     total_steps_if_candidate_added - 1) if total_steps_if_candidate_added > 1 else 0.0
            ]
            if len(single_step_feature_vec) < self.feature_dim:
                single_step_feature_vec.extend([0.0] * (self.feature_dim - len(single_step_feature_vec)))
            existing_step_features.append(single_step_feature_vec[:self.feature_dim])

            last_content_for_ncd = step_data['content']
            current_history_text_for_ncd_so_far += f"\n\nStep {step_idx} ({step_data['step_type']}):\n{step_data['content']}"

        candidate_step_type = "ThoughtAction"
        hypothetical_step_index = len(current_trajectory.steps) + 1
        total_steps_in_hyp_traj = hypothetical_step_index

        full_hypothetical_history_text = current_history_text_for_ncd_so_far + \
                                         f"\n\nStep {hypothetical_step_index} ({candidate_step_type}):\n{candidate_step_text}"

        candidate_feature_vector = self._create_feature_vector_for_step(
            step_content_text=candidate_step_text,
            step_type=candidate_step_type,
            step_index=hypothetical_step_index,
            total_steps_in_hypothetical_traj=total_steps_in_hyp_traj,
            previous_step_content_for_ncd=last_content_for_ncd,
            full_hypothetical_history_text_for_ncd=full_hypothetical_history_text,
            ncd_calculator=ncd_calculator
        )

        full_sequence_features = existing_step_features + [candidate_feature_vector]

        feature_tensor_unpadded = torch.tensor(full_sequence_features, dtype=torch.float32)
        actual_len = feature_tensor_unpadded.shape[0]
        model_device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

        padded_feature_sequence = torch.zeros((self.max_seq_len, self.feature_dim), dtype=torch.float32,
                                              device=model_device)
        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool, device=model_device)
        len_to_copy = min(actual_len, self.max_seq_len)

        if actual_len == 0:  # Handle empty sequence case
            return None

        start_idx = max(0, actual_len - self.max_seq_len)  # Truncate from beginning if too long
        end_idx_copy = min(actual_len, self.max_seq_len)  # How many elements to effectively copy

        padded_feature_sequence[:end_idx_copy, :] = feature_tensor_unpadded[start_idx: start_idx + end_idx_copy, :]
        padding_mask[:end_idx_copy] = False

        return padded_feature_sequence.unsqueeze(0), padding_mask.unsqueeze(0)

    def forward(self, sequences: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        batch_size, seq_len, current_input_feat_dim = sequences.shape
        if self.input_embedding.in_features != current_input_feat_dim:
            raise ValueError(
                f"Model input_embedding expects {self.input_embedding.in_features} features, got {current_input_feat_dim}.")

        x_emb = self.input_embedding(sequences)
        position_ids = torch.arange(0, seq_len, device=sequences.device).unsqueeze(0).repeat(batch_size, 1)
        pos_enc = self.positional_encoding(position_ids)
        x_emb = x_emb + pos_enc

        transformer_output = self.transformer_encoder(x_emb, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            actual_lengths = (~padding_mask).sum(dim=1)
            last_token_indices = actual_lengths - 1
            last_token_indices = torch.clamp(last_token_indices, min=0)
            batch_indices = torch.arange(batch_size, device=sequences.device)
            try:
                last_hidden_state = transformer_output[batch_indices, last_token_indices]
            except IndexError as e:
                print(
                    f"IndexError in forward: B={batch_size}, L={seq_len}, actual_lengths={actual_lengths}, indices={last_token_indices}")
                raise e  # Re-raise after printing context
        else:
            last_hidden_state = transformer_output[:, -1, :]

        predicted_next_complexities = self.complexity_regressor(last_hidden_state)
        success_logit = self.success_classifier(last_hidden_state)
        return predicted_next_complexities, success_logit

    def load_model(self, path: str, device: Optional[str] = None):
        print(f"ComplexityTransformer: Loading model state_dict from {path}")
        self.model_path = path
        try:
            map_location = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
            self.load_state_dict(torch.load(path, map_location=map_location))
            self.to(map_location)
            self.eval()
            print(f"  ComplexityTransformer model loaded successfully onto {map_location}.")
        except FileNotFoundError:
            print(f"  ERROR: Model file not found at {path}.")
        except Exception as e:
            print(f"  ERROR: Failed to load model from {path}: {e}.")

    def score_candidates(self, current_trajectory: Trajectory, candidates_text: List[str],
                         ncd_calculator: NcdCalculator) -> List[float]:
        if not hasattr(self, 'input_embedding'):
            print("Warning: ComplexityTransformer seems uninitialized. Returning dummy scores.")
            return [0.5] * len(candidates_text) if candidates_text else []
        if not candidates_text: return []

        self.eval()
        scores = []
        with torch.no_grad():
            for candidate_text in candidates_text:
                prepared_input_tuple = self._prepare_sequence_for_scoring(current_trajectory, candidate_text,
                                                                          ncd_calculator)
                if prepared_input_tuple is None:
                    scores.append(0.0);
                    continue
                sequence_tensor, padding_mask_tensor = prepared_input_tuple
                _, success_logit = self.forward(sequence_tensor, padding_mask=padding_mask_tensor)
                success_prob = torch.sigmoid(success_logit).item()
                scores.append(success_prob)
        return scores

    def evaluate_trajectory_appropriateness(self, trajectory_data_dict: Dict[str, Any],
                                            ncd_calculator: NcdCalculator) -> Dict[str, Any]:
        self.eval()
        guidance_text = "Evaluation of full trajectory appropriateness:"

        all_step_features = []
        last_content_for_ncd_eval = trajectory_data_dict.get("problem_statement", "")
        current_history_text_for_ncd_eval = f"Problem: {last_content_for_ncd_eval}"

        for i, step_data in enumerate(trajectory_data_dict.get("steps", [])):
            step_idx = step_data.get('step_index', i + 1)
            total_steps = len(trajectory_data_dict.get("steps", []))

            raw_global_ncd_at_step = step_data['metadata'].get('ncd_global', 0.0)
            scaled_global_complexity = math.log1p(float(raw_global_ncd_at_step))

            single_step_feature_vec = [
                *encode_step_type(step_data['step_type']),
                float(step_data['metadata'].get('ncd_local', 0.0)),
                scaled_global_complexity,
                (step_idx - 1) / max(1, total_steps - 1) if total_steps > 1 else 0.0
            ]
            if len(single_step_feature_vec) < self.feature_dim:
                single_step_feature_vec.extend([0.0] * (self.feature_dim - len(single_step_feature_vec)))
            all_step_features.append(single_step_feature_vec[:self.feature_dim])

        if not all_step_features:
            return {'overall_score': 0.0, 'guidance': 'Could not extract features.'}

        feature_tensor_unpadded = torch.tensor(all_step_features, dtype=torch.float32)
        actual_len = feature_tensor_unpadded.shape[0]
        model_device = next(self.parameters()).device

        padded_feature_sequence = torch.zeros((self.max_seq_len, self.feature_dim), dtype=torch.float32,
                                              device=model_device)
        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool, device=model_device)
        len_to_copy = min(actual_len, self.max_seq_len)

        if actual_len == 0: return {'overall_score': 0.0, 'guidance': 'Empty feature sequence.'}

        start_idx = max(0, actual_len - self.max_seq_len)
        end_idx_copy = min(actual_len, self.max_seq_len)
        padded_feature_sequence[:end_idx_copy, :] = feature_tensor_unpadded[start_idx: start_idx + end_idx_copy, :]
        padding_mask[:end_idx_copy] = False

        sequence_tensor = padded_feature_sequence.unsqueeze(0)
        padding_mask_tensor = padding_mask.unsqueeze(0)

        with torch.no_grad():
            _, success_logit = self.forward(sequence_tensor, padding_mask=padding_mask_tensor)

        success_prob = torch.sigmoid(success_logit).item()
        guidance_text += f"\n  Predicted P(success): {success_prob:.3f}."

        return {'overall_score': success_prob, 'problematic_steps': [], 'guidance': guidance_text}