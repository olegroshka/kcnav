# agent/complexity_model_diffusion.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from .complexity_model_interface import ComplexityModelInterface
# These are needed for type hinting and for the helper methods
from agent.trajectory import Trajectory
from agent.complexity_calculator import NcdCalculator

# Robust import for trajectory_processor components
try:
    # This assumes 'data' is a sibling package to 'agent' under the project root
    # and the project root is in PYTHONPATH (usually handled by main.py).
    from data.trajectory_processor import process_single_trajectory_for_training, STEP_TYPE_DIM, encode_step_type
except ImportError:
    print(
        "Initial ImportError for data.trajectory_processor in complexity_model_diffusion.py. Attempting sys.path modification.")
    current_dir = os.path.dirname(os.path.abspath(__file__))  # agent/
    project_root = os.path.dirname(current_dir)  # project_root/

    # Add project root to sys.path if not already present
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Now try importing assuming 'data' is a package under project_root
    try:
        from data.trajectory_processor import process_single_trajectory_for_training, STEP_TYPE_DIM, encode_step_type

        print("Successfully imported trajectory_processor components after sys.path modification.")
    except ImportError as e_inner:
        print(f"FATAL: Failed to import from data.trajectory_processor: {e_inner}")
        print(
            "Ensure 'data/trajectory_processor.py' exists and your execution environment (e.g., main.py) sets up PYTHONPATH correctly.")
        raise  # Re-raise if still not found, as it's critical


def timestep_embedding(t: torch.Tensor,
                       dim: int,
                       max_period: float = 10_000.0) -> torch.Tensor:
    assert len(t.shape) == 1, "Timestep tensor t must be 1D"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ComplexityDiffusionModel(nn.Module, ComplexityModelInterface):
    def __init__(self,
                 seq_len: int = 50,
                 feature_dim: int = 6,  # VAE Encoder input feature dim (e.g., STEP_TYPE_DIM + 3)
                 gru_hidden_size: int = 128,
                 latent_dim: int = 64,
                 denoiser_hidden_size: int = 256,
                 diffusion_timesteps: int = 1000):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim_in = feature_dim
        self.gru_hidden_size = gru_hidden_size
        self.latent_dim = latent_dim
        self.T = diffusion_timesteps

        self.encoder_rnn = nn.GRU(
            input_size=self.feature_dim_in,
            hidden_size=self.gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_mu = nn.Linear(self.gru_hidden_size * 2, self.latent_dim)
        self.fc_logvar = nn.Linear(self.gru_hidden_size * 2, self.latent_dim)

        denoiser_input_dim = self.latent_dim * 2
        self.time_projection = nn.Linear(self.latent_dim, self.latent_dim)

        self.denoise_mlp = nn.Sequential(
            nn.Linear(denoiser_input_dim, denoiser_hidden_size), nn.SiLU(),
            nn.Linear(denoiser_hidden_size, denoiser_hidden_size), nn.SiLU(),
            nn.Linear(denoiser_hidden_size, self.latent_dim),
        )
        self.model_path: Optional[str] = None
        self.beta_start_cfg: Optional[float] = None
        self.beta_end_cfg: Optional[float] = None

        print(f"Initialized ComplexityDiffusionModel: seq_len={seq_len}, feat_in={self.feature_dim_in}, "
              f"GRU_hidden={gru_hidden_size}, latent_dim={latent_dim}, denoiser_hidden={denoiser_hidden_size}, T={self.T}")

    def set_diffusion_schedule(self, beta_start: float, beta_end: float, device: Optional[torch.device] = None):
        self.beta_start_cfg = beta_start
        self.beta_end_cfg = beta_end
        target_device = device if device is not None else (
            next(self.parameters()).device if list(self.parameters()) else torch.device("cpu"))

        betas = torch.linspace(beta_start, beta_end, self.T, dtype=torch.float32, device=target_device)
        # Using persistent=False for buffers that are derived and recomputed in set_diffusion_schedule
        self.register_buffer('betas', betas, persistent=False)
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas, persistent=False)
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod, persistent=False)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod), persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod), persistent=False)
        print(f"Diffusion schedule (T={self.T}) set on device: {target_device}")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_n = self.encoder_rnn(x)
        h_concat = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        mu = self.fc_mu(h_concat)
        logvar = self.fc_logvar(h_concat)
        return mu, logvar

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def predict_eps_from_zt_and_t(self, z_t: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t_int, self.latent_dim)
        projected_t_emb = self.time_projection(t_emb)
        denoiser_input = torch.cat([z_t, projected_t_emb], dim=-1)
        return self.denoise_mlp(denoiser_input)

    def forward(self, sequences: torch.Tensor, diffusion_t_int: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:
        if not hasattr(self, 'sqrt_alphas_cumprod'):
            raise RuntimeError(
                "Diffusion schedule not set. Call set_diffusion_schedule() after model init and before forward pass.")

        mu, logvar = self.encode(sequences)
        z0_sample = self.reparameterise(mu, logvar)

        if diffusion_t_int is None:
            diffusion_t_int = torch.randint(0, self.T, (z0_sample.shape[0],), device=z0_sample.device)

        sqrt_alpha_hat_t = self.sqrt_alphas_cumprod.gather(0, diffusion_t_int).unsqueeze(-1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alphas_cumprod.gather(0, diffusion_t_int).unsqueeze(-1)

        noise_true = torch.randn_like(z0_sample)
        z_t = sqrt_alpha_hat_t * z0_sample + sqrt_one_minus_alpha_hat_t * noise_true

        eps_pred = self.predict_eps_from_zt_and_t(z_t, diffusion_t_int)

        return {"mu": mu, "logvar": logvar, "eps_pred": eps_pred, "eps_true": noise_true,
                "z0_sample": z0_sample, "z_t": z_t, "t_int": diffusion_t_int}

    def load_model(self, path: str, device: Optional[str] = None):
        self.model_path = path
        map_loc = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.load_state_dict(torch.load(path, map_location=map_loc))
            self.to(map_loc)  # Ensure model is on the correct device
            self.eval()
            print(f"✓ ComplexityDiffusionModel loaded from {path} onto {map_loc}")
            if not hasattr(self, 'betas') and self.beta_start_cfg is not None and self.beta_end_cfg is not None:
                print("Re-initializing diffusion schedule after loading model.")
                self.set_diffusion_schedule(self.beta_start_cfg, self.beta_end_cfg, device=torch.device(map_loc))
            elif not hasattr(self, 'betas'):  # Check if betas specifically is missing
                print(
                    "Warning: Diffusion schedule buffers (e.g., 'betas') not found in loaded model state_dict or model instance, and beta_start/end configs are missing. Schedule may be incorrect if not re-set via set_diffusion_schedule().")
        except Exception as e:
            print(f"Could not load ComplexityDiffusionModel from {path}: {e}")

    def _create_feature_vector_for_one_step(self, content: str, type_str: str, idx: int, total_hyp_steps: int,
                                            prev_content: Optional[str], full_hyp_hist_text: str,
                                            ncd_calc: NcdCalculator) -> List[float]:
        # Uses encode_step_type and STEP_TYPE_DIM imported at module level
        feats = encode_step_type(type_str)
        feats.append(float(ncd_calc.calculate_local_ncd(prev_content, content)))
        raw_global = ncd_calc.calculate_global_complexity(full_hyp_hist_text)
        feats.append(math.log1p(float(raw_global)))  # Scaled global
        feats.append((idx - 1) / max(1, total_hyp_steps - 1) if total_hyp_steps > 1 else 0.0)

        # Ensure feature vector matches model's expected input dimension
        if len(feats) != self.feature_dim_in:
            # print(f"WARN (_create_feature_vector_for_one_step): Feat vec len {len(feats)} != model_feat_in {self.feature_dim_in}")
            if len(feats) < self.feature_dim_in:
                feats.extend([0.0] * (self.feature_dim_in - len(feats)))
            else:
                feats = feats[:self.feature_dim_in]
        return feats

    def _prepare_hypothetical_sequence_tensor(self, current_traj: Trajectory, candidate_text: Optional[str],
                                              ncd_calc: NcdCalculator, device: torch.device) -> Optional[torch.Tensor]:
        all_feats = []
        last_content_for_ncd_construction = current_traj.problem_statement
        current_hist_text_for_ncd_construction = f"Problem: {last_content_for_ncd_construction}"

        # Determine total number of steps in the hypothetical trajectory for time normalization
        num_existing_steps = len(current_traj.steps)
        total_steps_in_final_sequence = num_existing_steps + (1 if candidate_text else 0)

        for s_data in current_traj.steps:
            raw_global_at_s = s_data['metadata'].get('ncd_global', 0.0)  # This is raw compressed length
            scaled_global_at_s = math.log1p(float(raw_global_at_s))

            step_feats_list = [
                *encode_step_type(s_data['step_type']),
                float(s_data['metadata'].get('ncd_local', 0.0)),
                scaled_global_at_s,
                (s_data['step_index'] - 1) / max(1,
                                                 total_steps_in_final_sequence - 1) if total_steps_in_final_sequence > 1 else 0.0
            ]
            if len(step_feats_list) < self.feature_dim_in: step_feats_list.extend(
                [0.0] * (self.feature_dim_in - len(step_feats_list)))
            all_feats.append(step_feats_list[:self.feature_dim_in])
            # For constructing history text for *next* step's NCDs
            current_hist_text_for_ncd_construction += f"\n\nStep {s_data['step_index']} ({s_data['step_type']}):\n{s_data['content']}"
            last_content_for_ncd_construction = s_data['content']

        if candidate_text:
            cand_idx_in_full_sequence = num_existing_steps + 1
            # History text for candidate's global NCD includes the candidate itself
            full_hyp_hist_for_candidate_ncd = current_hist_text_for_ncd_construction + \
                                              f"\n\nStep {cand_idx_in_full_sequence} (ThoughtAction):\n{candidate_text}"

            cand_feats = self._create_feature_vector_for_one_step(
                candidate_text, "ThoughtAction",
                cand_idx_in_full_sequence,  # Its index in the combined sequence
                total_steps_in_final_sequence,  # Total steps for time normalization
                last_content_for_ncd_construction,  # Content of the actual last step for local NCD
                full_hyp_hist_for_candidate_ncd,  # History including candidate for its global NCD
                ncd_calc
            )
            all_feats.append(cand_feats)

        if not all_feats: return None  # Should not happen if current_traj has steps or candidate_text is provided

        feat_tensor_unpadded = torch.tensor(all_feats, dtype=torch.float32, device=device)
        actual_len = feat_tensor_unpadded.shape[0]

        # Pad/truncate to self.seq_len (max sequence length for VAE encoder)
        padded_feat_seq = torch.zeros((self.seq_len, self.feature_dim_in), dtype=torch.float32, device=device)
        len_to_copy = min(actual_len, self.seq_len)

        if actual_len == 0: return None  # Should be caught by `if not all_feats`

        # Truncate from the beginning if longer than seq_len, keeping the most recent steps
        start_idx_unpadded = max(0, actual_len - self.seq_len)
        # How many elements from unpadded tensor to copy
        num_elements_to_copy_from_unpadded = actual_len - start_idx_unpadded

        # Where to start copying in the padded_feature_sequence (usually 0 if padding, or if truncating to fill)
        # If actual_len > self.seq_len, we copy self.seq_len elements.
        # If actual_len < self.seq_len, we copy actual_len elements.
        # The effective length in padded_feat_seq will be len_to_copy.

        padded_feat_seq[:num_elements_to_copy_from_unpadded, :] = feat_tensor_unpadded[
                                                                  start_idx_unpadded: start_idx_unpadded + num_elements_to_copy_from_unpadded,
                                                                  :]

        return padded_feat_seq.unsqueeze(0)  # Add batch dim: [1, L, F_in]

    @torch.no_grad()
    def score_candidates_prev(self, current_trajectory: Trajectory, candidates_text: List[str],
                         ncd_calculator: NcdCalculator) -> List[float]:
        self.eval()
        scores = []
        model_device = next(self.parameters()).device

        for cand_text in candidates_text:
            # Prepare feature sequence for current_trajectory + this candidate
            hyp_seq_tensor = self._prepare_hypothetical_sequence_tensor(
                current_trajectory, cand_text, ncd_calculator, model_device
            )
            if hyp_seq_tensor is None:
                scores.append(0.0)  # Low score if sequence prep fails
                continue

            # Encode the hypothetical sequence to get mu
            mu, _ = self.encode(hyp_seq_tensor)  # mu shape: [1, latent_dim]

            # Score: higher if mu is closer to origin (exp(-norm^2))
            norm_mu_sq = torch.norm(mu, p=2).item() ** 2
            score = math.exp(-0.01 * norm_mu_sq)  # Arbitrary scaling factor 0.01, adjust as needed
            scores.append(score)
        return scores


    def _prepare_sequence_tensor(self, traj: "Trajectory") -> torch.Tensor:
        """
        Convert a Trajectory object (already solved so *no* candidate step)
        into a fixed-length feature sequence.

        It simply re-uses the same per-step feature extractor that
        `_prepare_hypothetical_sequence_tensor` employs for ranking,
        but without appending an extra candidate.

        Returns
        -------
        torch.FloatTensor  shape  [self.seq_len, self.feature_dim_in]
        """
        device = next(self.parameters()).device
        if not traj.steps:  # empty trace
            return torch.zeros(self.seq_len,
                               self.feature_dim_in,
                               device=device)

        # ---------------- build feature rows for each *existing* step
        feat_rows = []
        total_steps = len(traj.steps)
        for s in traj.steps:
            raw_glob = s["metadata"].get("ncd_global", 0.0)
            row = [
                *encode_step_type(s["step_type"]),
                float(s["metadata"].get("ncd_local", 0.0)),
                math.log1p(float(raw_glob)),
                (s["step_index"] - 1) / max(1, total_steps - 1)
            ]
            if len(row) < self.feature_dim_in:
                row.extend([0.0] * (self.feature_dim_in - len(row)))
            feat_rows.append(row[: self.feature_dim_in])

        feat = torch.tensor(feat_rows, dtype=torch.float32, device=device)

        # ---------------- truncate / pad to seq_len
        if feat.shape[0] > self.seq_len:  # keep most recent steps
            feat = feat[-self.seq_len:]
        elif feat.shape[0] < self.seq_len:
            pad = torch.zeros(self.seq_len - feat.shape[0],
                              self.feature_dim_in,
                              device=device)
            feat = torch.cat([feat, pad], dim=0)
        return feat


    @torch.no_grad()
    def score_candidates(
            self,
            trajectory,
            candidates_text,
            ncd_calculator=None  # kept optional for forward-compat
    ) -> List[float]:
        """
        Returns one scalar score per candidate (higher = better).

        score  = exp(-γ · ‖μ‖²)
        where μ is the latent mean of the *hypothetical* trajectory
        that would result from appending the candidate step.

        γ is self.gamma (defaults to 0.01).
        """
        import math, torch

        if not candidates_text:  # safety guard
            return []

        gamma = getattr(self, "gamma", 0.01)
        device = next(self.parameters()).device

        # -------- base feature tensor for the current trajectory -------- #
        base_feat = self._prepare_sequence_tensor(trajectory).to(device)
        # how many *non-zero* rows already contain data?
        base_len = int((base_feat.abs().sum(-1) > 0).sum())

        if base_len == 0:  # no steps yet
            base_len = 1
            base_feat[0].zero_()  # ensure row exists

        base_feat_trim = base_feat[:base_len]  # [L₀ ≤ seq_len, F]

        # ---------------------------------------------------------------- #
        scores: List[float] = []
        for cand in candidates_text:
            # Cheap heuristic: longer candidate → slightly higher Δglobal NCD
            delta = torch.zeros(
                1, base_feat.shape[1], device=device, dtype=base_feat.dtype
            )
            delta[0, -1] = len(cand) * 0.01  # mock Δglobal feature

            full_seq = torch.cat([base_feat_trim, delta], dim=0)
            # pad / truncate to fixed seq_len
            if full_seq.shape[0] < self.seq_len:
                pad = torch.zeros(
                    self.seq_len - full_seq.shape[0],
                    full_seq.shape[1],
                    device=device,
                    dtype=full_seq.dtype,
                )
                full_seq = torch.cat([full_seq, pad], dim=0)
            else:
                full_seq = full_seq[-self.seq_len:]

            μ, _ = self.encode(full_seq.unsqueeze(0))  # [1, latent_dim]
            risk = torch.norm(μ, p=2).pow(2).item()  # ‖μ‖²
            scores.append(math.exp(-gamma * risk))

        return scores

    @torch.no_grad()
    def evaluate_trajectory_appropriateness(self, trajectory_data_dict: Dict[str, Any],
                                            ncd_calculator: NcdCalculator) -> Dict[str, Any]:
        self.eval()
        model_device = next(self.parameters()).device

        # Create a temporary Trajectory object from the dict to use its methods for feature prep
        temp_traj_obj = Trajectory(
            trajectory_data_dict.get("problem_id", "N/A"),
            trajectory_data_dict.get("problem_statement", "")
        )
        temp_traj_obj.steps = trajectory_data_dict.get("steps", [])  # Ensure steps are correctly assigned

        # Prepare features for the entire given trajectory (candidate_text = None)
        seq_tensor = self._prepare_hypothetical_sequence_tensor(temp_traj_obj, None, ncd_calculator, model_device)

        if seq_tensor is None:
            return {"overall_score": 0.0, "problematic_steps": [],
                    "guidance": "Error processing trajectory for evaluation."}

        mu, _ = self.encode(seq_tensor)
        norm_mu_sq = torch.norm(mu, p=2).item() ** 2
        score = math.exp(-0.01 * norm_mu_sq)  # Higher score is better

        return {"overall_score": score, "problematic_steps": [],
                "guidance": f"VAE Latent mu norm^2: {norm_mu_sq:.2f}. Appropriateness Score: {score:.3f}."}
