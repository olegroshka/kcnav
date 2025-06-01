#!/usr/bin/env python
# training/train_diffusion_complexity_model.py
from __future__ import annotations
import argparse, yaml, math, time, json, os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# Robust imports for project modules
try:
    from agent.complexity_model_diffusion import ComplexityDiffusionModel
    from data.dataset_preparation import TrajectoryFeatureDataset
    from data.trajectory_processor import STEP_TYPE_DIM
except ImportError:
    print(
        "ERROR: Could not import modules in train_diffusion_complexity_model.py. Ensure PYTHONPATH includes project root.")
    PROJECT_ROOT_FALLBACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT_FALLBACK not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FALLBACK)
    from agent.complexity_model_diffusion import ComplexityDiffusionModel
    from data.dataset_preparation import TrajectoryFeatureDataset
    from data.trajectory_processor import STEP_TYPE_DIM


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl_elementwise = 1 + logvar - mu.pow(2) - logvar.exp()
    return -0.5 * torch.sum(kl_elementwise, dim=-1).mean()


def save_ckpt(model: torch.nn.Module, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"✓ Checkpoint saved → {filepath}")


def collate_fn_for_diffusion_training(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    sequences = torch.stack([item['sequence'] for item in batch])
    # padding_masks = torch.stack([item['padding_mask'] for item in batch]) # Optional
    return {"sequences": sequences}  # "padding_masks": padding_masks


def train_diffusion_model(config: dict):
    device = torch.device("cuda" if config.get('use_gpu', True) and torch.cuda.is_available() else "cpu")
    experiment_name = config.get('experiment_name', 'DiffusionTrain')  # Get experiment name from main config
    print(f"--- Training Complexity VAE+Diffusion Model ({experiment_name}) ---")
    print(f"Using device: {device}")

    trajectory_data_sources = config.get('complexity_training_data_path')
    valid_source_config = False
    if isinstance(trajectory_data_sources, str):
        if os.path.isdir(trajectory_data_sources) or os.path.isfile(trajectory_data_sources): valid_source_config = True
    elif isinstance(trajectory_data_sources, list):
        if trajectory_data_sources and all(
                isinstance(p, str) and (os.path.isdir(p) or os.path.isfile(p)) for p in trajectory_data_sources):
            valid_source_config = True
        elif not trajectory_data_sources:
            print("ERROR: 'complexity_training_data_path' is an empty list.")
        else:
            print("ERROR: Not all items in 'complexity_training_data_path' list are valid paths.")
    if not trajectory_data_sources or not valid_source_config:
        print(f"ERROR: 'complexity_training_data_path' invalid: {trajectory_data_sources}");
        sys.exit(1)

    calculated_vae_input_feature_dim = STEP_TYPE_DIM + 3
    config_vae_input_feat_dim = config.get('diffusion_feature_dim')
    if config_vae_input_feat_dim is None:
        print(
            f"Config 'diffusion_feature_dim' (for VAE input) not set. Using calculated: {calculated_vae_input_feature_dim}")
        config['diffusion_feature_dim'] = calculated_vae_input_feature_dim  # Update in-memory config
    elif config_vae_input_feat_dim != calculated_vae_input_feature_dim:
        print(
            f"Warning: Config 'diffusion_feature_dim' ({config_vae_input_feat_dim}) != calculated VAE input dim ({calculated_vae_input_feature_dim}). Using configured: {config_vae_input_feat_dim}.")
    current_vae_input_feature_dim = config['diffusion_feature_dim']

    print("Loading and processing trajectory data...")
    full_dataset = TrajectoryFeatureDataset(
        trajectory_data_sources=trajectory_data_sources,
        feature_dim=current_vae_input_feature_dim,
        max_seq_len=config.get('diffusion_seq_len', 50),
        only_successful_trajectories=config.get('complexity_train_only_successful', True),
        limit_trajectories_per_source=config.get('limit_trajectories_per_source',
                                                 config.get('complexity_train_limit_trajectories', None))
    )
    if len(full_dataset) == 0: print("ERROR: No data loaded. Exiting."); sys.exit(1)

    val_split_ratio = config.get('complexity_train_val_split_ratio', 0.1)
    num_total, num_val = len(full_dataset), int(val_split_ratio * len(full_dataset))
    num_train = num_total - num_val
    if num_val == 0 and num_total > 10: num_val = max(1, int(0.05 * num_total)); num_train = num_total - num_val
    if num_train <= 0: print(f"ERROR: Not enough training data. Total:{num_total}, Train:{num_train}"); sys.exit(1)

    print(f"Splitting dataset: {num_train} train, {num_val} val samples.")
    train_ds, val_ds = random_split(full_dataset, [num_train, num_val],
                                    generator=torch.Generator().manual_seed(config.get('seed', 42)))

    train_loader = DataLoader(train_ds, batch_size=config['training_batch_size'], shuffle=True,
                              collate_fn=collate_fn_for_diffusion_training,
                              num_workers=config.get('dataloader_num_workers', 0))
    val_loader = DataLoader(val_ds, batch_size=config['training_batch_size'], shuffle=False,
                            collate_fn=collate_fn_for_diffusion_training,
                            num_workers=config.get('dataloader_num_workers', 0)) if num_val > 0 else None

    model = ComplexityDiffusionModel(
        seq_len=config.get('diffusion_seq_len', 50),
        feature_dim=current_vae_input_feature_dim,
        gru_hidden_size=config.get('diffusion_hidden_size', 128),
        latent_dim=config.get('diffusion_latent_dim', 64),
        denoiser_hidden_size=config.get('diffusion_denoiser_hidden_size', config.get('diffusion_hidden_size', 128) * 2),
        diffusion_timesteps=config.get('diffusion_timesteps', 1000),
    ).to(device)
    model.set_diffusion_schedule(config.get('diffusion_beta_start', 0.0001), config.get('diffusion_beta_end', 0.02),
                                 device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training_learning_rate']))

    target_lambda_kl = config.get('diffusion_loss_weight_kl', 1.0)
    lambda_eps = config.get('diffusion_loss_weight_eps', 1.0)
    kl_warmup_epochs = config.get('diffusion_kl_warmup_epochs', 0)

    best_val_loss = math.inf
    output_dir_models = Path(config.get('models_output_dir', 'models'))
    output_dir_models.mkdir(parents=True, exist_ok=True)

    base_save_name = config.get('complexity_model_save_name', "complexity_model.pt")
    if "diffusion" not in base_save_name.lower():
        name_part, ext_part = os.path.splitext(base_save_name)
        best_model_filename = f"{name_part}_diffusion_best{ext_part}"
    elif "_best" not in base_save_name:
        best_model_filename = base_save_name.replace(".pt",
                                                     "_best.pt") if ".pt" in base_save_name else f"{base_save_name}_best.pt"
    else:
        best_model_filename = base_save_name
    best_model_save_path = output_dir_models / best_model_filename

    print(
        f"Starting VAE+Diffusion training for {config['training_num_epochs']} epochs. Best model: {best_model_filename}")
    for epoch in range(1, config['training_num_epochs'] + 1):
        model.train()
        ep_loss, ep_kl, ep_eps = 0.0, 0.0, 0.0
        current_lambda_kl = target_lambda_kl
        if kl_warmup_epochs > 0 and epoch <= kl_warmup_epochs:
            current_lambda_kl = target_lambda_kl * (epoch / kl_warmup_epochs)

        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch}/{config['training_num_epochs']} [Training, KL_w={current_lambda_kl:.4f}]")
        for batch_data in progress_bar:
            sequences = batch_data["sequences"].to(device)
            optimizer.zero_grad()
            model_outputs = model(sequences)
            loss_kl = kl_divergence(model_outputs["mu"], model_outputs["logvar"])
            loss_eps = F.mse_loss(model_outputs["eps_pred"], model_outputs["eps_true"])
            total_loss = (current_lambda_kl * loss_kl) + (lambda_eps * loss_eps)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip_norm', 1.0))
            optimizer.step()
            ep_loss += total_loss.item();
            ep_kl += loss_kl.item();
            ep_eps += loss_eps.item()
            progress_bar.set_postfix(
                {"L": f"{total_loss.item():.4f}", "KL": f"{loss_kl.item():.4f}", "MSE_eps": f"{loss_eps.item():.4f}"})

        len_loader = max(1, len(train_loader))
        print(
            f"Epoch {epoch} Train: Avg Loss={(ep_loss / len_loader):.4f}, Avg KL={(ep_kl / len_loader):.4f} (w={current_lambda_kl:.4f}), Avg MSE_eps={(ep_eps / len_loader):.4f}")

        avg_val_loss_this_epoch = float('inf')
        if val_loader:
            model.eval()
            val_ep_loss, val_ep_kl, val_ep_eps = 0.0, 0.0, 0.0
            with torch.no_grad():
                val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Validation]")
                for batch_data in val_progress_bar:
                    sequences = batch_data["sequences"].to(device)
                    model_outputs = model(sequences)
                    loss_kl = kl_divergence(model_outputs["mu"], model_outputs["logvar"])
                    loss_eps = F.mse_loss(model_outputs["eps_pred"], model_outputs["eps_true"])
                    total_loss = (current_lambda_kl * loss_kl) + (lambda_eps * loss_eps)
                    val_ep_loss += total_loss.item();
                    val_ep_kl += loss_kl.item();
                    val_ep_eps += loss_eps.item()
                    val_progress_bar.set_postfix({"ValL": f"{total_loss.item():.4f}"})
            len_loader = max(1, len(val_loader))
            avg_val_loss_this_epoch = val_ep_loss / len_loader
            print(
                f"Epoch {epoch} Valid: Avg Loss={avg_val_loss_this_epoch:.4f}, Avg KL={(val_ep_kl / len_loader):.4f}, Avg MSE_eps={(val_ep_eps / len_loader):.4f}")
            if avg_val_loss_this_epoch < best_val_loss:
                best_val_loss = avg_val_loss_this_epoch
                save_ckpt(model, best_model_save_path)
                print(f"  Best val_loss: {best_val_loss:.4f}. Model saved: {best_model_save_path.name}")
        else:
            if epoch % config.get("save_every_n_epochs_no_val", 5) == 0 or epoch == config['training_num_epochs']:
                current_epoch_save_name = best_model_save_path.name.replace("_best",
                                                                            f"_epoch{epoch}") if "_best" in best_model_save_path.name else f"{best_model_save_path.name}_epoch{epoch}.pt"
                save_ckpt(model, output_dir_models / current_epoch_save_name)

    final_model_name_suffix = "_final.pt"
    final_model_save_path_str = ""
    if "_best.pt" in best_model_save_path.name:
        final_model_save_path_str = best_model_save_path.name.replace("_best.pt", final_model_name_suffix)
    elif ".pt" in best_model_save_path.name:
        final_model_save_path_str = best_model_save_path.name.replace(".pt", final_model_name_suffix)
    else:
        final_model_save_path_str = f"{best_model_save_path.name}{final_model_name_suffix}"
    final_model_save_path = output_dir_models / final_model_save_path_str

    # Save final model if it was the last epoch and it wasn't the best, or if no validation
    if not val_loader or (val_loader and abs(best_val_loss - avg_val_loss_this_epoch) > 1e-6 and epoch == config[
        'training_num_epochs']):
        save_ckpt(model, final_model_save_path)
        print(f"  Final model state from epoch {epoch} saved as {final_model_save_path.name}.")
    elif val_loader and abs(best_val_loss - avg_val_loss_this_epoch) < 1e-6 and epoch == config['training_num_epochs']:
        if best_model_save_path.name != final_model_save_path.name:
            try:
                os.rename(best_model_save_path, final_model_save_path)
            except OSError:
                save_ckpt(model, final_model_save_path)
            print(f"  Best model from last epoch is now final: {final_model_save_path.name}")
    elif not val_loader and epoch == config['training_num_epochs']:  # Ensure last epoch model is saved if no validation
        save_ckpt(model, final_model_save_path)
        print(f"  Final model state from epoch {epoch} saved as {final_model_save_path.name} (no validation).")

    print("--- Complexity VAE+Diffusion Model Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE+Diffusion Complexity Model")
    parser.add_argument("--config", required=True, help="Path to the main YAML config file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            main_config = yaml.safe_load(f)
        print(f"Loaded main configuration from: {args.config}")
    except Exception as e:
        print(f"FATAL ERROR loading config file {args.config}: {e}"); sys.exit(1)

    if main_config.get('complexity_model_type', '').lower() != "diffusion":
        print(f"Script for 'diffusion' model. Config type: {main_config.get('complexity_model_type')}. Exiting.");
        sys.exit(0)  # Exit gracefully
    if not main_config.get('complexity_training_data_path'):
        print("ERROR: 'complexity_training_data_path' missing.");
        sys.exit(1)

    seed = main_config.get('seed', 42)
    torch.manual_seed(seed);
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    train_diffusion_model(main_config)
