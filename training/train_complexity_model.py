# training/train_complexity_model.py
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
from typing import Dict, Any, Tuple, Optional, List  # For type hints
import numpy as np  # For potential metric calculations
from tqdm import tqdm  # For progress bars

# Adjust import paths - assuming run from project root via main.py
try:
    from agent.complexity_model_transformer import ComplexityTransformer
    # from agent.complexity_model_diffusion import ComplexityDiffusionModel # Keep for future
    from data.dataset_preparation import TrajectoryFeatureDataset  # Our new dataset
except ImportError:
    print("ERROR: Could not import modules in train_complexity_model.py. Ensure PYTHONPATH includes project root.")
    PROJECT_ROOT_FALLBACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT_FALLBACK not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FALLBACK)
    from agent.complexity_model_transformer import ComplexityTransformer
    from data.dataset_preparation import TrajectoryFeatureDataset


def collate_fn_for_training(batch: List[Dict[str, Any]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for the TrajectoryFeatureDataset.
    It takes a list of dictionaries (output of __getitem__) and stacks them into batch tensors.
    """
    sequences = torch.stack([item['sequence'] for item in batch])
    padding_masks = torch.stack([item['padding_mask'] for item in batch])
    next_complexity_targets = torch.stack([item['next_complexity_target'] for item in batch])
    trajectory_success_targets = torch.stack([item['trajectory_success_target'] for item in batch])

    return sequences, padding_masks, next_complexity_targets, trajectory_success_targets


def train_complexity_model(config: dict):
    model_type = config.get('complexity_model_type', 'transformer').lower()
    if model_type != "transformer":
        print(
            f"This training script is currently configured for 'transformer' model type. Found '{model_type}'. Exiting.")
        return

    print(f"--- Training Complexity Transformer ---")
    device = torch.device("cuda" if config.get('use_gpu', True) and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Prepare Data ---
    # Paths to collected trajectory data (JSON or JSONL files)
    # The trajectory_output_dir from Phase 2 config now becomes an input data source.
    # Example: "data/collected_trajectories/NCD_Math_Project_Phase2_DataCollection_Gemma2_9B"
    trajectory_data_source = config.get('complexity_training_data_path')
    if not trajectory_data_source or not (
            os.path.isdir(trajectory_data_source) or os.path.isfile(trajectory_data_source)):
        print(f"ERROR: 'complexity_training_data_path' not specified or invalid in config: {trajectory_data_source}")
        sys.exit(1)

    # Feature dimension must match what trajectory_processor.py and ComplexityTransformer expect.
    # From trajectory_processor: STEP_TYPE_DIM (e.g., 3) + local_ncd (1) + global_ncd (1) + time_step (1) = 6
    # Ensure this matches config['transformer_feature_dim']
    calculated_feature_dim = STEP_TYPE_DIM + 3  # Based on trajectory_processor.py
    if config.get('transformer_feature_dim') != calculated_feature_dim:
        print(f"Warning: Config 'transformer_feature_dim' ({config.get('transformer_feature_dim')}) "
              f"does not match calculated feature dim ({calculated_feature_dim}) from processor. "
              f"Using calculated: {calculated_feature_dim}.")
    current_feature_dim = calculated_feature_dim

    print("Loading and processing trajectory data for training...")
    full_dataset = TrajectoryFeatureDataset(
        trajectory_files_or_dir=trajectory_data_source,
        feature_dim=current_feature_dim,
        max_seq_len=config.get('transformer_max_seq_len', 50),  # Max sequence length for Transformer input
        only_successful_trajectories=config.get('complexity_train_only_successful', True),
        limit_trajectories=config.get('complexity_train_limit_trajectories', None)  # e.g., 1000 for faster initial test
    )

    if len(full_dataset) == 0:
        print("ERROR: No data points loaded by TrajectoryFeatureDataset. Check data path and processing. Exiting.")
        sys.exit(1)

    # Split into training and validation sets
    val_split_ratio = config.get('complexity_train_val_split_ratio', 0.1)
    num_total = len(full_dataset)
    num_val = int(val_split_ratio * num_total)
    num_train = num_total - num_val

    if num_val == 0 and num_total > 5:  # Ensure some validation data if dataset is not tiny
        num_val = max(1, int(0.05 * num_total))  # At least 5% or 1 sample for validation
        num_train = num_total - num_val

    if num_train <= 0:
        print(f"ERROR: Not enough data for training split after validation split. Total: {num_total}, Val: {num_val}")
        sys.exit(1)

    print(f"Splitting dataset: {num_train} training samples, {num_val} validation samples.")
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val],
                                              generator=torch.Generator().manual_seed(config.get('seed', 42)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_batch_size'],
        shuffle=True,
        collate_fn=collate_fn_for_training,
        num_workers=config.get('dataloader_num_workers', 0)  # 0 for main process, >0 for multiprocessing
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training_batch_size'],
        shuffle=False,
        collate_fn=collate_fn_for_training,
        num_workers=config.get('dataloader_num_workers', 0)
    ) if num_val > 0 else None

    # --- Initialize Model ---
    model = ComplexityTransformer(
        feature_dim=current_feature_dim,  # Use the dimension derived from data processing
        hidden_size=config.get('transformer_hidden_size', 128),
        num_layers=config.get('transformer_num_layers', 2),
        nhead=config.get('transformer_nhead', 4),
        dropout_rate=config.get('transformer_dropout_rate', 0.1),
        max_seq_len_for_pos_encoding=config.get('transformer_max_seq_len', 50)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training_learning_rate'])
    # Loss functions
    complexity_loss_fn = torch.nn.MSELoss()  # For NCD regression (2 values)
    success_loss_fn = torch.nn.BCEWithLogitsLoss()  # For trajectory success classification (1 logit)

    # Weighting for combined loss (can be tuned)
    complexity_loss_weight = config.get('complexity_loss_weight', 1.0)
    success_loss_weight = config.get('success_loss_weight', 1.0)

    print(f"Starting training for {config['training_num_epochs']} epochs...")
    best_val_loss = float('inf')

    for epoch in range(config['training_num_epochs']):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_complexity_loss = 0.0
        epoch_train_success_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training_num_epochs']} [Training]")
        for batch_data in progress_bar:
            sequences, padding_masks, next_complexity_targets, traj_success_targets = \
                [d.to(device) for d in batch_data]

            optimizer.zero_grad()

            predicted_next_complexities, success_logit = model(sequences, padding_mask=padding_masks)

            loss_complexity = complexity_loss_fn(predicted_next_complexities, next_complexity_targets)
            loss_success = success_loss_fn(success_logit, traj_success_targets)

            total_loss = (complexity_loss_weight * loss_complexity) + \
                         (success_loss_weight * loss_success)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip_norm', 1.0))  # Gradient clipping
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_train_complexity_loss += loss_complexity.item()
            epoch_train_success_loss += loss_success.item()
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "CmplxL": f"{loss_complexity.item():.4f}",
                "SuccL": f"{loss_success.item():.4f}"
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_complexity_loss = epoch_train_complexity_loss / len(train_loader)
        avg_train_success_loss = epoch_train_success_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Training: Avg Loss={avg_train_loss:.4f}, "
              f"Avg ComplexityLoss={avg_train_complexity_loss:.4f}, Avg SuccessLoss={avg_train_success_loss:.4f}")

        # --- Validation Step ---
        if val_loader:
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_complexity_loss = 0.0
            epoch_val_success_loss = 0.0
            all_success_preds = []
            all_success_targets = []

            with torch.no_grad():
                val_progress_bar = tqdm(val_loader,
                                        desc=f"Epoch {epoch + 1}/{config['training_num_epochs']} [Validation]")
                for batch_data in val_progress_bar:
                    sequences, padding_masks, next_complexity_targets, traj_success_targets = \
                        [d.to(device) for d in batch_data]

                    predicted_next_complexities, success_logit = model(sequences, padding_mask=padding_masks)

                    loss_complexity = complexity_loss_fn(predicted_next_complexities, next_complexity_targets)
                    loss_success = success_loss_fn(success_logit, traj_success_targets)
                    total_loss = (complexity_loss_weight * loss_complexity) + \
                                 (success_loss_weight * loss_success)

                    epoch_val_loss += total_loss.item()
                    epoch_val_complexity_loss += loss_complexity.item()
                    epoch_val_success_loss += loss_success.item()

                    all_success_preds.extend(torch.sigmoid(success_logit).cpu().numpy())
                    all_success_targets.extend(traj_success_targets.cpu().numpy())
                    val_progress_bar.set_postfix({"ValLoss": f"{total_loss.item():.4f}"})

            avg_val_loss = epoch_val_loss / len(val_loader)
            avg_val_complexity_loss = epoch_val_complexity_loss / len(val_loader)
            avg_val_success_loss = epoch_val_success_loss / len(val_loader)

            # Calculate success prediction accuracy (example metric)
            success_preds_binary = (np.array(all_success_preds) > 0.5).astype(int)
            success_targets_binary = np.array(all_success_targets).astype(int)
            val_success_accuracy = np.mean(success_preds_binary == success_targets_binary) * 100 if len(
                success_targets_binary) > 0 else 0.0

            print(f"Epoch {epoch + 1} Validation: Avg Loss={avg_val_loss:.4f}, "
                  f"Avg CmplxLoss={avg_val_complexity_loss:.4f}, Avg SuccLoss={avg_val_success_loss:.4f}, "
                  f"SuccAcc={val_success_accuracy:.2f}%")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # --- Save Best Model ---
                output_dir_models = config.get('models_output_dir', 'models')
                os.makedirs(output_dir_models, exist_ok=True)
                model_save_name = config.get('complexity_model_save_name', f"{model_type}_complexity_model_best.pt")
                save_path = os.path.join(output_dir_models, model_save_name)
                try:
                    torch.save(model.state_dict(), save_path)
                    print(f"  Best validation loss: {best_val_loss:.4f}. Model saved to {save_path}")
                except Exception as e:
                    print(f"  ERROR saving best model: {e}")
        else:  # No validation loader
            # --- Save Model after each epoch if no validation ---
            output_dir_models = config.get('models_output_dir', 'models')
            os.makedirs(output_dir_models, exist_ok=True)
            model_save_name = config.get('complexity_model_save_name',
                                         f"{model_type}_complexity_model_epoch{epoch + 1}.pt")
            save_path = os.path.join(output_dir_models, model_save_name)  # Save epoch-specific model
            try:
                torch.save(model.state_dict(), save_path)
                print(f"  Model after epoch {epoch + 1} saved to {save_path}")
            except Exception as e:
                print(f"  ERROR saving model: {e}")

    print("--- Complexity Model Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Complexity Transformer Model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration for complexity model training from {args.config}")
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading config file {args.config}: {e}")
        sys.exit(1)

    # Validate essential config for this script
    if config.get('complexity_model_type', 'transformer').lower() != "transformer":
        print(
            "This script is for training the 'transformer' complexity model. Update config or use a different script.")
    elif not config.get('complexity_training_data_path'):
        print("ERROR: 'complexity_training_data_path' (path to collected trajectories) must be specified in config.")
    else:
        # Set random seed for reproducibility
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)  # For numpy used in metrics
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_complexity_model(config)