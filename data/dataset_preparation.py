# data/dataset_preparation.py
import json
import os
import re
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union  # Added Union
import torch

# Conditional import for trajectory_processor
if __name__ == '__main__' and __package__ is None:
    import sys

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
    from data.trajectory_processor import process_single_trajectory_for_training, STEP_TYPE_DIM
else:
    from .trajectory_processor import process_single_trajectory_for_training, STEP_TYPE_DIM


# --- GSM8K Loading (kept for reference) ---
def extract_gsm8k_answer(answer_text: str) -> str:
    parts = answer_text.split("####")
    final_numeric_answer = parts[-1].strip().replace(",", "")
    try:
        num = float(final_numeric_answer)
        return str(int(num)) if num.is_integer() else str(num)
    except ValueError:
        return final_numeric_answer


def load_gsm8k_dataset(file_path: str) -> List[Dict[str, Any]]:
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    question, full_ans = data.get("question"), data.get("answer", "")
                    if question and full_ans:
                        dataset.append({
                            "id": f"gsm8k_{os.path.basename(file_path).split('.')[0]}_{i}",
                            "problem": question,
                            "answer": extract_gsm8k_answer(full_ans)
                        })
                except json.JSONDecodeError:
                    print(f"Warn: JSON decode error line {i + 1} in {file_path}.")
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {file_path}")
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")
    return dataset


def load_dataset(file_path: str, dataset_type: Optional[str] = None) -> List[Dict[str, Any]]:
    dt = dataset_type or ("gsm8k" if "gsm8k" in file_path.lower() else "unknown")
    if dt == "gsm8k": return load_gsm8k_dataset(file_path)
    print(f"ERROR: Unknown dataset type '{dt}' for {file_path}.")
    return []


# --- Dataset for Complexity Model Training ---
class TrajectoryFeatureDataset(Dataset):
    def __init__(self,
                 # Can now be a single path (str) or a list of paths (List[str])
                 trajectory_data_sources: Union[str, List[str]],
                 feature_dim: int,
                 max_seq_len: int = 50,
                 only_successful_trajectories: bool = True,
                 limit_trajectories_per_source: Optional[int] = None):  # Renamed for clarity
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.data_points = []

        print(f"Initializing TrajectoryFeatureDataset...")
        if isinstance(trajectory_data_sources, str):
            sources_to_process = [trajectory_data_sources]
            print(f"  Single source provided: {trajectory_data_sources}")
        elif isinstance(trajectory_data_sources, list):
            sources_to_process = trajectory_data_sources
            print(f"  Multiple sources provided ({len(sources_to_process)}):")
            for src in sources_to_process: print(f"    - {src}")
        else:
            raise ValueError("trajectory_data_sources must be a string or a list of strings.")

        print(f"  Feature dim: {feature_dim}, Max seq len: {max_seq_len}")
        print(f"  Only successful trajectories: {only_successful_trajectories}")
        if limit_trajectories_per_source is not None:
            print(f"  Limiting to loading first {limit_trajectories_per_source} trajectories *per source*.")

        total_trajectories_considered = 0
        total_data_points_added = 0

        for source_path in sources_to_process:
            print(f"  Processing source: {source_path}")
            current_source_traj_files = []
            if os.path.isdir(source_path):
                for filename in sorted(os.listdir(source_path)):  # Sort for consistent loading order
                    if filename.endswith(".json") and not filename.startswith("_"):
                        current_source_traj_files.append(os.path.join(source_path, filename))
            elif os.path.isfile(source_path) and source_path.endswith(".jsonl"):
                current_source_traj_files.append(source_path)
            else:
                print(f"Warning: Source path {source_path} is not a valid directory or .jsonl file. Skipping.")
                continue

            print(f"    Found {len(current_source_traj_files)} potential trajectory file(s)/JSONL in this source.")

            trajectories_loaded_from_this_source = 0
            for i, filepath in enumerate(current_source_traj_files):
                if limit_trajectories_per_source is not None and \
                        trajectories_loaded_from_this_source >= limit_trajectories_per_source:
                    print(
                        f"    Reached limit_trajectories_per_source ({limit_trajectories_per_source}) for {source_path}. Moving to next source if any.")
                    break  # Stop processing files from this source

                total_trajectories_considered += 1  # Counts files/jsonl lines attempted before limit

                if filepath.endswith(".jsonl"):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f_jsonl:
                            for line_num, line in enumerate(f_jsonl):
                                if limit_trajectories_per_source is not None and \
                                        trajectories_loaded_from_this_source >= limit_trajectories_per_source:
                                    break  # Stop processing lines from this JSONL
                                try:
                                    traj_data = json.loads(line)
                                    if only_successful_trajectories and not traj_data.get("is_correct_assessment",
                                                                                          False):
                                        continue

                                    processed_traj_steps = process_single_trajectory_for_training(
                                        traj_data, self.feature_dim
                                    )
                                    self.data_points.extend(processed_traj_steps)
                                    if processed_traj_steps:  # Only count if it yielded data points
                                        trajectories_loaded_from_this_source += 1
                                except json.JSONDecodeError:
                                    print(
                                        f"    Warning: JSON decode error in {filepath}, line {line_num + 1}. Skipping.")
                                except Exception as e_proc:
                                    print(
                                        f"    Warning: Error processing line from {filepath}, line {line_num + 1}: {e_proc}.")
                    except Exception as e_file:
                        print(f"    Warning: Could not read/process {filepath}: {e_file}")

                elif filepath.endswith(".json"):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f_json:
                            traj_data = json.load(f_json)

                        if only_successful_trajectories and not traj_data.get("is_correct_assessment", False):
                            continue

                        processed_traj_steps = process_single_trajectory_for_training(
                            traj_data, self.feature_dim
                        )
                        self.data_points.extend(processed_traj_steps)
                        if processed_traj_steps:  # Only count if it yielded data points
                            trajectories_loaded_from_this_source += 1
                    except Exception as e:
                        print(f"    Warning: Could not load/process trajectory {filepath}: {e}")

                if trajectories_loaded_from_this_source > 0 and trajectories_loaded_from_this_source % 100 == 0:
                    print(
                        f"      Processed {trajectories_loaded_from_this_source} trajectories from current source ({source_path})...")

            total_data_points_added = len(self.data_points)
            print(
                f"    Finished processing source {source_path}. Loaded {trajectories_loaded_from_this_source} trajectories from it. Total data points so far: {total_data_points_added}.")

        print(
            f"TrajectoryFeatureDataset initialized. Total training data points: {len(self.data_points)} from {total_trajectories_considered} trajectories considered across all sources.")

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx) -> Dict[str, Any]:
        # ... (the rest of __getitem__ remains the same as the previous version) ...
        data_point = self.data_points[idx]
        feature_seq_list = data_point["feature_sequence"]

        padded_feature_sequence = torch.zeros((self.max_seq_len, self.feature_dim), dtype=torch.float32)
        seq_len = len(feature_seq_list)
        len_to_copy = min(seq_len, self.max_seq_len)
        if len_to_copy > 0:
            try:
                feature_tensor_unpadded = torch.tensor(feature_seq_list, dtype=torch.float32)
                padded_feature_sequence[:len_to_copy, :] = feature_tensor_unpadded[:len_to_copy, :]
            except Exception as e:
                print(
                    f"Error converting feature_seq_list to tensor for data_point at index {idx}, problem_id {data_point.get('problem_id')}. Error: {e}")
                print(f"Feature sequence list was: {feature_seq_list}")

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        padding_mask[:len_to_copy] = False

        next_complexity_target = torch.tensor([
            data_point["next_step_ncd_local_target"],
            data_point["next_step_ncd_global_target"]
        ], dtype=torch.float32)

        trajectory_success_target = torch.tensor([data_point["trajectory_success_target"]], dtype=torch.float32)

        return {
            "sequence": padded_feature_sequence,
            "padding_mask": padding_mask,
            "next_complexity_target": next_complexity_target,
            "trajectory_success_target": trajectory_success_target,
            "problem_id": data_point.get("problem_id", "N/A"),
            "target_step_original_index": data_point.get("target_step_index", -1)
        }


# __main__ test block remains the same
if __name__ == '__main__':
    # ... (your existing __main__ test block) ...
    print("\n--- Testing TrajectoryFeatureDataset (Direct Execution of dataset_preparation.py) ---")

    # Create a dummy trajectory_processor.py if it's not found, for this test only
    try:
        from data.trajectory_processor import process_single_trajectory_for_training, STEP_TYPE_DIM
    except ImportError:
        print("Creating dummy trajectory_processor for test...")
        # ... (dummy tp content as before, or ensure the real one is discoverable) ...
        pass

    dummy_traj_dir_main = "temp_dummy_trajectories_main_test"
    dummy_traj_dir_secondary = "temp_dummy_trajectories_secondary_test"
    os.makedirs(dummy_traj_dir_main, exist_ok=True)
    os.makedirs(dummy_traj_dir_secondary, exist_ok=True)

    EXPECTED_FEATURE_DIM_FOR_PROC = STEP_TYPE_DIM + 3 if 'STEP_TYPE_DIM' in globals() else 6
    print(
        f"Using EXPECTED_FEATURE_DIM_FOR_PROC = {EXPECTED_FEATURE_DIM_FOR_PROC} (STEP_TYPE_DIM={STEP_TYPE_DIM if 'STEP_TYPE_DIM' in globals() else 'Unknown'})")

    dummy_traj_data_1 = {  # Belongs to main source
        "problem_id": "dummy_main_1", "is_correct_assessment": True,
        "steps": [{"step_index": 1, "step_type": "ThoughtAction", "content": "TA1",
                   "metadata": {"ncd_local": 0.5, "ncd_global": 10}},
                  {"step_index": 2, "step_type": "Observation", "content": "O1",
                   "metadata": {"ncd_local": 0.2, "ncd_global": 15}}]
    }
    with open(os.path.join(dummy_traj_dir_main, "traj_main1.json"), "w") as f:
        json.dump(dummy_traj_data_1, f)

    dummy_traj_data_2 = {  # Belongs to secondary source
        "problem_id": "dummy_secondary_1", "is_correct_assessment": True,
        "steps": [{"step_index": 1, "step_type": "ThoughtAction", "content": "TA_sec1",
                   "metadata": {"ncd_local": 0.4, "ncd_global": 12}}]
    }  # This will yield 0 data points as it only has 1 step.
    with open(os.path.join(dummy_traj_dir_secondary, "traj_sec1.json"), "w") as f:
        json.dump(dummy_traj_data_2, f)

    dummy_traj_data_3 = {  # Belongs to secondary source, 2 steps -> 1 data point
        "problem_id": "dummy_secondary_2", "is_correct_assessment": True,
        "steps": [{"step_index": 1, "step_type": "ThoughtAction", "content": "TA_sec2a",
                   "metadata": {"ncd_local": 0.3, "ncd_global": 8}},
                  {"step_index": 2, "step_type": "Observation", "content": "O_sec2",
                   "metadata": {"ncd_local": 0.1, "ncd_global": 10}}]
    }
    with open(os.path.join(dummy_traj_dir_secondary, "traj_sec2.json"), "w") as f:
        json.dump(dummy_traj_data_3, f)

    print("\nLoading from multiple trajectory sources:")
    list_of_sources = [dummy_traj_dir_main, dummy_traj_dir_secondary]
    feature_dataset_multi = TrajectoryFeatureDataset(
        trajectory_data_sources=list_of_sources,
        feature_dim=EXPECTED_FEATURE_DIM_FOR_PROC,
        max_seq_len=5,
        only_successful_trajectories=True,
        limit_trajectories_per_source=None  # Load all from these small dummy dirs
    )
    # Expected data points:
    # traj_main1 (2 steps) -> 1 data point
    # traj_sec1 (1 step) -> 0 data points
    # traj_sec2 (2 steps) -> 1 data point
    # Total = 2 data points
    print(f"Loaded {len(feature_dataset_multi)} data points from multiple sources.")
    assert len(feature_dataset_multi) == 2, "Should load 2 data points from the multi-source dummy data."

    if len(feature_dataset_multi) > 0:
        sample_multi = feature_dataset_multi[0]
        print("Sample from multi-source dataset (first point):")
        for key, val in sample_multi.items():
            print(f"  {key}: {val.shape if isinstance(val, torch.Tensor) else val}")

    import shutil

    if os.path.exists(dummy_traj_dir_main): shutil.rmtree(dummy_traj_dir_main)
    if os.path.exists(dummy_traj_dir_secondary): shutil.rmtree(dummy_traj_dir_secondary)
    print(f"\nCleaned up dummy directories.")